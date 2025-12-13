"""
WeLT Trainer for generation-based evaluation.

Minimal extension of Trainer that adds support for generation-based metrics
by overriding prediction_step to do generation, and using compute_metrics callback.
"""
import logging
import math

import evaluate
import torch
from transformers import GenerationConfig, Trainer

from welt.processor import TextImageProcessor

logger = logging.getLogger(__name__)


class WeLTTrainer(Trainer):
    """
    Minimal trainer extension for WeLT generation-based evaluation.

    Uses standard Trainer evaluation flow:
    1. Override prediction_step to generate text predictions
    2. Use compute_metrics callback for metric computation
    3. All logging, callbacks, progress bars work automatically

    Expected dataset format for generation-based evaluation:
    - prefix: Text to use as input for generation
    - completion: Gold reference text for metric computation

    Note: Generation-based evaluation requires predict_with_generate=True
    in training arguments. If False, only loss-based metrics will be computed.
    """

    def __init__(
        self,
        processor: TextImageProcessor,
        eval_metrics: list[str] | None = None,
        max_generated_words: int = 50,
        bytes_generation_config: GenerationConfig | None = None,
        log_samples: int = 3,
        **kwargs
    ):
        """
        Initialize WeLTTrainer.

        Args:
            processor: TextImageProcessor for tokenization and image rendering
            eval_metrics: List of metric names to load (e.g., ["bleu", "rouge"])
            max_generated_words: Maximum words to generate during evaluation
            bytes_generation_config: Optional GenerationConfig for bytes decoder (e.g., beam search)
            log_samples: Number of prediction samples to log (0 to disable)
            **kwargs: Additional arguments passed to Trainer
        """
        # Reserve compute_metrics slot - we'll set it after loading metrics
        if "compute_metrics" not in kwargs and eval_metrics:
            kwargs["compute_metrics"] = None

        super().__init__(**kwargs)

        self.processor = processor
        self.max_generated_words = max_generated_words
        self.bytes_generation_config = bytes_generation_config
        self.log_samples = log_samples

        # Configure trainer to handle our custom dataset columns
        self.args.label_names = ["labels_output"]

        # Load evaluation metrics
        self.loaded_metrics = {}
        if eval_metrics:
            for metric_name in eval_metrics:
                try:
                    self.loaded_metrics[metric_name] = evaluate.load(metric_name)
                    logger.info(f"Loaded metric: {metric_name}")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to load metric '{metric_name}': {e}")

        # Set compute_metrics callback if we loaded any metrics
        if self.loaded_metrics and self.compute_metrics is None:
            self.compute_metrics = self._compute_metrics_callback

        # Warn if metrics are loaded but predict_with_generate is disabled
        if self.loaded_metrics and not self.args.predict_with_generate:
            logger.warning(
                "eval_metrics are provided but predict_with_generate=False. "
                "Generation-based metrics will not be computed. "
                "Set predict_with_generate=True to enable generation-based evaluation."
            )

        # Track samples for logging
        self._logged_samples_this_eval = False

        # Store predictions and labels across batches
        self._eval_predictions = []
        self._eval_labels = []
        self._eval_sample_count = 0

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to generate text predictions for evaluation.

        This method is called by Trainer.evaluation_loop for each batch.
        We generate text predictions and store them for later metric computation.
        """
        # Extract custom fields not needed for loss computation
        prefixes = inputs.pop("prefix", None)
        completions = inputs.pop("completion", None)
        inputs.pop("text", None)

        # Count samples in this batch
        if prefixes is not None:
            self._eval_sample_count += len(prefixes)

        # Move inputs to device
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}

        # Compute loss
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

            # Handle NaN/Inf losses
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Encountered {'NaN' if torch.isnan(loss) else 'Inf'} loss in batch")
                loss = torch.tensor(0.0, device=model.device)

        # Generate predictions if predict_with_generate is enabled
        # Only do generation when: (1) predict_with_generate is True, (2) we have prefixes,
        # (3) and either we have metrics or prediction_loss_only is False
        predictions_text = []
        should_generate = (
            self.args.predict_with_generate
            and prefixes is not None
            and (self.loaded_metrics or not prediction_loss_only)
        )

        if should_generate:
            with torch.no_grad():
                # Process prefixes for generation
                generation_inputs = self.processor(prefixes, collated=True, packed=False)
                generation_inputs = {
                    k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in generation_inputs.items()
                }

                generation_kwargs = {
                    "processor": self.processor,
                    "max_generated_words": self.max_generated_words,
                }
                if self.bytes_generation_config is not None:
                    generation_kwargs["bytes_generation_config"] = self.bytes_generation_config

                predictions_text = model.generate(**generation_inputs, **generation_kwargs)

                # Store predictions and labels for compute_metrics
                self._eval_predictions.extend(predictions_text)
                if completions is not None:
                    self._eval_labels.extend(completions)

        # Log samples once per evaluation
        if not self._logged_samples_this_eval and predictions_text and self.log_samples > 0:
            self._log_samples(predictions_text, prefixes, completions)
            self._logged_samples_this_eval = True

        # Return loss; predictions/labels are stored in instance variables for compute_metrics
        # Return None to avoid Trainer's automatic tensor gathering (not compatible with strings)
        return (loss, None, None)

    def evaluate(self, eval_dataset=None, **kwargs):
        """
        Override evaluate to add generation metrics and perplexity.

        Calls parent evaluate (which uses prediction_step), then computes
        generation metrics from stored predictions and logs all metrics.
        """
        # Reset state
        self._logged_samples_this_eval = False
        self._eval_predictions = []
        self._eval_labels = []
        self._eval_sample_count = 0

        # Validate dataset
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is not None:
            self._validate_eval_dataset(eval_dataset)

        # Apply processor transform if needed
        if eval_dataset is not None:
            if not hasattr(eval_dataset, '_transforms') or eval_dataset._transforms is None:
                eval_dataset = eval_dataset.with_transform(self.processor)

        # Call parent evaluate - this handles loss computation and logging
        metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)

        # Track if we added any new metrics
        added_metrics = False

        # Compute generation metrics from stored predictions
        if self._eval_predictions and self._eval_labels and self.loaded_metrics:
            generation_metrics = self._compute_generation_metrics(
                self._eval_predictions,
                self._eval_labels
            )
            # Add generation metrics with eval_ prefix
            for key, value in generation_metrics.items():
                metrics[f"eval_{key}"] = value
            added_metrics = True

        # Add perplexity if we have loss
        if "eval_loss" in metrics:
            loss = metrics["eval_loss"]
            perplexity = math.exp(loss) if loss < 100 else float('inf')
            metrics["perplexity"] = perplexity
            added_metrics = True

        # Add eval_samples count
        if self._eval_sample_count > 0:
            metrics["eval_samples"] = self._eval_sample_count
        elif self._eval_predictions:
            metrics["eval_samples"] = len(self._eval_predictions)

        # CRITICAL: Log the additional metrics we computed
        # The parent's evaluate() already logged its metrics, but we added more
        # So we need to log them explicitly for them to appear in wandb/terminal
        if added_metrics and self.args.do_train:
            self.log(metrics)

        return metrics

    def _compute_metrics_callback(self, eval_preds):
        """
        Compute metrics callback for Trainer.

        Note: We don't use this directly since predictions are strings stored
        in instance variables, not tensors. Metrics are computed in evaluate().
        This is just a placeholder to satisfy the Trainer interface.
        """
        return {}

    def _log_samples(self, predictions, prefixes, completions):
        """Log sample predictions."""
        print("\n" + "="*60)
        print("Sample predictions:")
        print("="*60)
        for i in range(min(self.log_samples, len(predictions))):
            print(f"  Input: {prefixes[i] if prefixes else 'N/A'}")
            print(f"  Generated: {predictions[i]}")
            if completions and i < len(completions):
                print(f"  Reference: {completions[i]}")
            print()  # Empty line between samples

    def _validate_eval_dataset(self, eval_dataset):
        """Validate that eval dataset has required columns."""
        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided")  # noqa: TRY003

        if hasattr(eval_dataset, "features") and "prefix" not in eval_dataset.features:
            raise ValueError(  # noqa: TRY003
                "Evaluation dataset must have 'prefix' column for generation. "
                f"Found columns: {list(eval_dataset.features.keys())}"
            )

    def _compute_generation_metrics(
        self,
        predictions: list[str],
        references: list[str]
    ) -> dict[str, float]:
        """Compute generation-based metrics."""
        metrics = {}

        for metric_name, metric in self.loaded_metrics.items():
            try:
                # Try standard format first (flat list for ROUGE, CER, etc.)
                try:
                    result = metric.compute(predictions=predictions, references=references)
                except (ValueError, TypeError, KeyError):
                    # Try list-of-lists format (required by BLEU, SacreBLEU, etc.)
                    result = metric.compute(predictions=predictions, references=[[ref] for ref in references])

                # Extract scalar metric from result
                if isinstance(result, dict):
                    if "score" in result:
                        metrics[metric_name] = result["score"]
                    elif metric_name in result:
                        metrics[metric_name] = result[metric_name]
                    else:
                        # Use first numeric value found
                        for value in result.values():
                            if isinstance(value, int | float):
                                metrics[metric_name] = value
                                break
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to compute metric '{metric_name}': {e}")

        return metrics
