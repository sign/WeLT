"""
WeLT Trainer for generation-based evaluation.

Minimal extension of Trainer that adds support for generation-based metrics.
"""
import logging

import evaluate
import torch
from tqdm import tqdm
from transformers import GenerationConfig, Trainer

from welt.processor import TextImageProcessor

logger = logging.getLogger(__name__)


class WeLTTrainer(Trainer):
    """
    Minimal trainer extension for WeLT generation-based evaluation.

    Expected dataset format for generation-based evaluation:
    - prefix: Text to use as input for generation
    - completion: Gold reference text for metric computation
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
            **kwargs: Additional arguments passed to Seq2SeqTrainer
        """
        super().__init__(**kwargs)
        self.processor = processor
        self.max_generated_words = max_generated_words
        self.bytes_generation_config = bytes_generation_config
        self.log_samples = log_samples

        # Configure trainer to handle our custom dataset columns
        # Tell trainer the correct name of our labels output column
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

    def evaluate(self, eval_dataset=None, **kwargs):
        """
        Override evaluate to compute both standard eval_loss and generation-based metrics.

        This evaluation computes:
        1. eval_loss via the standard forward pass (manual computation)
        2. Generation-based metrics like BLEU, ROUGE, etc. (custom logic)
        """
        eval_dataset = eval_dataset or self.eval_dataset
        self._validate_eval_dataset(eval_dataset)

        # Apply processor transform once to avoid duplicate processing
        if not hasattr(eval_dataset, '_transforms') or eval_dataset._transforms is None:
            eval_dataset = eval_dataset.with_transform(self.processor)

        # Compute loss manually
        loss_metrics = self._compute_eval_loss(eval_dataset)

        # Generate predictions and compute generation-based metrics
        all_predictions, all_references, all_prefixes = self._generate_predictions(eval_dataset)

        # Log sample predictions
        self._log_sample_predictions(all_predictions, all_prefixes, all_references)

        # Compute generation-based metrics
        generation_metrics = self._compute_and_format_metrics(all_predictions, all_references)

        # Merge both sets of metrics
        return {**loss_metrics, **generation_metrics}

    def _compute_eval_loss(self, eval_dataset):
        """Compute evaluation loss manually."""
        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(eval_dataloader, desc="Computing loss", disable=not self.args.local_rank <= 0):
            # Remove columns that aren't model inputs (but keep processed model inputs)
            batch.pop("prefix", None)
            batch.pop("completion", None)
            batch.pop("text", None)

            with torch.no_grad():
                # Move batch to device
                batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

                # Determine batch size from first tensor in batch
                first_key = list(batch.keys())[0]
                first_value = batch[first_key]
                batch_size = (
                    first_value.size(0) if isinstance(first_value, torch.Tensor)
                    else len(first_value)
                )
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {"eval_loss": avg_loss}

    def _validate_eval_dataset(self, eval_dataset):
        """Validate that eval dataset has required columns."""
        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided")  # noqa: TRY003

        if hasattr(eval_dataset, "features") and "prefix" not in eval_dataset.features:
            raise ValueError(  # noqa: TRY003
                "Evaluation dataset must have 'prefix' column for generation. "
                f"Found columns: {list(eval_dataset.features.keys())}"
            )

    def _generate_predictions(self, eval_dataset):
        """Generate predictions for all examples in the dataset."""
        all_predictions = []
        all_references = []
        all_prefixes = []

        self.model.eval()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not self.args.local_rank <= 0):
            prefixes = batch.pop("prefix", None)
            completions = batch.pop("completion", None)

            if prefixes is None:
                continue

            with torch.no_grad():
                inputs = self.processor(prefixes, collated=True, packed=False)
                inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                         for k, v in inputs.items()}

                generation_kwargs = {
                    "processor": self.processor,
                    "max_generated_words": self.max_generated_words,
                }
                if self.bytes_generation_config is not None:
                    generation_kwargs["bytes_generation_config"] = self.bytes_generation_config

                predictions = self.model.generate(**inputs, **generation_kwargs)

                all_predictions.extend(predictions)
                all_prefixes.extend(prefixes)
                if completions is not None:
                    all_references.extend(completions)

        return all_predictions, all_references, all_prefixes

    def _log_sample_predictions(self, all_predictions, all_prefixes, all_references):
        """Log a few sample predictions for debugging."""
        if self.log_samples > 0 and all_predictions:
            logger.info("Sample predictions:")
            for i in range(min(self.log_samples, len(all_predictions))):
                logger.info(f"  Input: {all_prefixes[i]}")
                logger.info(f"  Generated: {all_predictions[i]}")
                if i < len(all_references):
                    logger.info(f"  Reference: {all_references[i]}")

    def _compute_and_format_metrics(self, all_predictions, all_references):
        """Compute metrics and format with eval_ prefix."""
        metrics = {}
        if all_references:
            metrics = self._compute_generation_metrics(all_predictions, all_references)
            logger.info(f"Generation metrics: {metrics}")
        else:
            logger.warning("No references found in dataset, skipping metric computation")

        return {f"eval_{k}": v for k, v in metrics.items()}

    def _compute_generation_metrics(
        self,
        predictions: list[str],
        references: list[str]
    ) -> dict[str, float]:
        """Compute generation-based metrics."""
        metrics = {}

        for metric_name, metric in self.loaded_metrics.items():
            try:
                # Try standard format first (flat list)
                # Some metrics (e.g., ROUGE, CER) expect flat lists
                try:
                    result = metric.compute(predictions=predictions, references=references)
                except (ValueError, TypeError, KeyError):
                    # If standard format fails, try list-of-lists format
                    # Some metrics (e.g., BLEU, SacreBLEU) expect references as list of lists
                    result = metric.compute(predictions=predictions, references=[[ref] for ref in references])

                # Extract scalar metrics from result
                if isinstance(result, dict):
                    # Try to find the main score
                    if "score" in result:
                        metrics[metric_name] = result["score"]
                    elif metric_name in result:
                        metrics[metric_name] = result[metric_name]
                    else:
                        # Use first numeric value
                        for _key, value in result.items():
                            if isinstance(value, int | float):
                                metrics[metric_name] = value
                                break

            except Exception as e:  # noqa: BLE001
                logger.warning(f"Failed to compute metric '{metric_name}': {e}")

        return metrics
