"""
WeLT Trainer for generation-based evaluation with accuracy metrics.

Minimal extension of Trainer that adds support for:
- Generation-based metrics (BLEU, ROUGE, SacreBLEU, ChrF, etc.)
- Byte-level accuracy (token-level) and word-level accuracy from logits
- Perplexity computation from loss

Overrides prediction_step to generate text predictions and store logits,
then computes all metrics in evaluate().
"""
import logging
import math

import evaluate
import torch
from torch.optim import AdamW
from transformers import GenerationConfig, Trainer

from welt.processor import TextImageProcessor

logger = logging.getLogger(__name__)


class WeLTTrainer(Trainer):
    """
    Minimal trainer extension for WeLT generation-based evaluation.

    Evaluation flow:
    1. Override prediction_step to:
       - Generate text predictions (if predict_with_generate=True)
       - Store logits and labels for accuracy computation
    2. Override evaluate to compute:
       - Generation-based metrics (BLEU, ROUGE, SacreBLEU, ChrF, etc.)
       - Byte-level and word-level accuracy from logits
       - Perplexity from loss
    3. All logging, callbacks, progress bars work automatically

    Expected dataset format for generation-based evaluation:
    - prefix: Text to use as input for generation
    - completion: Gold reference text for metric computation

    Computed metrics:
    - eval_loss: Cross-entropy loss
    - eval_byte_accuracy: Token/byte-level accuracy (always computed)
    - eval_word_accuracy: Word-level accuracy - all tokens in word must be correct (always computed)
    - eval_{metric}: Generation metrics (e.g., eval_sacrebleu, eval_chrf)
    - perplexity: exp(loss)

    Note: Generation-based evaluation requires predict_with_generate=True.
    If False, only loss and accuracy metrics will be computed.
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
            eval_metrics: List of generation metric names to load (e.g., ["sacrebleu", "chrf"])
                          Note: byte and word accuracy are always computed from logits automatically
            max_generated_words: Maximum words to generate during evaluation
            bytes_generation_config: Optional GenerationConfig for bytes decoder (e.g., beam search)
            log_samples: Number of prediction samples to log (0 to disable)
            **kwargs: Additional arguments passed to Trainer

        Raises:
            ValueError: If compute_metrics is provided (not supported)
        """
        # WeLTTrainer computes metrics internally - don't allow external compute_metrics
        if "compute_metrics" in kwargs:
            msg = (
                "compute_metrics is not supported for WeLTTrainer. "
                "Use eval_metrics parameter instead for generation metrics. "
                "Accuracy is computed automatically from logits."
            )
            raise ValueError(msg)

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
                    print(f"Loaded metric: {metric_name}")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Failed to load metric '{metric_name}': {e}")

        # Warn if metrics are loaded but predict_with_generate is disabled
        if self.loaded_metrics and not self.args.predict_with_generate:
            logger.warning(
                "eval_metrics are provided but predict_with_generate=False. "
                "Generation-based metrics will not be computed. "
                "Set predict_with_generate=True to enable generation-based evaluation."
            )

        # Initialize evaluation state
        self._reset_eval_state()

    def _reset_eval_state(self):
        """Reset evaluation state before each evaluation run."""
        self._logged_samples_this_eval = False
        self._eval_predictions = []
        self._eval_labels = []
        self._eval_sample_count = 0
        self._eval_logits = []
        self._eval_labels_for_accuracy = []

    def create_optimizer(self):
        """
        Create optimizer with 4 parameter groups (one per component).

        This improves optimization dynamics by isolating Adam state per component,
        resulting in ~3x better loss compared to the default optimizer.

        Hypothesis why does this work?
        With 4 param groups, Adam's momentum (m) and variance (v) estimates are computed separately per-group.
        The model has 4 distinct components:
        - **bytes_encoder** (529K params): Sees ~4096 samples/batch (32 batch * 128 words)
        - **latent_transformer** (3M params): Sees only 32 samples/batch
        - **bytes_decoder** (3.2M params): Sees ~4096 samples/batch
        - **mapping_layers** (99K params): Bridge layers

        The encoder/decoder process 128x more samples per optimizer step than the transformer.
        In a single param group, their gradient statistics may dominate Adam's m/v estimates,
        causing the transformer to receive suboptimal updates.
        """
        if self.optimizer is not None:
            return self.optimizer

        args = self.args

        # Validate optimizer type
        if "adamw" not in args.optim.lower():
            raise ValueError(
                f"WeLTTrainer only supports AdamW optimizer variants. "
                f"Got optim='{args.optim}'. Use 'adamw_torch' or 'adamw_torch_fused'."
            )

        # Build parameter groups by component
        model = self.model
        param_groups = []

        # bytes_encoder parameters
        if hasattr(model, 'bytes_encoder') and model.bytes_encoder is not None:
            encoder_params = [p for p in model.bytes_encoder.parameters() if p.requires_grad]
            if encoder_params:
                param_groups.append({
                    'params': encoder_params,
                    'lr': args.learning_rate,
                    'name': 'bytes_encoder'
                })

        # latent_transformer parameters
        if hasattr(model, 'latent_transformer') and model.latent_transformer is not None:
            transformer_params = [p for p in model.latent_transformer.parameters() if p.requires_grad]
            if transformer_params:
                param_groups.append({
                    'params': transformer_params,
                    'lr': args.learning_rate,
                    'name': 'latent_transformer'
                })

        # bytes_decoder parameters
        if hasattr(model, 'bytes_decoder') and model.bytes_decoder is not None:
            decoder_params = [p for p in model.bytes_decoder.parameters() if p.requires_grad]
            if decoder_params:
                param_groups.append({
                    'params': decoder_params,
                    'lr': args.learning_rate,
                    'name': 'bytes_decoder'
                })

        # Mapping layers
        mapping_params = []
        if hasattr(model, 'encoder_mapping'):
            mapping_params.extend([p for p in model.encoder_mapping.parameters() if p.requires_grad])
        if hasattr(model, 'decoder_mapping'):
            mapping_params.extend([p for p in model.decoder_mapping.parameters() if p.requires_grad])
        if mapping_params:
            param_groups.append({
                'params': mapping_params,
                'lr': args.learning_rate,
                'name': 'mapping_layers'
            })

        # Log the configuration
        print(f"\n{'=' * 60}")
        print("Component-specific parameter groups:")
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            print(f"  {group['name']}: lr={group['lr']:.2e}, params={num_params:,}")
        print(f"{'=' * 60}\n")

        # Create optimizer
        optimizer_kwargs = {
            'betas': (args.adam_beta1, args.adam_beta2),
            'eps': args.adam_epsilon,
            'weight_decay': args.weight_decay,
        }
        if "fused" in args.optim and torch.cuda.is_available():
            optimizer_kwargs['fused'] = True

        self.optimizer = AdamW(param_groups, **optimizer_kwargs)
        return self.optimizer

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to generate predictions and store data for metrics.

        This method is called by Trainer.evaluation_loop for each batch.
        It performs three operations:
        1. Computes loss from model outputs
        2. Stores logits and labels for accuracy computation
        3. Generates text predictions (if predict_with_generate=True) for generation metrics

        Returns:
            tuple: (loss, None, None) - predictions/labels stored in instance variables
        """
        # Extract custom fields and create model inputs (without mutating original batch)
        prefixes = inputs.get("prefix", None)
        completions = inputs.get("completion", None)

        # Count samples in this batch
        if prefixes is not None:
            self._eval_sample_count += len(prefixes)

        # Create model inputs without custom fields
        model_inputs = {
            k: v.to(model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
            if k not in ("prefix", "completion", "text")
        }

        # Compute loss and store logits for accuracy
        with torch.no_grad():
            outputs = model(**model_inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

            # Handle NaN/Inf losses
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Encountered {'NaN' if torch.isnan(loss) else 'Inf'} loss in batch")
                loss = torch.tensor(0.0, device=model.device)

            # Store logits for accuracy computation (always during evaluation)
            # Extract logits if available
            if hasattr(outputs, "logits") or "logits" in outputs:
                logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
                # Get predicted token IDs (argmax over vocabulary)
                pred_token_ids = logits.argmax(dim=-1)  # (batch, seq_len, vocab) -> (batch, seq_len)

                # Store predictions and labels for accuracy
                labels_output = inputs.get("labels_output")
                if labels_output is not None:
                    self._eval_logits.append(pred_token_ids.cpu())
                    self._eval_labels_for_accuracy.append(labels_output.cpu())

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

                # Store predictions and labels for generation metrics computation
                self._eval_predictions.extend(predictions_text)
                if completions is not None:
                    self._eval_labels.extend(completions)

        # Log samples once per evaluation
        if not self._logged_samples_this_eval and predictions_text and self.log_samples > 0:
            self._log_samples(predictions_text, prefixes, completions)
            self._logged_samples_this_eval = True

        # Return loss; predictions/labels stored in instance variables for evaluate()
        # Return None to avoid Trainer's automatic tensor gathering (not compatible with strings)
        return (loss, None, None)

    def evaluate(self, eval_dataset=None, **kwargs):
        """
        Override evaluate to compute accuracy, generation metrics, and perplexity.

        Workflow:
        1. Calls parent evaluate() which calls prediction_step() for each batch
        2. Computes generation metrics from stored string predictions
        3. Computes token-level accuracy from stored logits
        4. Computes perplexity from loss
        5. Returns all metrics (parent's callback system handles logging)

        Returns:
            dict: Metrics including eval_loss, eval_accuracy, eval_{metric}, perplexity
        """
        # Reset evaluation state
        self._reset_eval_state()

        # Prepare evaluation dataset
        eval_dataset = self._prepare_eval_dataset(eval_dataset)

        # Call parent evaluate - this handles loss computation and logging
        metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)

        # Add custom metrics (generation, accuracy, perplexity)
        additional_metrics = self._add_custom_metrics(metrics)

        # Log only the additional metrics we computed (not all metrics)
        # This allows them to appear in wandb and progress bars without creating
        # duplicate log entries that would break model card generation
        if additional_metrics and self.args.do_train:
            self.log(additional_metrics)

        return metrics

    def _prepare_eval_dataset(self, eval_dataset):
        """Prepare and validate evaluation dataset."""
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is not None:
            self._validate_eval_dataset(eval_dataset)
            # Apply processor transform if needed
            if not hasattr(eval_dataset, '_transforms') or eval_dataset._transforms is None:
                eval_dataset = eval_dataset.with_transform(self.processor)
        return eval_dataset

    def _add_custom_metrics(self, metrics):
        """Add custom metrics (generation, accuracy, perplexity) to metrics dict."""
        additional_metrics = {}

        # Compute generation metrics from stored predictions
        if self._eval_predictions and self._eval_labels and self.loaded_metrics:
            generation_metrics = self._compute_generation_metrics(
                self._eval_predictions,
                self._eval_labels
            )
            # Add generation metrics with eval_ prefix
            for key, value in generation_metrics.items():
                metric_key = f"eval_{key}"
                metrics[metric_key] = value
                additional_metrics[metric_key] = value

        # Add perplexity if we have loss
        if "eval_loss" in metrics:
            loss = metrics["eval_loss"]
            # Use 709 as threshold to avoid float overflow; exp(709) ~ 8.2e307 is the largest representable float
            perplexity = math.exp(loss) if loss < 709 else float('inf')
            metrics["perplexity"] = perplexity
            additional_metrics["perplexity"] = perplexity

        # Compute byte and word accuracy from stored logits
        if self._eval_logits and self._eval_labels_for_accuracy:
            accuracy_metrics = self._compute_accuracy(
                self._eval_logits,
                self._eval_labels_for_accuracy
            )
            metrics["eval_byte_accuracy"] = accuracy_metrics["byte_accuracy"]
            metrics["eval_word_accuracy"] = accuracy_metrics["word_accuracy"]
            additional_metrics["eval_byte_accuracy"] = accuracy_metrics["byte_accuracy"]
            additional_metrics["eval_word_accuracy"] = accuracy_metrics["word_accuracy"]

        # Add eval_samples count
        if self._eval_sample_count > 0:
            metrics["eval_samples"] = self._eval_sample_count
        elif self._eval_predictions:
            metrics["eval_samples"] = len(self._eval_predictions)

        return additional_metrics

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
            print("")  # Empty line between samples

    def _validate_eval_dataset(self, eval_dataset):
        """Validate that eval dataset has required columns."""
        if eval_dataset is None:
            raise ValueError("No evaluation dataset provided")  # noqa: TRY003

        # Check for 'prefix' column based on dataset type
        if hasattr(eval_dataset, "features"):
            # Standard Dataset with features
            if "prefix" not in eval_dataset.features:
                raise ValueError(  # noqa: TRY003
                    "Evaluation dataset must have 'prefix' column for generation. "
                    f"Found columns: {list(eval_dataset.features.keys())}"
                )
        else:
            # IterableDataset or transformed dataset - check first element
            try:
                # Try __getitem__ first (for indexable datasets)
                first = eval_dataset[0]
            except Exception:  # noqa: BLE001
                try:
                    # Try iter (for IterableDataset)
                    first = next(iter(eval_dataset))
                except Exception:  # noqa: BLE001
                    # Cannot validate - skip check
                    logger.warning(
                        "Cannot validate evaluation dataset: unable to access first element "
                        "to check for 'prefix' column. Ensure your dataset has 'prefix' column "
                        "for generation-based evaluation."
                    )
                    return

            if not isinstance(first, dict) or "prefix" not in first:
                raise ValueError(  # noqa: TRY003
                    "Evaluation dataset must have 'prefix' column for generation. "
                    f"First element: {first}"
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

    def _compute_accuracy(
        self,
        pred_token_ids: list[torch.Tensor],
        labels: list[torch.Tensor]
    ) -> dict[str, float]:
        """
        Compute byte-level and word-level accuracy from predictions and labels.

        Byte accuracy: Percentage of correctly predicted tokens
        Word accuracy: Percentage of words where ALL tokens are correctly predicted

        Args:
            pred_token_ids: List of predicted token ID tensors, shape (batch_size, num_words, tokens_per_word)
            labels: List of label tensors, shape (batch_size, num_words, tokens_per_word)

        Returns:
            dict with 'byte_accuracy' and 'word_accuracy' as floats between 0 and 1
        """
        pad_token_id = self.processor.tokenizer.pad_token_id

        total_bytes = 0
        correct_bytes = 0
        total_words = 0
        correct_words = 0

        # Process each batch separately (can't concatenate if tokens_per_word differs between batches)
        # but vectorize operations within each batch for efficiency
        for preds, label in zip(pred_token_ids, labels, strict=False):
            # preds/label shape: (batch_size, num_words, tokens_per_word)
            # Mask for non-padding tokens
            non_pad_mask = label != pad_token_id  # (batch_size, num_words, tokens_per_word)

            # Byte-level: count matching non-padding tokens across entire batch
            byte_matches = (preds == label) & non_pad_mask
            correct_bytes += byte_matches.sum().item()
            total_bytes += non_pad_mask.sum().item()

            # Word-level: a word is correct if all its non-padding tokens are correct
            word_nonpad = non_pad_mask.any(dim=2)  # (batch_size, num_words)
            word_matches = ((preds == label) | ~non_pad_mask).all(dim=2)  # (batch_size, num_words)
            valid_words = word_nonpad.sum().item()
            correct_words += (word_matches & word_nonpad).sum().item()
            total_words += valid_words

        byte_accuracy = correct_bytes / total_bytes if total_bytes > 0 else 0.0
        word_accuracy = correct_words / total_words if total_words > 0 else 0.0

        return {
            "byte_accuracy": byte_accuracy,
            "word_accuracy": word_accuracy
        }
