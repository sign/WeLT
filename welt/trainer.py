"""
Custom HuggingFace Trainer for WeLT model evaluation with generation metrics.

This trainer extends the Seq2SeqTrainer to handle WeLT's custom model forward
and generate methods, enabling evaluation with generation-based metrics like
BLEU, CER, ROUGE, etc.
"""

import logging
from typing import Any

import evaluate
import torch
from torch.utils.data import Dataset
from transformers import GenerationConfig, Seq2SeqTrainer, TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction

from welt.processor import TextImageProcessor

logger = logging.getLogger(__name__)


class WeLTTrainer(Seq2SeqTrainer):
    """
    Custom trainer for WeLT model that handles generation-based evaluation.

    The main differences from standard Seq2SeqTrainer:
    1. Custom prediction_step that uses WeLT's generate method with proper inputs
    2. Handles prefix/completion split from dataset for generation evaluation
    3. Computes metrics using generated text vs gold completion

    Expected dataset format from init_datasets:
        - text: The full text (prefix + completion) - used for forward loss
        - prefix: The input to feed to the model for generation
        - completion: The gold reference to compare against generated output
    """

    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        processing_class: TextImageProcessor | None = None,
        model_init=None,
        compute_metrics=None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        eval_metrics: list[str] | None = None,
        generation_config: GenerationConfig | None = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        self.processor = processing_class
        self.eval_metrics = eval_metrics or []
        self.generation_config = generation_config or GenerationConfig(
            temperature=1.0,
            top_p=1.0,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
        )

        # Load metric objects
        self._metrics = {}
        for metric_name in self.eval_metrics:
            try:
                self._metrics[metric_name] = evaluate.load(metric_name)
            except (FileNotFoundError, ValueError, ImportError) as e:
                logger.warning(f"Failed to load metric {metric_name}: {e}")

    def prediction_step(
        self,
        model,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Perform a prediction step using generation.

        This overrides the parent method to:
        1. Use WeLT's custom generate method with proper input format
        2. Return generated text tokens for metric computation

        Args:
            model: The model to use for prediction
            inputs: Dictionary containing model inputs and labels
            prediction_loss_only: If True, only return loss (skip generation)
            ignore_keys: Keys to ignore in the input dictionary

        Returns:
            Tuple of (loss, generated_tokens, completion)
        """
        model.eval()

        # Extract completion (gold label) before modifying inputs
        completion = inputs.pop("completion", None)

        # Move inputs to device
        inputs = self._prepare_inputs(inputs)

        with torch.inference_mode():
            # Compute loss using forward pass
            outputs = model(**inputs)
            loss = outputs.loss

            if prediction_loss_only:
                return (loss, None, None)

            # Generate predictions using WeLT's custom generate method
            generated_texts = model.generate(
                input_ids=inputs["input_ids"],
                input_attention_mask=inputs["input_attention_mask"],
                input_images=inputs["input_images"],
                input_images_dimensions=inputs["input_images_dimensions"],
                attention_mask=inputs["attention_mask"],
                processor=self.processor,
                bytes_generation_config=self.generation_config,
                max_generated_words=self.args.generation_max_length or 10,
            )

        # Store generated texts for metric computation
        # We return None for generated_tokens since we work with text directly
        # The completion is the gold reference for metrics
        return (loss, generated_texts, completion)

    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Custom evaluation loop that handles text-based generation metrics.

        This collects generated texts and completions across batches, then computes
        metrics on the full set of predictions.
        """
        model = self.model
        model.eval()

        all_predictions: list[str] = []
        all_completions: list[str] = []
        total_loss = 0.0
        num_batches = 0
        num_samples = 0

        for inputs in dataloader:
            loss, generated_texts, completions = self.prediction_step(
                model,
                inputs,
                prediction_loss_only=False,
                ignore_keys=ignore_keys,
            )

            if loss is not None:
                total_loss += loss.item()
                num_batches += 1

            if generated_texts is not None:
                all_predictions.extend(generated_texts)
                num_samples += len(generated_texts)

            if completions is not None:
                all_completions.extend(completions)

        # Compute metrics
        metrics = {}
        if num_batches > 0:
            metrics[f"{metric_key_prefix}_loss"] = total_loss / num_batches

        if all_predictions and all_completions:
            eval_metrics = self.compute_generation_metrics(all_predictions, all_completions)
            for key, value in eval_metrics.items():
                metrics[f"{metric_key_prefix}_{key}"] = value

        return EvalLoopOutput(
            predictions=all_predictions,
            label_ids=all_completions,
            metrics=metrics,
            num_samples=num_samples,
        )

    def compute_generation_metrics(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        """
        Compute generation metrics comparing predictions to references.

        Args:
            predictions: List of generated text strings
            references: List of gold label strings

        Returns:
            Dictionary mapping metric names to their computed values
        """
        results = {}

        for metric_name, metric in self._metrics.items():
            try:
                # SacreBLEU expects references as list of lists
                is_sacrebleu = metric_name.lower() == "sacrebleu"
                refs = [[r] for r in references] if is_sacrebleu else references

                result = metric.compute(predictions=predictions, references=refs)

                # Flatten nested results
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            results[f"{metric_name}_{key}"] = value
                        elif key == "score":
                            results[metric_name] = value
                else:
                    results[metric_name] = result

            except (ValueError, TypeError, ZeroDivisionError) as e:
                logger.warning(f"Failed to compute metric {metric_name}: {e}")

        return results

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> Any:
        """
        Get evaluation dataloader with proper collation.

        This ensures the dataloader uses our custom collator that handles
        the label field correctly (keeping it as a list of strings).
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        return super().get_eval_dataloader(eval_dataset)

    def create_compute_metrics_fn(self) -> callable:
        """
        Create a compute_metrics function compatible with HF Trainer.

        This is provided for compatibility with standard Trainer workflows,
        though our custom evaluation_loop handles metrics directly.
        """

        def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
            predictions, completions = eval_pred
            # If predictions are text strings stored as object array
            if hasattr(predictions, "tolist"):
                predictions = predictions.tolist()
            if hasattr(completions, "tolist"):
                completions = completions.tolist()

            return self.compute_generation_metrics(predictions, completions)

        return compute_metrics
