"""
Evaluation utilities for WeLT model training and testing.

This module provides functions for:
- Setting up evaluation metrics (accuracy, generation-based metrics like BLEU, chrF)
- Running evaluation on validation/test datasets
- Computing perplexity and generation metrics

The evaluation logic is designed to be reusable for both validation during training
and test set evaluation after training.
"""

import logging
import math

import evaluate
import numpy as np
import torch
from transformers import GenerationConfig, TrainingArguments, is_torch_xla_available

logger = logging.getLogger(__name__)


def _should_use_generation_metrics(data_args) -> bool:
    """Determine if we should use generation-based metrics (BLEU, chrF, etc.)."""
    return (
        data_args.metric_name is not None
        and isinstance(data_args.dataset_text_template, list)
    )


def _load_metrics(metric_names: list[str], cache_dir: str | None = None):
    """Load evaluation metrics by name."""
    return [evaluate.load(name.strip(), cache_dir=cache_dir) for name in metric_names]


def _compute_metric_results(metrics_list, metric_names: list[str], predictions, references):
    """
    Compute all metrics and return results dictionary.

    This function is reusable for both training evaluation and test evaluation.
    """
    results = {}
    for metric, name in zip(metrics_list, metric_names, strict=False):
        metric_result = metric.compute(predictions=predictions, references=references)
        # Handle different metric return formats
        if "score" in metric_result:
            results[name] = round(metric_result["score"], 4)
        elif name in metric_result:
            results[name] = round(metric_result[name], 4)
        else:
            # Fallback: use first value in result dict
            results[name] = round(list(metric_result.values())[0], 4)
    return results


def _postprocess_text_for_metrics(predictions, references):
    """
    Postprocess generated text and references for metric computation.

    Strips whitespace and formats references as lists (for metrics that expect multiple refs).
    This is reusable for both validation and test evaluation.
    """
    predictions = [pred.strip() for pred in predictions]
    references = [[ref.strip()] if isinstance(ref, str) else ref for ref in references]
    return predictions, references


def setup_evaluation_functions(
    training_args: TrainingArguments,
    data_args,
    tokenizer,
    cache_dir: str | None = None,
):
    """
    Configure trainer evaluation hooks and return compute functions.

    Returns:
        Tuple of (compute_metrics, preprocess_logits_for_metrics) functions
    """
    # Configure trainer evaluation settings
    training_args.include_for_metrics = ["loss"]
    training_args.label_names = ["labels_output"]
    training_args.eval_do_concat_batches = False

    pad_token_id = tokenizer.pad_token_id
    use_generation = _should_use_generation_metrics(data_args)

    if use_generation:
        # Setup for generation-based metrics (BLEU, chrF, etc.)
        metric_names = [m.strip() for m in data_args.metric_name.split(",")]

        # Validate metric_for_best_model if specified
        if (
            data_args.metric_for_best_model is not None
            and data_args.metric_for_best_model not in metric_names
        ):
            msg = (
                f"metric_for_best_model='{data_args.metric_for_best_model}' "
                f"is not in metric_name ({metric_names}). "
                "Add it to metric_name or remove metric_for_best_model."
            )
            raise ValueError(msg)  # noqa: TRY003

        metrics_list = _load_metrics(metric_names, cache_dir)

        def compute_metrics(eval_preds):
            """Compute generation-based metrics from predicted token IDs."""
            if not metrics_list:
                return {}

            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]

            # Replace -100 (ignore index) with pad token for decoding
            preds = np.where(preds != -100, preds, pad_token_id)
            labels = np.where(labels != -100, labels, pad_token_id)

            # Decode predictions and labels
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Postprocess and compute metrics
            decoded_preds, decoded_labels = _postprocess_text_for_metrics(
                decoded_preds, decoded_labels
            )
            results = _compute_metric_results(
                metrics_list, metric_names, decoded_preds, decoded_labels
            )

            # Add generation length statistic
            prediction_lens = [np.count_nonzero(pred != pad_token_id) for pred in preds]
            results["gen_len"] = round(np.mean(prediction_lens), 4)

            return results

        training_args.predict_with_generate = True
        return (
            compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
            None,
        )

    # Setup for accuracy-based metrics (teacher forcing)
    def preprocess_logits_for_metrics(logits, labels):
        """Extract argmax predictions from logits."""
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    accuracy_metric = evaluate.load("accuracy", cache_dir=cache_dir)

    def compute_metrics(eval_preds):
        """Compute token-level accuracy."""
        all_preds = eval_preds.predictions
        all_labels = eval_preds.label_ids

        flat_preds = []
        flat_labels = []

        for preds, labels in zip(all_preds, all_labels, strict=False):
            # Flatten batch predictions
            preds = preds.reshape(-1)
            labels = labels.reshape(-1)

            # Remove padding tokens
            mask = labels != pad_token_id
            preds = preds[mask]
            labels = labels[mask]

            flat_preds.append(torch.tensor(preds))
            flat_labels.append(torch.tensor(labels))

        flat_preds = torch.cat(flat_preds)
        flat_labels = torch.cat(flat_labels)

        return accuracy_metric.compute(predictions=flat_preds, references=flat_labels)

    return (
        compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics if training_args.do_eval and not is_torch_xla_available() else None,
    )


def evaluate_model(
    trainer,
    model,
    processor,
    data_args,
    training_args,
    eval_dataset,
    raw_eval_dataset=None,
    cache_dir: str | None = None,
):
    """
    Run evaluation on the provided dataset and log metrics.
    
    Supports two evaluation modes:
    1. Teacher forcing with perplexity/accuracy (standard training evaluation)
    2. Generation-based with BLEU/chrF/etc. (requires prefix/completion split)
    
    This function can be reused for validation during training or test evaluation.
    
    Args:
        trainer: HuggingFace Trainer instance
        model: The model to evaluate
        processor: Text processor for the model
        data_args: Data configuration arguments
        training_args: Training configuration arguments
        eval_dataset: Processed dataset for evaluation
        raw_eval_dataset: Unprocessed dataset (needed for generation metrics)
        cache_dir: Directory for caching evaluation metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("*** Evaluate ***")

    # Determine evaluation mode
    has_generation_data = (
        raw_eval_dataset is not None
        and "completion" in raw_eval_dataset.column_names
    )
    use_generation = _should_use_generation_metrics(data_args) and has_generation_data

    if use_generation:
        metrics = _evaluate_with_generation(
            trainer=trainer,
            model=model,
            processor=processor,
            data_args=data_args,
            training_args=training_args,
            eval_dataset=eval_dataset,
            raw_eval_dataset=raw_eval_dataset,
            cache_dir=cache_dir,
        )
    else:
        # Standard teacher forcing evaluation
        metrics = trainer.evaluate()
        metrics["perplexity"] = _compute_perplexity(metrics["eval_loss"])

    # Add sample count to metrics
    max_eval_samples = (
        data_args.max_eval_samples
        if data_args.max_eval_samples is not None
        else len(eval_dataset)
    )
    if data_args.streaming:
        metrics["eval_samples"] = max_eval_samples
    else:
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    # Log and save metrics
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    return metrics


def _compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss, handling overflow."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def _extract_prefixes_and_references(dataset, streaming: bool = False):
    """
    Extract prefixes and references from a dataset with prefix/completion columns.
    
    This is reusable for both validation and test evaluation.
    
    Args:
        dataset: Dataset with 'prefix' and 'completion' columns
        streaming: Whether the dataset is streaming
        
    Returns:
        Tuple of (prefixes, references) lists
    """
    if streaming:
        prefixes = []
        references = []
        for item in dataset:
            prefixes.append(item["prefix"])
            references.append(item["completion"])
    else:
        prefixes = [item["prefix"] for item in dataset]
        references = [item["completion"] for item in dataset]

    return prefixes, references


def _setup_generation_config(training_args):
    """Setup generation configuration from training arguments."""
    gen_config_arg = getattr(training_args, "generation_config", None)

    if isinstance(gen_config_arg, GenerationConfig):
        generation_config = gen_config_arg
    elif gen_config_arg is not None:
        generation_config = GenerationConfig.from_pretrained(gen_config_arg)
    else:
        generation_config = GenerationConfig()

    # Override num_beams if specified
    if getattr(training_args, "generation_num_beams", None) is not None:
        generation_config.num_beams = training_args.generation_num_beams
    elif generation_config.num_beams is None:
        generation_config.num_beams = 1

    return generation_config


def generate_predictions(
    model,
    processor,
    prefixes: list[str],
    batch_size: int,
    max_generated_words: int,
    generation_config: GenerationConfig,
):
    """
    Generate predictions for a list of input prefixes.
    
    This function is reusable for both validation and test evaluation.
    
    Args:
        model: The model to use for generation
        processor: Text processor for the model
        prefixes: List of input prefixes to complete
        batch_size: Batch size for generation
        max_generated_words: Maximum number of words to generate
        generation_config: Generation configuration
        
    Returns:
        List of generated text strings
    """
    model.eval()
    generated_texts = []

    with torch.no_grad():
        for i in range(0, len(prefixes), batch_size):
            batch_prefixes = prefixes[i : i + batch_size]
            inputs = processor(batch_prefixes, collated=True, packed=False)
            batch_outputs = model.generate(
                **inputs,
                processor=processor,
                max_generated_words=max_generated_words,
                bytes_generation_config=generation_config,
            )
            generated_texts.extend(batch_outputs)

    return generated_texts


def _evaluate_with_generation(
    trainer,
    model,
    processor,
    data_args,
    training_args,
    eval_dataset,
    raw_eval_dataset,
    cache_dir: str | None = None,
):
    """
    Evaluate model using generation-based metrics (BLEU, chrF, etc.).
    
    This performs actual text generation from prefixes and compares to references.
    Also computes perplexity using teacher forcing for comparison.
    """
    # Extract prefixes and references from dataset
    prefixes, references = _extract_prefixes_and_references(
        raw_eval_dataset,
        streaming=data_args.streaming
    )

    # Setup generation configuration
    generation_config = _setup_generation_config(training_args)
    generation_length = getattr(training_args, "generation_max_length", None)
    max_generated_words = generation_length // 10 if generation_length else 50

    # Generate predictions
    generated_texts = generate_predictions(
        model=model,
        processor=processor,
        prefixes=prefixes,
        batch_size=training_args.per_device_eval_batch_size,
        max_generated_words=max_generated_words,
        generation_config=generation_config,
    )

    # Postprocess and compute metrics
    predictions, formatted_references = _postprocess_text_for_metrics(
        generated_texts,
        references
    )

    metric_names = [m.strip() for m in data_args.metric_name.split(",")]
    metrics_list = _load_metrics(metric_names, cache_dir)
    metrics = _compute_metric_results(
        metrics_list,
        metric_names,
        predictions,
        formatted_references
    )

    # Also compute perplexity using teacher forcing
    perplexity_metrics = trainer.evaluate(eval_dataset)
    metrics["perplexity"] = _compute_perplexity(perplexity_metrics["eval_loss"])
    metrics["eval_loss"] = perplexity_metrics["eval_loss"]

    return metrics

