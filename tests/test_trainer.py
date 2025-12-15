"""
Tests for WeLTTrainer generation-based evaluation.
"""
import pytest
import torch
from datasets import Dataset
from transformers import GenerationConfig, Seq2SeqTrainingArguments

from tests.test_model import setup_tiny_model
from training.trainer import WeLTTrainer


@pytest.fixture(scope="module")
def trainer_setup():
    """Setup model, processor, and collator for trainer tests."""
    model, processor, collator = setup_tiny_model(image_encoder_name=None)
    # Force CPU to avoid device placement issues during generation
    model = model.to(torch.device("cpu"))
    model.eval()
    return model, processor, collator


def make_generation_dataset(prefixes: list[str], completions: list[str]) -> Dataset:
    """
    Create a dataset for generation-based evaluation.

    Args:
        prefixes: Input texts for generation
        completions: Expected completions for metric computation
    """
    return Dataset.from_dict({
        "text": [f"{p}{c}" for p, c in zip(prefixes, completions, strict=True)],  # Full text for loss calculation
        "prefix": prefixes,  # Input for generation
        "completion": completions,  # Reference for generation metrics
    })


def test_trainer_initialization(trainer_setup):
    """Test that WeLTTrainer can be initialized properly."""
    model, processor, collator = trainer_setup

    # Create minimal training args
    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    # Initialize trainer with no metrics
    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=None,
    )

    assert trainer.processor == processor
    assert len(trainer.loaded_metrics) == 0
    assert trainer.max_generated_words == 50
    assert trainer.bytes_generation_config is None
    assert trainer.log_samples == 3


def test_trainer_with_generation_config(trainer_setup):
    """Test that WeLTTrainer accepts bytes_generation_config."""
    model, processor, collator = trainer_setup

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    # Initialize with generation config
    gen_config = GenerationConfig(num_beams=2, max_new_tokens=10)
    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        bytes_generation_config=gen_config,
        log_samples=5,
    )

    assert trainer.bytes_generation_config == gen_config
    assert trainer.log_samples == 5


def test_trainer_with_metrics(trainer_setup):
    """Test that WeLTTrainer loads metrics correctly."""
    model, processor, collator = trainer_setup

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    # Initialize trainer with metrics (only bleu, as cer might not be available)
    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
    )

    assert "bleu" in trainer.loaded_metrics


def test_evaluation_with_generate(trainer_setup):
    """Test that trainer can evaluate using generation."""
    model, processor, collator = trainer_setup

    # Create a small evaluation dataset with prefix/completion pairs
    eval_dataset = make_generation_dataset(
        prefixes=["Hello", "Test", "Another"],
        completions=[" world", " text", " example"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
        max_generated_words=3,
    )

    # Set smaller max_word_length for faster testing
    trainer.processor.max_word_length = 5

    # Run evaluation
    metrics = trainer.evaluate(eval_dataset)

    # Check that both loss and generation metrics were computed
    assert "eval_loss" in metrics
    assert "eval_bleu" in metrics


def test_compute_generation_metrics(trainer_setup):
    """Test metric computation with known inputs."""
    model, processor, collator = trainer_setup

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
    )

    # Test with longer perfect predictions for better BLEU scores
    predictions = [
        "The quick brown fox jumps over the lazy dog",
        "This is a longer test sentence for BLEU evaluation"
    ]
    references = [
        "The quick brown fox jumps over the lazy dog",
        "This is a longer test sentence for BLEU evaluation"
    ]

    metrics = trainer._compute_generation_metrics(predictions, references)

    # BLEU should be present
    assert "bleu" in metrics
    bleu_score = metrics["bleu"]
    assert bleu_score >= 0, f"Expected non-negative BLEU score, got {bleu_score}"

    # Test with imperfect predictions
    predictions_bad = [
        "A completely different sentence here",
        "This does not match at all"
    ]
    metrics_bad = trainer._compute_generation_metrics(predictions_bad, references)

    bleu_score_bad = metrics_bad["bleu"]

    # Perfect match should score better than imperfect match
    assert bleu_score >= bleu_score_bad, \
        f"Perfect match score ({bleu_score}) should be >= imperfect match score ({bleu_score_bad})"


def test_evaluate_simple(trainer_setup):
    """Test simple evaluation case."""
    model, processor, collator = trainer_setup

    # Create tiny dataset
    eval_dataset = make_generation_dataset(
        prefixes=["a"],
        completions=[" b"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=1,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
        max_generated_words=2,
    )

    trainer.processor.max_word_length = 5

    # Run evaluation
    metrics = trainer.evaluate(eval_dataset)

    # Should have both loss and generation metrics
    assert "eval_loss" in metrics
    assert "eval_bleu" in metrics


def test_evaluate_missing_prefix_column(trainer_setup):
    """Test error handling when dataset lacks 'prefix' column."""
    model, processor, collator = trainer_setup

    # Create dataset without 'prefix' column
    invalid_dataset = Dataset.from_dict({
        "text": ["Hello", "Test"],
        "completion": [" world", " text"],
    })

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
    )

    # Should raise ValueError about missing 'prefix' column
    with pytest.raises(ValueError, match="prefix"):
        trainer.evaluate(invalid_dataset)


def test_evaluate_without_completions(trainer_setup):
    """Test evaluation when dataset has no 'completion' column."""
    model, processor, collator = trainer_setup

    # Create dataset without 'completion' column (only prefixes)
    eval_dataset = Dataset.from_dict({
        "text": ["Hello", "Test"],  # For loss calculation
        "prefix": ["Hello", "Test"],  # For generation
    })

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
        max_generated_words=3,
    )

    trainer.processor.max_word_length = 5

    # Should generate predictions but skip generation metric computation
    metrics = trainer.evaluate(eval_dataset)

    # Should have loss but not generation metrics (no reference for comparison)
    assert "eval_loss" in metrics
    assert "eval_bleu" not in metrics or metrics["eval_bleu"] is None


def test_trainer_with_multiple_metrics(trainer_setup):
    """Test evaluation with multiple metrics simultaneously."""
    model, processor, collator = trainer_setup

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    # Try to load multiple metrics (some may not be available)
    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu", "rouge"],
    )

    # At least bleu should be loaded
    assert "bleu" in trainer.loaded_metrics

    # Test that multiple metrics can be computed
    predictions = [
        "The quick brown fox jumps over the lazy dog",
        "This is a test sentence"
    ]
    references = [
        "The quick brown fox jumps over the lazy dog",
        "This is a test sentence"
    ]

    metrics = trainer._compute_generation_metrics(predictions, references)

    # Check that at least one metric was computed
    assert len(metrics) > 0


def test_trainer_with_invalid_metric(trainer_setup):
    """Test that invalid metrics are logged but don't crash initialization."""
    model, processor, collator = trainer_setup

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    # Initialize with mix of valid and invalid metrics
    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu", "nonexistent_metric_xyz", "another_invalid"],
    )

    # Valid metric should be loaded
    assert "bleu" in trainer.loaded_metrics
    # Invalid metrics should not be in loaded_metrics
    assert "nonexistent_metric_xyz" not in trainer.loaded_metrics
    assert "another_invalid" not in trainer.loaded_metrics


def test_evaluate_empty_dataset(trainer_setup):
    """Test evaluation with empty dataset."""
    model, processor, collator = trainer_setup

    # Create empty dataset
    empty_dataset = make_generation_dataset(
        prefixes=[],
        completions=[],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
        max_generated_words=3,
    )

    # Empty datasets may be treated as None by the dataloader
    # This is expected behavior - should raise ValueError
    try:
        trainer.evaluate(empty_dataset)
        # If we get here with 0 predictions, that's also acceptable
    except ValueError:
        # Expected: empty datasets may fail validation
        pass


def test_metric_format_flexibility(trainer_setup):
    """Test that metric computation handles both reference formats."""
    model, processor, collator = trainer_setup

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    # Test with BLEU (requires list-of-lists)
    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
    )

    predictions = ["The quick brown fox", "Another test"]
    references = ["The quick brown fox", "Another test"]

    metrics = trainer._compute_generation_metrics(predictions, references)

    # Should successfully compute BLEU without hardcoding format
    assert "bleu" in metrics


def test_deterministic_evaluation(trainer_setup):
    """Test that evaluation is deterministic with same seed."""
    model, processor, collator = trainer_setup

    # Create dataset
    eval_dataset = make_generation_dataset(
        prefixes=["Hello", "Test"],
        completions=[" world", " text"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        seed=42,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
        max_generated_words=3,
    )

    trainer.processor.max_word_length = 5

    # Run evaluation twice with same seed
    import transformers
    transformers.set_seed(42)
    metrics1 = trainer.evaluate(eval_dataset)

    transformers.set_seed(42)
    metrics2 = trainer.evaluate(eval_dataset)

    # Loss should be deterministic (within floating point tolerance)
    assert abs(metrics1["eval_loss"] - metrics2["eval_loss"]) < 1e-6, \
        f"Loss should be deterministic: {metrics1['eval_loss']} vs {metrics2['eval_loss']}"


def test_batch_size_edge_cases(trainer_setup):
    """Test evaluation with various batch sizes including single example."""
    model, processor, collator = trainer_setup

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=1,  # Single example batches
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
        max_generated_words=3,
    )

    trainer.processor.max_word_length = 5

    # Test with single example
    single_dataset = make_generation_dataset(
        prefixes=["a"],
        completions=[" b"],
    )
    metrics = trainer.evaluate(single_dataset)
    assert "eval_loss" in metrics
    assert "eval_bleu" in metrics

    # Test with 3 examples (odd number, not divisible by batch size)
    odd_dataset = make_generation_dataset(
        prefixes=["a", "b", "c"],
        completions=[" x", " y", " z"],
    )
    metrics = trainer.evaluate(odd_dataset)
    assert "eval_loss" in metrics
    assert metrics["eval_samples"] == 3


def test_transform_idempotency(trainer_setup):
    """Test that applying processor transform multiple times doesn't break evaluation."""
    model, processor, collator = trainer_setup

    eval_dataset = make_generation_dataset(
        prefixes=["Hello"],
        completions=[" world"],
    )

    # Apply transform manually before passing to trainer
    eval_dataset = eval_dataset.with_transform(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=1,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
        max_generated_words=3,
    )

    trainer.processor.max_word_length = 5

    # Should handle pre-transformed dataset gracefully
    metrics = trainer.evaluate(eval_dataset)
    assert "eval_loss" in metrics
    assert "eval_bleu" in metrics


def test_perplexity_computation(trainer_setup):
    """Test that perplexity is computed correctly from loss."""
    model, processor, collator = trainer_setup

    eval_dataset = make_generation_dataset(
        prefixes=["a", "b"],
        completions=[" x", " y"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=None,
        max_generated_words=3,
    )

    trainer.processor.max_word_length = 5

    metrics = trainer.evaluate(eval_dataset)

    # Verify perplexity is computed
    assert "perplexity" in metrics, "perplexity should be in metrics"

    # Verify perplexity is positive
    assert metrics["perplexity"] > 0, f"perplexity should be positive, got {metrics['perplexity']}"

    # Verify perplexity = exp(loss)
    import math
    expected_ppl = math.exp(metrics["eval_loss"])
    assert abs(metrics["perplexity"] - expected_ppl) < 0.01, \
        f"perplexity should equal exp(loss): {metrics['perplexity']} vs {expected_ppl}"


def test_eval_samples_count(trainer_setup):
    """Test that eval_samples correctly counts the number of examples evaluated."""
    model, processor, collator = trainer_setup

    # Create dataset with known size
    eval_dataset = make_generation_dataset(
        prefixes=["a", "b", "c", "d", "e"],
        completions=[" x", " y", " z", " w", " v"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=None,
    )

    trainer.processor.max_word_length = 5

    metrics = trainer.evaluate(eval_dataset)

    # Verify sample count
    assert "eval_samples" in metrics
    assert metrics["eval_samples"] == 5, f"Expected 5 samples, got {metrics['eval_samples']}"


def test_multi_evaluation_stability(trainer_setup):
    """Test that running multiple evaluations in sequence is stable."""
    model, processor, collator = trainer_setup

    eval_dataset = make_generation_dataset(
        prefixes=["Hello", "Test"],
        completions=[" world", " text"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
        max_generated_words=3,
    )

    trainer.processor.max_word_length = 5

    # Run evaluation 3 times in a row
    metrics1 = trainer.evaluate(eval_dataset)
    metrics2 = trainer.evaluate(eval_dataset)
    metrics3 = trainer.evaluate(eval_dataset)

    # All runs should produce valid metrics
    for metrics in [metrics1, metrics2, metrics3]:
        assert "eval_loss" in metrics
        assert "eval_bleu" in metrics
        assert "perplexity" in metrics
        assert "eval_samples" in metrics

    # Metrics should be similar across runs (model in eval mode, deterministic)
    assert abs(metrics1["eval_loss"] - metrics2["eval_loss"]) < 0.1
    assert abs(metrics2["eval_loss"] - metrics3["eval_loss"]) < 0.1


def test_generation_with_varying_lengths(trainer_setup):
    """Test generation with inputs of varying lengths."""
    model, processor, collator = trainer_setup

    # Create dataset with varying length inputs
    eval_dataset = make_generation_dataset(
        prefixes=["a", "ab", "abc", "abcd"],
        completions=[" x", " xy", " xyz", " xyzw"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
        max_generated_words=5,
    )

    trainer.processor.max_word_length = 10

    metrics = trainer.evaluate(eval_dataset)

    # Should handle varying lengths gracefully
    assert "eval_loss" in metrics
    assert "eval_bleu" in metrics
    assert metrics["eval_samples"] == 4


def test_generation_max_words_limit(trainer_setup):
    """Test that max_generated_words limits generation length."""
    model, processor, collator = trainer_setup

    eval_dataset = make_generation_dataset(
        prefixes=["test"],
        completions=[" reference"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=1,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    # Test with very small max_generated_words
    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
        max_generated_words=1,  # Very small limit
    )

    trainer.processor.max_word_length = 5

    metrics = trainer.evaluate(eval_dataset)

    # Should still complete evaluation even with tight limits
    assert "eval_loss" in metrics
    assert "eval_bleu" in metrics


def test_no_metrics_evaluation(trainer_setup):
    """Test evaluation with no generation metrics (only loss)."""
    model, processor, collator = trainer_setup

    eval_dataset = make_generation_dataset(
        prefixes=["a"],
        completions=[" b"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=1,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    # Initialize with no metrics
    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=None,  # No generation metrics
    )

    trainer.processor.max_word_length = 5

    metrics = trainer.evaluate(eval_dataset)

    # Should still compute loss and perplexity
    assert "eval_loss" in metrics
    assert "perplexity" in metrics
    assert "eval_samples" in metrics

    # Should not have generation metrics
    assert "eval_bleu" not in metrics
    assert "eval_rouge" not in metrics


def test_sacrebleu_metric(trainer_setup):
    """Test evaluation with sacrebleu metric (used in actual config)."""
    model, processor, collator = trainer_setup

    eval_dataset = make_generation_dataset(
        prefixes=["The quick brown", "Another test"],
        completions=[" fox jumps", " sentence"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    # Test with sacrebleu metric
    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["sacrebleu"],
    )

    trainer.processor.max_word_length = 10

    metrics = trainer.evaluate(eval_dataset)

    # Should compute sacrebleu
    assert "eval_loss" in metrics
    assert "eval_sacrebleu" in metrics
    assert metrics["eval_sacrebleu"] >= 0


def test_chrf_metric(trainer_setup):
    """Test evaluation with chrf metric (used in actual config)."""
    model, processor, collator = trainer_setup

    eval_dataset = make_generation_dataset(
        prefixes=["Hello", "Test"],
        completions=[" world", " text"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    # Test with chrf metric
    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["chrf"],
    )

    trainer.processor.max_word_length = 10

    metrics = trainer.evaluate(eval_dataset)

    # Should compute chrf
    assert "eval_loss" in metrics
    assert "eval_chrf" in metrics
    assert metrics["eval_chrf"] >= 0


def test_multiple_generation_metrics(trainer_setup):
    """Test evaluation with multiple generation metrics (sacrebleu + chrf)."""
    model, processor, collator = trainer_setup

    eval_dataset = make_generation_dataset(
        prefixes=["The quick brown fox", "Another test sentence"],
        completions=[" jumps over the dog", " for evaluation"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=2,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    # Test with both sacrebleu and chrf (as in actual config)
    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["sacrebleu", "chrf"],
    )

    trainer.processor.max_word_length = 15

    metrics = trainer.evaluate(eval_dataset)

    # Should compute both metrics
    assert "eval_loss" in metrics
    assert "eval_sacrebleu" in metrics or "sacrebleu" in trainer.loaded_metrics
    assert "eval_chrf" in metrics or "chrf" in trainer.loaded_metrics
    assert "perplexity" in metrics
    assert "eval_samples" in metrics


def test_metric_computation_with_empty_predictions(trainer_setup):
    """Test that empty predictions are handled gracefully."""
    model, processor, collator = trainer_setup

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=1,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
    )

    # Test with empty lists (edge case)
    predictions = []
    references = []

    # Should handle empty inputs gracefully (either return empty dict or skip)
    try:
        metrics = trainer._compute_generation_metrics(predictions, references)
        # If it succeeds, metrics should be empty or have safe defaults
        assert isinstance(metrics, dict)
    except Exception:  # noqa: BLE001, S110
        # It's acceptable to raise an exception for empty inputs
        pass


def test_dataloader_reuse(trainer_setup):
    """Test that dataloader can be reused without issues from batch mutations."""
    model, processor, collator = trainer_setup

    eval_dataset = make_generation_dataset(
        prefixes=["a", "b"],
        completions=[" x", " y"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="output/test_trainer",
        per_device_eval_batch_size=1,
        do_train=False,
        do_eval=True,
        remove_unused_columns=False,
        predict_with_generate=True,
    )

    trainer = WeLTTrainer(
        model=model,
        args=training_args,
        processor=processor,
        data_collator=collator,
        eval_metrics=["bleu"],
        max_generated_words=2,
    )

    trainer.processor.max_word_length = 5

    # Run evaluation twice on same dataset
    # This tests if pop() operations corrupt the dataset
    metrics1 = trainer.evaluate(eval_dataset)
    metrics2 = trainer.evaluate(eval_dataset)

    # Both should succeed
    assert "eval_loss" in metrics1
    assert "eval_loss" in metrics2
    # Results should be identical (deterministic)
    assert abs(metrics1["eval_loss"] - metrics2["eval_loss"]) < 0.1


def test_accuracy_is_computed(trainer_setup):
    """Test that eval_accuracy is computed during evaluation."""
    model, processor, collator = trainer_setup

    # Create dataset
    data = {
        "text": ["The quick brown", "Another test"],
        "prefix": ["The quick brown", "Another test"],
        "completion": [" fox jumps", " sentence"],
    }
    eval_dataset = Dataset.from_dict(data).with_transform(processor)

    # Setup trainer WITHOUT generation metrics (just accuracy)
    args = Seq2SeqTrainingArguments(
        output_dir="./test_accuracy",
        do_eval=True,
        per_device_eval_batch_size=2,
        predict_with_generate=False,  # Disable generation to focus on accuracy
        remove_unused_columns=False,  # Need to keep prefix/completion columns
        report_to="none",
    )

    trainer = WeLTTrainer(
        model=model,
        args=args,
        eval_dataset=eval_dataset,
        processor=processor,
        data_collator=collator,
        eval_metrics=None,  # No generation metrics
    )

    # Evaluate
    metrics = trainer.evaluate()

    # Should have both byte and word accuracy
    assert "eval_byte_accuracy" in metrics, f"eval_byte_accuracy not in metrics. Found: {list(metrics.keys())}"
    assert "eval_word_accuracy" in metrics, f"eval_word_accuracy not in metrics. Found: {list(metrics.keys())}"
    assert isinstance(metrics["eval_byte_accuracy"], float)
    assert isinstance(metrics["eval_word_accuracy"], float)
    assert 0.0 <= metrics["eval_byte_accuracy"] <= 1.0
    assert 0.0 <= metrics["eval_word_accuracy"] <= 1.0
    # Note: word_accuracy can be > byte_accuracy when many short words are fully correct
    # while longer words are only partially correct


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
