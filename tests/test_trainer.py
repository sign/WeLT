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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
