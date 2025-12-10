"""
Tests for WeLTTrainer custom trainer class.

Tests cover the custom methods that differ from standard Seq2SeqTrainer:
- prediction_step: Custom generation using WeLT's generate method
- evaluation_loop: Collects text predictions and computes metrics
- compute_generation_metrics: Metric computation on text predictions
"""

import tempfile

import pytest
import torch
from datasets import Dataset
from transformers import GenerationConfig, Seq2SeqTrainingArguments

from welt.model_utils import setup_model
from welt.trainer import WeLTTrainer


def setup_tiny_model(image_encoder_name="WinKawaks/vit-tiny-patch16-224", **kwargs):
    """Set up a tiny version of the WordLatentTransformer model for testing."""
    return setup_model(
        image_encoder_name=image_encoder_name,
        bytes_encoder_name="prajjwal1/bert-tiny",
        latent_transformer_name="sbintuitions/tiny-lm",
        bytes_decoder_name="sbintuitions/tiny-lm",
        load_pretrained=False,
        **kwargs
    )


def make_eval_dataset(texts: list[str], completions: list[str]) -> Dataset:
    """Create an evaluation dataset with text and completion columns."""
    return Dataset.from_dict({"text": texts, "completion": completions})


def get_training_args(output_dir: str, use_cpu: bool = True) -> Seq2SeqTrainingArguments:
    """Create minimal training arguments for testing."""
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        generation_max_length=5,  # Limit generated words to prevent OOM
        remove_unused_columns=False,
        report_to="none",
        use_cpu=use_cpu,  # Force CPU for consistent testing
    )


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move batch tensors to the specified device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class TestWeLTTrainerInit:
    """Tests for WeLTTrainer initialization."""

    def test_trainer_initializes_with_model_and_processor(self):
        """Test that trainer initializes correctly with model and processor."""
        model, processor, collator = setup_tiny_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
            )

            assert trainer.model is model
            assert trainer.processor is processor

    def test_trainer_initializes_with_eval_metrics(self):
        """Test that trainer loads specified evaluation metrics."""
        model, processor, collator = setup_tiny_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                eval_metrics=["bleu", "cer"],
            )

            assert "bleu" in trainer._metrics
            assert "cer" in trainer._metrics

    def test_trainer_initializes_with_generation_config(self):
        """Test that trainer accepts custom generation config."""
        model, processor, collator = setup_tiny_model()

        gen_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                generation_config=gen_config,
            )

            assert trainer.generation_config.temperature == 0.7
            assert trainer.generation_config.top_p == 0.9
            assert trainer.generation_config.do_sample is True

    def test_trainer_handles_invalid_metric_gracefully(self):
        """Test that trainer handles invalid metric names gracefully."""
        model, processor, collator = setup_tiny_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            # Should not raise, just log warning
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                eval_metrics=["nonexistent_metric_xyz"],
            )

            assert "nonexistent_metric_xyz" not in trainer._metrics


class TestComputeGenerationMetrics:
    """Tests for compute_generation_metrics method."""

    def test_compute_metrics_with_exact_match(self):
        """Test metrics computation with exact matching predictions."""
        model, processor, collator = setup_tiny_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                eval_metrics=["cer"],
            )

            predictions = ["hello", "world"]
            references = ["hello", "world"]

            metrics = trainer.compute_generation_metrics(predictions, references)

            # CER should be 0 for exact matches
            assert "cer" in metrics
            assert metrics["cer"] == 0.0

    def test_compute_metrics_with_mismatched_predictions(self):
        """Test metrics computation with different predictions."""
        model, processor, collator = setup_tiny_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                eval_metrics=["cer"],
            )

            predictions = ["helo", "word"]  # Missing characters
            references = ["hello", "world"]

            metrics = trainer.compute_generation_metrics(predictions, references)

            # CER should be > 0 for mismatches
            assert "cer" in metrics
            assert metrics["cer"] > 0.0

    def test_compute_metrics_with_sacrebleu(self):
        """Test that sacrebleu references are wrapped correctly."""
        model, processor, collator = setup_tiny_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                eval_metrics=["sacrebleu"],
            )

            predictions = ["hello world"]
            references = ["hello world"]

            metrics = trainer.compute_generation_metrics(predictions, references)

            # Should have computed sacrebleu without error
            assert any("sacrebleu" in k for k in metrics.keys())

    def test_compute_metrics_with_multiple_metrics(self):
        """Test computation of multiple metrics at once."""
        model, processor, collator = setup_tiny_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                eval_metrics=["cer", "bleu"],
            )

            predictions = ["hello", "test"]
            references = ["hello", "test"]

            metrics = trainer.compute_generation_metrics(predictions, references)

            assert "cer" in metrics
            # BLEU may have nested keys like bleu_bleu or just bleu
            assert any("bleu" in k.lower() for k in metrics.keys())

    def test_compute_metrics_empty_lists(self):
        """Test metrics computation with empty prediction lists."""
        model, processor, collator = setup_tiny_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                eval_metrics=["cer"],
            )

            # Empty lists should not crash, may return empty or error gracefully
            try:
                metrics = trainer.compute_generation_metrics([], [])
                # If it returns, should be a dict
                assert isinstance(metrics, dict)
            except (ValueError, ZeroDivisionError):
                # Some metrics may fail on empty input, which is acceptable
                pass


class TestPredictionStep:
    """Tests for prediction_step method."""

    def test_prediction_step_returns_loss_and_predictions(self):
        """Test that prediction_step returns loss and generated texts."""
        # Use model without image encoder to avoid device mismatch during generation
        # (image rendering happens on CPU, but model may be on GPU)
        model, processor, collator = setup_tiny_model(image_encoder_name=None)
        model.eval()

        # Limit word length to prevent sequence explosion
        processor.max_word_length = 5

        # Create a simple batch
        texts = ["a b"]
        completions = ["c"]
        dataset = make_eval_dataset(texts, completions)
        dataset = dataset.with_transform(processor)
        batch = collator([dataset[0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
            )

            loss, generated_texts, returned_completions = trainer.prediction_step(
                trainer.model,
                batch,
                prediction_loss_only=False,
            )

            assert loss is not None
            assert isinstance(loss, torch.Tensor)
            assert generated_texts is not None
            assert isinstance(generated_texts, list)
            assert returned_completions is not None

    def test_prediction_step_loss_only_mode(self):
        """Test prediction_step with prediction_loss_only=True."""
        model, processor, collator = setup_tiny_model()
        model.eval()

        texts = ["a b"]
        completions = ["c"]
        dataset = make_eval_dataset(texts, completions)
        dataset = dataset.with_transform(processor)
        batch = collator([dataset[0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
            )

            loss, generated_texts, returned_completions = trainer.prediction_step(
                model,
                batch,
                prediction_loss_only=True,
            )

            assert loss is not None
            assert generated_texts is None
            assert returned_completions is None

    def test_prediction_step_moves_inputs_to_device(self):
        """Test that prediction_step handles device placement."""
        model, processor, collator = setup_tiny_model()
        model.eval()

        texts = ["a b"]
        completions = ["c"]
        dataset = make_eval_dataset(texts, completions)
        dataset = dataset.with_transform(processor)
        batch = collator([dataset[0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
            )

            # Get the device after trainer initialization (trainer may move model)
            device = next(trainer.model.parameters()).device

            # Should not raise device mismatch errors
            loss, _, _ = trainer.prediction_step(
                trainer.model,
                batch,
                prediction_loss_only=True,
            )

            assert loss.device == device


class TestEvaluationLoop:
    """Tests for evaluation_loop method."""

    def test_evaluation_loop_returns_metrics_dict(self):
        """Test that evaluation_loop returns a metrics dictionary."""
        # Use model without image encoder to avoid device mismatch during generation
        model, processor, collator = setup_tiny_model(image_encoder_name=None)
        model.eval()

        # Limit word length to prevent sequence explosion
        processor.max_word_length = 5

        texts = ["a b", "c d"]
        completions = ["x", "y"]
        dataset = make_eval_dataset(texts, completions)
        dataset = dataset.with_transform(processor)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                eval_dataset=dataset,
                eval_metrics=["cer"],
            )

            dataloader = trainer.get_eval_dataloader()
            output = trainer.evaluation_loop(
                dataloader,
                description="Test evaluation",
            )

            assert isinstance(output.metrics, dict)
            assert "eval_loss" in output.metrics

    def test_evaluation_loop_computes_generation_metrics(self):
        """Test that evaluation_loop computes generation metrics."""
        # Use model without image encoder to avoid device mismatch during generation
        model, processor, collator = setup_tiny_model(image_encoder_name=None)
        model.eval()

        # Limit word length to prevent sequence explosion
        processor.max_word_length = 5

        texts = ["a b", "c d"]
        completions = ["x", "y"]
        dataset = make_eval_dataset(texts, completions)
        dataset = dataset.with_transform(processor)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                eval_dataset=dataset,
                eval_metrics=["cer"],
            )

            dataloader = trainer.get_eval_dataloader()
            output = trainer.evaluation_loop(
                dataloader,
                description="Test evaluation",
            )

            # Should have CER metric with eval_ prefix
            assert any("cer" in k for k in output.metrics.keys())

    def test_evaluation_loop_handles_custom_metric_prefix(self):
        """Test that evaluation_loop uses custom metric prefix."""
        # Use model without image encoder to avoid device mismatch during generation
        model, processor, collator = setup_tiny_model(image_encoder_name=None)
        model.eval()

        # Limit word length to prevent sequence explosion
        processor.max_word_length = 5

        texts = ["a b"]
        completions = ["x"]
        dataset = make_eval_dataset(texts, completions)
        dataset = dataset.with_transform(processor)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                eval_dataset=dataset,
                eval_metrics=["cer"],
            )

            dataloader = trainer.get_eval_dataloader()
            output = trainer.evaluation_loop(
                dataloader,
                description="Test evaluation",
                metric_key_prefix="test",
            )

            assert "test_loss" in output.metrics
            assert any(k.startswith("test_") for k in output.metrics.keys())


class TestCreateComputeMetricsFn:
    """Tests for create_compute_metrics_fn method."""

    def test_create_compute_metrics_fn_returns_callable(self):
        """Test that create_compute_metrics_fn returns a callable."""
        model, processor, collator = setup_tiny_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                eval_metrics=["cer"],
            )

            compute_metrics_fn = trainer.create_compute_metrics_fn()

            assert callable(compute_metrics_fn)

    def test_compute_metrics_fn_works_with_eval_prediction(self):
        """Test that the returned function works with EvalPrediction format."""
        model, processor, collator = setup_tiny_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                eval_metrics=["cer"],
            )

            compute_metrics_fn = trainer.create_compute_metrics_fn()

            # Simulate EvalPrediction with text data
            class MockEvalPrediction:
                def __init__(self, predictions, completions):
                    self.predictions = predictions
                    self.label_ids = completions

                def __iter__(self):
                    return iter([self.predictions, self.label_ids])

            eval_pred = MockEvalPrediction(
                predictions=["hello", "world"],
                completions=["hello", "world"],
            )

            metrics = compute_metrics_fn(eval_pred)

            assert isinstance(metrics, dict)
            assert "cer" in metrics


class TestIntegration:
    """Integration tests for WeLTTrainer."""

    @pytest.mark.slow
    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow from dataset to metrics."""
        # Use model without image encoder to avoid device mismatch during generation
        model, processor, collator = setup_tiny_model(image_encoder_name=None)
        model.eval()

        # Limit word length to prevent sequence explosion
        processor.max_word_length = 5

        # Create evaluation dataset
        texts = ["hello", "world"]
        completions = ["x", "y"]
        dataset = make_eval_dataset(texts, completions)
        dataset = dataset.with_transform(processor)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = get_training_args(tmpdir)
            trainer = WeLTTrainer(
                model=model,
                args=args,
                data_collator=collator,
                processing_class=processor,
                eval_dataset=dataset,
                eval_metrics=["cer"],
            )

            # Run full evaluation
            metrics = trainer.evaluate()

            assert isinstance(metrics, dict)
            # Should have loss and at least one metric
            assert len(metrics) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
