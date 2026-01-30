"""
End-to-end test for the training script.

This module tests the complete training pipeline by running a minimal training
run that completes in ~10 seconds, verifying the entire train.py workflow.
"""
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from welt_training.train import train


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for training output."""
    temp_dir = tempfile.mkdtemp(prefix="test_train_")
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_basic_training_with_eval_chrf(temp_output_dir):
    """
    Test that a basic training run works end-to-end and reports eval_chrf.

    This test:
    1. Loads the known-good string-repetition config
    2. Modifies it for fast testing (10 steps, small dataset)
    3. Trains for 10 steps (very fast)
    4. Runs evaluation
    5. Verifies that eval_chrf metric is reported
    """
    # Load the base config from the experiment
    base_config_path = Path(__file__).parent.parent / "welt_training/experiments/easy-tasks/string-repetition.yaml"
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Modify config for fast testing
    config.update({
        # Output to temp directory
        "output_dir": temp_output_dir,

        # Minimal training for speed
        "max_steps": 10,
        "max_train_samples": 10,
        "max_eval_samples": 5,

        # Disable reporting
        "report_to": "none",

        # Disable checkpointing
        "save_strategy": "no",

        # Evaluate after training (not at start)
        "eval_on_start": False,
        "eval_steps": 10,

        # Test specifically with chrf metric
        "eval_metrics": ["chrf"],
        "metric_for_best_model": None,  # Disable since we're not saving checkpoints

        # Minimal dataloader for speed
        "dataloader_num_workers": 0,
        "dataloader_prefetch_factor": None,
        "dataloader_pin_memory": False,
        "dataloader_persistent_workers": False,

        # Reduce logging
        "logging_steps": 5,
        "log_samples": 1,

        # Disable bf16 for compatibility
        "bf16": False,
    })

    # Write modified config to temp file
    config_path = Path(temp_output_dir) / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run training
    train(args=str(config_path))

    # Verify that training completed and metrics were saved
    output_dir = Path(temp_output_dir)

    # Check that eval metrics file was created
    eval_results_path = output_dir / "eval_results.json"
    assert eval_results_path.exists(), "eval_results.json should be created"

    # Load and verify metrics
    import json
    with open(eval_results_path) as f:
        eval_metrics = json.load(f)

    # Verify eval_chrf is present
    assert "eval_chrf" in eval_metrics, \
        f"eval_chrf should be in metrics. Found: {list(eval_metrics.keys())}"

    # Verify eval_chrf is a valid number
    chrf_score = eval_metrics["eval_chrf"]
    assert isinstance(chrf_score, int | float), \
        f"eval_chrf should be numeric, got {type(chrf_score)}"
    assert 0 <= chrf_score <= 100, \
        f"eval_chrf should be between 0 and 100, got {chrf_score}"

    # Verify other expected metrics are present
    assert "eval_loss" in eval_metrics, "eval_loss should be present"
    assert "eval_samples" in eval_metrics, "eval_samples should be present"
    assert "perplexity" in eval_metrics, "perplexity should be present"

    print("\n✓ Training completed successfully!")
    print(f"✓ eval_chrf = {chrf_score:.2f}")
    print(f"✓ eval_loss = {eval_metrics['eval_loss']:.4f}")
    print(f"✓ eval_samples = {eval_metrics['eval_samples']}")
    print(f"✓ All metrics: {list(eval_metrics.keys())}")


def test_training_without_generation_metrics(temp_output_dir):
    """Test that training works without generation-based metrics (backward compatibility)."""
    base_config_path = Path(__file__).parent.parent / "welt_training/experiments/easy-tasks/string-repetition.yaml"
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Modify config to disable generation metrics
    config.update({
        "output_dir": temp_output_dir,
        "max_steps": 5,
        "max_train_samples": 10,
        "max_eval_samples": 5,
        "report_to": "none",
        "save_strategy": "no",
        "eval_on_start": False,
        "eval_steps": 5,
        "eval_metrics": None,  # No generation metrics
        "metric_for_best_model": None,  # Disable since no generation metrics
        "dataloader_num_workers": 0,
        "dataloader_prefetch_factor": None,
        "dataloader_pin_memory": False,
        "dataloader_persistent_workers": False,
        "logging_steps": 5,
        "log_samples": 0,
        "bf16": False,
    })

    config_path = Path(temp_output_dir) / "test_config_no_metrics.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run training
    train(args=str(config_path))

    # Verify training completed
    output_dir = Path(temp_output_dir)
    eval_results_path = output_dir / "eval_results.json"
    assert eval_results_path.exists()

    import json
    with open(eval_results_path) as f:
        eval_metrics = json.load(f)

    # Should have loss and perplexity but no generation metrics
    assert "eval_loss" in eval_metrics
    assert "perplexity" in eval_metrics
    assert "eval_samples" in eval_metrics
    # Should not have generation metrics
    assert "eval_chrf" not in eval_metrics
    assert "eval_sacrebleu" not in eval_metrics


def test_training_with_sacrebleu(temp_output_dir):
    """Test training with sacrebleu metric specifically."""
    base_config_path = Path(__file__).parent.parent / "welt_training/experiments/easy-tasks/string-repetition.yaml"
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    config.update({
        "output_dir": temp_output_dir,
        "max_steps": 5,
        "max_train_samples": 10,
        "max_eval_samples": 5,
        "report_to": "none",
        "save_strategy": "no",
        "eval_on_start": False,
        "eval_steps": 5,
        "eval_metrics": ["sacrebleu"],  # Only sacrebleu
        "metric_for_best_model": None,
        "dataloader_num_workers": 0,
        "dataloader_prefetch_factor": None,
        "dataloader_pin_memory": False,
        "dataloader_persistent_workers": False,
        "logging_steps": 5,
        "log_samples": 1,
        "bf16": False,
    })

    config_path = Path(temp_output_dir) / "test_config_sacrebleu.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Run training
    train(args=str(config_path))

    # Verify training completed
    output_dir = Path(temp_output_dir)
    eval_results_path = output_dir / "eval_results.json"
    assert eval_results_path.exists()

    import json
    with open(eval_results_path) as f:
        eval_metrics = json.load(f)

    # Should have sacrebleu
    assert "eval_sacrebleu" in eval_metrics
    assert 0 <= eval_metrics["eval_sacrebleu"] <= 100
    assert "eval_loss" in eval_metrics
    assert "perplexity" in eval_metrics


def test_training_with_streaming(temp_output_dir):
    """Test that training works with streaming=True."""
    base_config_path = Path(__file__).parent.parent / "welt_training/experiments/easy-tasks/string-repetition.yaml"
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    config.update({
        "output_dir": temp_output_dir,
        "max_steps": 5,
        "max_train_samples": 10,
        "max_eval_samples": 5,
        "report_to": "none",
        "save_strategy": "no",
        "eval_on_start": False,
        "eval_steps": 5,
        "do_eval": True,
        "predict_with_generate": True,
        "metric_for_best_model": "chrf",
        "eval_metrics": ["sacrebleu", "chrf"],
        "dataloader_num_workers": 0,
        "dataloader_prefetch_factor": None,
        "dataloader_pin_memory": False,
        "dataloader_persistent_workers": False,
        "logging_steps": 5,
        "log_samples": 1,
        "bf16": False,
        "streaming": True,
    })

    config_path = Path(temp_output_dir) / "test_config_streaming.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    train(args=str(config_path))

    output_dir = Path(temp_output_dir)
    eval_results_path = output_dir / "eval_results.json"
    assert eval_results_path.exists()

    import json
    with open(eval_results_path) as f:
        eval_metrics = json.load(f)

    assert "eval_chrf" in eval_metrics
    assert "eval_loss" in eval_metrics
    assert "perplexity" in eval_metrics



def test_training_determinism(temp_output_dir):
    """Test that training with same seed produces similar results."""
    base_config_path = Path(__file__).parent.parent / "welt_training/experiments/easy-tasks/string-repetition.yaml"
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    config.update({
        "output_dir": temp_output_dir,
        "max_steps": 3,
        "max_train_samples": 5,
        "max_eval_samples": 3,
        "report_to": "none",
        "save_strategy": "no",
        "eval_on_start": False,
        "eval_steps": 3,
        "eval_metrics": ["bleu"],
        "metric_for_best_model": None,
        "dataloader_num_workers": 0,
        "dataloader_prefetch_factor": None,
        "dataloader_pin_memory": False,
        "dataloader_persistent_workers": False,
        "logging_steps": 3,
        "log_samples": 0,
        "bf16": False,
        "seed": 12345,  # Fixed seed for determinism
    })

    # Run 1
    config_path1 = Path(temp_output_dir) / "test_config_run1.yaml"
    with open(config_path1, "w") as f:
        yaml.dump(config, f)

    train(args=str(config_path1))

    import json
    with open(Path(temp_output_dir) / "eval_results.json") as f:
        metrics1 = json.load(f)

    # Clean up for run 2
    shutil.rmtree(temp_output_dir, ignore_errors=True)
    Path(temp_output_dir).mkdir(parents=True, exist_ok=True)

    # Run 2 with same config
    config_path2 = Path(temp_output_dir) / "test_config_run2.yaml"
    with open(config_path2, "w") as f:
        yaml.dump(config, f)

    train(args=str(config_path2))

    with open(Path(temp_output_dir) / "eval_results.json") as f:
        metrics2 = json.load(f)

    # Results should be similar (within tolerance for floating point differences)
    assert abs(metrics1["eval_loss"] - metrics2["eval_loss"]) < 0.5, \
        f"Losses should be similar: {metrics1['eval_loss']} vs {metrics2['eval_loss']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
