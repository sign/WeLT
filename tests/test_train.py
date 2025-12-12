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

from training.train import train


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
    base_config_path = Path(__file__).parent.parent / "training/experiments/easy-tasks/string-repetition.yaml"
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
        "report_to": [],

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
    assert isinstance(chrf_score, (int, float)), \
        f"eval_chrf should be numeric, got {type(chrf_score)}"
    assert 0 <= chrf_score <= 100, \
        f"eval_chrf should be between 0 and 100, got {chrf_score}"

    # Verify other expected metrics are present
    assert "eval_loss" in eval_metrics, "eval_loss should be present"
    assert "eval_samples" in eval_metrics, "eval_samples should be present"
    assert "perplexity" in eval_metrics, "perplexity should be present"

    print(f"\n✓ Training completed successfully!")
    print(f"✓ eval_chrf = {chrf_score:.2f}")
    print(f"✓ eval_loss = {eval_metrics['eval_loss']:.4f}")
    print(f"✓ eval_samples = {eval_metrics['eval_samples']}")
    print(f"✓ All metrics: {list(eval_metrics.keys())}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
