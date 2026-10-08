import glob

import pytest

from welt_training.extendable_yaml import load_yaml


def test_nested_extends_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("model:\n  a: 1\n  b: 2\ndata:\n  seq_length: 128\n")
    (tmp_path / "middle.yaml").write_text("# A comment before $extends\n$extends: ./base.yaml\nmodel:\n  b: 3\n")
    (tmp_path / "top.yaml").write_text("$extends: ./middle.yaml\noutput_dir: out\n")
    config = load_yaml(str(tmp_path / "top.yaml"),
                       ["model.a=null", "train.train_iters=5", "data.name=en-he", "optimizer.lr=1e-4"])
    assert config == {"model": {"a": None, "b": 3}, "data": {"seq_length": 128, "name": "en-he"},
                      "output_dir": "out", "train": {"train_iters": 5}, "optimizer": {"lr": 1e-4}}


CONFIGS = sorted(glob.glob("welt_training/experiments/*/*.yaml"))


@pytest.mark.parametrize("path", CONFIGS)
def test_experiment_configs_build(path):
    """Every shipped config builds a Megatron-Bridge config (unknown options raise)."""
    pytest.importorskip("megatron.bridge", reason="Requires the NeMo container")
    from welt_training import baseline, train

    config = load_yaml(path)
    (baseline if "transformer" in config["model"] else train).build(config)  # baseline: a causal LM


def test_null_sections_and_model_overrides():
    pytest.importorskip("megatron.bridge", reason="Requires the NeMo container")
    from welt_training import train

    config = load_yaml("welt_training/experiments/easy-tasks/string-repetition.yaml",
                       ["optimizer=null", "model.tensor_model_parallel_size=2"])
    cfg, _ = train.build(config)
    assert cfg.model.tensor_model_parallel_size == 2
    with pytest.raises(TypeError, match="unexpected keyword"):
        train.build(load_yaml("welt_training/experiments/easy-tasks/string-repetition.yaml", ["train.typo=1"]))
    with pytest.raises(ValueError, match="Unknown config sections"):
        train.build(load_yaml("welt_training/experiments/easy-tasks/string-repetition.yaml", ["optimiser.lr=0.1"]))
