import glob
from pathlib import Path

import pytest

from welt_training.extendable_yaml import load_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_nested_extends_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("model:\n  a: 1\n  b: 2\ndata:\n  seq_length: 128\n")
    (tmp_path / "middle.yaml").write_text("# A comment before $extends\n$extends: ./base.yaml\nmodel:\n  b: 3\n")
    (tmp_path / "top.yaml").write_text("$extends: ./middle.yaml\noutput_dir: out\n")
    config = load_yaml(str(tmp_path / "top.yaml"),
                       ["model.a=null", "train.train_iters=5", "data.name=en-he", "optimizer.lr=1e-4"])
    assert config == {"model": {"a": None, "b": 3}, "data": {"seq_length": 128, "name": "en-he"},
                      "output_dir": "out", "train": {"train_iters": 5}, "optimizer": {"lr": 1e-4}}


CONFIGS = sorted([*glob.glob(str(ROOT / "welt_training/experiments/*/*.yaml")),
                  str(ROOT / "benchmarks/welt-bench.yaml")])


@pytest.mark.parametrize("path", CONFIGS)
def test_experiment_configs_reference_existing_models(path):
    """Model configs given as JSON files exist (paths are relative to the repository)."""
    for value in load_yaml(path)["model"].values():
        if isinstance(value, str) and value.endswith(".json"):
            assert (ROOT / value).exists(), value


@pytest.mark.parametrize("path", CONFIGS)
def test_experiment_configs_build(path, monkeypatch):
    """Every shipped config builds a Megatron-Bridge config (unknown options raise)."""
    pytest.importorskip("megatron.bridge", reason="Requires the NeMo container")
    from welt_training import baseline, train

    config = load_yaml(path)
    monkeypatch.chdir(ROOT)  # Model JSON paths are relative to the repository
    (baseline if "transformer" in config["model"] else train).build(config)  # baseline: a causal LM


def test_null_sections_and_model_overrides(monkeypatch):
    pytest.importorskip("megatron.bridge", reason="Requires the NeMo container")
    from welt_training import train

    monkeypatch.chdir(ROOT)
    config = load_yaml("welt_training/experiments/easy-tasks/string-repetition.yaml",
                       ["optimizer=null", "model.hidden_dropout=0.2"])
    cfg, _ = train.build(config)
    assert cfg.model.hidden_dropout == 0.2
    with pytest.raises(TypeError, match="unexpected keyword"):
        train.build(load_yaml("welt_training/experiments/easy-tasks/string-repetition.yaml", ["train.typo=1"]))
    with pytest.raises(ValueError, match="Unknown config sections"):
        train.build(load_yaml("welt_training/experiments/easy-tasks/string-repetition.yaml", ["optimiser.lr=0.1"]))


def test_baseline_model_overrides(monkeypatch):
    pytest.importorskip("megatron.bridge", reason="Requires the NeMo container")
    from welt_training import baseline

    monkeypatch.chdir(ROOT)
    path = "welt_training/experiments/language-modeling/climbmix-baseline.yaml"
    cfg, _ = baseline.build(load_yaml(path, ["model.hidden_dropout=0.2"]))
    assert cfg.model.hidden_dropout == 0.2
    with pytest.raises(ValueError, match="Unknown model option"):
        baseline.build(load_yaml(path, ["model.typo=1"]))
