import json

import pytest
import yaml
from datasets import Dataset

from welt_training import evaluate


@pytest.fixture
def config(tmp_path):
    for split, rows in [("train", range(0, 10)), ("validation", range(100, 104))]:
        Dataset.from_dict({"a": [f"a{i}" for i in rows], "b": [f"b{i}" for i in rows]}).to_json(
            tmp_path / f"{split}.jsonl")
    path = tmp_path / "welt.yaml"
    path.write_text(yaml.safe_dump({"data": {
        "dataset_name": "json",
        "data_files": {"train": str(tmp_path / "train.jsonl"), "validation": str(tmp_path / "validation.jsonl")},
        "dataset_text_template": ["{a} ", "{b}"], "seq_length": 8}}))
    return path


def test_evaluate_generates_completions_of_validation_prefixes(config, monkeypatch):
    requests = []

    def generate(url, texts, max_generated_words):
        requests.append(texts)
        # Correct completions, except for the first prefix
        return {"outputs": ["wrong", *(text.replace("a", "b").strip() for text in texts[1:])],
                "generated_words": len(texts)}

    monkeypatch.setattr(evaluate, "generate", generate)
    results = evaluate.evaluate(str(config), "http://welt", max_samples=10)
    assert requests[-1] == ["a100 ", "a101 ", "a102 ", "a103 "]  # The validation split only
    assert results["samples"] == 4
    assert results["exact_match"] == 0.75
    assert results["examples"][1] == {"prefix": "a101 ", "reference": "b101", "prediction": "b101"}
    json.dumps(results)  # Serializable


def test_evaluation_stays_within_the_training_holdout(config, monkeypatch):
    """Without a validation split, the first max_eval_samples train examples are held out from training."""
    data = yaml.safe_load(config.read_text())["data"]
    data |= {"data_files": {"train": data["data_files"]["train"]}, "max_eval_samples": 2}
    config.write_text(yaml.safe_dump({"data": data}))
    monkeypatch.setattr(evaluate, "generate", lambda url, texts, max_generated_words: {
        "outputs": texts, "generated_words": len(texts)})
    results = evaluate.evaluate(str(config), "http://welt", max_samples=10)
    assert [example["prefix"] for example in results["examples"]] == ["a0 ", "a1 "]


def test_evaluate_requires_prefix_and_completion(config):
    data = yaml.safe_load(config.read_text())["data"] | {"dataset_text_template": "{a} {b}"}
    config.write_text(yaml.safe_dump({"data": data}))
    with pytest.raises(ValueError, match="prefix, completion"):
        evaluate.evaluate(str(config), "http://welt")
