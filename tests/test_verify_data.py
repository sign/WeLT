import gzip
import json
import shutil
import tempfile

import pytest

from welt_training.verify_data import verify


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="test_verify_data_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def write_metadata(path, metadata):
    with open(path, "w") as f:
        json.dump(metadata, f)


def write_shard(path, examples):
    with gzip.open(path, "wt") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def make_metadata(split, num_examples, num_shards, **overrides):
    base = {
        "format": "welt-preprocessed-v1",
        "split": split,
        "num_examples": num_examples,
        "num_shards": num_shards,
        "total_units": num_examples * 10,
        "unit_type": "words",
        "source_dataset": "test-dataset",
        "source_config": "test-config",
        "source_split": "train",
        "language": "eng_Latn",
        "max_seq_length": None,
        "seed": 42,
        "text_column": "text",
        "text_template": None,
        "created_with_another_split": False,
    }
    base.update(overrides)
    return base


def test_verify_empty_dir(temp_dir):
    passed, messages = verify(temp_dir)
    assert not passed
    assert any("No" in m for m in messages)


def test_verify_consistent_single_split(temp_dir):
    examples = [{"text": f"example {i}"} for i in range(5)]
    write_shard(f"{temp_dir}/ds-train-00000000.jsonl.gz", examples)
    write_metadata(f"{temp_dir}/ds-train-metadata.json", make_metadata("train", 5, 1))

    passed, messages = verify(temp_dir)
    assert passed


def test_verify_shard_count_mismatch(temp_dir):
    examples = [{"text": "hello"}]
    write_shard(f"{temp_dir}/ds-train-00000000.jsonl.gz", examples)
    write_metadata(f"{temp_dir}/ds-train-metadata.json", make_metadata("train", 1, 2))

    passed, messages = verify(temp_dir)
    assert not passed
    assert any("Expected 2 shards" in m for m in messages)


def test_verify_example_count_mismatch(temp_dir):
    examples = [{"text": "hello"}, {"text": "world"}]
    write_shard(f"{temp_dir}/ds-train-00000000.jsonl.gz", examples)
    write_metadata(f"{temp_dir}/ds-train-metadata.json", make_metadata("train", 5, 1))

    passed, messages = verify(temp_dir)
    assert not passed
    assert any("Expected 5 examples" in m for m in messages)


def test_verify_contamination_warning(temp_dir):
    """Splits from same source created separately should warn."""
    write_shard(f"{temp_dir}/ds-train-00000000.jsonl.gz", [{"text": "a"}])
    write_shard(f"{temp_dir}/ds-validation-00000000.jsonl.gz", [{"text": "b"}])
    write_metadata(f"{temp_dir}/ds-train-metadata.json",
                   make_metadata("train", 1, 1, created_with_another_split=False))
    write_metadata(f"{temp_dir}/ds-validation-metadata.json",
                   make_metadata("validation", 1, 1, created_with_another_split=False))

    passed, messages = verify(temp_dir)
    assert passed  # warning, not failure
    assert any("WARN" in m and "overlap" in m for m in messages)


def test_verify_no_contamination_when_created_together(temp_dir):
    """Splits created together should pass contamination check."""
    write_shard(f"{temp_dir}/ds-train-00000000.jsonl.gz", [{"text": "a"}])
    write_shard(f"{temp_dir}/ds-validation-00000000.jsonl.gz", [{"text": "b"}])
    write_metadata(f"{temp_dir}/ds-train-metadata.json",
                   make_metadata("train", 1, 1, created_with_another_split=True))
    write_metadata(f"{temp_dir}/ds-validation-metadata.json",
                   make_metadata("validation", 1, 1, created_with_another_split=True))

    passed, messages = verify(temp_dir)
    assert passed
    assert any("no overlap" in m for m in messages)


def test_verify_no_contamination_different_sources(temp_dir):
    """Splits from different sources should skip contamination check."""
    write_shard(f"{temp_dir}/ds1-train-00000000.jsonl.gz", [{"text": "a"}])
    write_shard(f"{temp_dir}/ds2-validation-00000000.jsonl.gz", [{"text": "b"}])
    write_metadata(f"{temp_dir}/ds1-train-metadata.json",
                   make_metadata("train", 1, 1, source_dataset="dataset-A"))
    write_metadata(f"{temp_dir}/ds2-validation-metadata.json",
                   make_metadata("validation", 1, 1, source_dataset="dataset-B"))

    passed, messages = verify(temp_dir)
    assert passed
    # No contamination message since sources differ
    assert not any("overlap" in m for m in messages)
    assert not any("contamination" in m for m in messages)
