import glob
import gzip
import json
import shutil
import tempfile

import pytest
from datasets import load_dataset

from welt_training.data_utils import load_prepared_data
from welt_training.prepare_data import get_shard_prefix, main

WIKITEXT_DATASET = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"


# --- Helpers ---


def read_shard_examples(output_dir, pattern="*.jsonl.gz"):
    """Read all examples from shards matching *pattern* in *output_dir*."""
    examples = []
    for path in sorted(glob.glob(f"{output_dir}/{pattern}")):
        with gzip.open(path, "rt") as f:
            for line in f:
                examples.append(json.loads(line))
    return examples


def read_metadata(output_dir, dataset_name=WIKITEXT_DATASET, dataset_config=WIKITEXT_CONFIG):
    """Load the metadata JSON produced by prepare_data."""
    prefix = get_shard_prefix(dataset_name, dataset_config)
    with open(f"{output_dir}/{prefix}-metadata.json") as f:
        return json.load(f)


def shard_paths(output_dir, pattern="*.jsonl.gz"):
    """Return sorted list of shard file paths matching *pattern*."""
    return sorted(glob.glob(f"{output_dir}/{pattern}"))


# --- get_shard_prefix ---


def test_get_shard_prefix_with_org():
    assert get_shard_prefix("HuggingFaceFW/fineweb", None) == "fineweb"


def test_get_shard_prefix_with_config():
    assert get_shard_prefix("HuggingFaceFW/fineweb", "sample-10BT") == "fineweb-sample-10BT"


def test_get_shard_prefix_no_org():
    assert get_shard_prefix("wikitext", "wikitext-2-raw-v1") == "wikitext-wikitext-2-raw-v1"


# --- Integration tests ---


@pytest.fixture
def temp_output_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_prepare_data_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_prepare_data_creates_shards(temp_output_dir, monkeypatch):
    """Test that welt-prepare-data creates sharded .jsonl.gz files that can be loaded."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", WIKITEXT_DATASET,
            "--dataset_config", WIKITEXT_CONFIG,
            "--train_split_units", "400",
            "--validation_split_units", "100",
            "--num_units_per_file", "200",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    shards = shard_paths(temp_output_dir)
    assert len(shards) >= 2, f"Expected at least 2 shards, got {len(shards)}"

    metadata = read_metadata(temp_output_dir)
    assert metadata["format"] == "welt-preprocessed-v1"
    assert metadata["total_units"] <= 500
    assert metadata["unit_type"] == "words"
    assert metadata["num_shards"] == len(shards)

    examples = read_shard_examples(temp_output_dir)
    for example in examples:
        assert "text" in example
        assert isinstance(example["text"], str)
    assert len(examples) == metadata["num_examples"]

    # Verify loading with HuggingFace datasets (same path as train.py)
    ds = load_dataset("json", data_files=shards, split="train")
    assert len(ds) == len(examples)
    assert "text" in ds.features


def test_prepare_data_with_language(temp_output_dir, monkeypatch):
    """Test that --language stores language metadata in each example."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", WIKITEXT_DATASET,
            "--dataset_config", WIKITEXT_CONFIG,
            "--train_split_units", "160",
            "--validation_split_units", "40",
            "--language", "eng_Latn",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    for example in read_shard_examples(temp_output_dir):
        assert example["language"] == "eng_Latn"

    metadata = read_metadata(temp_output_dir)
    assert metadata["language"] == "eng_Latn"


def test_prepare_data_unit_type_chars(temp_output_dir, monkeypatch):
    """Test that --unit_type chars counts characters instead of words."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", WIKITEXT_DATASET,
            "--dataset_config", WIKITEXT_CONFIG,
            "--train_split_units", "400",
            "--validation_split_units", "100",
            "--unit_type", "chars",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    metadata = read_metadata(temp_output_dir)
    assert metadata["unit_type"] == "chars"
    assert metadata["total_units"] <= 500


def test_prepare_data_with_max_seq_length(temp_output_dir, monkeypatch):
    """Test that --max_seq_length splits long documents into chunks."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", WIKITEXT_DATASET,
            "--dataset_config", WIKITEXT_CONFIG,
            "--train_split_units", "400",
            "--validation_split_units", "100",
            "--max_seq_length", "32",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    metadata = read_metadata(temp_output_dir)
    assert metadata["max_seq_length"] == 32
    assert metadata["total_units"] <= 500

    # Verify each example has at most max_seq_length words
    from words_segmentation.tokenizer import WordsSegmentationTokenizer
    pretokenizer = WordsSegmentationTokenizer(max_bytes=126)

    for example in read_shard_examples(temp_output_dir):
        words = pretokenizer.tokenize(example["text"])
        assert len(words) <= 32


def test_prepare_data_with_validation_split(temp_output_dir, monkeypatch):
    """Test that --validation_split_units creates split-aware shards."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", WIKITEXT_DATASET,
            "--dataset_config", WIKITEXT_CONFIG,
            "--train_split_units", "800",
            "--validation_split_units", "200",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    prefix = get_shard_prefix(WIKITEXT_DATASET, WIKITEXT_CONFIG)

    # Verify split-aware shard files were created
    train_files = shard_paths(temp_output_dir, f"{prefix}-train-*.jsonl.gz")
    val_files = shard_paths(temp_output_dir, f"{prefix}-validation-*.jsonl.gz")
    assert len(train_files) >= 1, f"Expected at least 1 train shard, got {len(train_files)}"
    assert len(val_files) >= 1, f"Expected at least 1 validation shard, got {len(val_files)}"

    # No legacy (unsplit) shards should exist
    all_shards = shard_paths(temp_output_dir)
    assert len(all_shards) == len(train_files) + len(val_files)

    # Count examples per split
    train_examples = read_shard_examples(temp_output_dir, f"{prefix}-train-*.jsonl.gz")
    val_examples = read_shard_examples(temp_output_dir, f"{prefix}-validation-*.jsonl.gz")
    for example in train_examples + val_examples:
        assert "text" in example

    total_examples = len(train_examples) + len(val_examples)
    assert total_examples > 0

    # Verify validation fraction is roughly correct (20% +/- tolerance)
    val_fraction = len(val_examples) / total_examples
    assert 0.05 < val_fraction < 0.45, f"Expected ~20% validation, got {val_fraction:.1%}"

    # Verify metadata
    metadata = read_metadata(temp_output_dir)
    assert metadata["format"] == "welt-preprocessed-v1"
    assert metadata["validation_split_units"] == 200
    assert metadata["num_examples"] == total_examples
    assert "splits" in metadata
    assert metadata["splits"]["train"]["num_examples"] == len(train_examples)
    assert metadata["splits"]["validation"]["num_examples"] == len(val_examples)
    assert metadata["splits"]["train"]["num_shards"] == len(train_files)
    assert metadata["splits"]["validation"]["num_shards"] == len(val_files)


def test_load_prepared_data_split_aware(temp_output_dir, monkeypatch):
    """Test that load_prepared_data detects and loads split-aware shards."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", WIKITEXT_DATASET,
            "--dataset_config", WIKITEXT_CONFIG,
            "--train_split_units", "800",
            "--validation_split_units", "200",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    # load_prepared_data should detect split-aware files and load them directly
    result = load_prepared_data(temp_output_dir)
    assert "train" in result
    assert "validation" in result
    assert len(result["train"]) > 0
    assert len(result["validation"]) > 0
    assert "text" in result["train"].features
    assert "text" in result["validation"].features

    # Total should match what was prepared
    metadata = read_metadata(temp_output_dir)
    assert len(result["train"]) + len(result["validation"]) == metadata["num_examples"]


def test_load_prepared_data_requires_validation_shards(temp_output_dir):
    """Test that load_prepared_data raises when validation shards are missing."""
    with pytest.raises(ValueError, match="train"):
        load_prepared_data(temp_output_dir)
