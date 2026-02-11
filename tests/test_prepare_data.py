import glob
import gzip
import json
import shutil
import tempfile

import pytest
from datasets import load_dataset

from welt_training.data_utils import load_prepared_data
from welt_training.prepare_data import get_shard_prefix, main

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
            "--dataset_name", "wikitext",
            "--dataset_config", "wikitext-2-raw-v1",
            "--max_total_units", "500",
            "--num_units_per_file", "200",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    # Verify shards were created
    shard_files = sorted(glob.glob(f"{temp_output_dir}/*.jsonl.gz"))
    assert len(shard_files) >= 2, f"Expected at least 2 shards, got {len(shard_files)}"

    # Verify metadata
    prefix = get_shard_prefix("wikitext", "wikitext-2-raw-v1")
    with open(f"{temp_output_dir}/{prefix}-metadata.json") as f:
        metadata = json.load(f)
    assert metadata["format"] == "welt-preprocessed-v1"
    assert metadata["total_units"] <= 500
    assert metadata["unit_type"] == "words"
    assert metadata["num_shards"] == len(shard_files)

    # Verify each shard is valid gzipped JSONL with a "text" field
    total_examples = 0
    for path in shard_files:
        with gzip.open(path, "rt") as f:
            for line in f:
                example = json.loads(line)
                assert "text" in example
                assert isinstance(example["text"], str)
                total_examples += 1
    assert total_examples == metadata["num_examples"]

    # Verify loading with HuggingFace datasets (same path as train.py)
    ds = load_dataset("json", data_files=shard_files, split="train")
    assert len(ds) == total_examples
    assert "text" in ds.features


def test_prepare_data_with_language(temp_output_dir, monkeypatch):
    """Test that --language stores language metadata in each example."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", "wikitext",
            "--dataset_config", "wikitext-2-raw-v1",
            "--max_total_units", "200",
            "--language", "eng_Latn",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    shard_files = sorted(glob.glob(f"{temp_output_dir}/*.jsonl.gz"))
    for path in shard_files:
        with gzip.open(path, "rt") as f:
            for line in f:
                example = json.loads(line)
                assert example["language"] == "eng_Latn"

    prefix = get_shard_prefix("wikitext", "wikitext-2-raw-v1")
    with open(f"{temp_output_dir}/{prefix}-metadata.json") as f:
        metadata = json.load(f)
    assert metadata["language"] == "eng_Latn"


def test_prepare_data_unit_type_chars(temp_output_dir, monkeypatch):
    """Test that --unit_type chars counts characters instead of words."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", "wikitext",
            "--dataset_config", "wikitext-2-raw-v1",
            "--max_total_units", "500",
            "--unit_type", "chars",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    prefix = get_shard_prefix("wikitext", "wikitext-2-raw-v1")
    with open(f"{temp_output_dir}/{prefix}-metadata.json") as f:
        metadata = json.load(f)
    assert metadata["unit_type"] == "chars"
    assert metadata["total_units"] <= 500


def test_prepare_data_with_max_seq_length(temp_output_dir, monkeypatch):
    """Test that --max_seq_length splits long documents into chunks."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", "wikitext",
            "--dataset_config", "wikitext-2-raw-v1",
            "--max_total_units", "500",
            "--max_seq_length", "32",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    prefix = get_shard_prefix("wikitext", "wikitext-2-raw-v1")
    with open(f"{temp_output_dir}/{prefix}-metadata.json") as f:
        metadata = json.load(f)
    assert metadata["max_seq_length"] == 32
    assert metadata["total_units"] <= 500

    # Verify each example has at most max_seq_length words
    from words_segmentation.tokenizer import WordsSegmentationTokenizer
    pretokenizer = WordsSegmentationTokenizer(max_bytes=126)

    shard_files = sorted(glob.glob(f"{temp_output_dir}/*.jsonl.gz"))
    for path in shard_files:
        with gzip.open(path, "rt") as f:
            for line in f:
                example = json.loads(line)
                words = pretokenizer.tokenize(example["text"])
                assert len(words) <= 32


def test_prepare_data_with_validation_split(temp_output_dir, monkeypatch):
    """Test that --validation_split_percentage creates split-aware shards."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", "wikitext",
            "--dataset_config", "wikitext-2-raw-v1",
            "--max_total_units", "1000",
            "--validation_split_percentage", "20",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    prefix = get_shard_prefix("wikitext", "wikitext-2-raw-v1")

    # Verify split-aware shard files were created
    train_files = sorted(glob.glob(f"{temp_output_dir}/{prefix}-train-*.jsonl.gz"))
    val_files = sorted(glob.glob(f"{temp_output_dir}/{prefix}-validation-*.jsonl.gz"))
    assert len(train_files) >= 1, f"Expected at least 1 train shard, got {len(train_files)}"
    assert len(val_files) >= 1, f"Expected at least 1 validation shard, got {len(val_files)}"

    # No legacy (unsplit) shards should exist
    all_shards = sorted(glob.glob(f"{temp_output_dir}/*.jsonl.gz"))
    assert len(all_shards) == len(train_files) + len(val_files)

    # Count examples per split
    train_examples = 0
    for path in train_files:
        with gzip.open(path, "rt") as f:
            for line in f:
                example = json.loads(line)
                assert "text" in example
                train_examples += 1

    val_examples = 0
    for path in val_files:
        with gzip.open(path, "rt") as f:
            for line in f:
                example = json.loads(line)
                assert "text" in example
                val_examples += 1

    total_examples = train_examples + val_examples
    assert total_examples > 0

    # Verify validation fraction is roughly correct (20% +/- tolerance)
    val_fraction = val_examples / total_examples
    assert 0.05 < val_fraction < 0.45, f"Expected ~20% validation, got {val_fraction:.1%}"

    # Verify metadata
    with open(f"{temp_output_dir}/{prefix}-metadata.json") as f:
        metadata = json.load(f)
    assert metadata["format"] == "welt-preprocessed-v1"
    assert metadata["validation_split_percentage"] == 20
    assert metadata["num_examples"] == total_examples
    assert "splits" in metadata
    assert metadata["splits"]["train"]["num_examples"] == train_examples
    assert metadata["splits"]["validation"]["num_examples"] == val_examples
    assert metadata["splits"]["train"]["num_shards"] == len(train_files)
    assert metadata["splits"]["validation"]["num_shards"] == len(val_files)


def test_load_prepared_data_split_aware(temp_output_dir, monkeypatch):
    """Test that load_prepared_data detects and loads split-aware shards."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", "wikitext",
            "--dataset_config", "wikitext-2-raw-v1",
            "--max_total_units", "1000",
            "--validation_split_percentage", "20",
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
    prefix = get_shard_prefix("wikitext", "wikitext-2-raw-v1")
    with open(f"{temp_output_dir}/{prefix}-metadata.json") as f:
        metadata = json.load(f)
    assert len(result["train"]) + len(result["validation"]) == metadata["num_examples"]


def test_load_prepared_data_legacy(temp_output_dir, monkeypatch):
    """Test that load_prepared_data falls back to legacy mode for unsplit shards."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", "wikitext",
            "--dataset_config", "wikitext-2-raw-v1",
            "--max_total_units", "500",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    # Without --validation_split_percentage, shards have no split marker
    result = load_prepared_data(temp_output_dir, validation_split_percentage=10, seed=42)
    assert "train" in result
    assert "validation" in result
    assert len(result["train"]) > 0
    assert len(result["validation"]) > 0
