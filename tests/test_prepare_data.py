import glob
import gzip
import json
import shutil
import tempfile

import pytest
from datasets import load_dataset

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
