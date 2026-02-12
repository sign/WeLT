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


def read_metadata(output_dir, split_name, dataset_name=WIKITEXT_DATASET, dataset_config=WIKITEXT_CONFIG):
    """Load the per-split metadata JSON produced by prepare_data."""
    prefix = get_shard_prefix(dataset_name, dataset_config)
    with open(f"{output_dir}/{prefix}-{split_name}-metadata.json") as f:
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
            "--max_seq_length", "1024",
            "--language", "eng_Latn",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    shards = shard_paths(temp_output_dir)
    assert len(shards) >= 2, f"Expected at least 2 shards, got {len(shards)}"

    train_meta = read_metadata(temp_output_dir, "train")
    val_meta = read_metadata(temp_output_dir, "validation")
    assert train_meta["format"] == "welt-preprocessed-v1"
    assert train_meta["total_units"] + val_meta["total_units"] <= 500
    assert train_meta["unit_type"] == "words"
    assert train_meta["num_shards"] + val_meta["num_shards"] == len(shards)

    examples = read_shard_examples(temp_output_dir)
    for example in examples:
        assert "text" in example
        assert isinstance(example["text"], str)
    assert len(examples) == train_meta["num_examples"] + val_meta["num_examples"]

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
            "--max_seq_length", "1024",
            "--language", "eng_Latn",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    for example in read_shard_examples(temp_output_dir):
        assert example["language"] == "eng_Latn"

    assert read_metadata(temp_output_dir, "train")["language"] == "eng_Latn"
    assert read_metadata(temp_output_dir, "validation")["language"] == "eng_Latn"


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
            "--max_seq_length", "1024",
            "--language", "eng_Latn",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    meta = read_metadata(temp_output_dir, "validation")
    assert meta["unit_type"] == "chars"
    assert meta["total_units"] <= 500


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
            "--language", "eng_Latn",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    train_meta = read_metadata(temp_output_dir, "train")
    val_meta = read_metadata(temp_output_dir, "validation")
    assert train_meta["max_seq_length"] == 32
    assert train_meta["total_units"] + val_meta["total_units"] <= 500

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
            "--max_seq_length", "1024",
            "--language", "eng_Latn",
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

    # Verify per-split metadata
    train_meta = read_metadata(temp_output_dir, "train")
    val_meta = read_metadata(temp_output_dir, "validation")
    assert train_meta["format"] == "welt-preprocessed-v1"
    assert train_meta["num_examples"] == len(train_examples)
    assert val_meta["num_examples"] == len(val_examples)
    assert train_meta["num_shards"] == len(train_files)
    assert val_meta["num_shards"] == len(val_files)


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
            "--max_seq_length", "1024",
            "--language", "eng_Latn",
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
    train_meta = read_metadata(temp_output_dir, "train")
    val_meta = read_metadata(temp_output_dir, "validation")
    assert len(result["train"]) + len(result["validation"]) == train_meta["num_examples"] + val_meta["num_examples"]


def test_load_prepared_data_requires_some_shards(temp_output_dir):
    """Test that load_prepared_data raises when no shards exist."""
    with pytest.raises(ValueError, match="No"):
        load_prepared_data(temp_output_dir)


def test_prepare_data_train_only(temp_output_dir, monkeypatch):
    """Test that setting validation_split_units=0 creates only train shards."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", WIKITEXT_DATASET,
            "--dataset_config", WIKITEXT_CONFIG,
            "--train_split_units", "400",
            "--max_seq_length", "1024",
            "--language", "eng_Latn",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    prefix = get_shard_prefix(WIKITEXT_DATASET, WIKITEXT_CONFIG)
    train_files = shard_paths(temp_output_dir, f"{prefix}-train-*.jsonl.gz")
    val_files = shard_paths(temp_output_dir, f"{prefix}-validation-*.jsonl.gz")
    assert len(train_files) >= 1
    assert len(val_files) == 0

    metadata = read_metadata(temp_output_dir, "train")
    assert metadata["num_examples"] > 0
    assert not glob.glob(f"{temp_output_dir}/*-validation-metadata.json")

    result = load_prepared_data(temp_output_dir)
    assert "train" in result
    assert "validation" not in result


def test_prepare_data_validation_only(temp_output_dir, monkeypatch):
    """Test that setting train_split_units=0 creates only validation shards."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", WIKITEXT_DATASET,
            "--dataset_config", WIKITEXT_CONFIG,
            "--validation_split_units", "200",
            "--max_seq_length", "1024",
            "--language", "eng_Latn",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    prefix = get_shard_prefix(WIKITEXT_DATASET, WIKITEXT_CONFIG)
    train_files = shard_paths(temp_output_dir, f"{prefix}-train-*.jsonl.gz")
    val_files = shard_paths(temp_output_dir, f"{prefix}-validation-*.jsonl.gz")
    assert len(train_files) == 0
    assert len(val_files) >= 1

    metadata = read_metadata(temp_output_dir, "validation")
    assert metadata["num_examples"] > 0
    assert not glob.glob(f"{temp_output_dir}/*-train-metadata.json")

    result = load_prepared_data(temp_output_dir)
    assert "train" not in result
    assert "validation" in result


def test_prepare_data_with_id_column(temp_output_dir, monkeypatch):
    """Test that --id_column preserves the source column as 'id' in output."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "welt-prepare-data",
            "--dataset_name", "HuggingFaceFW/fineweb-2",
            "--dataset_config", "tur_Latn",
            "--train_split_units", "400",
            "--id_column", "id",
            "--max_seq_length", "1024",
            "--language", "tur_Latn",
            "--seed", "42",
            "--output_path", temp_output_dir,
        ],
    )
    main()

    for example in read_shard_examples(temp_output_dir):
        assert "id" in example
