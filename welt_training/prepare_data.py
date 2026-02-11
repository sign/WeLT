"""
Data preparation script for offline use.

Downloads HuggingFace datasets, samples raw text with unit-based limits,
and saves sharded .jsonl.gz files for offline training.
"""

import argparse
import gzip
import json
import logging
from pathlib import Path

from datasets import load_dataset
from words_segmentation.tokenizer import WordsSegmentationTokenizer

logger = logging.getLogger(__name__)


def get_shard_prefix(dataset_name: str, dataset_config: str | None) -> str:
    """Derive a shard filename prefix from dataset name and config."""
    name = dataset_name.split("/")[-1]
    if dataset_config:
        return f"{name}-{dataset_config}"
    return name


class ShardWriter:
    """Manages writing sharded .jsonl.gz files for a single data split."""

    def __init__(self, output_path: Path, prefix: str, split_name: str | None, num_units_per_file: int | None):
        self.output_path = output_path
        self.prefix = prefix
        self.split_name = split_name
        self.num_units_per_file = num_units_per_file
        self.shard_index = 0
        self.shard_units = 0
        self.total_units = 0
        self.num_examples = 0
        self._current_file = self._open_shard()

    def _shard_path(self, index: int) -> Path:
        if self.split_name:
            return self.output_path / f"{self.prefix}-{self.split_name}-{index:08d}.jsonl.gz"
        return self.output_path / f"{self.prefix}-{index:08d}.jsonl.gz"

    def _open_shard(self):
        path = self._shard_path(self.shard_index)
        logger.info(f"Writing shard: {path.name}")
        return gzip.open(path, "wt")

    def write(self, record: dict, text_units: int):
        self._current_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.shard_units += text_units
        self.total_units += text_units
        self.num_examples += 1

        if self.num_units_per_file is not None and self.shard_units >= self.num_units_per_file:
            self._current_file.close()
            logger.info(f"Completed shard {self.shard_index} ({self.shard_units} units)")
            self.shard_index += 1
            self.shard_units = 0
            self._current_file = self._open_shard()

    def close(self):
        self._current_file.close()
        # Remove empty last shard
        if self.shard_units == 0 and self.shard_index > 0:
            self._shard_path(self.shard_index).unlink()
            self.shard_index -= 1
        # Remove shard file if no examples were written at all
        if self.num_examples == 0:
            self._shard_path(self.shard_index).unlink(missing_ok=True)

    @property
    def num_shards(self) -> int:
        if self.num_examples == 0:
            return 0
        return self.shard_index + 1


def stream_texts(args):
    """Stream raw text examples from a HuggingFace dataset."""
    load_kwargs = {
        "path": args.dataset_name,
        "split": args.dataset_split,
        "streaming": True,
    }
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config

    logger.info(f"Loading dataset: {args.dataset_name} (config: {args.dataset_config}, split: {args.dataset_split})")
    stream = load_dataset(**load_kwargs)

    # Shuffle with seed
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    for example in stream:
        if args.text_template:
            text = args.text_template.format(**example)
        else:
            text = example[args.text_column]

        if text:
            yield text


def stream_examples(args, pretokenizer: WordsSegmentationTokenizer):
    """Stream text examples, optionally chunked by max_seq_length.

    Uses word segmentation to split text into words (handles Thai and other
    languages without whitespace). When max_seq_length is set, long documents
    are split into chunks of at most max_seq_length words.

    Yields (text, num_words) tuples.
    """
    for text in stream_texts(args):
        words = pretokenizer.tokenize(text)

        if args.max_seq_length is None:
            yield text, len(words)
            continue

        for i in range(0, len(words), args.max_seq_length):
            chunk_words = words[i:i + args.max_seq_length]
            if args.drop_remainder and len(chunk_words) < args.max_seq_length:
                continue
            yield "".join(chunk_words), len(chunk_words)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare HuggingFace datasets for offline training."
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset identifier (required)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="Dataset config name (optional)",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Split to use (default: 'train')",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Column containing text (default: 'text')",
    )
    parser.add_argument(
        "--text_template",
        type=str,
        default=None,
        help="Python format string template (optional)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language tag to store with each example (e.g., 'eng_Latn')",
    )

    # Processing arguments
    parser.add_argument(
        "--unit_type",
        type=str,
        choices=["words", "chars"],
        default="words",
        help="Unit type for counting (default: 'words')",
    )
    parser.add_argument(
        "--train_split_units",
        type=int,
        required=True,
        help="Number of units for the train split.",
    )
    parser.add_argument(
        "--num_units_per_file",
        type=int,
        default=None,
        help="Max units per shard file. If not set, all data goes into one file.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Max words per example. Long documents are split using word segmentation.",
    )
    parser.add_argument(
        "--max_bytes_per_word",
        type=int,
        default=126,
        help="Max UTF-8 bytes per word. Words exceeding this are split. "
             "Should match training config: max_word_length - 2 (default: 128 - 2 = 126).",
    )
    parser.add_argument(
        "--drop_remainder",
        action="store_true",
        help="Drop partial chunks when splitting documents by max_seq_length",
    )
    parser.add_argument(
        "--validation_split_units",
        type=int,
        required=True,
        help="Number of units for the validation split. "
             "Data is already shuffled before splitting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10000,
        help="Buffer size for streaming shuffle (default: 10000)",
    )

    # Output arguments
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory path (required)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Create output directory
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    prefix = get_shard_prefix(args.dataset_name, args.dataset_config)
    pretokenizer = WordsSegmentationTokenizer(max_bytes=args.max_bytes_per_word)

    logger.info("Starting data preparation...")
    logger.info(f"  unit_type={args.unit_type}, train_split_units={args.train_split_units}, "
                f"num_units_per_file={args.num_units_per_file}, max_seq_length={args.max_seq_length}, "
                f"language={args.language}")

    # Create shard writers
    max_total_units = args.train_split_units + args.validation_split_units
    logger.info(f"  validation_split_units={args.validation_split_units}, max_total_units={max_total_units}")
    train_writer = ShardWriter(output_path, prefix, "train", args.num_units_per_file)
    val_writer = ShardWriter(output_path, prefix, "validation", args.num_units_per_file)

    total_units = 0
    total_examples = 0

    for text, num_words in stream_examples(args, pretokenizer):
        text_units = num_words if args.unit_type == "words" else len(text)

        # Check global limit
        if total_units + text_units > max_total_units:
            logger.info(f"Reached total units limit ({max_total_units})")
            break

        record = {"text": text}
        if args.language:
            record["language"] = args.language

        # Fill validation first, then route to train
        if val_writer.total_units < args.validation_split_units:
            val_writer.write(record, text_units)
        else:
            train_writer.write(record, text_units)

        total_units += text_units
        total_examples += 1

        if total_examples % 10000 == 0:
            logger.info(f"Processed {total_examples} examples, {total_units} total {args.unit_type}")

    train_writer.close()

    if total_examples == 0:
        logger.warning("No examples were written. Check dataset and filter settings.")

    # Save metadata
    metadata = {
        "format": "welt-preprocessed-v1",
        "num_examples": total_examples,
        "total_units": total_units,
        "unit_type": args.unit_type,
        "num_shards": train_writer.num_shards,
        "source_dataset": args.dataset_name,
        "source_config": args.dataset_config,
        "source_split": args.dataset_split,
        "language": args.language,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "text_column": args.text_column,
        "text_template": args.text_template,
    }

    val_writer.close()
    metadata["num_shards"] += val_writer.num_shards
    metadata["validation_split_units"] = args.validation_split_units
    metadata["splits"] = {
        "train": {
            "num_examples": train_writer.num_examples,
            "total_units": train_writer.total_units,
            "num_shards": train_writer.num_shards,
        },
        "validation": {
            "num_examples": val_writer.num_examples,
            "total_units": val_writer.total_units,
            "num_shards": val_writer.num_shards,
        },
    }

    metadata_path = output_path / f"{prefix}-metadata.json"
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Data preparation complete!")
    logger.info(f"  - Examples: {total_examples}")
    logger.info(f"  - Total {args.unit_type}: {total_units}")
    logger.info(f"  - Shards: {metadata['num_shards']}")
    logger.info(f"  - Output: {output_path}")


if __name__ == "__main__":
    main()
