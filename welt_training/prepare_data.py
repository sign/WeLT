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
        "--max_total_units",
        type=int,
        default=None,
        help="Max total units to sample. If not set, processes the entire dataset.",
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
    logger.info(f"  unit_type={args.unit_type}, max_total_units={args.max_total_units}, "
                f"num_units_per_file={args.num_units_per_file}, max_seq_length={args.max_seq_length}, "
                f"language={args.language}")

    shard_index = 0
    shard_units = 0
    total_units = 0
    num_examples = 0

    def open_shard(index: int):
        path = output_path / f"{prefix}-{index:08d}.jsonl.gz"
        logger.info(f"Writing shard: {path.name}")
        return gzip.open(path, "wt")

    current_file = open_shard(shard_index)

    for text, num_words in stream_examples(args, pretokenizer):
        text_units = num_words if args.unit_type == "words" else len(text)

        # Check global limit
        if args.max_total_units is not None and total_units + text_units > args.max_total_units:
            logger.info(f"Reached max_total_units limit ({args.max_total_units})")
            break

        record = {"text": text}
        if args.language:
            record["language"] = args.language

        current_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        shard_units += text_units
        total_units += text_units
        num_examples += 1

        if num_examples % 10000 == 0:
            logger.info(f"Processed {num_examples} examples, {total_units} total {args.unit_type}")

        # Check shard limit
        if args.num_units_per_file is not None and shard_units >= args.num_units_per_file:
            current_file.close()
            logger.info(f"Completed shard {shard_index} ({shard_units} {args.unit_type})")
            shard_index += 1
            shard_units = 0
            current_file = open_shard(shard_index)

    current_file.close()

    # Remove empty last shard
    last_shard_path = output_path / f"{prefix}-{shard_index:08d}.jsonl.gz"
    if shard_units == 0 and shard_index > 0:
        last_shard_path.unlink()
        shard_index -= 1

    num_shards = shard_index + 1

    if num_examples == 0:
        logger.warning("No examples were written. Check dataset and filter settings.")

    # Save metadata
    metadata = {
        "format": "welt-preprocessed-v1",
        "num_examples": num_examples,
        "total_units": total_units,
        "unit_type": args.unit_type,
        "num_shards": num_shards,
        "source_dataset": args.dataset_name,
        "source_config": args.dataset_config,
        "source_split": args.dataset_split,
        "language": args.language,
        "max_seq_length": args.max_seq_length,
        "seed": args.seed,
        "text_column": args.text_column,
        "text_template": args.text_template,
    }

    metadata_path = output_path / f"{prefix}-metadata.json"
    logger.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Data preparation complete!")
    logger.info(f"  - Examples: {num_examples}")
    logger.info(f"  - Total {args.unit_type}: {total_units}")
    logger.info(f"  - Shards: {num_shards}")
    logger.info(f"  - Output: {output_path}")


if __name__ == "__main__":
    main()
