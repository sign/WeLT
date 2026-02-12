"""Verify integrity of prepared data directories.

Checks metadata consistency, shard counts, example counts,
and warns about potential data contamination between splits.
"""

import argparse
import glob
import gzip
import json
import os
import sys


def discover_splits(data_path):
    """Find all per-split metadata files and return {split_name: metadata_dict}."""
    splits = {}
    for path in sorted(glob.glob(os.path.join(data_path, "*-metadata.json"))):
        with open(path) as f:
            metadata = json.load(f)
        split_name = metadata["split"]
        splits[split_name] = metadata
    return splits


def count_shard_examples(shard_files):
    """Count total examples across a list of .jsonl.gz shard files."""
    count = 0
    for path in shard_files:
        with gzip.open(path, "rt") as f:
            for _ in f:
                count += 1
    return count


def verify(data_path):
    """Run all verification checks. Returns (passed: bool, messages: list[str])."""
    messages = []
    passed = True

    # 1. Discover splits
    splits = discover_splits(data_path)
    if not splits:
        messages.append("FAIL: No *-metadata.json files found.")
        return False, messages

    messages.append(f"Found splits: {', '.join(splits)}")

    # 2. Per-split shard consistency
    for split_name, metadata in splits.items():
        pattern = os.path.join(data_path, f"*-{split_name}-*.jsonl.gz")
        shard_files = sorted(glob.glob(pattern))

        expected_shards = metadata["num_shards"]
        actual_shards = len(shard_files)
        if actual_shards != expected_shards:
            messages.append(
                f"FAIL [{split_name}]: Expected {expected_shards} shards, found {actual_shards}."
            )
            passed = False
        else:
            messages.append(f"OK   [{split_name}]: {actual_shards} shard(s)")

        expected_examples = metadata["num_examples"]
        actual_examples = count_shard_examples(shard_files)
        if actual_examples != expected_examples:
            messages.append(
                f"FAIL [{split_name}]: Expected {expected_examples} examples, found {actual_examples}."
            )
            passed = False
        else:
            messages.append(f"OK   [{split_name}]: {actual_examples} example(s)")

    # 3. Data contamination check
    if "train" in splits and "validation" in splits:
        train_meta = splits["train"]
        val_meta = splits["validation"]
        same_source = (
            train_meta["source_dataset"] == val_meta["source_dataset"]
            and train_meta["source_config"] == val_meta["source_config"]
            and train_meta["source_split"] == val_meta["source_split"]
        )
        if same_source:
            if not train_meta.get("created_with_another_split") or not val_meta.get("created_with_another_split"):
                messages.append(
                    "WARN: Train and validation share the same source but were not created together. "
                    "Examples may overlap."
                )
            else:
                messages.append("OK   [contamination]: Splits created together, no overlap risk.")

    return passed, messages


def main():
    parser = argparse.ArgumentParser(description="Verify integrity of prepared data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to prepared data directory.")
    args = parser.parse_args()

    passed, messages = verify(args.data_path)
    for msg in messages:
        print(msg)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
