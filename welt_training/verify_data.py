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

from welt_training.data_utils import find_shard_files


def discover_metadata(data_path):
    """Find all per-split metadata files and return ``[(prefix, metadata)]``.

    Each metadata file is named ``{prefix}-{split}-metadata.json``.  The prefix
    is derived from the filename and the ``split`` field inside the JSON so that
    multiple datasets in one directory are handled correctly.
    """
    entries = []
    for path in sorted(glob.glob(os.path.join(data_path, "*-metadata.json"))):
        with open(path) as f:
            metadata = json.load(f)
        split_name = metadata["split"]
        filename = os.path.basename(path)
        suffix = f"-{split_name}-metadata.json"
        prefix = filename[: -len(suffix)]
        entries.append((prefix, metadata))
    return entries


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

    # 1. Discover metadata (prefix-aware)
    entries = discover_metadata(data_path)
    if not entries:
        messages.append("FAIL: No *-metadata.json files found.")
        return False, messages

    prefixes = sorted({p for p, _ in entries})
    splits = sorted({m["split"] for _, m in entries})
    messages.append(f"Found {len(entries)} metadata file(s): {len(prefixes)} dataset(s), splits: {', '.join(splits)}")

    # 2. Per-dataset, per-split shard consistency
    for prefix, metadata in entries:
        split_name = metadata["split"]
        label = f"{prefix}/{split_name}"
        shard_files = find_shard_files(data_path, split_name, prefix=prefix)

        expected_shards = metadata["num_shards"]
        actual_shards = len(shard_files)
        if actual_shards != expected_shards:
            messages.append(
                f"FAIL [{label}]: Expected {expected_shards} shards, found {actual_shards}."
            )
            passed = False
        else:
            messages.append(f"OK   [{label}]: {actual_shards} shard(s)")

        expected_examples = metadata["num_examples"]
        actual_examples = count_shard_examples(shard_files)
        if actual_examples != expected_examples:
            messages.append(
                f"FAIL [{label}]: Expected {expected_examples} examples, found {actual_examples}."
            )
            passed = False
        else:
            messages.append(f"OK   [{label}]: {actual_examples} example(s)")

    # 3. Data contamination check (per dataset)
    by_prefix = {}
    for prefix, metadata in entries:
        by_prefix.setdefault(prefix, {})[metadata["split"]] = metadata

    for prefix, split_metas in by_prefix.items():
        if "train" not in split_metas or "validation" not in split_metas:
            continue
        train_meta = split_metas["train"]
        val_meta = split_metas["validation"]
        same_source = (
            train_meta["source_dataset"] == val_meta["source_dataset"]
            and train_meta["source_config"] == val_meta["source_config"]
            and train_meta["source_split"] == val_meta["source_split"]
        )
        if same_source:
            if not train_meta.get("created_with_another_split") or not val_meta.get("created_with_another_split"):
                messages.append(
                    f"WARN [{prefix}]: Train and validation share the same source but were not created together. "
                    "Examples may overlap."
                )
            else:
                messages.append(f"OK   [{prefix}/contamination]: Splits created together, no overlap risk.")

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
