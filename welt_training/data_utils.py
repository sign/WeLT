import glob
import logging
import os

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_prepared_data(prepared_data_path: str, validation_split_percentage: int | None = None, seed: int = 42):
    """Load preprocessed shards produced by prepare_data.py.

    Supports two shard naming conventions:

    - **Split-aware** (preferred): Files named ``{prefix}-train-*.jsonl.gz``
      and ``{prefix}-validation-*.jsonl.gz``.  Train and validation datasets
      are loaded directly from their respective shards, ensuring each source
      dataset contributes proportionally to both splits.  The
      ``validation_split_percentage`` parameter is ignored in this mode.

    - **Legacy**: Files named ``{prefix}-*.jsonl.gz`` without split markers.
      A random ``train_test_split`` is applied using
      ``validation_split_percentage``.

    Args:
        prepared_data_path: Directory containing ``*.jsonl.gz`` shard files.
        validation_split_percentage: If set, split the data into train/validation
            (only used for legacy shards without split markers).
        seed: Random seed for the train/test split (legacy mode only).

    Returns:
        A dict with ``"train"`` (and optionally ``"validation"``) datasets.
    """
    # Check for split-aware files produced with --validation_split_percentage
    train_files = sorted(glob.glob(os.path.join(prepared_data_path, "*-train-*.jsonl.gz")))
    validation_files = sorted(glob.glob(os.path.join(prepared_data_path, "*-validation-*.jsonl.gz")))

    if train_files:
        logger.info(
            f"Loading split-aware data: {len(train_files)} train shard(s), "
            f"{len(validation_files)} validation shard(s) from {prepared_data_path}"
        )
        result = {"train": load_dataset("json", data_files=train_files, split="train")}
        if validation_files:
            result["validation"] = load_dataset("json", data_files=validation_files, split="train")
        return result

    # Legacy mode: all shards in a single pool
    data_files = sorted(glob.glob(os.path.join(prepared_data_path, "*.jsonl.gz")))
    if not data_files:
        raise ValueError(f"No .jsonl.gz files found in {prepared_data_path}")
    logger.info(f"Loading prepared data from {len(data_files)} shard(s) in {prepared_data_path}")
    train_data = load_dataset("json", data_files=data_files, split="train")

    if validation_split_percentage:
        split = train_data.train_test_split(
            test_size=validation_split_percentage / 100, seed=seed
        )
        return {"train": split["train"], "validation": split["test"]}
    return {"train": train_data}
