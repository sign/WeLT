import glob
import logging
import os

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_prepared_data(prepared_data_path: str, validation_split_percentage: int | None = None, seed: int = 42):
    """Load preprocessed shards produced by prepare_data.py.

    Args:
        prepared_data_path: Directory containing ``*.jsonl.gz`` shard files.
        validation_split_percentage: If set, split the data into train/validation.
        seed: Random seed for the train/test split.

    Returns:
        A dict with ``"train"`` (and optionally ``"validation"``) datasets.
    """
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
