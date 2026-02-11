import glob
import logging
import os

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_prepared_data(prepared_data_path: str):
    """Load preprocessed shards produced by prepare_data.py.

    Loads ``{prefix}-train-*.jsonl.gz`` shards as the train split and
    ``{prefix}-validation-*.jsonl.gz`` shards as the validation split.

    Both train and validation shards are required. Prepare data with
    ``--validation_split_units`` to produce them.

    Args:
        prepared_data_path: Directory containing ``*.jsonl.gz`` shard files.

    Returns:
        A dict with ``"train"`` and ``"validation"`` datasets.
    """
    train_files = sorted(glob.glob(os.path.join(prepared_data_path, "*-train-*.jsonl.gz")))
    validation_files = sorted(glob.glob(os.path.join(prepared_data_path, "*-validation-*.jsonl.gz")))

    if not train_files:
        raise ValueError(
            f"No *-train-*.jsonl.gz files found in {prepared_data_path}. "
            "Prepare data with --train_split_units and --validation_split_units."
        )
    if not validation_files:
        raise ValueError(
            f"No *-validation-*.jsonl.gz files found in {prepared_data_path}. "
            "Prepare data with --validation_split_units to create a validation split."
        )

    logger.info(
        f"Loading prepared data: {len(train_files)} train shard(s), "
        f"{len(validation_files)} validation shard(s) from {prepared_data_path}"
    )
    return {
        "train": load_dataset("json", data_files=train_files, split="train"),
        "validation": load_dataset("json", data_files=validation_files, split="train"),
    }
