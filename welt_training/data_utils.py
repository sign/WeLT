import glob
import logging
import os

from datasets import load_dataset

logger = logging.getLogger(__name__)


def extract_text(example: dict, text_column: str = "text", text_template: str | None = None) -> str:
    """Extract text from a dataset example using a column name or format template."""
    if text_template is not None:
        return text_template.format(**example)
    return example[text_column]


def load_prepared_data(prepared_data_path: str):
    """Load preprocessed shards produced by prepare_data.py.

    Loads ``{prefix}-train-*.jsonl.gz`` shards as the train split and/or
    ``{prefix}-validation-*.jsonl.gz`` shards as the validation split.

    At least one split must be present. Missing splits are omitted from the result.

    Args:
        prepared_data_path: Directory containing ``*.jsonl.gz`` shard files.

    Returns:
        A dict with ``"train"`` and/or ``"validation"`` datasets.
    """
    train_files = sorted(glob.glob(os.path.join(prepared_data_path, "*-train-*.jsonl.gz")))
    validation_files = sorted(glob.glob(os.path.join(prepared_data_path, "*-validation-*.jsonl.gz")))

    if not train_files and not validation_files:
        raise ValueError(
            f"No *-train-*.jsonl.gz or *-validation-*.jsonl.gz files found in {prepared_data_path}. "
            "Prepare data with --train_split_units and/or --validation_split_units."
        )

    result = {}
    if train_files:
        result["train"] = load_dataset("json", data_files=train_files, split="train")
    if validation_files:
        result["validation"] = load_dataset("json", data_files=validation_files, split="train")

    logger.info(
        f"Loading prepared data: {len(train_files)} train shard(s), "
        f"{len(validation_files)} validation shard(s) from {prepared_data_path}"
    )
    return result
