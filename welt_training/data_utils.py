"""Text datasets for WeLT: pretokenize into words and pack; prepared shards."""
import glob
import logging
import os

from datasets import Dataset, load_dataset
from utf8_tokenizer.tokenizer import UTF8Tokenizer

from welt.processor import TextImageProcessor

logger = logging.getLogger(__name__)

PAD_WORD = UTF8Tokenizer().pad_token  # Each its own sequence, without a label (see TextImageProcessor)


def extract_text(example: dict, text_column: str = "text", text_template: str | None = None) -> str:
    """Extract text from a dataset example using a column name or format template."""
    if text_template is not None:
        return text_template.format(**example)
    return example[text_column]


def find_shard_files(data_path: str, split_name: str, prefix: str | None = None) -> list[str]:
    """Return sorted shard files for a given split in a prepared data directory.

    When *prefix* is given, only shards for that specific dataset are matched.
    Without it, shards from all datasets in the directory are returned (used by
    :func:`load_prepared_data` to load multi-dataset mixtures).
    """
    name = f"{prefix}-{split_name}" if prefix else f"*-{split_name}"
    return sorted(glob.glob(os.path.join(data_path, f"{name}-*.jsonl.gz")))


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
    train_files = find_shard_files(prepared_data_path, "train")
    validation_files = find_shard_files(prepared_data_path, "validation")

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


def pack_words(batch: dict[str, list], seq_length: int) -> dict[str, list]:
    """Greedily pack word sequences (truncated to seq_length) into blocks of exactly seq_length words.
    Each block is right-padded with PAD words, each its own sequence, for constant shapes."""
    packed = {"words": [], "seq_lengths": []}

    def flush(words, lengths):
        if words:
            pad = seq_length - len(words)
            packed["words"].append(words + [PAD_WORD] * pad)
            packed["seq_lengths"].append(lengths + [1] * pad)

    words, lengths = [], []
    for sequence in batch["words"]:
        sequence = sequence[:seq_length]
        if len(words) + len(sequence) > seq_length:
            flush(words, lengths)
            words, lengths = [], []
        words += sequence
        lengths.append(len(sequence))
    flush(words, lengths)
    return packed


def pack_dataset(processor: TextImageProcessor, dataset: Dataset, seq_length: int, num_proc: int | None = None):
    dataset = dataset.map(lambda example: {"words": processor.pretokenize(example["text"])},
                          remove_columns=dataset.column_names, num_proc=num_proc, desc="Pretokenizing texts into words")
    # ponytail: greedy in-order packing; best-fit-decreasing would waste fewer PAD words
    return dataset.map(pack_words, batched=True, batch_size=1000, remove_columns=dataset.column_names,
                       fn_kwargs={"seq_length": seq_length}, num_proc=num_proc, desc="Packing words")
