"""Text datasets for WeLT: load, format, pretokenize into words and pack. Independent of Megatron."""
import glob
import logging
import os
from dataclasses import dataclass
from itertools import islice

from datasets import Dataset, load_dataset

from welt.processor import TextImageProcessor

logger = logging.getLogger(__name__)

PAD_WORD = "\x00"


@dataclass(kw_only=True)
class TextDataConfig:
    seq_length: int  # Maximum words per (packed) example
    max_word_length: int = 128  # Maximum bytes per word, including BOS and EOS
    render_images: bool = False
    pretokenizer_name: str | None = None

    dataset_name: str | None = None
    dataset_config_name: str | None = None
    # A python format string over dataset columns, or [prefix, completion] which are concatenated for training
    dataset_text_template: str | list[str] | None = None
    train_file: str | None = None
    validation_file: str | None = None
    prepared_data_path: str | None = None
    validation_split_percentage: int = 5
    streaming: bool = False  # Streams the dataset, materializing max_train_samples / max_eval_samples
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    preprocessing_num_workers: int | None = None
    trust_remote_code: bool = False
    def processor(self) -> TextImageProcessor:
        return TextImageProcessor.create(max_word_length=self.max_word_length, max_seq_length=self.seq_length,
                                         render_images=self.render_images, pretokenizer_name=self.pretokenizer_name,
                                         trust_remote_code=self.trust_remote_code)



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
    dataset = processor.pretokenize_dataset(dataset.select_columns(["text"]), num_proc=num_proc)
    # ponytail: greedy in-order packing; best-fit-decreasing would waste fewer PAD words
    return dataset.map(pack_words, batched=True, batch_size=1000, remove_columns=dataset.column_names,
                       fn_kwargs={"seq_length": seq_length}, num_proc=num_proc, desc="Packing words")


def load_raw_datasets(args: TextDataConfig) -> dict:
    if args.prepared_data_path is not None:
        return load_prepared_data(args.prepared_data_path)

    if args.dataset_name is not None:
        load_args = dict(path=args.dataset_name, name=args.dataset_config_name,
                         trust_remote_code=args.trust_remote_code)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = next(iter(data_files.values())).split(".")[-1]
        load_args = dict(path="text" if extension == "txt" else extension, data_files=data_files)

    if args.streaming:
        streams = load_dataset(**load_args, streaming=True)
        limits = {"train": args.max_train_samples, "validation": args.max_eval_samples}
        if "validation" not in streams:
            # Hold out the first examples of the stream for validation
            num_valid = args.max_eval_samples or 1000
            streams = {"validation": streams["train"].take(num_valid), "train": streams["train"].skip(num_valid)}
        for split, limit in limits.items():
            if limit is None:
                raise ValueError(f"streaming requires max_{'train' if split == 'train' else 'eval'}_samples")
        return {split: Dataset.from_list(list(islice(streams[split], limits[split]))) for split in limits}

    raw = load_dataset(**load_args)
    if "validation" not in raw:
        split = raw["train"].train_test_split(test_size=args.validation_split_percentage / 100, seed=42)
        raw = {"train": split["train"], "validation": split["test"]}
    return dict(raw)


def load_text_datasets(args: TextDataConfig) -> dict[str, Dataset]:
    """Load the train and validation splits, as datasets with a single 'text' column."""
    raw = load_raw_datasets(args)

    template = args.dataset_text_template
    if isinstance(template, list | tuple):
        template = "".join(template)

    limits = {"train": args.max_train_samples, "validation": args.max_eval_samples}
    texts = {}
    for split in ("train", "validation"):
        if split not in raw:
            continue
        dataset = raw[split]
        if limits[split] is not None and limits[split] < len(dataset):
            dataset = dataset.select(range(limits[split]))
        text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]
        dataset = dataset.map(lambda example, column=text_column: {"text": extract_text(example, column, template)},
                              remove_columns=dataset.column_names, num_proc=args.preprocessing_num_workers,
                              desc=f"Formatting {split} split")
        texts[split] = dataset.filter(lambda example: len(example["text"]) > 0)
    return texts
