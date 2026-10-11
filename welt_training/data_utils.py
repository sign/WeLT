"""Text datasets for WeLT: load, format, pretokenize into words and pack. Independent of Megatron."""
import glob
import logging
import os
from dataclasses import dataclass
from importlib.metadata import version
from itertools import islice

from datasets import Dataset, load_dataset
from utf8_tokenizer.tokenizer import UTF8Tokenizer

from welt.processor import TextImageProcessor

logger = logging.getLogger(__name__)

PAD_WORD = UTF8Tokenizer().pad_token  # Each its own sequence, without a label (see TextImageProcessor)


@dataclass(kw_only=True)
class TextDataConfig:
    """Where the texts come from, and how they are formatted."""
    seq_length: int  # Words (WeLT) or tokens (baseline) per packed example

    dataset_name: str | None = None
    dataset_config_name: str | None = None
    # A python format string over dataset columns, or [prefix, completion] which are concatenated for training
    dataset_text_template: str | list[str] | None = None
    train_file: str | None = None
    validation_file: str | None = None
    prepared_data_path: str | None = None  # Shards made by welt-prepare-data
    validation_split_percentage: int = 5
    streaming: bool = False  # Streams the dataset, materializing max_train_samples / max_eval_samples
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    preprocessing_num_workers: int | None = None
    trust_remote_code: bool = False
    dataloader_type: str = "cyclic"  # Megatron's sampler: shuffled, repeating epochs
    samples_per_eval: int | None = None  # Set from the validation config: each evaluation covers the dataset once


def repeated_length(num_examples: int, min_length: int) -> int:
    """The smallest number of samples >= min_length that repeats every example equally often."""
    if num_examples == 0:
        raise ValueError("Empty dataset (e.g. fewer texts than one packed example)")
    return -(-max(min_length, 1) // num_examples) * num_examples


def dataset_lengths(datasets: dict, context, samples_per_eval: int | None) -> dict[str, int]:
    """Samples drawn from each split: train repeats examples equally up to the train samples. Validation has
    samples_per_eval samples, so that every evaluation (one epoch of Megatron's sampler) covers the whole set."""
    for split, dataset in datasets.items():
        if len(dataset) == 0:
            raise ValueError(f"Empty {split} dataset (e.g. fewer texts than one packed example)")
    lengths = {}
    if "train" in datasets:
        lengths["train"] = repeated_length(len(datasets["train"]), context.train_samples)
    if "validation" in datasets:
        num_examples = len(datasets["validation"])
        lengths["validation"] = samples_per_eval or repeated_length(num_examples, context.valid_samples)
        if samples_per_eval is not None and samples_per_eval < num_examples:
            logger.warning(f"Evaluations see {samples_per_eval} of {num_examples} validation examples, "
                           "increase validation.eval_iters to see them all")
    return lengths


def _take(load_args: dict, split: str, skip: int, limit: int):
    yield from islice(load_dataset(**load_args, split=split, streaming=True).skip(skip), limit)


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
    """Greedily pack word sequences into blocks of exactly seq_length words. Longer sequences are split into
    seq_length chunks, each its own sequence. Each block is right-padded with PAD words, each its own sequence,
    for constant shapes."""
    packed = {"words": [], "seq_lengths": []}

    def flush(words, lengths):
        if words:
            pad = seq_length - len(words)
            packed["words"].append(words + [PAD_WORD] * pad)
            packed["seq_lengths"].append(lengths + [1] * pad)

    # ponytail: a chunk's last word is trained as a document end; carry the next chunk's first word as its label
    # (a label without an input word) if long documents matter
    chunks = (document[start:start + seq_length] for document in batch["words"]
              for start in range(0, len(document), seq_length))
    words, lengths = [], []
    for sequence in chunks:
        if len(words) + len(sequence) > seq_length:
            flush(words, lengths)
            words, lengths = [], []
        words += sequence
        lengths.append(len(sequence))
    flush(words, lengths)
    return packed


def pack_dataset(processor: TextImageProcessor, dataset: Dataset, seq_length: int, num_proc: int | None = None):
    # The cache fingerprint hashes the processor's state, not its code: the words-segmentation version keeps the cache
    # from serving words split by an older version
    dataset = dataset.map(lambda example, segmentation_version: {"words": processor.pretokenize(example["text"])},
                          fn_kwargs={"segmentation_version": version("words-segmentation")},
                          remove_columns=dataset.column_names, num_proc=num_proc, desc="Pretokenizing texts into words")
    # ponytail: greedy in-order packing; best-fit-decreasing would waste fewer PAD words
    return dataset.map(pack_words, batched=True, batch_size=1000, remove_columns=dataset.column_names,
                       fn_kwargs={"seq_length": seq_length}, num_proc=num_proc, desc="Packing words")


def load_raw_datasets(args: TextDataConfig) -> dict:
    if args.prepared_data_path is not None:
        return with_validation(load_prepared_data(args.prepared_data_path), args.validation_split_percentage)

    if args.dataset_name is not None:
        load_args = dict(path=args.dataset_name, name=args.dataset_config_name,
                         trust_remote_code=args.trust_remote_code)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = next(iter(data_files.values())).removesuffix(".gz").rsplit(".", 1)[-1]
        load_args = dict(path={"txt": "text", "jsonl": "json"}.get(extension, extension), data_files=data_files)

    if args.streaming:
        if args.max_train_samples is None or args.max_eval_samples is None:
            raise ValueError("streaming requires max_train_samples and max_eval_samples")
        # Without a validation split, the first examples of the train stream are held out for validation
        has_validation = "validation" in load_dataset(**load_args, streaming=True)
        splits = {"train": ("train", 0 if has_validation else args.max_eval_samples, args.max_train_samples),
                  "validation": ("validation" if has_validation else "train", 0, args.max_eval_samples)}
        # Materialized to (cached, memory-mapped) arrow files
        return {name: Dataset.from_generator(_take, gen_kwargs=dict(load_args=load_args, split=split, skip=skip,
                                                                    limit=limit))
                for name, (split, skip, limit) in splits.items()}

    return with_validation(dict(load_dataset(**load_args)), args.validation_split_percentage)


def with_validation(raw: dict, validation_split_percentage: int) -> dict:
    """Holds out a part of the train split as validation, if there is no validation split."""
    if "validation" in raw:
        return raw
    split = raw["train"].train_test_split(test_size=validation_split_percentage / 100, seed=42)
    return {"train": split["train"], "validation": split["test"]}


def load_text_datasets(args: TextDataConfig) -> dict[str, Dataset]:
    """Load the train and validation splits, as datasets with a single 'text' column."""
    raw = load_raw_datasets(args)

    template = args.dataset_text_template
    if isinstance(template, list):
        template = "".join(template)

    limits = {"train": args.max_train_samples, "validation": args.max_eval_samples}
    texts = {}
    for split in ("train", "validation"):
        if split not in raw:
            continue
        dataset = raw[split]
        if limits[split] is not None and limits[split] < len(dataset):
            dataset = dataset.select(range(limits[split]))
        if template is None and "text" not in dataset.column_names:
            raise ValueError(f"No 'text' column (columns: {dataset.column_names}), set data.dataset_text_template")
        dataset = dataset.map(lambda example: {"text": extract_text(example, "text", template)},
                              remove_columns=dataset.column_names, num_proc=args.preprocessing_num_workers,
                              desc=f"Formatting {split} split")
        texts[split] = dataset.filter(lambda example: len(example["text"]) > 0)
    return texts
