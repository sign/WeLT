"""Text datasets for WeLT: load, format, pretokenize into words, pack, and process into model inputs."""
import logging
from dataclasses import dataclass
from functools import partial
from itertools import islice

import torch
from datasets import Dataset, load_dataset
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider

from welt.collator import collate_fn
from welt.processor import TextImageProcessor
from welt_training.data_utils import extract_text, load_prepared_data

logger = logging.getLogger(__name__)

PAD_WORD = "\x00"


@dataclass(kw_only=True)
class WeLTDatasetProvider(DatasetProvider):
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
    dataloader_type: str = "cyclic"

    def processor(self) -> TextImageProcessor:
        return TextImageProcessor.create(max_word_length=self.max_word_length, max_seq_length=self.seq_length,
                                         render_images=self.render_images, pretokenizer_name=self.pretokenizer_name,
                                         trust_remote_code=self.trust_remote_code)

    def build_datasets(self, context: DatasetBuildContext):
        processor = self.processor()
        texts = load_text_datasets(self)
        train = texts.get("train")
        valid = texts.get("validation")
        if train is not None:
            train = WordsDataset(pack_dataset(processor, train, self.seq_length, self.preprocessing_num_workers),
                                 processor)
        if valid is not None:
            valid = WordsDataset(pack_dataset(processor, valid, self.seq_length, self.preprocessing_num_workers),
                                 processor)
        return train, valid, None


class WordsDataset(torch.utils.data.Dataset):
    """Packed words examples, processed lazily (in the dataloader workers) into model inputs."""

    def __init__(self, dataset: Dataset, processor: TextImageProcessor):
        self.dataset = dataset
        self.processor = processor
        self.collate_fn = partial(collate_fn, pad_value=processor.tokenizer.pad_token_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = self.dataset[int(index)]
        return self.processor.process_single_example(words=example["words"], seq_lengths=example["seq_lengths"])


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


def _load_raw_datasets(args: WeLTDatasetProvider) -> dict:
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


def load_text_datasets(args: WeLTDatasetProvider) -> dict[str, Dataset]:
    """Load the train and validation splits, as datasets with a single 'text' column."""
    raw = _load_raw_datasets(args)

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
        dataset = dataset.map(lambda example: {"text": extract_text(example, text_column, template)},
                              remove_columns=dataset.column_names, num_proc=args.preprocessing_num_workers,
                              desc=f"Formatting {split} split")
        texts[split] = dataset.filter(lambda example: len(example["text"]) > 0)
    return texts
