"""Megatron-Bridge dataset provider for WeLT: packed words, processed into model inputs."""
import logging
from dataclasses import dataclass
from functools import partial

import torch
from datasets import Dataset
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider

from welt.collator import collate_fn
from welt.processor import TextImageProcessor
from welt_training.data_utils import TextDataConfig, load_text_datasets, pack_dataset

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class WeLTDatasetProvider(TextDataConfig, DatasetProvider):
    dataloader_type: str = "cyclic"

    def build_datasets(self, context: DatasetBuildContext):
        processor = self.processor()
        texts = load_text_datasets(self)
        train = texts.get("train")
        valid = texts.get("validation")
        if train is not None:
            train = WordsDataset(pack_dataset(processor, train, self.seq_length, self.preprocessing_num_workers),
                                 processor, min_length=context.train_samples)
        if valid is not None:
            valid = WordsDataset(pack_dataset(processor, valid, self.seq_length, self.preprocessing_num_workers),
                                 processor, min_length=context.valid_samples)
        return train, valid, None


class WordsDataset(torch.utils.data.Dataset):
    """Packed words examples, processed lazily (in the dataloader workers) into model inputs.
    Repeats the examples up to min_length, as Megatron samplers need at least a global batch."""

    def __init__(self, dataset: Dataset, processor: TextImageProcessor, min_length: int = 0):
        self.dataset = dataset
        self.processor = processor
        self.length = max(len(dataset), min_length)
        self.collate_fn = partial(collate_fn, pad_value=processor.tokenizer.pad_token_id)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        example = self.dataset[int(index) % len(self.dataset)]
        return self.processor.process_single_example(words=example["words"], seq_lengths=example["seq_lengths"])
