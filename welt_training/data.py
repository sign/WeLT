"""Megatron-Bridge dataset provider for WeLT: texts packed into examples of words, processed into model inputs."""
from dataclasses import dataclass
from functools import partial

import torch
from datasets import Dataset
from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider

from welt.collator import collate_fn
from welt.processor import TextImageProcessor
from welt_training.data_utils import TextDataConfig, dataset_lengths, load_text_datasets, pack_dataset


@dataclass(kw_only=True)
class WeLTDatasetProvider(TextDataConfig, DatasetProvider):
    max_word_length: int = 128  # Bytes per word, including BOS and EOS
    render_images: bool = False  # For an image encoder
    pretokenizer_name: str | None = None  # A HF tokenizer splitting texts into words, defaults to words-segmentation
    dataloader_type: str = "cyclic"

    def processor(self) -> TextImageProcessor:
        return TextImageProcessor.create(max_word_length=self.max_word_length,
                                         render_images=self.render_images, pretokenizer_name=self.pretokenizer_name,
                                         trust_remote_code=self.trust_remote_code)

    def build_datasets(self, context: DatasetBuildContext):
        processor = self.processor()
        texts = load_text_datasets(self)
        datasets = {split: pack_dataset(processor, texts[split], self.seq_length, self.preprocessing_num_workers)
                    for split in texts}
        lengths = dataset_lengths(datasets, context, self.samples_per_eval)
        return (*(WordsDataset(datasets[split], processor, lengths[split]) if split in datasets else None
                  for split in ("train", "validation")), None)


class WordsDataset(torch.utils.data.Dataset):
    """Packed words examples, processed lazily (in the dataloader workers) into model inputs.
    Repeats the examples up to length, as Megatron samplers draw a fixed number of samples."""

    def __init__(self, dataset: Dataset, processor: TextImageProcessor, length: int):
        self.dataset = dataset
        self.processor = processor
        self.length = length
        self.collate_fn = partial(collate_fn, pad_value=processor.tokenizer.pad_token_id)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        example = self.dataset[int(index) % len(self.dataset)]
        return self.processor.process_single_example(words=example["words"], seq_lengths=example["seq_lengths"])
