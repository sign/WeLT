import json
import os
from collections import defaultdict

import datasets
import torch
from cachetools import LRUCache
from datasets import Dataset
from pixel_renderer import PixelRendererProcessor
from transformers import AutoImageProcessor, AutoTokenizer, ImageProcessingMixin, PreTrainedTokenizer, ProcessorMixin
from utf8_tokenizer.tokenizer import UTF8Tokenizer
from words_segmentation.tokenizer import WordsSegmentationTokenizer  # noqa: F401 - for registering AutoTokenizer

from welt.attention import (
    get_attention_mask_for_packed_sequence,
    get_position_ids_for_packed_sequence,
    get_shift_blocks,
)
from welt.collator import collate_fn, stack_pad_tensors
from welt.noop import NoopImageProcessor

PROCESSOR_CONFIG_NAME = "processor_config.json"

ATTRIBUTE_LOADERS = {
    "pretokenizer": AutoTokenizer.from_pretrained,
    "tokenizer": AutoTokenizer.from_pretrained,
    "renderer": PixelRendererProcessor.from_pretrained,
    "image_processor": AutoImageProcessor.from_pretrained,
}


class TextImageProcessor(ProcessorMixin):
    name = "text-image-processor"

    attributes = [
        "pretokenizer",
        "tokenizer",
        "renderer",
        "image_processor"
    ]
    pretokenizer_class = "AutoTokenizer"
    tokenizer_class = "AutoTokenizer"
    renderer_class = "PixelRendererProcessor"
    image_processor_class = "AutoImageProcessor"

    def __init__(self,
                 pretokenizer: PreTrainedTokenizer,
                 tokenizer: UTF8Tokenizer,
                 renderer: PixelRendererProcessor,
                 image_processor: ImageProcessingMixin,
                 max_seq_length: int = 128,
                 max_word_length: int = 32,
                 cache_size: int = 10000):
        self.chat_template = None

        assert tokenizer.bos_token_id is not None, "Tokenizer must have a BOS token"
        assert tokenizer.eos_token_id is not None, "Tokenizer must have an EOS token"

        self.pretokenizer = pretokenizer
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.image_processor = image_processor

        self.max_word_length = max_word_length
        self.max_seq_length = max_seq_length
        self.cache_size = cache_size

        self.images_cache = LRUCache(maxsize=self.cache_size)

    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
        os.makedirs(save_directory, exist_ok=True)

        for attr_name in self.attributes:
            attr = getattr(self, attr_name)
            attr_dir = os.path.join(save_directory, attr_name)
            os.makedirs(attr_dir, exist_ok=True)
            if hasattr(attr, "_set_processor_class"):
                attr._set_processor_class(self.__class__.__name__)
            attr.save_pretrained(attr_dir)

        output = {k: v for k, v in self.__dict__.items()
                  if k not in self.attributes and isinstance(v, (int, float, str, bool))}
        output["processor_class"] = self.__class__.__name__
        config_file = os.path.join(save_directory, PROCESSOR_CONFIG_NAME)
        with open(config_file, "w") as f:
            json.dump(output, f, indent=2, sort_keys=True)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_file = os.path.join(pretrained_model_name_or_path, PROCESSOR_CONFIG_NAME)
        with open(config_file) as f:
            config = json.load(f)

        attrs = {}
        for attr_name, loader in ATTRIBUTE_LOADERS.items():
            attr_dir = os.path.join(pretrained_model_name_or_path, attr_name)
            attrs[attr_name] = loader(attr_dir)

        init_kwargs = {k: v for k, v in config.items() if k != "processor_class"}
        return cls(**attrs, **init_kwargs)

    def render_texts(self, texts: list[str], device=None) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.image_processor, NoopImageProcessor):
            return torch.empty(1, device=device), torch.empty(1, device=device)

        device_kwargs = {"device": device} if device else {}
        images = [self.images_cache.get(text, None) for text in texts]

        # Render all missing texts and group by size for efficient batching
        render_groups = defaultdict(list)
        index_groups = defaultdict(list)
        for i, v in enumerate(images):
            if v is None:
                render = self.renderer.render_text(texts[i])
                index_groups[render.shape].append(i)
                render_groups[render.shape].append(render)

        # Process each shape group and update cache
        for shape, renders in render_groups.items():
            processed = self.image_processor(renders, return_tensors="pt", do_center_crop=False, do_resize=False)
            pixel_values = processed.pixel_values.to(**device_kwargs)
            for i, pixel_value in zip(index_groups[shape], pixel_values, strict=True):
                self.images_cache[texts[i]] = pixel_value
                images[i] = pixel_value

        image_dimensions = torch.tensor([img.shape[-2:] for img in images], dtype=torch.long, device=device)
        return stack_pad_tensors(images), image_dimensions

    def pretokenize(self, text: str) -> list[str]:
        # Add BOS token at the start
        return self.pretokenizer.tokenize(self.tokenizer.bos_token + text)

    def pretokenize_dataset(self, dataset: Dataset, num_proc=4) -> Dataset:
        """Pretokenize a dataset in place, adding a 'words' column."""

        def tokenize_example(example):
            example["words"] = self.pretokenize(example["text"])
            return example

        map_kwargs = {}
        if isinstance(dataset, datasets.Dataset):
            # these args are not available for IterableDataset
            map_kwargs["num_proc"] = num_proc
            map_kwargs["desc"] = "Pretokenizing texts into 'words'"

        return dataset.map(tokenize_example,
                           batched=False,
                           remove_columns=["text"],
                           **map_kwargs)

    def get_sequence_labels(self, words: list[str], seq_lengths: list[int] = None) -> list[str]:
        """
        Generate labels for word-level sequences.

        Tokens inside shift blocks (between ShiftOut and ShiftIn control tokens) are masked
        with empty labels to prevent training on "known" tokens that are already visible via
        self-attention. The ShiftIn token itself keeps its label to predict the next word.

        Args:
            words: List of word strings to generate labels for
            seq_lengths: Optional list of sequence lengths for packed sequences
            pack: If True, use packed mode (longer context labels), else unpacked (next word only)

        Returns:
            List of label strings corresponding to each word
        """
        if seq_lengths is None:
            seq_lengths = [len(words)]

        labels = []

        # Process each sequence separately, to support efficient packing
        offset = 0
        for length in seq_lengths:
            # Next word as label, last word has no label
            labels += words[offset + 1:offset + length] + [""]

            offset += length

        return labels

    def tokenize_words(self, words: list[str], device=None):
        return self.tokenizer.torch(
            words,
            padding=True,
            add_special_tokens=True,
            device=device,
            # Truncation happens mostly in pre-tokenization.
            # This is just for additional safety, for UTF-16 use cases.
            max_length=self.max_word_length,
            truncation=True,
        )

    def process_single_example(self, words: list[str], seq_lengths: list[int]):
        labels = self.get_sequence_labels(words, seq_lengths)

        # Tokenize words with BOS and EOS tokens
        tokenized = self.tokenize_words(words)  # Tokenized inputs
        tokenized_labels = self.tokenize_words(labels)  # Tokenized outputs

        # Mask labels inside shift blocks (except for ShiftIn token)
        for start, end in get_shift_blocks(words):
            # Excludes end (ShiftIn token)
            tokenized_labels.input_ids[start:end] = 0 # PAD token id
            tokenized_labels.attention_mask[start:end] = 0 # no attention for PAD tokens

        # Render images
        input_images, input_images_dimensions = self.render_texts(words)

        return {
            "input_ids": tokenized.input_ids,
            "input_attention_mask": tokenized.attention_mask,  # Attention within each word
            # Attention across words
            "attention_mask": get_attention_mask_for_packed_sequence(seq_lengths, words=words),
            "position_ids": get_position_ids_for_packed_sequence(seq_lengths),
            "input_images": input_images,
            "input_images_dimensions": input_images_dimensions,
            "labels_input": tokenized_labels.input_ids[:, :-1],  # Remove EOS token from input labels
            "labels_attention_mask": tokenized_labels.attention_mask[:, :-1],  # Remove EOS token from attention mask
            "labels_output": tokenized_labels.input_ids[:, 1:]  # Remove BOS token from output labels
        }

    def __call__(self,
                 batch: dict[str, list[str]] | dict[str] | str | list[str],
                 collated=False) -> dict[str, torch.Tensor]:
        if isinstance(batch, str):
            batch = {"text": [batch]}

        if isinstance(batch, list):
            batch = {"text": batch}

        if "text" in batch and isinstance(batch["text"], str):
            batch["text"] = [batch["text"]]

        # Copy batch before modifying to avoid mutating the input
        if "text" in batch and "words" not in batch:
            batch = batch.copy()
            words = [self.pretokenize(t) for t in batch["text"]]
            batch["words"] = words
            batch["seq_lengths"] = [[len(w)] for w in words]

        dicts = [self.process_single_example(words=words, seq_lengths=seq_lengths)
                 for words, seq_lengths in zip(batch["words"], batch["seq_lengths"], strict=False)]

        if collated:
            return collate_fn(dicts)

        new_batch = {}
        for key in dicts[0].keys():
            new_batch[key] = [d[key] for d in dicts]

        # Preserve extra fields from the original batch (e.g., "prefix", "completion")
        for key in batch:
            if key not in new_batch and key not in {"text", "words", "seq_lengths"}:
                new_batch[key] = batch[key]

        return new_batch
