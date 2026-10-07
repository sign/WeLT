import json
import os

import datasets
import torch
from cachetools import LRUCache
from datasets import Dataset
from font_download import FontConfig
from font_download.example_fonts.noto_sans import FONTS_NOTO_SANS
from pixel_renderer import PixelRendererProcessor
from transformers import AutoTokenizer, PreTrainedTokenizer
from utf8_tokenizer.tokenizer import UTF8Tokenizer
from words_segmentation.tokenizer import WordsSegmentationTokenizer

from welt.attention import get_attention_mask_for_packed_sequence, get_shift_blocks
from welt.collator import collate_fn, stack_pad_tensors

PROCESSOR_CONFIG_NAME = "processor_config.json"
PATCH_SIZE = 16  # pixel_renderer renders lines of 16px height, widths rounded to 16px


def patchify(image, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """(H, W, C) uint8 render -> (H/p * W/p, p*p*C) uint8 patches, row-major."""
    image = torch.from_numpy(image)
    h, w, c = image.shape
    patches = image.reshape(h // patch_size, patch_size, w // patch_size, patch_size, c)
    return patches.permute(0, 2, 1, 3, 4).reshape(-1, patch_size * patch_size * c)


class TextImageProcessor:
    """Turns text into word-level byte tensors, rendered word patches, and labels."""

    def __init__(self,
                 pretokenizer: PreTrainedTokenizer,
                 tokenizer: UTF8Tokenizer,
                 renderer: PixelRendererProcessor | None,
                 max_seq_length: int = 128,
                 max_word_length: int = 32,
                 cache_size: int = 10000):
        assert tokenizer.bos_token_id is not None, "Tokenizer must have a BOS token"
        assert tokenizer.eos_token_id is not None, "Tokenizer must have an EOS token"

        self.pretokenizer = pretokenizer
        self.tokenizer = tokenizer
        self.renderer = renderer

        self.max_word_length = max_word_length
        self.max_seq_length = max_seq_length
        self.cache_size = cache_size

        self.patches_cache = LRUCache(maxsize=self.cache_size)

    @classmethod
    def create(cls, max_word_length: int, max_seq_length: int, render_images: bool,
               pretokenizer_name: str | None = None, trust_remote_code: bool = False):
        if pretokenizer_name is not None:
            pretokenizer = AutoTokenizer.from_pretrained(pretokenizer_name, use_fast=True,
                                                         trust_remote_code=trust_remote_code)
        else:
            pretokenizer = WordsSegmentationTokenizer(max_bytes=max_word_length - 2)  # BOS and EOS
        renderer = PixelRendererProcessor(font=FontConfig(sources=FONTS_NOTO_SANS)) if render_images else None
        return cls(pretokenizer=pretokenizer, tokenizer=UTF8Tokenizer(), renderer=renderer,
                   max_seq_length=max_seq_length, max_word_length=max_word_length)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.pretokenizer.save_pretrained(os.path.join(save_directory, "pretokenizer"))
        if self.renderer is not None:
            self.renderer.save_pretrained(os.path.join(save_directory, "renderer"))
        config = {"max_seq_length": self.max_seq_length, "max_word_length": self.max_word_length,
                  "cache_size": self.cache_size}
        with open(os.path.join(save_directory, PROCESSOR_CONFIG_NAME), "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)

    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, PROCESSOR_CONFIG_NAME)) as f:
            config = json.load(f)
        renderer_dir = os.path.join(path, "renderer")
        renderer = PixelRendererProcessor.from_pretrained(renderer_dir) if os.path.isdir(renderer_dir) else None
        return cls(pretokenizer=AutoTokenizer.from_pretrained(os.path.join(path, "pretokenizer")),
                   tokenizer=UTF8Tokenizer(), renderer=renderer, **config)

    def render_texts(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Render words into (num_words, max_patches, 768) uint8 patches, and each word's (rows, columns) of patches."""
        patches, shapes = [], []
        for text in texts:
            rendered = self.patches_cache.get(text)
            if rendered is None:
                image = self.renderer.render_text(text)
                rendered = patchify(image), (image.shape[0] // PATCH_SIZE, image.shape[1] // PATCH_SIZE)
                self.patches_cache[text] = rendered
            patches.append(rendered[0])
            shapes.append(rendered[1])
        return stack_pad_tensors(patches), torch.tensor(shapes, dtype=torch.long)

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
        Generate labels for word-level sequences: the next word, per packed sequence.
        The last word of each sequence has an empty label.
        """
        if seq_lengths is None:
            seq_lengths = [len(words)]

        labels = []
        offset = 0
        for length in seq_lengths:
            labels += words[offset + 1:offset + length] + [""]
            offset += length

        return labels

    def tokenize_words(self, words: list[str], device=None):
        return self.tokenizer.torch(
            words,
            padding=True,
            add_special_tokens=True,
            device=device,
            # Truncation happens mostly in pre-tokenization. This is just for additional safety.
            max_length=self.max_word_length,
            truncation=True,
        )

    def process_single_example(self, words: list[str], seq_lengths: list[int]):
        labels = self.get_sequence_labels(words, seq_lengths)

        # Tokenize words with BOS and EOS tokens
        tokenized = self.tokenize_words(words)  # Tokenized inputs
        tokenized_labels = self.tokenize_words(labels)  # Tokenized outputs

        # Packed fixed-size chunks use PAD words as isolated sequences. Their
        # empty labels would otherwise contribute synthetic EOS targets.
        for index, word in enumerate(words):
            if word == self.tokenizer.pad_token:
                tokenized_labels.input_ids[index] = self.tokenizer.pad_token_id
                tokenized_labels.attention_mask[index] = 0

        # Mask labels inside shift blocks (except for ShiftIn token)
        # Tokens inside shift blocks are visible via self-attention, so they are "known".
        for start, end in get_shift_blocks(words):
            tokenized_labels.input_ids[start:end] = self.tokenizer.pad_token_id
            tokenized_labels.attention_mask[start:end] = 0

        example = {
            "input_ids": tokenized.input_ids,
            "input_attention_mask": tokenized.attention_mask,  # Attention within each word
            # Attention across words
            "attention_mask": get_attention_mask_for_packed_sequence(seq_lengths, words=words),
            "labels_input": tokenized_labels.input_ids[:, :-1],  # Remove EOS token from input labels
            "labels_attention_mask": tokenized_labels.attention_mask[:, :-1],
            "labels_output": tokenized_labels.input_ids[:, 1:]  # Remove BOS token from output labels
        }
        if self.renderer is not None:
            example["input_patches"], example["input_patches_shape"] = self.render_texts(words)
        return example

    def __call__(self,
                 batch: dict[str, list[str]] | str | list[str],
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
                 for words, seq_lengths in zip(batch["words"], batch["seq_lengths"], strict=True)]

        if collated:
            return collate_fn(dicts, pad_value=self.tokenizer.pad_token_id)

        return {key: [d[key] for d in dicts] for key in dicts[0]}
