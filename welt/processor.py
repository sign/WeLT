import json
import os

import torch
import torch.nn.functional as F  # noqa: N812
from cachetools import LRUCache
from font_download import FontConfig
from font_download.example_fonts.noto_sans import FONTS_NOTO_SANS
from pixel_renderer import PixelRendererProcessor
from transformers import AutoTokenizer, PreTrainedTokenizer
from utf8_tokenizer.tokenizer import UTF8Tokenizer
from words_segmentation.tokenizer import WordsSegmentationTokenizer

from welt.attention import get_attention_mask_for_packed_sequence, get_shift_blocks

PROCESSOR_CONFIG_NAME = "processor_config.json"
PATCH_SIZE = 16  # pixel_renderer renders lines of 16px height, widths rounded to 16px


MAX_PATCH_POSITION = 256  # Rows and columns of patches beyond it share its position embedding


def patch_positions(shapes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 2) rows and columns of patches of N images -> the row and column of each of their row-major patches,
    packed: two (total patches,) tensors, capped at MAX_PATCH_POSITION - 1."""
    counts = shapes.prod(dim=-1)
    within = torch.arange(int(counts.sum()), device=shapes.device) - torch.repeat_interleave(
        F.pad(counts.cumsum(0), (1, 0))[:-1], counts)
    columns = torch.repeat_interleave(shapes[:, 1], counts)
    return (within // columns).clamp(max=MAX_PATCH_POSITION - 1), (within % columns).clamp(max=MAX_PATCH_POSITION - 1)


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack examples' tensors, right-padding every dimension to the largest size with zeros."""
    return {key: torch.nested.nested_tensor([item[key] for item in batch]).to_padded_tensor(0) for key in batch[0]}


def patchify(image, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """(H, W, C) uint8 render -> (H/p * W/p, p*p*C) uint8 patches, row-major."""
    image = torch.from_numpy(image)
    h, w, c = image.shape
    patches = image.reshape(h // patch_size, patch_size, w // patch_size, patch_size, c)
    return patches.permute(0, 2, 1, 3, 4).reshape(-1, patch_size * patch_size * c)


class TextImageProcessor:
    """Turns text into word-level byte tensors, rendered word patches, and labels."""

    def __init__(self, pretokenizer: PreTrainedTokenizer, renderer: PixelRendererProcessor | None,
                 max_word_length: int = 32):
        self.pretokenizer = pretokenizer
        self.tokenizer = UTF8Tokenizer()
        self.renderer = renderer
        self.max_word_length = max_word_length
        self.patches_cache = LRUCache(maxsize=10_000)  # Rendering a word takes ~30µs

    @classmethod
    def create(cls, max_word_length: int, render_images: bool,
               pretokenizer_name: str | None = None, trust_remote_code: bool = False):
        if pretokenizer_name is not None:
            pretokenizer = AutoTokenizer.from_pretrained(pretokenizer_name, use_fast=True,
                                                         trust_remote_code=trust_remote_code)
        else:
            pretokenizer = WordsSegmentationTokenizer(max_bytes=max_word_length - 2)  # BOS and EOS
        renderer = PixelRendererProcessor(font=FontConfig(sources=FONTS_NOTO_SANS)) if render_images else None
        return cls(pretokenizer=pretokenizer, renderer=renderer, max_word_length=max_word_length)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.pretokenizer.save_pretrained(os.path.join(save_directory, "pretokenizer"))
        if self.renderer is not None:
            self.renderer.save_pretrained(os.path.join(save_directory, "renderer"))
        config = {"max_word_length": self.max_word_length}
        with open(os.path.join(save_directory, PROCESSOR_CONFIG_NAME), "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)

    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, PROCESSOR_CONFIG_NAME)) as f:
            config = json.load(f)
        renderer_dir = os.path.join(path, "renderer")
        renderer = PixelRendererProcessor.from_pretrained(renderer_dir) if os.path.isdir(renderer_dir) else None
        return cls(pretokenizer=AutoTokenizer.from_pretrained(os.path.join(path, "pretokenizer")), renderer=renderer,
                   max_word_length=config["max_word_length"])

    def render_texts(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Render words into their 16x16 patches, packed: (total patches, 768) uint8, each word's row-major patches in
        turn (without padding), and each word's (rows, columns) of patches."""
        patches, shapes = [], []
        for text in texts:
            rendered = self.patches_cache.get(text)
            if rendered is None:
                image = self.renderer.render_text(text)
                rendered = patchify(image), (image.shape[0] // PATCH_SIZE, image.shape[1] // PATCH_SIZE)
                self.patches_cache[text] = rendered
            patches.append(rendered[0])
            shapes.append(rendered[1])
        return torch.cat(patches), torch.tensor(shapes, dtype=torch.long)

    def pretokenize(self, text: str) -> list[str]:
        """Split a text (prefixed with BOS) into words."""
        text = self.tokenizer.bos_token + text
        if isinstance(self.pretokenizer, WordsSegmentationTokenizer):
            return self.pretokenizer.tokenize(text)
        # Other tokenizers' tokens can be encoded (e.g. byte-level BPE "Ġworld"), words are spans of the text instead
        offsets = self.pretokenizer(text, add_special_tokens=False, return_offsets_mapping=True).offset_mapping
        starts = sorted({0, *(start for start, _ in offsets)})
        return [text[start:end] for start, end in zip(starts, [*starts[1:], len(text)], strict=True) if end > start]

    @staticmethod
    def get_sequence_labels(words: list[str], seq_lengths: list[int]) -> list[str]:
        """The next word of each word, per packed sequence. The last word of each sequence has an empty label."""
        labels = []
        offset = 0
        for length in seq_lengths:
            labels += words[offset + 1:offset + length] + [""]
            offset += length
        return labels

    def tokenize_words(self, words: list[str]):
        return self.tokenizer.torch(
            words,
            padding=True,
            add_special_tokens=True,
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

    def __call__(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """A collated batch of texts, each its own sequence."""
        words = [self.pretokenize(text) for text in texts]
        return collate_fn([self.process_single_example(w, [len(w)]) for w in words])
