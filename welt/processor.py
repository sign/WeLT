from collections import defaultdict

import datasets
import torch
from cachetools import LRUCache
from datasets import Dataset
from pixel_renderer import PixelRendererProcessor
from transformers import ImageProcessingMixin, PreTrainedTokenizer, ProcessorMixin
from utf8_tokenizer.tokenizer import UTF8Tokenizer
from words_segmentation.tokenizer import WordsSegmentationTokenizer  # noqa: F401 - for registering AutoTokenizer

from welt.attention import (
    get_attention_mask_for_packed_sequence,
    get_position_ids_for_packed_sequence,
    get_shift_blocks,
)
from welt.collator import collate_fn, stack_pad_tensors
from welt.noop import NoopImageProcessor


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
        super().__init__(pretokenizer=pretokenizer,
                         tokenizer=tokenizer,
                         renderer=renderer,
                         image_processor=image_processor)

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
            # TODO : make dtype configurable
            pixel_values = processed.pixel_values.to(dtype=torch.bfloat16, **device_kwargs)
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

        # Mask labels inside shift blocks (except for ShiftIn token)
        for start, end in get_shift_blocks(words):
            for i in range(start, end):  # Excludes end (ShiftIn token)
                labels[i] = ""

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
