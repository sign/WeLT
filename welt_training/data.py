"""Megatron-Bridge dataset provider for WeLT: streamed texts, split into words and packed into examples on the fly."""
from collections.abc import Iterable
from dataclasses import dataclass

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider
from megatron.core import parallel_state

from welt.collator import collate_fn
from welt.processor import TextImageProcessor
from welt_training.data_utils import TextDataConfig, build_iterators, pack_words


@dataclass(kw_only=True)
class WeLTDatasetProvider(TextDataConfig, DatasetProvider):
    max_word_length: int = 128  # Bytes per word, including BOS and EOS
    render_images: bool = False  # For an image encoder
    pretokenizer_name: str | None = None  # A HF tokenizer splitting texts into words, defaults to words-segmentation
    dataloader_type: str = "external"  # Iterators of micro batches (build_iterators)
    # Batches vary in shape (bytes per word, patches per image): PyTorch caches a pinned buffer for every size, without
    # freeing them, so pinning grows memory throughout training
    pin_memory: bool = False

    def processor(self) -> TextImageProcessor:
        return TextImageProcessor.create(max_word_length=self.max_word_length,
                                         render_images=self.render_images, pretokenizer_name=self.pretokenizer_name,
                                         trust_remote_code=self.trust_remote_code)

    def build_datasets(self, context: DatasetBuildContext):
        processor = self.processor()

        def make_examples(texts: Iterable[str]):
            for words, seq_lengths in pack_words(map(processor.pretokenize, texts), self.seq_length):
                yield processor.process_single_example(words=words, seq_lengths=seq_lengths)

        train, validation = build_iterators(self, make_examples, collate_fn,
                                            rank=parallel_state.get_data_parallel_rank(),
                                            world_size=parallel_state.get_data_parallel_world_size())
        return train, validation, None
