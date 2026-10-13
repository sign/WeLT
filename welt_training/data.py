"""Megatron-Bridge dataset provider for WeLT: streamed texts, split into words and packed into examples on the fly."""
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import partial

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider
from megatron.core import parallel_state

from welt.processor import TextImageProcessor, collate_fn
from welt_training.data_utils import TextDataConfig, build_iterators, pack_words


@dataclass(kw_only=True)
class WeLTDatasetProvider(TextDataConfig, DatasetProvider):
    max_word_length: int = 128  # Bytes per word, including BOS and EOS
    render_images: bool = False  # For an image encoder
    pretokenizer_name: str | None = None  # A HF tokenizer splitting texts into words, defaults to words-segmentation
    dataloader_type: str = "external"  # Iterators of micro batches (build_iterators)
    # Pinned in the loader's thread: the copy from the workers' shared memory (page faults) is off the training step
    pin_memory: bool = True

    def processor(self) -> TextImageProcessor:
        return TextImageProcessor.create(max_word_length=self.max_word_length,
                                         render_images=self.render_images, pretokenizer_name=self.pretokenizer_name,
                                         trust_remote_code=self.trust_remote_code)

    def make_examples(self, texts: Iterable[str], processor: TextImageProcessor | None = None) -> Iterator[dict]:
        """Packed examples of the texts' words (see pack_words), as in training."""
        processor = processor or self.processor()
        for packed in pack_words(map(processor.pretokenize, texts), self.seq_length):
            yield processor.process_single_example(*packed)

    def build_datasets(self, context: DatasetBuildContext):
        make_examples = partial(self.make_examples, processor=self.processor())
        train, validation = build_iterators(self, make_examples, collate_fn,
                                            rank=parallel_state.get_data_parallel_rank(),
                                            world_size=parallel_state.get_data_parallel_world_size())
        return train, validation, None
