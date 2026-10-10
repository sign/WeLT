"""Text streams for WeLT and the baseline: a HF dataset streamed, formatted, shuffled, and split across data parallel
ranks and dataloader workers. Examples are made from the texts on the fly, in the dataloader workers.
Independent of Megatron."""
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import count, cycle, islice

import torch
from datasets import IterableDataset, get_dataset_split_names, load_dataset
from datasets.distributed import split_dataset_by_node
from utf8_tokenizer.tokenizer import UTF8Tokenizer

logger = logging.getLogger(__name__)

PAD_WORD = UTF8Tokenizer().pad_token  # Each its own sequence, without a label (see TextImageProcessor)


@dataclass(kw_only=True)
class TextDataConfig:
    """Where the texts come from, and how they are formatted."""
    seq_length: int  # Words (WeLT) or tokens (baseline) per packed example

    dataset_name: str  # A HF dataset, or a file format ("json", "text", "csv", ...) with data_files
    dataset_config_name: str | None = None
    data_files: str | list[str] | dict[str, str | list[str]] | None = None
    # A python format string over dataset columns, or [prefix, completion] which are concatenated for training
    dataset_text_template: str | list[str] | None = None
    # Validation texts: the first of the validation split, or held out from the start of the train split
    max_eval_samples: int = 256
    shuffle_buffer_size: int = 10_000
    seed: int = 42
    trust_remote_code: bool = False

    # Set by the training config: micro batches drawn per evaluation, per data parallel rank, and their size
    eval_micro_batches: int = 1
    micro_batch_size: int = 1

    def examples(self, split: str) -> IterableDataset:
        """The raw examples of the train or validation split, streamed."""
        has_validation = "validation" in get_dataset_split_names(self.dataset_name, self.dataset_config_name,
                                                                 data_files=self.data_files)
        dataset = load_dataset(self.dataset_name, self.dataset_config_name, data_files=self.data_files,
                               split="validation" if split == "validation" and has_validation else "train",
                               streaming=True)
        if split == "validation":
            return dataset.take(self.max_eval_samples)
        return dataset if has_validation else dataset.skip(self.max_eval_samples)

    def texts(self, split: str) -> IterableDataset:
        """The formatted (non-empty) texts of a split, streamed, as {"text": ...} examples."""
        template = self.dataset_text_template
        template = "".join(template) if isinstance(template, list) else template

        def format_text(example):
            if template is None and "text" not in example:
                raise ValueError(f"No 'text' column (columns: {list(example)}), set data.dataset_text_template")
            return {"text": example["text"] if template is None else template.format(**example)}

        dataset = self.examples(split)
        return dataset.map(format_text, remove_columns=dataset.column_names).filter(lambda e: len(e["text"]) > 0)


def train_texts(config: TextDataConfig, rank: int, world_size: int) -> Iterator[str]:
    """Endless shuffled train texts of this data parallel rank and dataloader worker: each worker reads its rank's
    stream (shuffled alike) and keeps every num_workers-th text, so all workers make examples however many files
    the dataset has. Epochs reshuffle."""
    # ponytail: resuming from a checkpoint restarts the stream; track the texts consumed if exact resumption matters
    dataset = split_dataset_by_node(config.texts("train").shuffle(seed=config.seed,
                                                                  buffer_size=config.shuffle_buffer_size),
                                    rank=rank, world_size=world_size)
    worker = torch.utils.data.get_worker_info()
    worker_id, num_workers = (worker.id, worker.num_workers) if worker is not None else (0, 1)
    for epoch in count():
        dataset.set_epoch(epoch)
        # .iter(), unlike __iter__, does not split the stream across dataloader workers by files
        texts = (text for batch in dataset.iter(batch_size=1000) for text in batch["text"])
        empty = True
        for text in islice(texts, worker_id, None, num_workers):
            empty = False
            yield text
        if empty:
            raise ValueError(f"No train texts for data parallel rank {rank}, dataloader worker {worker_id}")


class TrainExamples(torch.utils.data.IterableDataset):
    """Examples made by make_examples from this rank's (and dataloader worker's) endless train texts."""

    def __init__(self, config: TextDataConfig, make_examples: Callable[[Iterable[str]], Iterator[dict]],
                 rank: int, world_size: int):
        self.config, self.make_examples, self.rank, self.world_size = config, make_examples, rank, world_size

    def __iter__(self):
        return self.make_examples(train_texts(self.config, self.rank, self.world_size))


def build_iterators(config, make_examples: Callable[[Iterable[str]], Iterator[dict]], collate_fn,
                    rank: int, world_size: int):
    """Megatron's (external) train and validation iterators of micro batches, for a DatasetProvider with dataloader
    options (num_workers, pin_memory, persistent_workers). Validation: the examples of the validation texts, split
    across ranks, repeated to fill config.eval_micro_batches, the same ones at every evaluation."""
    train = torch.utils.data.DataLoader(
        TrainExamples(config, make_examples, rank, world_size), batch_size=config.micro_batch_size,
        collate_fn=collate_fn, num_workers=config.num_workers, pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers and config.num_workers > 0)
    if config.eval_micro_batches == 0:  # No evaluation
        return iter(train), None

    texts = [example["text"] for example in config.texts("validation")]
    examples = list(make_examples(texts))[rank::world_size]
    if not examples:
        raise ValueError(f"No validation examples for data parallel rank {rank} (from {len(texts)} texts)")
    batches = [collate_fn(examples[i:i + config.micro_batch_size])
               for i in range(0, len(examples), config.micro_batch_size)]
    if len(batches) > config.eval_micro_batches:
        logger.warning(f"Evaluations see {config.eval_micro_batches} of {len(batches)} validation micro batches, "
                       "increase validation.eval_iters to see them all")
    validation = [batches[i % len(batches)] for i in range(config.eval_micro_batches)]
    return iter(train), cycle(validation)


def pack_words(sequences: Iterable[list[str]], seq_length: int) -> Iterator[tuple[list[str], list[int]]]:
    """Greedily pack word sequences, in order, into blocks of exactly seq_length words: (words, sequence lengths).
    Longer sequences are split into seq_length chunks, each its own sequence. Each block is right-padded with PAD
    words, each its own sequence, for constant shapes."""
    # ponytail: a chunk's last word is trained as a document end; carry the next chunk's first word as its label
    # (a label without an input word) if long documents matter
    words, lengths = [], []
    for document in sequences:
        for start in range(0, len(document), seq_length):
            sequence = document[start:start + seq_length]
            if len(words) + len(sequence) > seq_length:
                yield words + [PAD_WORD] * (seq_length - len(words)), lengths + [1] * (seq_length - len(words))
                words, lengths = [], []
            words += sequence
            lengths.append(len(sequence))
    if words:
        yield words + [PAD_WORD] * (seq_length - len(words)), lengths + [1] * (seq_length - len(words))
