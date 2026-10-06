"""Numerical and dataloader regression checks without pretrained assets."""

import math
from types import SimpleNamespace

import pytest
import torch
from accelerate.data_loader import prepare_data_loader
from accelerate.state import GradientState
from datasets import Dataset
from torch.utils.data import DataLoader, IterableDataset
from utf8_tokenizer import CharacterCausalLMWrapper, CharacterEmbedding, UTF8Tokenizer, UTF16Tokenizer, UTF32Tokenizer

from welt.model import WordLatentTransformerForCausalLM
from welt_training.streaming import CustomIterableDataset, take_streaming_dataset
from welt_training.trainer import WeLTTrainer


@pytest.mark.parametrize("size", [1, 3, 4, 5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize("split_batches", [False, True])
def test_distributed_padding_counts_real_samples(size, split_batches):
    """Use Accelerate's real last-batch state, for both sharding modes."""
    counts = []
    observed = []
    for rank in range(2):
        trainer = WeLTTrainer.__new__(WeLTTrainer)
        trainer.accelerator = SimpleNamespace(
            num_processes=2, process_index=rank, gradient_state=GradientState())
        loader = prepare_data_loader(
            DataLoader(list(range(size)), batch_size=4),
            device=torch.device("cpu"), num_processes=2, process_index=rank,
            split_batches=split_batches, dispatch_batches=False,
        )
        for batch in loader:
            count = trainer._real_batch_count(len(batch))
            counts.append(count)
            observed.extend(batch[:count].tolist())
    assert sum(counts) == size
    assert sorted(observed) == list(range(size))


@pytest.mark.parametrize(("encoding", "width"), [("UTF-8", 1), ("UTF-16", 2), ("UTF-32", 4)])
@pytest.mark.parametrize("unicode_content", [False, True])
def test_bpb_and_accuracy_use_each_byte_distribution(encoding, width, unicode_content):
    """Compare counters to explicitly summed content-byte log probabilities."""
    trainer = WeLTTrainer.__new__(WeLTTrainer)
    trainer.accelerator = SimpleNamespace(num_processes=1, unwrap_model=lambda model: model)
    trainer.processor = SimpleNamespace(tokenizer={
        "UTF-8": UTF8Tokenizer, "UTF-16": UTF16Tokenizer, "UTF-32": UTF32Tokenizer}[encoding]())
    trainer._reset_eval_state()
    embedding = CharacterEmbedding(embedding_size=16, num_bytes=width) if width > 1 else None
    decoder = SimpleNamespace(char_embedding=embedding)
    decoder.compute_loss = lambda logits, labels: CharacterCausalLMWrapper.compute_loss(decoder, logits, labels)
    model = SimpleNamespace(config=SimpleNamespace(encoding=encoding), bytes_decoder=decoder)
    content_id = 65 if width == 1 or not unicode_content else 0x1F642 if width == 4 else 0x03A9
    utf8_byte_count = len(chr(content_id).encode("utf-8"))
    labels = torch.tensor([[[content_id, 3, 0]]])
    byte_labels = embedding._split_to_bytes(labels) if embedding is not None else labels.unsqueeze(-1)
    logits = torch.zeros(1, 1, 3, width, 256)
    logits.scatter_(-1, byte_labels.unsqueeze(-1), 2.0)
    # Make EOS deliberately wrong: its loss must not enter BPB.
    logits[:, :, 1] = -7
    expected_nats = -logits[:, :, :1].log_softmax(-1).gather(
        -1, byte_labels[:, :, :1].unsqueeze(-1)).sum().item()
    trainer._accumulate_accuracy_and_bpb(model, {"labels_output": labels}, logits.flatten(-2))
    assert trainer._eval_total_nats == pytest.approx(expected_nats)
    assert trainer._eval_total_content_bytes == utf8_byte_count
    assert trainer._eval_correct_bytes == 1
    metrics = {}
    trainer._add_custom_metrics(metrics)
    assert metrics["eval_bits_per_byte"] == pytest.approx(expected_nats / (utf8_byte_count * math.log(2)))


class RawTorchDataset(IterableDataset):
    def __iter__(self):
        yield from [{"text": "one"}, {"text": "two"}, {"text": "three"}]


@pytest.mark.parametrize("kind", ["hf", "torch", "custom"])
def test_raw_iterables_are_transformed_and_sharded(kind):
    raw = Dataset.from_dict({"text": ["one", "two", "three"]}).to_iterable_dataset()
    if kind == "torch":
        raw = RawTorchDataset()
    elif kind == "custom":
        raw = CustomIterableDataset(raw)
    observed = []
    for rank in range(2):
        trainer = WeLTTrainer.__new__(WeLTTrainer)
        trainer.eval_dataset = raw
        trainer.loaded_metrics = {}
        trainer.accelerator = SimpleNamespace(num_processes=2, process_index=rank)
        trainer.args = SimpleNamespace(
            per_device_eval_batch_size=2, dataloader_num_workers=0, dataloader_pin_memory=False)
        trainer.processor = lambda batch: {"processed": [text.upper() for text in batch["text"]]}
        trainer.data_collator = lambda rows: [row["processed"] for row in rows]
        prepared = trainer._prepare_eval_dataset(None)
        for batch in trainer.get_eval_dataloader(prepared):
            observed.extend(batch)
    assert sorted(observed) == ["ONE", "THREE", "TWO"]


def test_limited_stream_can_repeat_epochs():
    raw = Dataset.from_dict({"text": ["one", "two", "three"]}).to_iterable_dataset()
    limited = take_streaming_dataset(raw, 2)
    for epoch in [0, 1, 2]:
        limited.set_epoch(epoch)
        assert list(limited) == [{"text": "one"}, {"text": "two"}]


@pytest.mark.parametrize(("encoding", "width"), [("UTF-8", 1), ("UTF-16", 2), ("UTF-32", 4)])
def test_entropy_uses_character_tokens_and_separate_byte_distributions(encoding, width):
    tokenizer = {"UTF-8": UTF8Tokenizer, "UTF-16": UTF16Tokenizer, "UTF-32": UTF32Tokenizer}[encoding]()
    words = ["a", "Ω🙂"]
    tokenized = tokenizer.torch(words, padding=True, add_special_tokens=True)

    def decode(_latents, input_ids, attention_mask):
        assert torch.equal(input_ids[0], tokenized.input_ids[:, :-1])
        assert torch.equal(attention_mask[0], tokenized.attention_mask[:, :-1])
        return torch.zeros(*input_ids.shape, width * 256)

    model = SimpleNamespace(config=SimpleNamespace(encoding=encoding), parallel_causal_decode=decode)
    entropies, byte_labels = WordLatentTransformerForCausalLM._compute_generation_entropy(
        model, [torch.zeros(1, 1, 16)] * 2, words, tokenizer, torch.device("cpu"))
    byte_count = sum(len(word.encode(tokenizer.encoding)) for word in words)
    assert len(byte_labels) == byte_count
    assert entropies == pytest.approx([8.0] * byte_count)
