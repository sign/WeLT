import math

import pytest
import torch

pytest.importorskip("megatron.bridge", reason="Requires the NeMo container")

from welt_training import baseline, train  # noqa: E402


def test_welt_loss_func_metrics():
    pad, eos = train.TOKENIZER.pad_token_id, train.TOKENIZER.eos_token_id
    # Words "ab", "c", then the end of the document (an empty word)
    labels = torch.tensor([[97, 98, eos, pad], [99, eos, pad, pad], [eos, pad, pad, pad]])
    losses = torch.tensor([[1.0, 2.0, 3.0, 9.0], [4.0, 5.0, 9.0, 9.0], [6.0, 9.0, 9.0, 9.0]])
    correct = torch.tensor([[True, True, True, False], [True, False, False, False], [True, False, False, False]])
    loss, num_tokens, report = train.loss_func(losses, correct, labels)
    assert num_tokens == 6
    assert loss == 21.0  # Padding excluded
    # Word ends are predicted (like spaces in the baseline), the document end is not, per byte of text
    assert report["bits per byte"].tolist() == pytest.approx([(1 + 2 + 3 + 4 + 5) / math.log(2), 3])
    assert report["byte accuracy"].tolist() == [5, 6]
    assert report["word accuracy"].tolist() == [2, 3]


def test_baseline_byte_lengths_and_chunks():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    lengths = baseline.token_byte_lengths(tokenizer)
    texts = ["Hello wörld", "שלום עולם"]
    ids = tokenizer(texts, add_special_tokens=False).input_ids
    assert [int(lengths[i].sum()) for i in map(torch.tensor, ids)] == [len(t.encode("utf-8")) for t in texts]
    assert lengths[tokenizer.eos_token_id] == 0

    chunks = baseline.chunk_tokens({"text": texts}, tokenizer, length=4)["input_ids"]
    flat = [t for ids_ in ids for t in [*ids_, tokenizer.eos_token_id]]
    assert all(len(chunk) == 4 for chunk in chunks)
    assert [t for chunk in chunks for t in chunk] == flat[:len(chunks) * 4]


def test_baseline_loss_func_bits_per_byte():
    losses = torch.tensor([[2.0, 3.0, 1.0]])
    label_bytes = torch.tensor([[1, 4, 0]])  # The last label is EOS
    loss, num_tokens, report = baseline.loss_func(losses, label_bytes)
    assert loss == 6.0
    assert num_tokens == 3
    assert report["bits per byte"].tolist() == pytest.approx([5 / math.log(2), 5])
