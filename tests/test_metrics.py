import math

import pytest
import torch

pytest.importorskip("megatron.bridge", reason="Requires the NeMo container")

from utf8_tokenizer.tokenizer import EOS_TOKEN_ID, PAD_TOKEN_ID  # noqa: E402

from welt_training import baseline, train  # noqa: E402


def test_welt_loss_func_metrics():
    pad, eos = PAD_TOKEN_ID, EOS_TOKEN_ID
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


def test_baseline_loss_func_bits_per_byte():
    losses = torch.tensor([[2.0, 3.0, 1.0, 1.5, 9.0]])
    loss_mask = torch.tensor([[1, 1, 1, 1, 0]])  # The last label is padding
    # The third label is EOS; the fourth ends a character split across tokens (0 bytes, but scored text)
    label_bytes = torch.tensor([[1, 4, 0, 0, 0]])
    text_mask = torch.tensor([[1, 1, 0, 1, 0]])
    loss, num_tokens, report = baseline.loss_func(losses, loss_mask, label_bytes, text_mask)
    assert loss == 7.5
    assert num_tokens == 4
    assert report["bits per byte"].tolist() == pytest.approx([6.5 / math.log(2), 5])
