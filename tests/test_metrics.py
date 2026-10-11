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


def test_baseline_counts_the_bytes_of_its_texts():
    """Each token counts the text since the previous one: every byte once, also with characters split across tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    texts = ["Hello wörld 👋 שלום 日本語 🧑‍🤝‍🧑", "a"]
    (ids, _, nbytes), = baseline.token_chunks(texts, tokenizer, length=64)
    assert sum(nbytes) == sum(len(text.encode()) for text in texts)
    assert [n for i, n in zip(ids, nbytes, strict=True) if i == tokenizer.eos_token_id] == [0] * ids.count(
        tokenizer.eos_token_id)


def test_baseline_scores_the_bytes_welt_scores():
    """Every token is a label once (chunks overlap by one token); shift blocks are not scored, as in WeLT."""
    from transformers import AutoTokenizer

    from welt.processor import TextImageProcessor, next_word_labels

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    texts = ["<en>\x0eHello world\x0f<he> שלום עולם", "no shift block here", "<a>\x0eb\x0f c"]
    examples = list(baseline.token_examples(texts, tokenizer, length=4))
    labels = [token for example in examples for token in example["labels"][example["loss_mask"].bool()].tolist()]
    # The labels: each document (after the first, after its EOS) without the text of its shift block
    assert tokenizer.decode(labels) == tokenizer.eos_token.join(baseline.SHIFT_BLOCK.sub("\x0e", t) for t in texts)
    baseline_bytes = sum(int(example["label_bytes"].sum()) for example in examples)

    processor = TextImageProcessor.create(max_word_length=32, render_images=False)
    welt_bytes = 0
    for text in texts:
        words = processor.pretokenize(text)
        example = processor.process_single_example(words, [len(words)])
        labels = next_word_labels(*(example[k][None] for k in ("input_ids", "sequence_ids", "label_mask")),
                                  bos=2, eos=3, pad=0)[:, :, 1:]  # Without BOS
        welt_bytes += int(((labels != train.TOKENIZER.pad_token_id) & (labels != train.TOKENIZER.eos_token_id)).sum())
    assert baseline_bytes == welt_bytes == sum(len(baseline.SHIFT_BLOCK.sub("\x0e", t).encode()) for t in texts)


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
