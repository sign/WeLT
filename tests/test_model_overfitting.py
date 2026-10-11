"""Overfit a tiny WeLT on a few texts, and check it learned character, word, and byte level conditioning."""
import pytest
import torch

pytest.importorskip("megatron.bridge", reason="Requires the NeMo container")

from tests.conftest import build_model  # noqa: E402
from welt.collator import collate_fn  # noqa: E402
from welt.processor import TextImageProcessor  # noqa: E402
from welt_training.data_utils import pack_words  # noqa: E402

TRAIN_TEXTS = ["a b", "b a", "a cat", "a dog"]


def train(model, processor, packed: bool, steps: int = 600):
    if packed:
        examples = pack_words({"words": [processor.pretokenize(text) for text in TRAIN_TEXTS]}, seq_length=7)
        batch = collate_fn([processor.process_single_example(w, lengths)
                            for w, lengths in zip(examples["words"], examples["seq_lengths"], strict=True)])
    else:
        batch = processor(TRAIN_TEXTS)
    batch = {k: v.cuda() for k, v in batch.items()}

    torch.manual_seed(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        losses, _, labels = model(**batch)
        loss = (losses * (labels != 0)).sum() / (labels != 0).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()


def text_losses(model, processor, texts: list[str]) -> dict[str, float]:
    """Mean byte loss of each text, on its own."""
    results = {}
    for text in texts:
        batch = {k: v.cuda() for k, v in processor([text]).items()}
        with torch.no_grad():
            losses, _, labels = model(**batch)
        mask = labels != 0
        results[text] = ((losses * mask).sum() / mask.sum()).item()
    return results


# (packing, encoders): both encoders, packed and not, and each encoder alone
@pytest.fixture(scope="module", params=[("packed", "full_model"), ("unpacked", "full_model"),
                                        ("packed", "no_bytes_encoder"), ("packed", "no_image_encoder")])
def configured(request, megatron, tiny_config):
    packing, name = request.param
    torch.manual_seed(1)  # The checks are margins of a tiny overfit model
    model = build_model(tiny_config, image_encoder=name != "no_image_encoder",
                        bytes_encoder=name != "no_bytes_encoder").float()
    processor = TextImageProcessor.create(max_word_length=16, render_images=name != "no_image_encoder")
    train(model, processor, packed=packing == "packed")
    return model, processor, name


def test_character_level_conditioning(configured):
    model, processor, name = configured
    if name == "no_bytes_encoder":
        pytest.skip("A tiny image encoder cannot distinguish single-character renders ('a' vs 'b')")
    losses = text_losses(model, processor, ["a b", "b a", "a a", "b b"])
    assert losses["a b"] < losses["a a"]
    assert losses["b a"] < losses["b b"]


def test_word_level_conditioning(configured):
    model, processor, _ = configured
    losses = text_losses(model, processor, ["a cat", "a dog", "a dat", "a cog"])
    assert losses["a cat"] < losses["a dat"]
    assert losses["a dog"] < losses["a cog"]


def test_byte_level_conditioning(configured):
    """After 'c', 'cat' is more likely than 'cog', and after 'd', 'dog' than 'dat'."""
    model, processor, _ = configured
    losses = text_losses(model, processor, ["a cat", "a cog", "a dog", "a dat"])
    assert losses["a cat"] < losses["a cog"]
    assert losses["a dog"] < losses["a dat"]
