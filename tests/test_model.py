from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

pytest.importorskip("megatron.bridge", reason="Requires the NeMo container")

from tests.conftest import ORACLE_TEXTS, PAD, build_model, oracle_mask  # noqa: E402
from welt.model import FlexCoreAttention, latent_block_mask, packed_block_mask  # noqa: E402
from welt.processor import TextImageProcessor, collate_fn  # noqa: E402
from welt_training.data_utils import pack_words  # noqa: E402


@pytest.fixture(scope="module")
def model(megatron, tiny_config):
    torch.manual_seed(0)
    return build_model(tiny_config)


@pytest.fixture(scope="module")
def processor():
    return TextImageProcessor.create(max_word_length=16, render_images=True)


def word_losses(model, processor, texts: list[str]) -> torch.Tensor:
    """Per word (B, L) summed byte losses."""
    batch = {k: v.cuda() for k, v in processor(texts).items()}
    with torch.no_grad():
        losses, _, labels = model(**batch)
    per_word = torch.zeros(batch["label_mask"].shape, device=losses.device)
    per_word[batch["label_mask"]] = (losses * (labels != 0)).sum(dim=-1)
    return per_word


def test_forward_is_finite(model, processor):
    losses = word_losses(model, processor, ["hello world, how are you?", "short"])
    assert torch.isfinite(losses).all()
    assert (losses[0, :5] > 0).all()


def test_attention_no_look_ahead(model, processor):
    """Changing a later word does not change the losses of earlier words."""
    a = word_losses(model, processor, ["the quick brown fox jumps"])
    b = word_losses(model, processor, ["the quick brown cat jumps"])
    # Words: BOS, "the ", "quick ", "brown ", ... Word i predicts word i+1.
    torch.testing.assert_close(a[0, :3], b[0, :3])
    assert not torch.allclose(a[0, 3:5], b[0, 3:5])


def test_attention_does_look_back(model, processor):
    """Changing an earlier word changes the losses of later words."""
    a = word_losses(model, processor, ["the quick brown fox jumps"])
    b = word_losses(model, processor, ["a quick brown fox jumps"])
    assert not torch.allclose(a[0, 2:], b[0, 2:])


def test_loss_is_independent_of_batch(model, processor):
    texts = ["the quick brown fox", "jumps over the lazy dog and more words"]
    batched = word_losses(model, processor, texts)
    for i, text in enumerate(texts):
        alone = word_losses(model, processor, [text])
        torch.testing.assert_close(batched[i, :alone.shape[1]], alone[0], atol=2e-2, rtol=2e-2)


def test_backward_reaches_all_parameters(model, processor):
    model.train()
    try:
        batch = {k: v.cuda() for k, v in processor(["hello world, how are you?"]).items()}
        losses, _, labels = model(**batch)
        (losses * (labels != 0)).sum().backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        assert not missing, f"Parameters without gradients: {missing}"
    finally:
        model.zero_grad(set_to_none=True)
        model.eval()


ATTENTION_CONFIG = SimpleNamespace(softmax_scale=None, attention_dropout=0.0)


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize(("lengths", "kv_heads"), [
    ([3, 7, 1, 5], 4),  # Within one 128-token block
    ([3, 300, 7, 129, 1, 250], 2),  # Sequences across blocks, grouped query attention
])
def test_packed_attention_matches_padded_attention(megatron, causal, lengths, kv_heads):
    from megatron.core.packed_seq_params import PackedSeqParams

    torch.manual_seed(0)
    lengths = torch.tensor(lengths, device="cuda")
    total, heads, dim = int(lengths.sum()), 4, 16
    query = torch.randn(total, heads, dim, device="cuda", dtype=torch.bfloat16)
    key, value = (torch.randn(total, kv_heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                             max_seqlen_q=int(lengths.max()), max_seqlen_kv=int(lengths.max()))
    params.block_mask = packed_block_mask(cu_seqlens, causal)
    packed = FlexCoreAttention(ATTENTION_CONFIG)(query, key, value, packed_seq_params=params)

    expected = []
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True):
        q, k, v = (t[start:end].transpose(0, 1) for t in (query, key, value))
        expected.append(F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True).transpose(0, 1))
    torch.testing.assert_close(packed, torch.cat(expected), atol=2e-2, rtol=2e-2)


def packed_batch(processor, seq_length: int):
    """Two packed examples of the oracle texts (the second with PAD words), their words and sequence lengths."""
    packed = list(pack_words(map(processor.pretokenize, ORACLE_TEXTS * 20), seq_length))[:2]
    examples = [processor.process_single_example(words, seq_lengths) for words, seq_lengths in packed]
    return collate_fn(examples), packed


def test_latent_mask_matches_an_oracle_of_the_words(megatron, processor):
    from torch.nn.attention.flex_attention import create_mask

    batch, packed = packed_batch(processor, seq_length=40)
    block_mask = latent_block_mask(batch["sequence_ids"].cuda(), batch["block_ids"].cuda())
    dense = create_mask(block_mask.mask_mod, len(packed), None, 40, 40, device="cuda")[:, 0].cpu()
    for i, (words, seq_lengths) in enumerate(packed):
        torch.testing.assert_close(dense[i], oracle_mask(words, seq_lengths))


def test_masked_attention_matches_sdpa(megatron, processor):
    """FlexAttention with the latent mask (its sparse blocks too) against SDPA with the oracle's dense mask."""
    torch.manual_seed(0)
    seq, heads, dim = 200, 4, 16  # Over one 128-word block
    batch, packed = packed_batch(processor, seq_length=seq)
    batch = {k: v.cuda() for k, v in batch.items()}
    allowed = torch.stack([oracle_mask(words, lengths) for words, lengths in packed]).cuda()[:, None]
    query, key, value = (torch.randn(seq, len(packed), heads, dim, device="cuda", dtype=torch.bfloat16)
                         for _ in range(3))
    mask = torch.empty(0)
    mask.block_mask = latent_block_mask(batch["sequence_ids"], batch["block_ids"])
    out = FlexCoreAttention(ATTENTION_CONFIG)(query, key, value, attention_mask=mask)

    q, k, v = (t.permute(1, 2, 0, 3) for t in (query, key, value))
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed).permute(2, 0, 1, 3).flatten(2)
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


def test_packing_is_invisible(model, processor):
    """Each text's per-byte losses are the same alone as packed with the others and with PAD words: sequences,
    shift blocks and labels stay within their text."""
    words = [processor.pretokenize(text) for text in ORACLE_TEXTS]
    packed = processor.process_single_example(sum(words, []) + [PAD] * 3, [len(w) for w in words] + [1] * 3)
    with torch.no_grad():
        losses, _, labels = model(**{k: v.cuda() for k, v in collate_fn([packed]).items()})
        packed_losses = (losses * (labels != 0)).sum(dim=-1)
        alone = []
        for text in ORACLE_TEXTS:
            losses, _, labels = model(**{k: v.cuda() for k, v in processor([text]).items()})
            alone.append((losses * (labels != 0)).sum(dim=-1))
    torch.testing.assert_close(packed_losses, torch.cat(alone), atol=2e-2, rtol=2e-2)


def test_checkpoint_shards_every_transformer(megatron, tiny_config):
    """Every transformer's weights are saved by its own sharded_state_dict (Megatron's distributed checkpoint
    format, e.g. its layers stacked), not as plain tensors."""
    sharded = build_model(tiny_config).sharded_state_dict()
    for name in ["bytes_encoder.transformer", "image_encoder.transformer", "latent_transformer", "bytes_decoder"]:
        weight = sharded[f"{name}.decoder.layers.0.self_attention.linear_proj.weight"]
        assert weight.key == f"{name}.decoder.layers.self_attention.linear_proj.weight"  # Megatron's layer stacking
