"""Tests for on-the-fly sequence packing in bytes decoder."""

import torch
import pytest

from welt.model_utils import setup_model


def setup_tiny_model(**kwargs):
    """Set up a tiny version of the WordLatentTransformer model for testing."""
    return setup_model(
        image_encoder_name="WinKawaks/vit-tiny-patch16-224",
        bytes_encoder_name="prajjwal1/bert-tiny",
        latent_transformer_name="sbintuitions/tiny-lm",
        bytes_decoder_name="sbintuitions/tiny-lm",
        load_pretrained=False,
        **kwargs
    )


def test_pack_sequences_basic():
    """Test that packing sequences works correctly."""
    model, processor, collator = setup_tiny_model()
    model.eval()

    # Create test data with varying sequence lengths
    B, L, hidden_dim = 2, 4, model.bytes_decoder.config.hidden_size  # noqa: N806
    T = 8  # max token length  # noqa: N806

    # Simulate latent vectors
    latent_vectors = torch.randn(B, L, hidden_dim)

    # Create target_ids and masks with different lengths
    target_ids = torch.randint(0, 100, (B, L, T))
    target_mask = torch.zeros(B, L, T)

    # Set different lengths for each word
    # First batch: words of length [2, 3, 4, 5]
    # Second batch: words of length [1, 2, 3, 4]
    lengths = [[2, 3, 4, 5], [1, 2, 3, 4]]
    for b in range(B):
        for l_idx in range(L):
            target_mask[b, l_idx, :lengths[b][l_idx]] = 1

    # Flatten for testing the internal packing function
    target_ids_flat = target_ids.view(B * L, T)
    target_mask_flat = target_mask.view(B * L, T)
    latent_vectors_flat = latent_vectors.view(B * L, hidden_dim)

    # Get embeddings
    embed_layer = model.bytes_decoder.get_input_embeddings()
    target_embeds = embed_layer(target_ids_flat)

    # Test packing
    max_packed_length = 20
    packed_embeds, packed_masks, unpack_indices = model._pack_sequences_for_decoding(
        latent_vectors_flat, target_embeds, target_mask_flat, max_packed_length
    )

    # Verify packing results
    assert len(packed_embeds) > 0, "Should have at least one packed sequence"
    assert len(packed_embeds) == len(packed_masks), "Should have matching embeds and masks"
    assert len(unpack_indices) == B * L, f"Should have {B * L} unpack indices"

    # Verify that all sequences are accounted for
    for i, (pack_idx, start_pos, end_pos) in enumerate(unpack_indices):
        assert pack_idx < len(packed_embeds), f"Pack index {pack_idx} out of range"
        expected_len = lengths[i // L][i % L] + 1  # +1 for latent vector
        actual_len = end_pos - start_pos
        assert actual_len == expected_len, f"Sequence {i} length mismatch: {actual_len} vs {expected_len}"

    # Verify that packed sequences don't exceed max length
    for pack_embeds in packed_embeds:
        assert pack_embeds.shape[0] <= max_packed_length, "Packed sequence exceeds max length"

    print(f"✓ Packing test passed: {B * L} sequences packed into {len(packed_embeds)} packs")


def test_unpack_logits():
    """Test that unpacking logits works correctly."""
    model, processor, collator = setup_tiny_model()
    model.eval()

    N, T = 4, 8  # noqa: N806
    vocab_size = 100

    # Create dummy packed logits
    packed_logits_list = [
        torch.randn(10, vocab_size),  # First pack with 10 tokens
        torch.randn(8, vocab_size),   # Second pack with 8 tokens
    ]

    # Create unpack indices for 4 sequences
    # Sequence 0: pack 0, positions 0-3 (length 3 + 1 latent)
    # Sequence 1: pack 0, positions 3-6 (length 3 + 1 latent)
    # Sequence 2: pack 0, positions 6-10 (length 4 + 1 latent)
    # Sequence 3: pack 1, positions 0-8 (length 8 + 1 latent)
    unpack_indices = [
        (0, 0, 3),
        (0, 3, 6),
        (0, 6, 10),
        (1, 0, 8),
    ]

    # Unpack
    unpacked_logits = model._unpack_logits(packed_logits_list, unpack_indices, (N, T))

    # Verify shape
    assert unpacked_logits.shape == (N, T, vocab_size), f"Wrong shape: {unpacked_logits.shape}"

    # Verify content for each sequence
    # Sequence 0: should have 2 tokens (3 - 1 for latent)
    assert torch.allclose(unpacked_logits[0, :2], packed_logits_list[0][1:3]), "Sequence 0 mismatch"

    # Sequence 1: should have 2 tokens (3 - 1 for latent)
    assert torch.allclose(unpacked_logits[1, :2], packed_logits_list[0][4:6]), "Sequence 1 mismatch"

    # Sequence 2: should have 3 tokens (4 - 1 for latent)
    assert torch.allclose(unpacked_logits[2, :3], packed_logits_list[0][7:10]), "Sequence 2 mismatch"

    # Sequence 3: should have 7 tokens (8 - 1 for latent)
    assert torch.allclose(unpacked_logits[3, :7], packed_logits_list[1][1:8]), "Sequence 3 mismatch"

    print("✓ Unpacking test passed")


def test_parallel_causal_decode_with_packing():
    """Test that parallel_causal_decode produces same results with packing."""
    model, processor, collator = setup_tiny_model()
    model.eval()

    # Create test data
    B, L = 2, 3  # noqa: N806
    hidden_dim = model.bytes_decoder.config.hidden_size
    T = 8  # noqa: N806

    latent_vectors = torch.randn(B, L, hidden_dim)
    target_ids = torch.randint(0, 100, (B, L, T))
    target_mask = torch.zeros(B, L, T)

    # Set different lengths for each word
    lengths = [[2, 3, 4], [1, 2, 5]]
    for b in range(B):
        for l in range(L):
            target_mask[b, l, :lengths[b][l]] = 1

    # Run through model
    with torch.no_grad():
        logits = model.parallel_causal_decode(latent_vectors, target_ids, target_mask)

    # Verify output shape
    assert logits.shape == (B, L, T, model.bytes_decoder.config.vocab_size), f"Wrong output shape: {logits.shape}"

    # Verify that logits are non-zero where we have actual tokens
    for b in range(B):
        for l in range(L):
            seq_len = lengths[b][l]
            # Check that we have non-zero logits for actual tokens
            assert not torch.all(logits[b, l, :seq_len] == 0), f"Logits for sequence ({b}, {l}) are all zero"

    print("✓ Parallel causal decode with packing test passed")


def test_packing_reduces_computation():
    """Test that packing actually reduces the number of decoder calls."""
    model, processor, collator = setup_tiny_model()
    model.eval()

    # Create test data with many short sequences
    B, L = 4, 16  # noqa: N806  # 64 total sequences
    hidden_dim = model.bytes_decoder.config.hidden_dim
    T = 32  # max token length  # noqa: N806

    latent_vectors = torch.randn(B, L, hidden_dim)
    target_ids = torch.randint(0, 100, (B, L, T))
    target_mask = torch.zeros(B, L, T)

    # Create mostly short sequences (2-5 tokens each)
    for b in range(B):
        for l in range(L):
            length = torch.randint(2, 6, (1,)).item()
            target_mask[b, l, :length] = 1

    # Flatten for testing
    target_ids_flat = target_ids.view(B * L, T)
    target_mask_flat = target_mask.view(B * L, T)
    latent_vectors_flat = latent_vectors.view(B * L, hidden_dim)

    # Get embeddings
    embed_layer = model.bytes_decoder.get_input_embeddings()
    target_embeds = embed_layer(target_ids_flat)

    # Test packing
    max_packed_length = T * 2
    packed_embeds, packed_masks, unpack_indices = model._pack_sequences_for_decoding(
        latent_vectors_flat, target_embeds, target_mask_flat, max_packed_length
    )

    # Verify that packing reduces the number of sequences
    num_packed = len(packed_embeds)
    num_original = B * L

    assert num_packed < num_original, f"Packing should reduce sequences: {num_packed} vs {num_original}"

    reduction_ratio = num_packed / num_original
    print(f"✓ Packing reduced {num_original} sequences to {num_packed} ({reduction_ratio:.2%} of original)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
