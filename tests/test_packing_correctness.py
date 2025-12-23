"""Test that packing maintains training correctness."""

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


def create_unpacked_baseline(model, latent_vectors, target_ids, target_mask):
    """
    Create baseline results without packing by processing each sequence individually.
    This simulates the old behavior for comparison.
    """
    B, L, hidden_dim = latent_vectors.shape  # noqa: N806
    _, _, T = target_ids.shape  # noqa: N806
    
    # Flatten inputs
    target_ids_flat = target_ids.view(B * L, T)
    target_mask_flat = target_mask.view(B * L, T)
    latent_vectors_flat = latent_vectors.view(B * L, hidden_dim)
    
    # Get embeddings
    embed_layer = model.bytes_decoder.get_input_embeddings()
    target_embeds = embed_layer(target_ids_flat)
    
    # Process each sequence individually (no packing)
    all_logits = []
    for i in range(B * L):
        # Get actual length
        seq_len = target_mask_flat[i].sum().item()
        
        # Prepare sequence
        latent_vec = latent_vectors_flat[i].unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
        seq_embeds = target_embeds[i:i+1, :seq_len]  # (1, seq_len, embed_dim)
        combined = torch.cat([latent_vec, seq_embeds], dim=1)  # (1, 1 + seq_len, embed_dim)
        
        # Create mask
        seq_mask = torch.ones(1, 1 + seq_len, device=combined.device)
        
        # Forward pass
        outputs = model.bytes_decoder(
            inputs_embeds=combined,
            attention_mask=seq_mask,
            output_hidden_states=False
        )
        
        # Extract logits (skip latent position)
        logits = outputs.logits[0, 1:]  # (seq_len, vocab_size)
        
        # Pad to T
        padded_logits = torch.zeros(T, outputs.logits.shape[-1], device=logits.device, dtype=logits.dtype)
        padded_logits[:seq_len] = logits
        all_logits.append(padded_logits)
    
    # Stack and reshape
    all_logits = torch.stack(all_logits, dim=0)  # (B*L, T, vocab_size)
    return all_logits.view(B, L, T, -1)


def test_packing_produces_identical_results():
    """Test that packed and unpacked versions produce identical logits."""
    model, processor, collator = setup_tiny_model()
    model.eval()
    
    # Create test data with varying lengths
    B, L = 2, 4  # noqa: N806
    hidden_dim = model.bytes_decoder.config.hidden_size
    T = 8  # noqa: N806
    
    torch.manual_seed(42)
    latent_vectors = torch.randn(B, L, hidden_dim)
    target_ids = torch.randint(0, 100, (B, L, T))
    target_mask = torch.zeros(B, L, T)
    
    # Set varying lengths
    lengths = [[2, 3, 4, 5], [1, 2, 3, 6]]
    for b in range(B):
        for l in range(L):
            target_mask[b, l, :lengths[b][l]] = 1
    
    with torch.no_grad():
        # Get results with packing (new implementation)
        packed_logits = model.parallel_causal_decode(latent_vectors, target_ids, target_mask)
        
        # Get baseline results without packing
        unpacked_logits = create_unpacked_baseline(model, latent_vectors, target_ids, target_mask)
    
    # Compare results
    for b in range(B):
        for l in range(L):
            seq_len = lengths[b][l]
            # Only compare non-padded positions
            packed_seq = packed_logits[b, l, :seq_len]
            unpacked_seq = unpacked_logits[b, l, :seq_len]
            
            # Check if logits are close (allowing for small numerical differences)
            max_diff = (packed_seq - unpacked_seq).abs().max().item()
            
            # They should be identical (within floating point precision)
            assert max_diff < 1e-4, (
                f"Logits mismatch at ({b}, {l}): max_diff={max_diff:.6f}"
            )
    
    print("✓ Packing produces identical results to unpacked baseline")


def test_packing_loss_computation():
    """Test that loss computation is correct with packing."""
    model, processor, collator = setup_tiny_model()
    model.eval()
    
    # Create simple test data
    B, L = 1, 3  # noqa: N806
    hidden_dim = model.bytes_decoder.config.hidden_size
    T = 8  # noqa: N806
    vocab_size = model.bytes_decoder.config.vocab_size
    
    torch.manual_seed(42)
    latent_vectors = torch.randn(B, L, hidden_dim)
    target_ids = torch.randint(0, vocab_size, (B, L, T))
    target_mask = torch.zeros(B, L, T)
    
    # Set lengths
    lengths = [2, 3, 4]
    for l in range(L):
        target_mask[0, l, :lengths[l]] = 1
    
    # Create labels (same as target_ids but shifted)
    labels = target_ids.clone()
    
    with torch.no_grad():
        # Get logits
        logits = model.parallel_causal_decode(latent_vectors, target_ids, target_mask)
        
        # Compute loss manually for each sequence
        individual_losses = []
        for l in range(L):
            seq_len = lengths[l]
            seq_logits = logits[0, l, :seq_len]  # (seq_len, vocab_size)
            seq_labels = labels[0, l, :seq_len]  # (seq_len,)
            
            loss = torch.nn.functional.cross_entropy(
                seq_logits, seq_labels, reduction='mean'
            )
            individual_losses.append(loss.item())
        
        # Compute loss on entire batch (with masking)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_labels = labels.reshape(-1)
        flat_mask = target_mask.reshape(-1)
        
        # Mask out padding positions by setting labels to -100
        masked_labels = flat_labels.clone()
        masked_labels[flat_mask == 0] = -100
        
        batch_loss = torch.nn.functional.cross_entropy(
            flat_logits, masked_labels, ignore_index=-100, reduction='mean'
        )
        
        # The batch loss should be similar to the mean of individual losses
        mean_individual = sum(individual_losses) / len(individual_losses)
        
        print(f"Individual losses: {individual_losses}")
        print(f"Mean individual loss: {mean_individual:.6f}")
        print(f"Batch loss: {batch_loss.item():.6f}")
        
        # Should be relatively close (not exact due to different reduction strategies)
        assert abs(batch_loss.item() - mean_individual) < 0.5, \
            "Batch loss differs significantly from mean individual losses"
    
    print("✓ Loss computation is correct with packing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
