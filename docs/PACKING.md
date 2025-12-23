# On-the-fly Sequence Packing for Bytes Decoder

## Overview

This implementation adds on-the-fly sequence packing to the bytes decoder training, significantly reducing padding waste and improving training efficiency.

## Problem

Previously, the bytes decoder processed sequences with the following characteristics:
- Each word is padded to max length `T` (e.g., 32 tokens)
- For a batch with `B` samples and `L` words, we create `B×L` sequences
- Example: 128 batch size × 512 words = 65,536 sequences
- Each sequence padded to 32 tokens = 2,097,152 tokens total
- Most words are short (e.g., "a" = 2 tokens, "the" = 4 tokens), leading to significant padding waste

## Solution

The implementation packs multiple short sequences into single decoder passes:
- Calculate actual length of each word (from attention mask)
- Pack words sequentially until reaching `max_packed_length` (default: `T × 2`)
- Process packed sequences through the decoder
- Unpack results back to original shape

## Implementation

### New Methods

1. **`_pack_sequences_for_decoding`**
   - Input: Flattened latent vectors, embeddings, and attention masks
   - Output: Packed sequences, masks, and unpacking indices
   - Strategy: Greedy packing - add sequences until max length reached

2. **`_unpack_logits`**
   - Input: Packed logits and unpacking indices
   - Output: Logits in original (B, L, T, vocab_size) shape
   - Strategy: Extract and place logits using stored indices

3. **Modified `parallel_causal_decode`**
   - Now calls packing before decoder
   - Processes packed sequences in a loop
   - Unpacks results to original shape

### Key Design Decisions

1. **Greedy Packing**: Simple, efficient, and works well in practice
2. **max_packed_length = T × 2**: Conservative estimate allowing ~2 average words per pack
3. **No Cross-Pack Attention**: Each packed sequence is independent
4. **Zero Padding for Output**: Unpacked positions default to zero (ignored by loss)

## Performance

Based on simulations with realistic word length distributions:

### Typical English Text
- **Token Savings**: 82.9%
- **Pass Reduction**: 91.0%
- Example: 2,097,152 tokens → 359,306 tokens
- Example: 65,536 passes → 5,880 passes

### Very Short Words (Maximum Benefit)
- **Token Savings**: 90.8%
- **Pass Reduction**: 95.3%
- Example: 524,288 tokens → 48,260 tokens
- Example: 16,384 passes → 767 passes

### Mostly Long Words (Minimal Benefit)
- **Token Savings**: 35.2%
- **Pass Reduction**: 61.5%
- Example: 524,288 tokens → 339,896 tokens
- Example: 16,384 passes → 6,310 passes

## Correctness

The implementation maintains training correctness:
- Each word receives its corresponding latent vector
- Attention masks prevent cross-word attention
- Logits are correctly extracted and placed
- Loss computation remains unchanged

Tests verify:
- Packed results match unpacked baseline (within floating point precision)
- Loss computation is correct
- Edge cases (empty sequences) are handled

## Usage

The packing is automatic and transparent:
- No changes required to training code
- No changes to model configuration
- No changes to data processing
- Works with all existing datasets and configurations

## Testing

Comprehensive tests included:
- `tests/test_packing.py`: Unit tests for packing/unpacking logic
- `tests/test_packing_correctness.py`: Correctness verification
- `examples/demo_packing_efficiency.py`: Efficiency demonstration

Run tests:
```bash
pytest tests/test_packing.py tests/test_packing_correctness.py -v
```

Run efficiency demo:
```bash
python examples/demo_packing_efficiency.py
```

## Future Enhancements

Potential improvements:
1. Make `max_packed_length` configurable via model config
2. Implement smarter packing strategies (e.g., bin packing)
3. Add option to disable packing for debugging
4. Profile and optimize packing overhead
5. Support dynamic packing based on available memory

## References

- PyTorch `pack_padded_sequence`: Similar concept for RNNs
- HuggingFace `trl.pack_dataset`: Used for latent transformer packing
- Original issue: Train bytes decoder with on-the-fly packing
