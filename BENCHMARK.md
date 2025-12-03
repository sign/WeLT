# Generation Benchmark Results

Benchmark comparing generation performance across different implementations.

**Test Environment:**
- Model: Tiny test model (~12.5M parameters)
- Device: CPU (Apple Silicon)
- Max generated words: 5
- Runs: 20 (after 5 warmup runs)

## V1: Before (No KV-Cache)

The old implementation re-computed the full attention at each generation step without utilizing KV-cache.

| Test Case | Batch Size | Avg (ms) | Min (ms) | Max (ms) |
|-----------|------------|----------|----------|----------|
| single_short | 1 | 73.50 | 70.20 | 76.71 |
| single_medium | 1 | 80.13 | 75.58 | 92.59 |
| batch_4_short | 4 | 120.71 | 112.50 | 132.56 |
| batch_4_medium | 4 | 146.76 | 123.11 | 208.89 |
| batch_3_mixed | 3 | 125.32 | 113.26 | 157.48 |

## V2: With KV-Cache

Prefill/decode pattern with KV-cache for incremental generation.

| Test Case | Batch Size | Avg (ms) | Min (ms) | Max (ms) |
|-----------|------------|----------|----------|----------|
| single_short | 1 | 68.95 | 64.97 | 86.54 |
| single_medium | 1 | 71.62 | 69.28 | 75.12 |
| batch_4_short | 4 | 112.72 | 102.11 | 140.22 |
| batch_4_medium | 4 | 123.30 | 112.31 | 165.77 |
| batch_3_mixed | 3 | 113.39 | 109.04 | 122.13 |

## V3: Optimized (Current)

Additional optimizations:
- Vectorized attention mask construction (no Python loops)
- Reuse logits processor and stopping criteria across iterations
- Simplified word encoding (no growing tensor)
- Cache BOS embedding (created once, reused every iteration)
- Use `torch.inference_mode()` instead of `torch.no_grad()`
- Pre-allocate decode attention mask (slice instead of concatenate each iteration)

| Test Case | Batch Size | Avg (ms) | Min (ms) | Max (ms) |
|-----------|------------|----------|----------|----------|
| single_short | 1 | 62.02 | 57.72 | 83.71 |
| single_medium | 1 | 65.91 | 61.57 | 81.09 |
| batch_4_short | 4 | 90.18 | 87.32 | 98.47 |
| batch_4_medium | 4 | 104.35 | 98.37 | 121.39 |
| batch_3_mixed | 3 | 103.84 | 96.61 | 125.04 |

## Summary

| Test Case | V1 → V3 Speedup |
|-----------|-----------------|
| single_short | 1.19x |
| single_medium | 1.22x |
| batch_4_short | 1.34x |
| batch_4_medium | 1.41x |
| batch_3_mixed | 1.21x |

**Total speedup from V1 to V3: ~1.27x (27% faster)**

The biggest gains are on batch_4_medium (1.41x). Benefits expected to be more pronounced with:
- Longer sequences (more tokens to cache)
- More generation steps
- GPU execution (where memory bandwidth is more critical)
