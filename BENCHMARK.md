# Generation Benchmark Results

Benchmark comparing generation performance before and after the prefill/decode refactor.

**Test Environment:**
- Model: Tiny test model (~12.5M parameters)
- Device: CPU (Apple Silicon)
- Max generated words: 5
- Runs: 20 (after 5 warmup runs)

## Before (No KV-Cache)

The old implementation re-computed the full attention at each generation step without utilizing KV-cache.

| Test Case | Batch Size | Avg (ms) | Min (ms) | Max (ms) |
|-----------|------------|----------|----------|----------|
| single_short | 1 | 73.50 | 70.20 | 76.71 |
| single_medium | 1 | 80.13 | 75.58 | 92.59 |
| batch_4_short | 4 | 120.71 | 112.50 | 132.56 |
| batch_4_medium | 4 | 146.76 | 123.11 | 208.89 |
| batch_3_mixed | 3 | 125.32 | 113.26 | 157.48 |

## After (With KV-Cache)

The new implementation uses prefill/decode pattern with KV-cache for incremental generation.

| Test Case | Batch Size | Avg (ms) | Min (ms) | Max (ms) |
|-----------|------------|----------|----------|----------|
| single_short | 1 | 68.95 | 64.97 | 86.54 |
| single_medium | 1 | 71.62 | 69.28 | 75.12 |
| batch_4_short | 4 | 112.72 | 102.11 | 140.22 |
| batch_4_medium | 4 | 123.30 | 112.31 | 165.77 |
| batch_3_mixed | 3 | 113.39 | 109.04 | 122.13 |

## Summary

| Test Case | Speedup |
|-----------|---------|
| single_short | 1.07x |
| single_medium | 1.12x |
| batch_4_short | 1.07x |
| batch_4_medium | 1.19x |
| batch_3_mixed | 1.11x |

**Average speedup: ~1.11x (11% faster)**

The KV-cache implementation provides modest but consistent speedups. The benefit is expected to be more pronounced with:
- Longer sequences (more tokens to cache)
- More generation steps
- GPU execution (where memory bandwidth is more critical)
