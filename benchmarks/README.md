# Benchmarks

All numbers are on a single NVIDIA GB10 (DGX Spark: unified LPDDR5x memory, ~95 TFLOP/s measured bf16 matmul peak).
The GB10 is memory-bandwidth bound, so on e.g. H100s the relative gains of each optimization may differ.

## Time per training step

![Time per training step](step_time.png)

[`step_time.csv`](step_time.csv) lists every iteration, including the attempts that were not kept.
Regenerate the chart with `python benchmarks/plot.py`.

### Bench config: HuggingFace Trainer vs. Megatron-Bridge

Same model sizes and data in both stacks:
`Helsinki-NLP/opus-100` en-he, packed to 128 words per example, 32 bytes per word, micro batch 32, bf16,
a 10-layer 256-wide bytes encoder, a 6-layer 512-wide latent transformer, and a `sbintuitions/tiny-lm`-shaped
bytes decoder, from scratch. Time per step is the steady-state mean (after warmup).

| Stack | ms / step | Speedup | Model TFLOP/s |
|-------|----------:|--------:|--------------:|
| HF Trainer (`huggingface-transformers` tag, [`hf_baseline/`](hf_baseline)) | 1227 | 1.0x | 2.3 |
| Megatron-Bridge ([`welt-bench.yaml`](welt-bench.yaml)) | 127 | **9.7x** | 21.8 |

Both learn comparably: after 300 steps the per-byte loss is 0.92 for HF (mean of steps 251-300, linear LR decay)
and 0.86 for Megatron-Bridge (mean of steps 291-300, cosine LR decay).
The HF baseline encodes bytes with a BERT encoder, Megatron-Bridge with a Llama-like bidirectional encoder of the same size.

Reproduce:

```shell
# Megatron-Bridge
torchrun --nproc_per_node=1 -m welt_training.train benchmarks/welt-bench.yaml

# HF Trainer: bench_hf.py adds a timing callback to the old welt_training.trainer, so it runs in a checkout of
# the huggingface-transformers tag (set up per that checkout's README), with hf_baseline/ copied in
git worktree add ../WeLT-hf huggingface-transformers
mkdir -p ../WeLT-hf/benchmarks && cp -r benchmarks/hf_baseline ../WeLT-hf/benchmarks/
cd ../WeLT-hf && python benchmarks/hf_baseline/bench_hf.py benchmarks/hf_baseline/hf-bench.yaml
```

### What made it faster

1. **Megatron-Bridge** (2.0x): Transformer Engine layers with fused RoPE / SwiGLU / RMSNorm / cross entropy,
   distributed fused Adam, no HF Trainer overheads.
2. **Packed bytes** (2.4x): the bytes of all words are packed (THD) for the encoders and the decoder, instead of
   padding every word to the longest one in the batch (words average ~7 bytes; padded to 24-32).
3. **FlexAttention** (1.5x on MT): varlen flash attention is slow for tens of thousands of few-token sequences
   (its backward pads each sequence to a 128 tile). FlexAttention with a block mask built in O(tokens) from
   `cu_seqlens` is ~2x faster per layer, and also replaces Transformer Engine's unfused attention for the latent
   transformer's arbitrary (packed + bidirectional shift block) mask.
4. **Encode unique words** (1.3x on MT): the word encoders' output only depends on the word, and only ~31% of the
   words in a batch are distinct, so each distinct word is encoded once and its embedding is shared (exact).

Not kept: padded SDPA / matmul attention for the short sequences (memory-bound on GB10), FP8 (MXFP8 is unsupported on
SM12x; tensorwise FP8 needs token counts padded to multiples of 8, and matmuls are a minority of the step time here).

### Utilization

[`flops.py`](flops.py) counts the FLOPs of each transformer on its real (packed) tokens: "model" FLOPs encode every word
occurrence, "hardware" FLOPs encode each distinct word once.

| Config | ms / step | Model TFLOPs / step | Model TFLOP/s | Hardware TFLOP/s |
|--------|----------:|-------------------:|--------------:|-----------------:|
| Bench | 127 | 2.76 | 21.8 | - |
| Machine translation | 172 | 3.06 | 17.8 | 14.5 |

The remaining time is dominated by memory-bound kernels (SwiGLU, RoPE, norms) of the narrow (128-512 wide)
transformers, which GB10's bandwidth limits. Larger micro batches help a little (batch 128: 10% more samples/s).

### Multiple GPUs

[`parity.sh`](parity.sh) trains the machine translation config for 200 steps on the same global batches
(64 examples, micro batch 32, modality dropout off), on 2 H100s (on Modal, 16 CPUs):

| Parallelism | ms / step | Val. loss after 200 steps |
|-------------|----------:|--------------------------:|
| 1 GPU (2 micro batches) | 189 | 0.933 |
| Data parallel (DP=2) | 101 | 0.933 |
| Tensor + sequence parallel (TP=2) | 254 | 0.938 |

Data parallelism scales 1.9x. Tensor parallelism is slower for these narrow (128-512 wide) transformers: use it only
for models that do not fit on one GPU. Its checkpoints export (resharded) to the same vLLM format.

With micro batch 64, a step takes 103 ms on one H100 vs. 167 ms on the GB10: only 1.6x faster. A profile shows the
H100 is host bound: its kernels take ~30 ms of a ~125 ms step, the rest is the CPU launching ~1500 small kernels
(Transformer Engine modules cost 0.2-0.4 ms of CPU each for these narrow layers) and building the packed attention
masks. So on H100s, use larger micro batches (128: 1.7x the samples per second, 256: ~2.9x).
Follow-up: computing the packing metadata (lengths, unique words, block masks) in the dataloader and padding packed
shapes to fixed buckets would remove the host synchronizations and allow CUDA graphs.

## Tasks

[`run_task.sh`](run_task.sh) trains a config, exports it, and evaluates generation with vLLM
([`welt_training/evaluate.py`](../welt_training/evaluate.py): exact match and chrF of completions generated from the
validation prefixes), recording a row of [`tasks.csv`](tasks.csv).

| Task | Steps | ms / step | Train time | Val. bits / byte | Val. word acc. | Gen. exact match | Gen. chrF | Gen. words / s |
|------|------:|----------:|-----------:|-----------------:|---------------:|-----------------:|----------:|---------------:|
| **string-repetition**: repeat an English sentence, pretrained tiny LMs | 1500 | 50 | 1.3 min | 0.017 | 99.2% | 93.4% | 98.5 | 776 |
| **ocr**: write a sentence seen only as rendered word images | 3000 | 62 | 3.1 min | 0.038 | 97.7% | 80.5% | 95.4 | 867 |
| **letter-count**: count the letters of a word (Muon) | 3000 | 67 | 3.4 min | 0.0001 | 100% | 99.2% | 99.7 | 1336 |
| **machine-translation**: English to Hebrew, from scratch, image + bytes encoders | 10000 | 172 | 29 min | 0.626 | 64.8% | 5.9% | 41.0 | 441 |

Generation is greedy, on 256 validation examples, with vLLM (batched over all examples).

The causal LM [baseline](../welt_training/experiments/machine-translation/baseline.yaml) (the same 6-layer 512-wide
transformer over Pythia BPE tokens, same data and batch size, 10000 steps) reaches **1.016** validation bits per byte
at 729 ms / step, vs. **0.626** for WeLT at 172 ms / step.
