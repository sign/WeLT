# Language Modeling

Plain causal language modeling, without shift blocks (every byte is predicted and scored), WeLT against the subword
causal LM baseline with the same latent transformer, data, batches of about the same text, steps and optimizer.

| Data | WeLT | Baseline |
|------|------|----------|
| [opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) en-he sentence pairs, as text | [`opus-welt.yaml`](opus-welt.yaml): 128 words | [`opus-baseline.yaml`](opus-baseline.yaml): 352 Pythia tokens |
| [wikitext-103](https://huggingface.co/datasets/Salesforce/wikitext) (English Wikipedia) | [`wikitext-welt.yaml`](wikitext-welt.yaml): 256 words | [`wikitext-baseline.yaml`](wikitext-baseline.yaml): 288 Pythia tokens |

```shell
torchrun --nproc_per_node=1 -m welt_training.train welt_training/experiments/language-modeling/opus-welt.yaml
torchrun --nproc_per_node=1 -m welt_training.baseline welt_training/experiments/language-modeling/opus-baseline.yaml
```

Both log validation `bits per byte` (see the [README](../../../README.md#train)), which compare directly.

## Scaling: WeLT vs. the baseline on ClimbMix

[nvidia/Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) is web text stored as GPT-2 token
ids: `data.dataset_tokenizer: gpt2` decodes each streamed document back to text, which each model then segments its own
way. Five models, with latent transformers of the dimensions of Pythia's 14M, 31M, 70M, 160M, 410M and 1B
([`models/latent-*.json`](../models): 1.6M, 6.3M, 25M, 113M, 403M and 1.07B parameters without embeddings), all without
an image encoder:

| Model | Config | Example |
|-------|--------|---------|
| Baseline: Pythia tokens, embedding matrix and LM head | [`climbmix-baseline.yaml`](climbmix-baseline.yaml) | 340 tokens |
| WeLT over Pythia tokens (the same segmentation, bytes encoder and decoder) | [`climbmix-welt-pythia.yaml`](climbmix-welt-pythia.yaml) | 340 tokens |
| WeLT over words | [`climbmix-welt.yaml`](climbmix-welt.yaml) | 256 words |
| WeLT, matched params: bytes encoder and decoder each the size of the embedding table it replaces (50,304 × hidden) | `climbmix-welt.yaml` with [`models/bytes-<size>.json`](../models) | 256 words |
| WeLT, matched FLOPs: bytes encoder and decoder sized so that training FLOPs per step match the baseline's (within 5%) | `climbmix-welt.yaml` with [`models/bytes-flops-<size>.json`](../models) | 256 words |

Examples hold the same text (1.33 Pythia tokens per word). The first 32 documents are held out for validation: few
enough that each model's validation examples are a single micro batch, so all models score exactly the same bytes.
The stream is shuffled within a 10,000 document buffer, and ClimbMix stores documents by topic cluster, so long runs
drift across topics (alike for all models: comparisons at equal steps are fair).

```shell
torchrun --nproc_per_node=1 -m welt_training.train welt_training/experiments/language-modeling/climbmix-welt.yaml \
  model.latent_transformer=welt_training/experiments/models/latent-160m.json
# Matched params: also model.bytes_encoder=welt_training/experiments/models/bytes-160m.json model.bytes_decoder=(the same)
# Matched FLOPs: bytes-flops-160m.json instead
torchrun --nproc_per_node=1 -m welt_training.baseline welt_training/experiments/language-modeling/climbmix-baseline.yaml \
  model.transformer=welt_training/experiments/models/latent-160m.json
```

Steps 5643 and 5651 hold outlier batches (a document of long unique strings: 2.3x the median decoder bytes) whose
activations take 2-4x the usual memory: with large byte models, lower `train.micro_batch_size` (gradient accumulation,
the same training). The 1B models train on single H100s with [`modal_1b.py`](modal_1b.py) (Modal).
[`benchmarks/flops.py`](../../../benchmarks/flops.py) counts each run's FLOPs per step.

### Results

Validation bits per byte after 10,000 steps (the best per latent transformer in bold):

| Latent | Baseline | WeLT, Pythia tokens | WeLT, words | WeLT, matched params | WeLT, matched FLOPs |
|-------:|---------:|----:|----:|----:|----:|
| 1.6M | 1.238 | 1.203 | 1.194 | **1.165** | 1.242 |
| 6.3M | 1.145 | 1.146 | 1.145 | **1.097** | 1.156 |
| 25M | 1.077 | 1.088 | 1.098 | **1.043** | 1.083 |
| 113M | 1.038 | 1.057 | 1.075 | **1.020** | 1.040 |
| 403M | 1.009 | 1.024 | 1.049 | **1.007** | 1.015 |
| 1.07B | **1.024** |  |  | 1.038 | 1.036 |

- Quality follows scale (FLOPs or parameters, 92% of pairs each) far more than architecture.
- At equal parameters (matched params), WeLT is ahead up to 403M, by a shrinking margin (0.072 to 0.003), and 0.014
  behind at 1B, where every model is worse than at 403M (the learning rate and text of the smaller models).
- At equal FLOPs (matched FLOPs), WeLT is slightly behind at every size (0.002-0.013): the baseline's lookups are free.
- WeLT over Pythia tokens beats WeLT over words from 25M on (0.010-0.025), also at equal FLOPs.
- Trained 30,000 steps instead of 10,000, WeLT overtakes the baseline (25M: 1.058 vs. 1.075; 113M: 1.051 vs. 1.060).
- With the 25M latent transformer, bigger byte models keep helping (1.145 with 1.1M parameters each, 1.043 with 27M,
  1.034 with 54M), and the decoder carries most of it: a 27M decoder alone gains 0.048, a 27M encoder alone 0.011.

On a GB10, the baseline's steps are faster except with the 403M latent transformer: WeLT's small byte transformers are
bound by memory bandwidth.

## Earlier results


10000 steps each, on one GB10, with the BOS bytes decoder (before the latent took its place) and the baseline's
unpadded vocabulary with Megatron's cross entropy (padded and fused, its steps now take half the time), validation bits
per byte at the last step (and the best step):

| Data | WeLT | Baseline | ms / step (WeLT, baseline) |
|------|-----:|---------:|---------------------------:|
| opus-100 en-he | **1.077** (1.061) | 1.093 (1.073) | **244**, 517 |
| wikitext-103 | 1.199 | **1.165** | **197**, 226 |

The baseline's sequence lengths match WeLT's text per example: Pythia's tokenizer splits opus-100 (half Hebrew) into
2.7 tokens per word, but wikitext into 1.1. WeLT wins where subword tokens are short (better, and then 2.1x faster per step), and the baseline where they are nearly whole words.
