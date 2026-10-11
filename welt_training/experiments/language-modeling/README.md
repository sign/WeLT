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
way. Four models, at four latent transformer sizes with the dimensions of Pythia's 14M, 31M, 70M and 160M
([`models/latent-*.json`](../models)), all without an image encoder:

| Model | Config | Example |
|-------|--------|---------|
| Baseline: Pythia tokens, embedding matrix and LM head | [`climbmix-baseline.yaml`](climbmix-baseline.yaml) | 340 tokens |
| WeLT over Pythia tokens (the same segmentation, bytes encoder and decoder) | [`climbmix-welt-pythia.yaml`](climbmix-welt-pythia.yaml) | 340 tokens |
| WeLT over words | [`climbmix-welt.yaml`](climbmix-welt.yaml) | 256 words |
| WeLT over words, matched: bytes encoder and decoder each the size of the embedding table it replaces (50,304 × hidden) | [`climbmix-welt.yaml`](climbmix-welt.yaml) with [`models/bytes-*.json`](../models) | 256 words |

Examples hold the same text (1.33 Pythia tokens per word). The first 32 documents are held out for validation: few
enough that each model's validation examples are a single micro batch, so all models score exactly the same bytes.

```shell
torchrun --nproc_per_node=1 -m welt_training.train welt_training/experiments/language-modeling/climbmix-welt.yaml \
  model.latent_transformer=welt_training/experiments/models/latent-160m.json
# Matched: also model.bytes_encoder=welt_training/experiments/models/bytes-160m.json model.bytes_decoder=(the same)
torchrun --nproc_per_node=1 -m welt_training.baseline welt_training/experiments/language-modeling/climbmix-baseline.yaml \
  model.transformer=welt_training/experiments/models/latent-160m.json
```

## Results

10000 steps each, on one GB10, with the BOS bytes decoder (before the latent took its place), validation bits per byte at the last step (and the best step):

| Data | WeLT | Baseline | ms / step (WeLT, baseline) |
|------|-----:|---------:|---------------------------:|
| opus-100 en-he | **1.077** (1.061) | 1.093 (1.073) | **244**, 517 |
| wikitext-103 | 1.199 | **1.165** | **197**, 226 |

The baseline's sequence lengths match WeLT's text per example: Pythia's tokenizer splits opus-100 (half Hebrew) into
2.7 tokens per word, but wikitext into 1.1. WeLT wins where subword tokens are short (2.1x faster per step, and
better), and the baseline where they are nearly whole words.
