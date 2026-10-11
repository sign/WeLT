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

## Results

10000 steps each, on one GB10, with the BOS bytes decoder (before the latent took its place), validation bits per byte at the last step (and the best step):

| Data | WeLT | Baseline | ms / step (WeLT, baseline) |
|------|-----:|---------:|---------------------------:|
| opus-100 en-he | **1.077** (1.061) | 1.093 (1.073) | **244**, 517 |
| wikitext-103 | 1.199 | **1.165** | **197**, 226 |

The baseline's sequence lengths match WeLT's text per example: Pythia's tokenizer splits opus-100 (half Hebrew) into
2.7 tokens per word, but wikitext into 1.1. WeLT wins where subword tokens are short (2.1x faster per step, and
better), and the baseline where they are nearly whole words.
