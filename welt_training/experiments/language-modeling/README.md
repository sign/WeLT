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
