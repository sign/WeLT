# Machine Translation

English to Hebrew on [opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) (en-he), with the
latent transformer trained from scratch, comparing WeLT to a subword causal LM of the same size.

| Config | Model |
|--------|-------|
| [`machine-translation.yaml`](machine-translation.yaml) | WeLT: image + bytes encoders, a 6-layer 512-wide latent transformer ([`latent-70m.json`](../models/latent-70m.json)), 128 words per example |
| [`baseline.yaml`](baseline.yaml) | Causal LM: the same transformer over Pythia BPE tokens, 320 tokens per example (about as much text) |
| [`machine-translation-signed-spoken.yaml`](machine-translation-signed-spoken.yaml) | WeLT, SignWriting to spoken language ([signbank-plus](https://huggingface.co/datasets/sign/signbank-plus)) |

```shell
torchrun --nproc_per_node=1 -m welt_training.train welt_training/experiments/machine-translation/machine-translation.yaml
torchrun --nproc_per_node=1 -m welt_training.baseline welt_training/experiments/machine-translation/baseline.yaml
```

Both log `bits per byte` to the same W&B project (`welt-machine-translation`): the loss of every prediction
except the end of a document (WeLT's word ends included), per UTF-8 byte of text, so their validation values
compare directly. Generation quality of the WeLT model (chrF, exact match) comes from
`welt-export` and `welt-evaluate`, see the [README](../../../README.md#export--generate).
