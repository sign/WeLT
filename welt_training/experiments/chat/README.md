# Chat

Fine-tunes the pretrained [PleIAs/Monad](https://huggingface.co/PleIAs/Monad) as the latent transformer on
single-turn chat ([PleIAs/SYNTH](https://huggingface.co/datasets/PleIAs/SYNTH), streamed: 1M training examples),
with query, reasoning and answer delimited by control characters, as in the "Back to bytes" paper.
Uses the Muon optimizer.

```shell
torchrun --nproc_per_node=1 -m welt_training.train welt_training/experiments/chat/single-query.yaml
```
