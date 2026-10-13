# Pretraining

[`pile-pretrain-70m-no-image.yaml`](pile-pretrain-70m-no-image.yaml) pretrains a 70m latent transformer, bytes encoder
only, on the Pile (`monology/pile-uncopyrighted`, streamed), for Pythia's 143k steps. [`test-run.yaml`](test-run.yaml)
runs it for 100 small steps, as a check of the pipeline.

```shell
torchrun --nproc_per_node=1 -m welt_training.train welt_training/experiments/pretrain/test-run.yaml
```

For WeLT against a subword baseline at several sizes, see [language modeling](../language-modeling).
