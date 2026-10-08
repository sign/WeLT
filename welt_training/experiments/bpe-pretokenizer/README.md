# BPE Pretokenizer

The pretokenizer that splits text into words can be any HuggingFace tokenizer: here, words are the
[Pythia](https://huggingface.co/EleutherAI/pythia-14m) BPE tokens (`model.pretokenizer`), instead of
[words-segmentation](https://github.com/sign/words-segmentation), on the string repetition task.

```shell
torchrun --nproc_per_node=1 -m welt_training.train welt_training/experiments/bpe-pretokenizer/welt-bpe-14m.yaml
```
