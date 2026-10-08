# Easy Tasks

Small tasks that verify the model learns basic computation, in minutes on one GPU
(results in [benchmarks](../../../benchmarks#tasks)). Each config's `dataset_text_template` is
`[prefix, completion]`, so `welt-evaluate` measures generation on it.

| Config | Task |
|--------|------|
| [`string-repetition.yaml`](string-repetition.yaml) | Repeat an English sentence (opus-100), pretrained tiny LMs, bytes encoder only |
| [`ocr.yaml`](ocr.yaml) | Write a sentence seen only as rendered word images: a patch transformer image encoder from scratch, no bytes encoder |
| [`letter-count.yaml`](letter-count.yaml) | Count the letters of a word, with the Muon optimizer |

```shell
torchrun --nproc_per_node=1 -m welt_training.train welt_training/experiments/easy-tasks/string-repetition.yaml
```
