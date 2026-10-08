# Easy Tasks

Small tasks that verify the model learns basic computation, in minutes on one GPU
(results in [benchmarks](../../../benchmarks#tasks)). Each is evaluated with `welt-evaluate`.

| Config | Task |
|--------|------|
| [`string-repetition.yaml`](string-repetition.yaml) | Repeat an English sentence (opus-100), pretrained tiny LMs, bytes encoder only |
| [`ocr.yaml`](ocr.yaml) | Write a sentence seen only as rendered word images: a patch transformer image encoder from scratch, no bytes encoder |
| [`letter-count.yaml`](letter-count.yaml) | Count the letters of a word, with the Muon optimizer |
