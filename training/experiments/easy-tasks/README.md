# Easy Tasks

We define easy task to verify the model is able to perform basic computation.

### String Repetition

```bash
python -m training.train training/experiments/easy-tasks/string-repetition.yaml
```

### Letter Count

```bash
python -m training.train training/experiments/easy-tasks/letter-count.yaml
```
```bash
python -m training.evaluate training/experiments/easy-tasks/letter-count-eval.yaml
```
```bash
python -m training.train training/experiments/machine-translation/machine-translation-en-ne.yaml
```