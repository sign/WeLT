# Easy Tasks

We define easy task to verify the model is able to perform basic computation.

### String Repetition

```bash
export WANDB_PROJECT="string-repetition"
python -m training.train training/experiments/easy-tasks/string-repetition.yaml
```

### OCR

```bash
export WANDB_PROJECT="ocr"
python -m training.train training/experiments/easy-tasks/ocr.yaml
```