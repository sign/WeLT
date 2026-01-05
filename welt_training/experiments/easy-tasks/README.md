# Easy Tasks

We define easy task to verify the model is able to perform basic computation.

### String Repetition

```bash
export WANDB_PROJECT="string-repetition"
welt-train welt_training/experiments/easy-tasks/string-repetition.yaml
```

Or
```shell
mkdir -p output
docker build -t welt . && \
docker run -it --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$(pwd)/welt:/app/welt" \
  -v "$(pwd)/welt_training:/app/welt_training" \
  -v "$(pwd)/output:/app/output" \
  -v /shared/.cache:/root/.cache \
  -v ~/.netrc:/root/.netrc:ro \
  -e WANDB_PROJECT="string-repetition" \
  -e WANDB_NAME="full-run" \
  -e CONFIG="welt_training/experiments/easy-tasks/string-repetition.yaml" \
  welt 
```

### OCR

```bash
export WANDB_PROJECT="ocr"
welt-train welt_training/experiments/easy-tasks/ocr.yaml
```