# Chat

```shell
mkdir -p output && \
docker build -t welt . && \
docker run -it --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$(pwd)/welt:/app/welt" \
  -v "$(pwd)/welt_training:/app/welt_training" \
  -v "$(pwd)/output:/app/output" \
  -v /shared/.cache:/root/.cache \
  -v ~/.netrc:/root/.netrc:ro \
  -e WANDB_PROJECT="welt" \
  -e CONFIG="welt_training/experiments/chat/single-query.yaml" \
  welt
 ```
