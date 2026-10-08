#!/bin/bash
# Train, export, serve and evaluate an experiment config, recording timing and quality in benchmarks/tasks.csv
#   benchmarks/run_task.sh <config.yaml> <output_dir> [section.key=value ...]
set -euo pipefail
config=$1; output=$2; shift 2
mkdir -p "$output"
torchrun --nproc_per_node=1 -m welt_training.train "$config" output_dir="$output" "$@" 2>&1 | tee "$output/train.log"
welt-export "$output/checkpoints" --output "$output/export"
port=${PORT:-8199}
if curl -s localhost:$port > /dev/null; then echo "Port $port is in use, set PORT" >&2; exit 1; fi
welt-serve "$output/export" --port $port > "$output/serve.log" 2>&1 &
server=$!
trap 'kill $server; wait $server' EXIT  # Waits for vLLM to release the GPU
until curl -sf localhost:$port/health > /dev/null; do kill -0 $server; sleep 5; done  # Fails if it exited
welt-evaluate "$output/export/welt.yaml" --url http://localhost:$port --output "$output/eval.json"
python benchmarks/summarize.py "$config" "$output" >> benchmarks/tasks.csv
