#!/bin/bash
# Train, export and evaluate an experiment config, recording timing and quality in benchmarks/tasks.csv
#   benchmarks/run_task.sh <config.yaml> <output_dir> [section.key=value ...]
set -euo pipefail
config=$1; output=$2; shift 2
mkdir -p "$output"
torchrun --nproc_per_node=1 -m welt_training.train "$config" output_dir="$output" "$@" 2>&1 | tee "$output/train.log"
welt-export "$output/checkpoints" --output "$output/export"
welt-evaluate "$output/export" --output "$output/eval.json"
python benchmarks/summarize.py "$config" "$output" >> benchmarks/tasks.csv
