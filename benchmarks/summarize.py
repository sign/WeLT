"""Summarize a benchmarks/run_task.sh run as a CSV row: config, ms/step, validation metrics, generation metrics."""
import json
import re
import sys
from pathlib import Path

config, output = sys.argv[1], Path(sys.argv[2])
log = (output / "train.log").read_text()
step_times = [float(t) for t in re.findall(r"elapsed time per iteration \(ms\): ([\d.]+)", log)][1:]  # skip warmup
iterations = re.findall(r"iteration\s+(\d+)/", log)[-1]
validation = re.findall(r"validation loss at iteration \d+ on validation set \| (.*)", log)[-1]
metrics = {k.strip(): v for k, v in re.findall(r"([a-z ]+) value: ([\d.E+-]+)", validation)}
evaluation = json.loads((output / "eval.json").read_text())
print(",".join([
    Path(config).stem, iterations, f"{sum(step_times) / len(step_times):.1f}",
    *(f"{float(metrics[name]):.4f}" for name in ("lm loss", "bits per byte", "byte accuracy", "word accuracy")),
    f"{evaluation['exact_match']:.4f}", f"{evaluation['chrf']:.2f}", f"{evaluation['generated_words_per_second']:.1f}",
]))
