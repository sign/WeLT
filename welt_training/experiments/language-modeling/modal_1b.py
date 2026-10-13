"""ClimbMix at Pythia-1B's dimensions on Modal H100s: the baseline, WeLT matched in parameters, WeLT matched in FLOPs
(10,000 steps each, as the local runs). Outputs and logs in the "welt-runs" volume: /out/<run>.log, /out/<run>/.

    modal run --detach welt_training/experiments/language-modeling/modal_1b.py   # all three, in parallel
    modal volume get welt-runs <run>.log .      # progress (committed every 5 minutes)
"""

import pathlib
import subprocess
import threading

import modal

REPO = str(pathlib.Path(__file__).resolve().parents[3])  # The repository, built with its Dockerfile
M, L = "welt_training/experiments/models", "welt_training/experiments/language-modeling"
RUNS = {
    "cm-1b-baseline": f"welt_training.baseline {L}/climbmix-baseline.yaml model.transformer={M}/latent-1b.json "
    "train.micro_batch_size=16",
    "cm-1b-welt-matched": f"welt_training.train {L}/climbmix-welt.yaml model.latent_transformer={M}/latent-1b.json "
    f"model.bytes_encoder={M}/bytes-1b.json model.bytes_decoder={M}/bytes-1b.json "
    "train.micro_batch_size=8",
    "cm-1b-welt-flops": f"welt_training.train {L}/climbmix-welt.yaml model.latent_transformer={M}/latent-1b.json "
    f"model.bytes_encoder={M}/bytes-flops-1b.json model.bytes_decoder={M}/bytes-flops-1b.json "
    "train.micro_batch_size=8",
}

image = modal.Image.from_dockerfile(f"{REPO}/Dockerfile", context_dir=REPO).env(
    {
        "HF_HOME": "/hf",
        "WANDB_MODE": "disabled",
        "PYTHONPATH": "/app",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)
app = modal.App("welt-1b", image=image)
out = modal.Volume.from_name("welt-runs", create_if_missing=True)
hf = modal.Volume.from_name("welt-hf-cache", create_if_missing=True)


@app.function(gpu="H100", timeout=24 * 3600, cpu=16, memory=128 * 1024, volumes={"/out": out, "/hf": hf})
def train(name: str) -> int:
    module, *args = RUNS[name].split()
    command = [
        "torchrun",
        "--nproc_per_node=1",
        "-m",
        module,
        *args,
        f"output_dir=/out/{name}",
        "logger.wandb_project=null",
        "checkpoint.save_interval=10000",
    ]
    with open(f"/out/{name}.log", "w") as log:
        process = subprocess.Popen(command, cwd="/app", stdout=log, stderr=subprocess.STDOUT)
        stop = threading.Event()

        def commit():  # The log readable from outside while training
            while not stop.wait(300):
                out.commit()

        threading.Thread(target=commit, daemon=True).start()
        code = process.wait()
        stop.set()
    out.commit()
    return code


@app.local_entrypoint()
def main(runs: str = ",".join(RUNS)):
    calls = {name: train.spawn(name) for name in runs.split(",")}
    for name, call in calls.items():
        print(name, call.object_id)
