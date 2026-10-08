"""End-to-end: train a tiny WeLT with Megatron-Bridge."""
import os
import subprocess

import pytest
import torch
import yaml
from datasets import Dataset

from tests.conftest import free_port

pytest.importorskip("megatron.bridge", reason="Requires the NeMo container")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron requires a GPU")


def run(*args):
    # A port of its own: the in-process Megatron test fixture holds MASTER_PORT
    env = {**os.environ, "PYTHONPATH": os.getcwd(), "MASTER_PORT": str(free_port())}
    subprocess.run([*args], check=True, env=env)


@pytest.fixture(scope="module")
def trained(tmp_path_factory, tiny_config):
    root = tmp_path_factory.mktemp("e2e")
    data = root / "data.json"
    sentences = [f"number {i} is {'even' if i % 2 == 0 else 'odd'}" for i in range(200)]
    Dataset.from_dict({"text": sentences}).to_json(data)
    config = {
        "output_dir": str(root / "run"),
        "model": {"bytes_encoder": tiny_config, "image_encoder": tiny_config,
                  "latent_transformer": tiny_config, "bytes_decoder": tiny_config},
        "data": {"train_file": str(data), "seq_length": 32, "max_word_length": 16, "num_workers": 1},
        "train": {"train_iters": 20, "micro_batch_size": 4, "global_batch_size": 4},
        "validation": {"eval_interval": 10, "eval_iters": 1},
        "checkpoint": {"save_interval": 20},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run("torchrun", "--nproc_per_node=1", f"--master_port={free_port()}", "-m", "welt_training.train",
        str(config_path))
    return root


def test_training_saves_checkpoint(trained):
    checkpoints = trained / "run" / "checkpoints"
    assert (checkpoints / "latest_checkpointed_iteration.txt").read_text().strip() == "20"

