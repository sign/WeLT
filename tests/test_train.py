"""End-to-end: train a tiny WeLT with Megatron-Bridge, export it, and generate with vLLM."""
import os
import subprocess
import sys

import pytest
import torch
import yaml
from datasets import Dataset

pytest.importorskip("megatron.bridge", reason="Requires the NeMo container")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron and vLLM require a GPU")


def run(*args):
    subprocess.run([*args], check=True, env={**os.environ, "PYTHONPATH": os.getcwd()})


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

    run("torchrun", "--nproc_per_node=1", "--master_port=29613", "-m", "welt_training.train", str(config_path))
    run(sys.executable, "-m", "welt.export", str(root / "run" / "checkpoints"), "--output", str(root / "export"))
    return root


def test_training_saves_checkpoint(trained):
    checkpoints = trained / "run" / "checkpoints"
    assert (checkpoints / "latest_checkpointed_iteration.txt").read_text().strip() == "20"


def test_export_writes_vllm_models(trained):
    export = trained / "export"
    for name in ["bytes_encoder", "image_encoder", "latent_transformer", "bytes_decoder"]:
        assert (export / name / "config.json").exists()
        assert (export / name / "model.safetensors").exists()
    assert (export / "welt.safetensors").exists()
    assert (export / "processor" / "processor_config.json").exists()


def test_generate_with_vllm(trained):
    from welt.inference import WeLTGenerator

    generator = WeLTGenerator(str(trained / "export"), gpu_memory_utilization=0.05)
    outputs = generator.generate(["number 4 is", "number \x0E7\x0F is od"], max_generated_words=3)
    assert len(outputs) == 2
    assert all(isinstance(output, str) for output in outputs)
