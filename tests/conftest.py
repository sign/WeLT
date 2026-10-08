import json
import os
import socket

import pytest
import torch

TINY_LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "max_position_embeddings": 256,
    "vocab_size": 256,
    "tie_word_embeddings": False,
}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def build_model(config_path: str, image_encoder=True):
    """A WeLT model with every transformer from config_path, on GPU, without Megatron DDP."""
    from welt.model import WeLTModelProvider

    provider = WeLTModelProvider.from_hf(latent_transformer=config_path, bytes_decoder=config_path,
                                         bytes_encoder=config_path,
                                         image_encoder=config_path if image_encoder else None)
    provider.bf16 = True
    provider.seq_length = 64
    provider.gradient_accumulation_fusion = False  # Requires Megatron DDP
    provider.finalize()
    return provider.provide().cuda().bfloat16().eval()


@pytest.fixture(scope="session")
def megatron():
    """Single-process Megatron parallel state, on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("Megatron requires a GPU")
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", str(free_port()))
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(42)
    yield
    parallel_state.destroy_model_parallel()


@pytest.fixture(scope="session")
def tiny_config(tmp_path_factory):
    path = tmp_path_factory.mktemp("configs") / "tiny-llama.json"
    path.write_text(json.dumps(TINY_LLAMA))
    return str(path)
