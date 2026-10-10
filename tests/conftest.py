import json
import os
import socket

import pytest
import torch
from utf8_tokenizer.control import ControlTokens

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


def build_model(config_path: str, image_encoder=True, bytes_encoder=True):
    """A WeLT model with every transformer from config_path, on GPU, without Megatron DDP."""
    from welt.model import WeLTModelProvider

    provider = WeLTModelProvider.from_hf(latent_transformer=config_path, bytes_decoder=config_path,
                                         bytes_encoder=config_path if bytes_encoder else None,
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


SO, SI, PAD = ControlTokens.ShiftOut, ControlTokens.ShiftIn, ControlTokens.Null
ORACLE_TEXTS = ["<en>\x0eHello world, how are you?\x0f<he> שלום עולם",  # Translation-style shift block
                "plain text without a block",
                "\x0ea\x0f\x0eb c\x0f two blocks",
                "<ase>\x0eM518x529S14c20481x471S27106503x489\x0f<en> hello"]  # SignWriting


def oracle_labels(words: list[str], seq_lengths: list[int]) -> list[str | None]:
    """Each word's label, from the words alone: the next word of its sequence, "" after its last word, and None for
    PAD words and for the words of a shift block before its ShiftIn (they see each other)."""
    labels, start = [], 0
    for length in seq_lengths:
        sequence, inside = words[start:start + length], False
        for i, word in enumerate(sequence):
            inside = (inside or word == SO) and word != SI
            labels.append(None if word == PAD or inside else (sequence + [""])[i + 1])
        start += length
    return labels


def oracle_mask(words: list[str], seq_lengths: list[int]) -> torch.Tensor:
    """(L, L) the words each word sees in the latent transformer, from the words alone: itself and the earlier words
    of its sequence, and every word of its shift block (ShiftOut to ShiftIn)."""
    allowed = torch.zeros(len(words), len(words), dtype=torch.bool)
    start = 0
    for length in seq_lengths:
        shift_out = None
        for i in range(start, start + length):
            allowed[i, start:i + 1] = True
            if words[i] == SO:
                shift_out = i
            elif words[i] == SI and shift_out is not None:
                allowed[shift_out:i + 1, shift_out:i + 1] = True
                shift_out = None
        start += length
    return allowed
