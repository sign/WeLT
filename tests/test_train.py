"""End-to-end: train a tiny WeLT with Megatron-Bridge, export it, and generate with vLLM."""
import os
import subprocess
import sys

import pytest
import torch
import yaml
from datasets import Dataset

from tests.conftest import free_port

pytest.importorskip("megatron.bridge", reason="Requires the NeMo container")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron and vLLM require a GPU")


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
        # A large latent initialization makes its outputs sensitive to the bidirectional shift blocks (in bf16)
        "model": {"bytes_encoder": tiny_config, "image_encoder": tiny_config,
                  "latent_transformer": tiny_config, "bytes_decoder": tiny_config, "init_method_std": 0.3},
        "data": {"train_file": str(data), "seq_length": 32, "max_word_length": 16, "num_workers": 1},
        "train": {"train_iters": 20, "micro_batch_size": 4, "global_batch_size": 4},
        "validation": {"eval_interval": 10, "eval_iters": 1},
        "checkpoint": {"save_interval": 20},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    run("torchrun", "--nproc_per_node=1", f"--master_port={free_port()}", "-m", "welt_training.train",
        str(config_path))
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

    generator = WeLTGenerator(str(trained / "export"), kv_cache_gib=0.5)
    outputs = generator.generate(["number 4 is", "number \x0E7\x0F is od"], max_generated_words=3)
    assert len(outputs) == 2
    assert all(isinstance(output, str) for output in outputs)

    # Served by welt-serve's app, through its HTTP client
    import threading

    from werkzeug.serving import make_server

    from welt.server import create_app, generate

    server = make_server("localhost", 0, create_app(generator), threaded=True)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        served = generate(f"http://localhost:{server.port}", ["number 4 is", "number \x0E7\x0F is od"],
                          max_generated_words=3)
    finally:
        server.shutdown()
        thread.join()
    assert served["outputs"] == outputs  # Greedy

    # A word cache smaller than a sequence: words still needed are not evicted mid-step
    from cachetools import LRUCache

    generator.word_embeddings = LRUCache(maxsize=2)
    assert generator.generate(["number 4 is", "number \x0E7\x0F is od"], max_generated_words=3) == outputs


    # Sampled at a high temperature from random latents, words are still valid UTF-8 (ending at a character boundary)
    from vllm import SamplingParams

    latents = torch.randn(64, generator.weights["decoder_norm.weight"].size(0), device="cuda").bfloat16()
    sampling = SamplingParams(temperature=2.0, seed=0, max_tokens=generator.processor.max_word_length - 1,
                              stop_token_ids=[generator.tokenizer.eos_token_id], detokenize=False)
    words = generator._next_words(latents, [b""] * len(latents), sampling)
    assert any(max(word, default=0) >= 0x80 for word in words)  # Multi-byte characters were sampled
    for word in words:
        word.decode("utf-8")  # Raises on invalid UTF-8


def test_vllm_matches_megatron(trained, megatron, monkeypatch):
    """The exported model, served by vLLM, computes what the Megatron model computes."""
    from vllm import SamplingParams

    import welt.inference
    from welt.export import load_model

    model = load_model(str(trained / "run" / "checkpoints" / "iter_0000020")).cuda()

    generator = welt.inference.WeLTGenerator(str(trained / "export"), kv_cache_gib=0.5)
    # With a long bidirectional shift block, for a measurable effect on the last word's latent
    text = "number \x0e7 is the number after six and before eight, which is even\x0f odd"
    words = generator.processor.pretokenize(text)
    batch = {k: v.cuda() for k, v in generator.processor([text]).items()}
    with torch.no_grad():
        word_embeds = model.encode_words(batch["input_ids"], batch["input_attention_mask"],
                                         batch["input_patches"], batch["input_patches_shape"])
        latents = model.latent(word_embeds, batch["attention_mask"])[0]
        causal = torch.ones_like(batch["attention_mask"]).tril()  # Without the bidirectional shift block
        causal_latents = model.latent(word_embeds, causal)[0]
        first_byte_logits, _ = model.decode(latents[-1:], batch["input_ids"][0, :1, :1], torch.ones(1, 1).cuda())

    def error(a, b):
        return ((a.float().cuda() - b.float().cuda()).norm() / b.float().cuda().norm()).item()

    embeddings = generator._encode_words(words)
    assert error(torch.stack([embeddings[w] for w in words]), word_embeds[0]) < 0.02
    latent = generator._latents([words])[0]
    assert error(latent, latents[-1]) < 0.02
    assert error(generator._latents([words])[0], latent) < 1e-3  # Again, from the prefix cache
    # A prompt with the same words before, and the same ranges, but other words within the shift block, does not
    # reuse the cached keys and values of the block's first words (which attend to the rest of the block)
    other = generator.processor.pretokenize(text.replace("which is even", "which is odd"))
    generator._latents([other])
    assert error(generator._latents([words])[0], latent) < 1e-3

    # Negative control: without the bidirectional shift block, vLLM matches Megatron's causal latent instead
    monkeypatch.setattr(welt.inference, "get_shift_blocks", lambda _: [])
    vllm_causal = generator._latents([words])[0]
    assert error(vllm_causal, causal_latents[-1]) < 0.02
    # Each is closer to its own Megatron counterpart (the shift block changes the latent by a few percent)
    assert error(latent, latents[-1]) < error(latent, causal_latents[-1])
    assert error(vllm_causal, causal_latents[-1]) < error(vllm_causal, latents[-1])

    # The decoder predicts the same first byte (index 1, after BOS)
    greedy = SamplingParams(temperature=0, max_tokens=1, detokenize=False)
    next_word = generator._next_words(latent[None], [b""], greedy)[0]
    first_byte = first_byte_logits[1].argmax().item()
    assert next_word[:1] == (b"" if first_byte == generator.tokenizer.eos_token_id else bytes([first_byte]))
