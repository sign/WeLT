"""
Export a WeLT Megatron checkpoint for inference with vLLM.

    welt-export <output_dir>/checkpoints[/iter_N] --output <export_dir>

Writes a HuggingFace model directory per transformer, in its original architecture (e.g. Llama or Qwen3),
which vLLM serves:
- bytes_encoder/, image_encoder/: bidirectional (is_causal: false), served with CLS pooling
- latent_transformer/: causal, served with last token pooling, bidirectional shift blocks (is_mm_prefix_lm)
- bytes_decoder/: causal LM over bytes
and `welt.safetensors` with the remaining (small) layers, plus the processor.
"""
import argparse
import os
import shutil

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.training.model_load_save import load_megatron_model
from safetensors.torch import save_file

from welt.model import WeLTModel, hf_config

TRANSFORMERS = {  # Config key in welt.yaml: the GPTModel in WeLTModel
    "bytes_encoder": "bytes_encoder.transformer",
    "image_encoder": "image_encoder.transformer",
    "latent_transformer": "latent_transformer",
    "bytes_decoder": "bytes_decoder",
}


def load_model(checkpoint: str) -> WeLTModel:
    """A WeLTModel from a Megatron iter_* checkpoint directory, in eval mode."""
    model = load_megatron_model(checkpoint, skip_temp_dist_context=True)
    model = model[0] if isinstance(model, list) else model
    while hasattr(model, "module"):  # DDP / Float16Module wrappers
        model = model.module
    return model.eval()


def _save_placeholder_tokenizer(path: str, num_tokens: int):
    """vLLM requires a tokenizer, even though WeLT passes token ids or embeddings and never detokenizes."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    vocab = {f"<0x{i:02X}>": i for i in range(num_tokens)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<0x00>"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<0x02>", eos_token="<0x03>",
                            pad_token="<0x00>").save_pretrained(path)


def _save_transformer(model: WeLTModel, name: str, model_config: dict, path: str, trust_remote_code=False):
    gpt = model.get_submodule(TRANSFORMERS[name])
    config = hf_config(model_config[name], trust_remote_code)
    weights = dict(AutoBridge.from_hf_config(config).export_hf_weights([gpt], cpu=True, show_progress=False))
    hidden_size = config.hidden_size
    num_tokens = model.config.num_tokens
    config.vocab_size = num_tokens
    config.tie_word_embeddings = False
    config.dtype = "bfloat16"

    # The original architecture is kept, transformers with other roles than the decoder are served as pooling models
    # vLLM's Gemma models scale token embeddings by sqrt(hidden), while WeLT's byte embeddings replace Megatron's
    embedding_scale = hidden_size ** -0.5 if config.model_type.startswith("gemma") else 1.0
    if name == "bytes_decoder":
        weights["model.embed_tokens.weight"] = model.bytes_decoder_embedding.folded_weight().detach() * embedding_scale
        weights["lm_head.weight"] = gpt.output_layer.weight.detach()
    else:
        if name == "bytes_encoder":
            embeddings = model.bytes_encoder.embed.folded_weight().detach() * embedding_scale
        else:  # Inputs are given as embeddings
            embeddings = torch.zeros(num_tokens, hidden_size)
        weights["model.embed_tokens.weight"] = embeddings
        if name == "latent_transformer":
            config.is_mm_prefix_lm = True  # Bidirectional attention ranges, for shift blocks
        else:
            config.is_causal = False  # Encoders are bidirectional

    weights = {k: v.to("cpu", torch.bfloat16).contiguous() for k, v in weights.items()}
    os.makedirs(path, exist_ok=True)
    config.bos_token_id, config.eos_token_id, config.pad_token_id = 2, 3, 0  # UTF8Tokenizer special tokens
    config.save_pretrained(path)
    save_file(weights, os.path.join(path, "model.safetensors"))
    _save_placeholder_tokenizer(path, num_tokens)


def export(checkpoint: str, output: str):
    checkpoint = os.path.normpath(checkpoint)
    if not os.path.basename(checkpoint).startswith("iter_"):  # A checkpoints dir, use the latest
        with open(os.path.join(checkpoint, "latest_checkpointed_iteration.txt")) as f:
            checkpoint = os.path.join(checkpoint, f"iter_{int(f.read().strip()):07d}")
    run_dir = os.path.dirname(os.path.dirname(checkpoint))  # <run>/checkpoints/iter_N

    from welt_training.extendable_yaml import CONFIG_FILE_NAME, load_yaml
    config = load_yaml(os.path.join(run_dir, CONFIG_FILE_NAME))

    model = load_model(checkpoint)

    os.makedirs(output, exist_ok=True)
    transformer_prefixes = tuple(path + "." for path in TRANSFORMERS.values())
    others = {k: v for k, v in model.state_dict().items()
              if not k.startswith(transformer_prefixes) and isinstance(v, torch.Tensor) and "_extra_state" not in k}
    others["decoder_prompt_embeddings"] = model.bytes_decoder_embedding.folded_weight()  # Prompts are embeddings
    save_file({k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in others.items()},
              os.path.join(output, "welt.safetensors"))

    for name in TRANSFORMERS:
        if config["model"].get(name):
            _save_transformer(model, name, config["model"], os.path.join(output, name),
                              config["model"].get("trust_remote_code", False))

    shutil.copytree(os.path.join(run_dir, "processor"), os.path.join(output, "processor"), dirs_exist_ok=True)
    shutil.copy(os.path.join(run_dir, CONFIG_FILE_NAME), os.path.join(output, CONFIG_FILE_NAME))
    print(f"Exported {checkpoint} to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="Megatron checkpoints directory, or a specific iter_* directory")
    parser.add_argument("--output", required=True, help="Export directory")
    args = parser.parse_args()

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    from megatron.core import parallel_state
    torch.distributed.init_process_group("nccl")
    parallel_state.initialize_model_parallel()
    try:
        export(args.checkpoint, args.output)
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
