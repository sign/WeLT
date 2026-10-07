"""
Export a WeLT Megatron checkpoint for inference with vLLM.

    torchrun --nproc_per_node=1 -m welt.export <output_dir>/checkpoints [--iteration N] --output <export_dir>

Writes a HuggingFace (Llama-like) model directory per transformer, which vLLM serves:
- bytes_encoder/, image_encoder/: bidirectional, CLS pooling (LlamaBidirectionalModel)
- latent_transformer/: causal, last token pooling, bidirectional shift blocks (is_mm_prefix_lm)
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

TRANSFORMERS = {
    # module name in WeLTModel: (attribute path to the GPTModel, config key in welt.yaml)
    "bytes_encoder": ("bytes_encoder.transformer", "bytes_encoder"),
    "image_encoder": ("image_encoder.transformer", "image_encoder"),
    "latent_transformer": ("latent_transformer", "latent_transformer"),
    "bytes_decoder": ("bytes_decoder", "bytes_decoder"),
}


def _unwrap(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _save_placeholder_tokenizer(path: str, num_tokens: int):
    """vLLM requires a tokenizer, even though WeLT passes token ids or embeddings and never detokenizes."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    vocab = {f"<0x{i:02X}>": i for i in range(num_tokens)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<0x00>"))
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<0x02>", eos_token="<0x03>",
                            pad_token="<0x00>").save_pretrained(path)


def _save_transformer(model: WeLTModel, name: str, model_config: dict, path: str, trust_remote_code=False):
    module_path, config_key = TRANSFORMERS[name]
    gpt = model.get_submodule(module_path)
    config = hf_config(model_config[config_key], trust_remote_code)

    weights = {name: tensor.to(torch.bfloat16).contiguous() for name, tensor in
               AutoBridge.from_hf_config(config).export_hf_weights([gpt], cpu=True, show_progress=False)}
    hidden_size = config.hidden_size
    num_tokens = model.config.num_tokens
    config.vocab_size = num_tokens
    config.tie_word_embeddings = False
    config.dtype = "bfloat16"

    if name == "bytes_decoder":
        weights["model.embed_tokens.weight"] = model.bytes_decoder_embedding.folded_weight().detach()
        weights["lm_head.weight"] = gpt.output_layer.weight.detach()
        config.architectures = ["LlamaForCausalLM"]
    else:
        if name == "bytes_encoder":
            embeddings = model.bytes_encoder.embed.folded_weight().detach()
        else:  # Inputs are given as embeddings
            embeddings = torch.zeros(num_tokens, hidden_size)
        weights["model.embed_tokens.weight"] = embeddings
        if name == "latent_transformer":
            config.architectures = ["LlamaForCausalLM"]  # served as a pooling model
            config.is_mm_prefix_lm = True  # bidirectional attention ranges, for shift blocks
        else:
            config.architectures = ["LlamaBidirectionalModel"]
            config.pooling = "cls"

    weights = {k: v.to("cpu", torch.bfloat16).contiguous() for k, v in weights.items()}
    os.makedirs(path, exist_ok=True)
    config.bos_token_id, config.eos_token_id, config.pad_token_id = 2, 3, 0  # UTF8Tokenizer special tokens
    config.save_pretrained(path)
    save_file(weights, os.path.join(path, "model.safetensors"))
    _save_placeholder_tokenizer(path, num_tokens)


def export(checkpoint: str, output: str):
    run_dir = os.path.dirname(os.path.normpath(checkpoint))
    if not os.path.basename(os.path.normpath(checkpoint)).startswith("iter_"):  # A checkpoints dir, use the latest
        with open(os.path.join(checkpoint, "latest_checkpointed_iteration.txt")) as f:
            checkpoint = os.path.join(checkpoint, f"iter_{int(f.read().strip()):07d}")
    else:
        run_dir = os.path.dirname(run_dir)

    from welt_training.train import CONFIG_FILE_NAME, load_yaml
    config = load_yaml(os.path.join(run_dir, CONFIG_FILE_NAME))

    model = load_megatron_model(checkpoint, skip_temp_dist_context=True)
    model: WeLTModel = _unwrap(model[0] if isinstance(model, list) else model)
    model.eval()

    os.makedirs(output, exist_ok=True)
    transformer_prefixes = tuple(path + "." for path, _ in TRANSFORMERS.values())
    others = {k: v.to("cpu", torch.bfloat16).contiguous() for k, v in model.state_dict().items()
              if not k.startswith(transformer_prefixes) and isinstance(v, torch.Tensor) and "_extra_state" not in k}
    save_file(others, os.path.join(output, "welt.safetensors"))

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
