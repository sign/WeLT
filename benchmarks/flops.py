"""
Model FLOPs per training step for a WeLT config, counting each transformer on its real (packed) tokens.

    python benchmarks/flops.py <config.yaml> [--batches 20]

Per transformer: 6 * matmul params * tokens (forward + backward), plus attention 12 * layers * hidden * sum(len^2).
"Model" FLOPs encode every word occurrence,
"hardware" FLOPs encode each distinct word of a batch once (as WeLTModel does).
"""
import argparse
from itertools import islice

from transformers import AutoConfig
from utf8_tokenizer.tokenizer import PAD_TOKEN_ID

from welt.processor import collate_fn, next_word_labels
from welt_training.data_utils import pack_words, train_texts
from welt_training.extendable_yaml import load_yaml
from welt_training.train import build_dataset_provider


def matmul_params(config) -> tuple[int, int, int]:
    """(params in matmuls per token, layers, attention width) of a transformer config (Llama-like)."""
    hidden, layers = config.hidden_size, config.num_hidden_layers
    head_dim = getattr(config, "head_dim", None) or hidden // config.num_attention_heads
    kv = getattr(config, "num_key_value_heads", config.num_attention_heads) * head_dim
    attention = hidden * (config.num_attention_heads * head_dim) * 2 + hidden * kv * 2
    mlp = (3 if config.hidden_act == "silu" else 2) * hidden * config.intermediate_size  # Gated (SwiGLU) or not
    return layers * (attention + mlp), layers, config.num_attention_heads * head_dim


def transformer_flops(config, lengths) -> float:
    params, layers, width = matmul_params(config)
    tokens = sum(lengths)
    return 6 * params * tokens + 12 * layers * width * sum(n * n for n in lengths)


def step_flops(config: dict, batches: int = 20) -> dict[str, float]:
    model, data = config["model"], config["data"]
    provider = build_dataset_provider(model, data)
    processor = provider.processor()
    # The training examples (of a single data parallel rank), as the dataloader makes them
    texts = train_texts(provider, rank=0, world_size=1)
    examples = (processor.process_single_example(words, seq_lengths)
                for words, seq_lengths in pack_words(map(processor.pretokenize, texts), provider.seq_length))
    train_config = config.get("train") or {}
    micro_batch = train_config.get("micro_batch_size", 32)
    global_batch = train_config.get("global_batch_size", micro_batch)
    configs = {name: AutoConfig.from_pretrained(model[name], trust_remote_code=model.get("trust_remote_code", False))
               for name in ["bytes_encoder", "image_encoder", "latent_transformer", "bytes_decoder"] if model.get(name)}

    totals = {"model": 0.0, "hardware": 0.0}
    for _ in range(batches):
        batch = collate_fn(list(islice(examples, micro_batch)))
        ids, label_mask = batch["input_ids"], batch["label_mask"]
        valid = ids[..., 0] != PAD_TOKEN_ID  # Words have BOS, as in WeLTModel.encode_words
        lengths = (ids != PAD_TOKEN_ID).sum(-1)[valid].tolist()
        patches = batch["input_patches_shape"].prod(-1)[valid].tolist() if "input_patches_shape" in batch else lengths
        words = list(zip(map(tuple, ids[valid].tolist()), lengths, patches, strict=True))
        common = transformer_flops(configs["latent_transformer"], [ids.size(1)] * len(ids))
        # The bytes decoder runs on each label's latent and its bytes but the last (BOS, ..., without EOS)
        labels = next_word_labels(ids, batch["sequence_ids"], label_mask)[label_mask]
        common += transformer_flops(configs["bytes_decoder"], (1 + (labels[:, :-1] != PAD_TOKEN_ID).sum(-1)).tolist())
        for kind, encoded in [("model", words), ("hardware", list(dict.fromkeys(words)))]:
            flops = common
            if "bytes_encoder" in configs:
                flops += transformer_flops(configs["bytes_encoder"], [n for _, n, _ in encoded])
            if "image_encoder" in configs:
                flops += transformer_flops(configs["image_encoder"], [p + 1 for _, _, p in encoded])
            totals[kind] += flops
    return {kind: flops * global_batch / micro_batch / batches for kind, flops in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--ms_per_step", type=float, help="Also print achieved TFLOP/s")
    args = parser.parse_args()
    for kind, flops in step_flops(load_yaml(args.config), args.batches).items():
        line = f"{kind} TFLOPs per step: {flops / 1e12:.3f}"
        if args.ms_per_step:
            line += f", {flops / 1e12 / (args.ms_per_step / 1000):.2f} TFLOP/s"
        print(line)


if __name__ == "__main__":
    main()
