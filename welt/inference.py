"""
WeLT generation with vLLM, from a `welt.export` directory.

    python -m welt.inference <export_dir> "<text prompt>" ["<another prompt>" ...]

Each generation step runs, for all active prompts at once:
1. the encoders (vLLM pooling) on new words -> word embeddings (cached per word)
2. the latent transformer (vLLM pooling, prefix cached) on all word embeddings -> latent of the last word
3. the bytes decoder (vLLM generation) from the latent -> bytes of the next word
"""
import argparse
import os

# Engine processes are spawned, not forked: forking after CUDA / Megatron imports is unsafe
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from vllm import LLM, PoolingParams, SamplingParams  # noqa: E402
from vllm.config import PoolerConfig  # noqa: E402
from words_segmentation.pretokenizer import is_word_complete  # noqa: E402

from welt.attention import get_shift_blocks  # noqa: E402
from welt.processor import PATCH_SIZE, TextImageProcessor  # noqa: E402
from welt.vision import HFImageEncoder, is_vision_model  # noqa: E402
from welt.vllm_plugin import RANGES_KEY, restore_opentelemetry_context  # noqa: E402

PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3


class WeLTGenerator:
    def __init__(self, path: str, gpu_memory_utilization: float = 0.1, device: str = "cuda"):
        restore_opentelemetry_context()
        self.processor = TextImageProcessor.from_pretrained(os.path.join(path, "processor"))
        self.tokenizer = self.processor.tokenizer
        self.device = device

        engine_args = dict(gpu_memory_utilization=gpu_memory_utilization, dtype="bfloat16")
        # Context lengths default to each transformer's max_position_embeddings
        encoder_args = dict(runner="pooling", convert="embed", **engine_args,
                            pooler_config=PoolerConfig(seq_pooling_type="CLS", use_activation=False))
        self.encoders = {}
        self.vision = None  # A HF vision backbone, which vLLM does not serve on its own
        image_encoder = os.path.join(path, "image_encoder")
        if os.path.isdir(image_encoder) and is_vision_model(AutoConfig.from_pretrained(image_encoder)):
            self.vision = HFImageEncoder(image_encoder, pretrained=True).to(device, torch.bfloat16).eval()
        elif os.path.isdir(image_encoder):
            self.encoders["image_encoder"] = LLM(image_encoder, enable_prompt_embeds=True, **encoder_args)
        if os.path.isdir(os.path.join(path, "bytes_encoder")):
            self.encoders["bytes_encoder"] = LLM(os.path.join(path, "bytes_encoder"), **encoder_args)
        self.latent = LLM(os.path.join(path, "latent_transformer"), convert="embed", enable_prompt_embeds=True,
                          enable_prefix_caching=True, runner="pooling",
                          pooler_config=PoolerConfig(seq_pooling_type="LAST", use_activation=False), **engine_args)
        self.decoder = LLM(os.path.join(path, "bytes_decoder"), enable_prompt_embeds=True,
                           max_model_len=self.processor.max_word_length + 2, **engine_args)

        # The small layers around the transformers
        weights = load_file(os.path.join(path, "welt.safetensors"), device=device)
        self.encoder_mapping = self._linear(weights, "encoder_mapping")
        self.decoder_mapping = self._linear(weights, "decoder_mapping")
        self.encoder_norm = self._norm(weights, "encoder_norm")
        self.decoder_norm = self._norm(weights, "decoder_norm")
        self.patch_proj = self._linear(weights, "image_encoder.embed.proj") if self.has("image_encoder") else None
        self.patch_cls = weights.get("image_encoder.embed.cls")
        decoder_embedding = (weights["bytes_decoder_embedding.weight"] +
                             self._bits().to(weights["bytes_decoder_embedding.weight"].dtype)
                             @ weights["bytes_decoder_embedding.bit_proj"].T)
        self.decoder_embedding = decoder_embedding

        self.word_embeddings = {}

    def has(self, name):
        return name in self.encoders

    def _bits(self):
        return (torch.arange(256, device=self.device)[:, None] >> torch.arange(7, -1, -1, device=self.device) & 1)

    @staticmethod
    def _linear(weights, prefix):
        weight = weights[f"{prefix}.weight"]
        layer = nn.Linear(weight.size(1), weight.size(0), dtype=weight.dtype, device=weight.device)
        layer.load_state_dict({"weight": weight, "bias": weights[f"{prefix}.bias"]})
        return layer

    @staticmethod
    def _norm(weights, prefix):
        weight = weights[f"{prefix}.weight"]
        norm = nn.RMSNorm(weight.size(0), dtype=weight.dtype, device=weight.device)
        norm.load_state_dict({"weight": weight})
        return norm

    @staticmethod
    def _pooled(outputs) -> torch.Tensor:
        return torch.stack([torch.as_tensor(o.outputs.embedding) for o in outputs])

    @torch.inference_mode()
    def _encode_words(self, words: list[str]):
        """Cache the latent-space embedding of each new word."""
        words = list(dict.fromkeys(w for w in words if w not in self.word_embeddings))
        if not words:
            return
        params = PoolingParams(use_activation=False)
        embeds = []
        if self.vision is not None:
            patches, shapes = self.processor.render_texts(words)
            embeds.append(self.vision(patches.to(self.device), shapes.to(self.device)).float().cpu())
        if self.has("image_encoder"):
            patches, shapes = self.processor.render_texts(words)
            prompts = []
            for word_patches, count in zip(patches.to(self.device), shapes.prod(dim=-1), strict=True):
                projected = self.patch_proj(word_patches[:count].to(self.patch_proj.weight.dtype) / 127.5 - 1)
                prompts.append({"prompt_embeds": torch.cat([self.patch_cls[None], projected]).cpu()})
            embeds.append(self._pooled(self.encoders["image_encoder"].embed(prompts, pooling_params=params,
                                                                            use_tqdm=False)))
        if self.has("bytes_encoder"):
            tokenized = self.processor.tokenize_words(words)
            prompts = [{"prompt_token_ids": ids[mask.bool()].tolist()}
                       for ids, mask in zip(tokenized.input_ids, tokenized.attention_mask, strict=True)]
            embeds.append(self._pooled(self.encoders["bytes_encoder"].embed(prompts, pooling_params=params,
                                                                            use_tqdm=False)))
        embeds = torch.cat(embeds, dim=-1).to(self.device, self.encoder_mapping.weight.dtype)
        embeds = self.encoder_norm(self.encoder_mapping(embeds)).cpu()
        self.word_embeddings.update(zip(words, embeds, strict=True))

    @torch.inference_mode()
    def _latents(self, sequences: list[list[str]]) -> torch.Tensor:
        """The latent vector after the last word of each sequence, mapped to the bytes decoder."""
        self._encode_words([w for words in sequences for w in words])
        prompts = [{"prompt_embeds": torch.stack([self.word_embeddings[w] for w in words])} for words in sequences]
        params = [PoolingParams(use_activation=False, extra_kwargs={RANGES_KEY: list(get_shift_blocks(words))})
                  for words in sequences]
        hidden = self._pooled(self.latent.embed(prompts, pooling_params=params, use_tqdm=False))
        hidden = hidden.to(self.device, self.decoder_mapping.weight.dtype)
        return self.decoder_norm(self.decoder_mapping(hidden))

    @torch.inference_mode()
    def _next_words(self, latents: torch.Tensor, prefixes: list[bytes], sampling: SamplingParams) -> list[bytes]:
        prompts = []
        for latent, prefix in zip(latents, prefixes, strict=True):
            byte_ids = torch.tensor([self.tokenizer.bos_token_id, *prefix], device=self.device)
            prompts.append({"prompt_embeds": torch.cat([latent[None], self.decoder_embedding[byte_ids]]).cpu()})
        outputs = self.decoder.generate(prompts, sampling, use_tqdm=False)
        eos = self.tokenizer.eos_token_id
        return [prefix + bytes(t for t in o.outputs[0].token_ids if t != eos)
                for o, prefix in zip(outputs, prefixes, strict=True)]

    def generate(self, texts: list[str], max_generated_words: int = 50, temperature: float = 0.0,
                 seed: int | None = None) -> list[str]:
        sequences = [self.processor.pretokenize(text) for text in texts]
        # A prompt ending mid-word continues that word: its bytes become the decoder's prefix
        prefixes = [b""] * len(texts)
        for i, words in enumerate(sequences):
            if len(words) > 1 and not is_word_complete(words[-1]):
                prefixes[i] = words.pop().encode("utf-8")

        sampling = SamplingParams(temperature=temperature, seed=seed, detokenize=False,
                                  max_tokens=self.processor.max_word_length,
                                  stop_token_ids=[self.tokenizer.eos_token_id])
        generated = [[] for _ in texts]
        active = list(range(len(texts)))
        for _ in range(max_generated_words):
            if not active:
                break
            latents = self._latents([sequences[i] for i in active])
            next_words = self._next_words(latents, [prefixes[i] for i in active], sampling)
            still_active = []
            for i, word in zip(active, next_words, strict=True):
                continuation = word[len(prefixes[i]):]
                prefixes[i] = b""
                if not word:  # An empty word (immediate EOS) ends the generation
                    continue
                generated[i].append(continuation.decode("utf-8", errors="replace"))
                sequences[i].append(word.decode("utf-8", errors="replace"))
                still_active.append(i)
            active = still_active
        return ["".join(words) for words in generated]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Directory exported by welt.export")
    parser.add_argument("prompts", nargs="+")
    parser.add_argument("--max_generated_words", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    generator = WeLTGenerator(args.model)
    for prompt, output in zip(args.prompts, generator.generate(args.prompts, args.max_generated_words,
                                                                args.temperature), strict=True):
        print(repr(prompt), "->", repr(output))


if __name__ == "__main__":
    main()
