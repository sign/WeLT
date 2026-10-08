"""
WeLT generation with vLLM, from a `welt.export` directory.

    welt-generate <export_dir> "<text prompt>" ["<another prompt>" ...]

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
import torch.nn.functional as F  # noqa: E402, N812
from cachetools import LRUCache  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from vllm import LLM, PoolingParams, SamplingParams  # noqa: E402
from vllm.config import PoolerConfig  # noqa: E402
from words_segmentation.pretokenizer import is_word_complete  # noqa: E402

from welt.attention import get_shift_blocks  # noqa: E402
from welt.processor import TextImageProcessor  # noqa: E402
from welt.vllm_plugin import RANGES_KEY, restore_opentelemetry_context  # noqa: E402


class WeLTGenerator:
    def __init__(self, path: str, kv_cache_gib: float = 1.0, device: str = "cuda"):
        """kv_cache_gib: KV cache memory of each vLLM engine. A fixed size (rather than a fraction of the GPU) skips
        vLLM's memory profiling, which fails when other processes use the GPU."""
        restore_opentelemetry_context()
        self.processor = TextImageProcessor.from_pretrained(os.path.join(path, "processor"))
        self.tokenizer = self.processor.tokenizer
        self.device = device

        # gpu_memory_utilization only gates vLLM's startup check of free memory, given kv_cache_memory_bytes
        engine = dict(kv_cache_memory_bytes=int(kv_cache_gib * 2**30), gpu_memory_utilization=0.05, dtype="bfloat16")
        encoder = dict(runner="pooling", convert="embed", **engine,
                       pooler_config=PoolerConfig(seq_pooling_type="CLS", use_activation=False))
        self.image_encoder = self.bytes_encoder = None
        image_path, bytes_path = os.path.join(path, "image_encoder"), os.path.join(path, "bytes_encoder")
        if os.path.isdir(image_path):
            self.image_encoder = LLM(image_path, enable_prompt_embeds=True, **encoder)
        if os.path.isdir(bytes_path):
            self.bytes_encoder = LLM(bytes_path, max_model_len=self.processor.max_word_length, **encoder)
        # Chunked prefill must not split a bidirectional shift block across steps
        self.latent = LLM(os.path.join(path, "latent_transformer"), runner="pooling", convert="embed",
                          enable_prompt_embeds=True, enable_prefix_caching=True, enable_chunked_prefill=False,
                          pooler_config=PoolerConfig(seq_pooling_type="LAST", use_activation=False), **engine)
        self.decoder = LLM(os.path.join(path, "bytes_decoder"), enable_prompt_embeds=True,
                           max_model_len=self.processor.max_word_length + 1, **engine)

        self.weights = load_file(os.path.join(path, "welt.safetensors"), device=device)  # The layers around them

        self.word_embeddings = LRUCache(maxsize=100_000)

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        weight = self.weights[f"{name}.weight"]
        return F.linear(x.to(weight.dtype), weight, self.weights[f"{name}.bias"])

    def _map(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """WeLTModel's {encoder,decoder}_mapping and _norm, between the transformers."""
        x = self._linear(x, f"{name}_mapping")
        return F.rms_norm(x, x.shape[-1:], self.weights[f"{name}_norm.weight"], eps=1e-6)

    @staticmethod
    def _pooled(llm: LLM, prompts: list[dict], params: PoolingParams | list[PoolingParams]) -> torch.Tensor:
        return torch.stack([o.outputs.data for o in llm.encode(prompts, params, pooling_task="embed",
                                                                use_tqdm=False)])

    @torch.inference_mode()
    def _encode_words(self, words: list[str]):
        """Cache the latent-space embedding of each new word."""
        words = list(dict.fromkeys(w for w in words if w not in self.word_embeddings))
        if not words:
            return
        params = PoolingParams(use_activation=False)
        embeds = []
        if self.image_encoder is not None:
            patches, shapes = self.processor.render_texts(words)
            patches = patches.to(self.device)
            prompts = []
            for word_patches, count in zip(patches, shapes.prod(dim=-1), strict=True):
                projected = self._linear(word_patches[:count] / 127.5 - 1, "image_encoder.embed.proj")
                cls = self.weights["image_encoder.embed.cls"]
                prompts.append({"prompt_embeds": torch.cat([cls[None], projected]).cpu()})
            embeds.append(self._pooled(self.image_encoder, prompts, params))
        if self.bytes_encoder is not None:
            tokenized = self.processor.tokenize_words(words)
            prompts = [{"prompt_token_ids": ids[mask.bool()].tolist()}
                       for ids, mask in zip(tokenized.input_ids, tokenized.attention_mask, strict=True)]
            embeds.append(self._pooled(self.bytes_encoder, prompts, params))
        embeds = self._map(torch.cat(embeds, dim=-1).to(self.device), "encoder").cpu()
        self.word_embeddings.update(zip(words, embeds, strict=True))

    @torch.inference_mode()
    def _latents(self, sequences: list[list[str]]) -> torch.Tensor:
        """The latent vector after the last word of each sequence, mapped to the bytes decoder."""
        self._encode_words([w for words in sequences for w in words])
        prompts, params = [], []
        for words in sequences:
            ranges = list(get_shift_blocks(words))
            # Cached keys and values depend on the bidirectional ranges, which are not part of vLLM's cache key
            prompts.append({"prompt_embeds": torch.stack([self.word_embeddings[w] for w in words]),
                            "cache_salt": repr(ranges)})
            params.append(PoolingParams(use_activation=False, extra_kwargs={RANGES_KEY: ranges}))
        return self._map(self._pooled(self.latent, prompts, params).to(self.device), "decoder")

    @torch.inference_mode()
    def _next_words(self, latents: torch.Tensor, prefixes: list[bytes], sampling: SamplingParams) -> list[bytes]:
        prompts = []
        for latent, prefix in zip(latents, prefixes, strict=True):
            byte_ids = torch.tensor([self.tokenizer.bos_token_id, *prefix], device=self.device)
            embeddings = self.weights["decoder_prompt_embeddings"][byte_ids]
            prompts.append({"prompt_embeds": torch.cat([latent[None], embeddings]).cpu()})
        outputs = self.decoder.generate(prompts, sampling, use_tqdm=False)
        eos = self.tokenizer.eos_token_id
        return [prefix + bytes(t for t in o.outputs[0].token_ids if t != eos)
                for o, prefix in zip(outputs, prefixes, strict=True)]

    def generate(self, texts: list[str], max_generated_words: int = 50, temperature: float = 0.0,
                 seed: int | None = None) -> list[str]:
        """Greedy (or sampled, with temperature) continuation of each text, word by word."""
        sequences = [self.processor.pretokenize(text) for text in texts]
        # A prompt ending mid-word continues that word: its bytes become the decoder's prefix
        max_prefix = self.processor.max_word_length - 3  # BOS and EOS, and at least one new byte
        prefixes = [b""] * len(texts)
        for i, words in enumerate(sequences):
            if len(words) > 1 and not is_word_complete(words[-1]) and len(words[-1].encode()) <= max_prefix:
                prefixes[i] = words.pop().encode("utf-8")

        generated = [[] for _ in texts]
        active = list(range(len(texts)))
        for step in range(max_generated_words):
            if not active:
                break
            sampling = SamplingParams(temperature=temperature, seed=None if seed is None else seed + step,
                                      detokenize=False, max_tokens=self.processor.max_word_length - 1,
                                      stop_token_ids=[self.tokenizer.eos_token_id])
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
    outputs = generator.generate(args.prompts, args.max_generated_words, args.temperature)
    for prompt, output in zip(args.prompts, outputs, strict=True):
        print(repr(prompt), "->", repr(output))


if __name__ == "__main__":
    main()
