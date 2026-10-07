"""
WeLT on Megatron-Core.

Three Megatron GPT transformers, built and (optionally) initialized from HuggingFace checkpoints by Megatron-Bridge:
- bytes encoder: bidirectional transformer over the bytes of each word, BOS output is the word embedding
- image encoder: bidirectional transformer over 16x16 patches of each rendered word, CLS output is the word embedding
- latent transformer: causal transformer over word embeddings (packed sequences, prefix-LM shift blocks)
- bytes decoder: causal transformer generating each word's bytes from the latent vector of the previous word

Only the transformer blocks of each GPTModel are used: embeddings are replaced by word embeddings / byte embeddings,
and output layers by the mapping layers. Unused HF modules are deleted after loading.
"""
import dataclasses
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from torch import nn
from transformers import AutoConfig

PATCH_DIM = 16 * 16 * 3

# Fields copied from the WeLT (latent) config to every sub-transformer config
SHARED_CONFIG_FIELDS = (
    "tensor_model_parallel_size", "sequence_parallel", "bf16", "fp16", "params_dtype", "pipeline_dtype",
    "autocast_dtype", "use_cpu_initialization", "perform_initialization", "gradient_accumulation_fusion",
    "attention_backend", "recompute_granularity", "recompute_method", "recompute_num_layers", "recompute_modules",
)


def hf_config(name_or_path: str, trust_remote_code: bool = False):
    """A HF model id/path, or a JSON file of a HF config (must include "model_type" and "architectures")."""
    if name_or_path.endswith(".json"):
        import json
        with open(name_or_path) as f:
            kwargs = json.load(f)
        return AutoConfig.for_model(kwargs.pop("model_type"), **kwargs)
    return AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)


def transformer_provider(name_or_path: str, trust_remote_code: bool = False) -> GPTModelProvider:
    return AutoBridge.from_hf_config(hf_config(name_or_path, trust_remote_code)).to_megatron_provider(
        load_weights=False)


class ByteEmbedding(nn.Module):
    """Embedding table plus an additive (zero-initialized) projection of each token's 8 bits."""

    def __init__(self, num_embeddings: int, dim: int, init_std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, dim) * init_std)
        self.bit_proj = nn.Parameter(torch.zeros(dim, 8))
        shifts = torch.arange(7, -1, -1)
        self.register_buffer("bits", (torch.arange(num_embeddings)[:, None] >> shifts & 1).float(), persistent=False)

    def folded_weight(self) -> torch.Tensor:
        """Equivalent plain embedding table (used for export)."""
        return self.weight + self.bits.to(self.weight.dtype) @ self.bit_proj.T.to(self.weight.dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.folded_weight())


@dataclass
class WeLTModelProvider(GPTModelProvider):
    """The latent transformer's config, plus the configs of the other transformers."""
    bytes_encoder: GPTModelProvider | None = None
    image_encoder: GPTModelProvider | None = None
    bytes_decoder: GPTModelProvider | None = None

    # HF checkpoints to initialize each transformer from (None = random init)
    bytes_encoder_hf_path: str | None = None
    image_encoder_hf_path: str | None = None
    latent_transformer_hf_path: str | None = None
    bytes_decoder_hf_path: str | None = None

    num_tokens: int = 256
    pad_token_id: int = 0
    modality_dropout: float = 0.15

    @classmethod
    def from_hf(cls, latent_transformer: str, bytes_decoder: str,
                bytes_encoder: str | None = None, image_encoder: str | None = None,
                load_pretrained: bool = False, trust_remote_code: bool = False, **kwargs) -> "WeLTModelProvider":
        assert bytes_encoder or image_encoder, "At least one encoder must be provided"

        def provider(name):
            return transformer_provider(name, trust_remote_code) if name else None

        def pretrained(name):
            return name if load_pretrained and name and not name.endswith(".json") else None

        latent = provider(latent_transformer)
        fields = {f.name: getattr(latent, f.name) for f in dataclasses.fields(GPTModelProvider) if f.init}
        return cls(**fields,
                   bytes_encoder=provider(bytes_encoder),
                   image_encoder=provider(image_encoder),
                   bytes_decoder=provider(bytes_decoder),
                   bytes_encoder_hf_path=pretrained(bytes_encoder),
                   image_encoder_hf_path=pretrained(image_encoder),
                   latent_transformer_hf_path=pretrained(latent_transformer),
                   bytes_decoder_hf_path=pretrained(bytes_decoder),
                   **kwargs)

    def sub_providers(self):
        return [p for p in (self.bytes_encoder, self.image_encoder, self.bytes_decoder) if p is not None]

    def finalize(self):
        for sub in self.sub_providers():
            for name in SHARED_CONFIG_FIELDS:
                setattr(sub, name, getattr(self, name))
            sub.finalize()
        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "WeLTModel":
        assert self.pipeline_model_parallel_size == 1 and self.context_parallel_size == 1, \
            "WeLT does not support pipeline or context parallelism"
        return WeLTModel(self)


class PackedAttention(nn.Module):
    """Core attention over packed (THD) short sequences, e.g. the bytes of each word, with varlen flash attention.
    Replaces Transformer Engine's attention, whose cuDNN backend lacks THD backward on some GPUs (e.g. GB10)."""

    def __init__(self, config, causal: bool):
        super().__init__()
        self.causal = causal
        self.softmax_scale = config.softmax_scale
        self.dropout = config.attention_dropout

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None, attention_bias=None,
                packed_seq_params: PackedSeqParams = None, **kwargs):
        """(T, heads, dim) packed query, key, value -> (T, heads, dim)"""
        p = packed_seq_params
        return flash_attn_varlen_func(query, key, value, p.cu_seqlens_q, p.cu_seqlens_kv,
                                      p.max_seqlen_q, p.max_seqlen_kv, causal=self.causal,
                                      softmax_scale=self.softmax_scale,
                                      dropout_p=self.dropout if self.training else 0.0)


def build_transformer(provider: GPTModelProvider, hf_path: str | None, attention: str,
                      keep_embedding=False, keep_output_layer=False) -> GPTModel:
    """Build a Megatron GPTModel, load HF weights, then drop the parts WeLT replaces.
    attention: "arbitrary" (attention_mask, BSHD) or "causal"/"bidirectional" (packed short sequences, THD)"""
    provider.share_embeddings_and_output_weights = False  # embeddings and output layers are replaced
    assert provider.position_embedding_type == "rope", "Only RoPE transformers are supported"
    model = GPTModelProvider.provide(provider, pre_process=True, post_process=True)

    if hf_path is not None:
        # HF vocabularies differ from bytes, skip mismatched embedding & lm_head
        AutoBridge.from_hf_pretrained(hf_path).load_hf_weights(
            [model], allowed_mismatched_params=["embedding.word_embeddings.weight", "output_layer.weight"])

    for layer in model.decoder.layers:
        if attention == "arbitrary":
            layer.self_attention.attn_mask_type = AttnMaskType.arbitrary
        else:
            layer.self_attention.core_attention = PackedAttention(provider, causal=attention == "causal")

    if not keep_embedding:
        del model.embedding
    if not keep_output_layer:
        del model.output_layer
    return model


def run_transformer(model: GPTModel, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """(B, S, H) -> (B, S, H), including the final layer norm. attention_mask: True = masked."""
    hidden = hidden.transpose(0, 1).contiguous()
    rotary_pos_emb = model.rotary_pos_emb(hidden.size(0))
    hidden = model.decoder(hidden_states=hidden, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
    return hidden.transpose(0, 1)


def run_packed_transformer(model: GPTModel, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Runs the valid positions of (N, S, H) right-padded sequences, packed without padding (THD format).
    Returns (num_valid, H) outputs, including the final layer norm."""
    lengths = mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    max_length = mask.size(1)
    params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                             max_seqlen_q=max_length, max_seqlen_kv=max_length)
    rotary_pos_emb = model.rotary_pos_emb(max_length, packed_seq=True)
    hidden = model.decoder(hidden_states=hidden[mask].unsqueeze(1), attention_mask=None,
                           rotary_pos_emb=rotary_pos_emb, packed_seq_params=params)
    return hidden.squeeze(1)


class WordEncoder(nn.Module):
    """Bidirectional transformer, whose first position output is the word embedding."""

    def __init__(self, provider: GPTModelProvider, hf_path: str | None, embed: nn.Module):
        super().__init__()
        self.transformer = build_transformer(provider, hf_path, "bidirectional")
        self.embed = embed

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """inputs: (N, T, ...), mask: (N, T) True = valid, right padded -> (N, H)"""
        hidden = run_packed_transformer(self.transformer, self.embed(inputs), mask)
        first_positions = F.pad(mask.sum(dim=-1).cumsum(0), (1, 0))[:-1]
        return hidden[first_positions]


class PatchEmbedding(nn.Module):
    """uint8 16x16 RGB patches -> CLS + linear patch embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(PATCH_DIM, dim)
        self.cls = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        embeds = self.proj(patches.to(self.proj.weight.dtype) / 127.5 - 1)
        return torch.cat([self.cls.expand(len(embeds), 1, -1), embeds], dim=1)


class WeLTModel(MegatronModule):
    def __init__(self, config: WeLTModelProvider):
        super().__init__(config=config)
        self.pre_process = self.post_process = True

        self.bytes_encoder = None
        if config.bytes_encoder is not None:
            self.bytes_encoder = WordEncoder(config.bytes_encoder, config.bytes_encoder_hf_path,
                                             ByteEmbedding(config.num_tokens, config.bytes_encoder.hidden_size))

        self.image_encoder = None
        if config.image_encoder is not None:
            self.image_encoder = WordEncoder(config.image_encoder, config.image_encoder_hf_path,
                                             PatchEmbedding(config.image_encoder.hidden_size))

        self.latent_transformer = build_transformer(config, config.latent_transformer_hf_path, "arbitrary")

        decoder_config = config.bytes_decoder
        decoder_config.vocab_size = config.num_tokens
        self.bytes_decoder = build_transformer(decoder_config, config.bytes_decoder_hf_path, "causal",
                                               keep_embedding=True, keep_output_layer=True)
        # Byte embeddings replace the decoder's embedding, initialized from it
        self.bytes_decoder_embedding = ByteEmbedding(config.num_tokens, decoder_config.hidden_size)
        with torch.no_grad():
            self.bytes_decoder_embedding.weight.copy_(self.bytes_decoder.embedding.word_embeddings.weight)
        del self.bytes_decoder.embedding

        # Mapping layers
        encoders = [e for e in (self.image_encoder, self.bytes_encoder) if e is not None]
        encoder_dim = sum(e.transformer.config.hidden_size for e in encoders)
        self.encoder_mapping = nn.Linear(encoder_dim, config.hidden_size)
        self.encoder_norm = nn.RMSNorm(config.hidden_size)
        self.decoder_mapping = nn.Linear(config.hidden_size, decoder_config.hidden_size)
        self.decoder_norm = nn.RMSNorm(decoder_config.hidden_size)

    def set_input_tensor(self, input_tensor):
        pass  # No pipeline parallelism

    def _modality_scale(self, num_modalities: int, device) -> torch.Tensor:
        """Per-modality multiplier: drops modalities during training, rescaling the remaining ones."""
        keep = torch.ones(num_modalities, device=device)
        if self.training and num_modalities > 1 and self.config.modality_dropout > 0:
            keep = (torch.rand(num_modalities, device=device) >= self.config.modality_dropout).float()
            # Multiplying by zero, rather than skipping, keeps all parameters in the graph for DDP
            keep = keep * (num_modalities / keep.sum().clamp(min=1))
        return keep

    def encode_words(self,
                     input_ids: torch.Tensor,
                     input_attention_mask: torch.Tensor,
                     input_patches: torch.Tensor | None = None,
                     input_patches_count: torch.Tensor | None = None) -> torch.Tensor:
        """Word embeddings in the latent space: (B, L, T) bytes [+ (B, L, P, 768) patches] -> (B, L, H)"""
        B, L, T = input_ids.shape  # noqa: N806
        words_mask = input_attention_mask.view(B * L, T).bool()
        valid = words_mask[:, 0]  # Words have BOS, batch padding words do not

        embeds = []
        if self.image_encoder is not None:
            patches = input_patches.view(B * L, *input_patches.shape[2:])[valid]
            counts = input_patches_count.view(B * L)[valid]
            patches_mask = torch.arange(patches.size(1) + 1, device=counts.device)[None, :] <= counts[:, None]
            embeds.append(self.image_encoder(patches, patches_mask))
        if self.bytes_encoder is not None:
            embeds.append(self.bytes_encoder(input_ids.view(B * L, T)[valid], words_mask[valid]))

        scale = self._modality_scale(len(embeds), input_ids.device)
        embeds = torch.cat([e * s for e, s in zip(embeds, scale, strict=True)], dim=-1)
        word_embeds = embeds.new_zeros(B * L, embeds.size(-1))
        word_embeds[valid] = embeds
        return self.encoder_norm(self.encoder_mapping(word_embeds)).view(B, L, -1)

    def latent(self, word_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, L, H) word embeddings, (B, 1, L, L) True = attend -> (B, L, H_decoder) latent vectors"""
        hidden = run_transformer(self.latent_transformer, word_embeds, ~attention_mask)
        return self.decoder_norm(self.decoder_mapping(hidden))

    def decode(self, latents: torch.Tensor, labels_input: torch.Tensor, labels_mask: torch.Tensor):
        """Parallel causal decoding, each word's bytes conditioned on its latent vector.
        (N, H) latents, (N, T) input bytes and mask -> (num_valid, vocab) logits and (N, 1 + T) valid mask,
        where the logits at each latent position are a prediction of the first input byte (which is BOS)."""
        embeds = torch.cat([latents[:, None], self.bytes_decoder_embedding(labels_input)], dim=1)
        mask = F.pad(labels_mask.bool(), (1, 0), value=True)
        hidden = run_packed_transformer(self.bytes_decoder, embeds, mask)
        logits, _ = self.bytes_decoder.output_layer(hidden)
        return logits, mask

    def forward(self,
                input_ids: torch.Tensor,
                input_attention_mask: torch.Tensor,
                attention_mask: torch.Tensor,
                labels_input: torch.Tensor,
                labels_attention_mask: torch.Tensor,
                labels_output: torch.Tensor,
                input_patches: torch.Tensor | None = None,
                input_patches_count: torch.Tensor | None = None):
        """
        Args:
            input_ids: (B, L, T) bytes of each word
            input_attention_mask: (B, L, T) attention within each word
            attention_mask: (B, 1, L, L) attention across words, True = attend
            labels_input: (B, L, T') bytes decoder inputs (next word, without EOS)
            labels_attention_mask: (B, L, T')
            labels_output: (B, L, T') bytes decoder targets (next word, without BOS)
            input_patches: (B, L, P, 768) uint8 patches of the rendered words
            input_patches_count: (B, L) number of patches per word

        Returns:
            (N, T') per-byte losses and (N, T') per-byte correctness, for the N words with labels
        """
        word_embeds = self.encode_words(input_ids, input_attention_mask, input_patches, input_patches_count)
        latents = self.latent(word_embeds, attention_mask)

        # Only decode words that have a label
        has_label = labels_attention_mask.flatten(0, 1).any(dim=-1)
        latents = latents.flatten(0, 1)[has_label]
        labels_input = labels_input.flatten(0, 1)[has_label]
        labels_output = labels_output.flatten(0, 1)[has_label]

        logits, mask = self.decode(latents, labels_input, labels_attention_mask.flatten(0, 1)[has_label])
        # Position t (input byte t-1, or the latent for t=0) predicts output byte t-1
        targets = F.pad(labels_output, (1, 0), value=self.config.pad_token_id)
        packed_targets = targets[mask]
        # (S, B, V) logits and (B, S) labels, vocab-parallel cross entropy
        packed_losses = self.bytes_decoder.compute_language_model_loss(packed_targets[None], logits[:, None])[0]

        losses = torch.zeros_like(targets, dtype=packed_losses.dtype)
        losses[mask] = packed_losses
        correct = torch.zeros_like(targets, dtype=torch.bool)
        correct[mask] = logits.argmax(dim=-1) == packed_targets
        return losses[:, 1:], correct[:, 1:], labels_output
