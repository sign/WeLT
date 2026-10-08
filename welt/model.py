"""
WeLT on Megatron-Core.

Up to four Megatron GPT transformers, built (and optionally initialized from HF checkpoints) by Megatron-Bridge:
- bytes encoder: bidirectional transformer over the bytes of each word, BOS output is the word embedding
- image encoder: bidirectional transformer over 16x16 patches of each rendered word, CLS output is the word embedding
- latent transformer: causal transformer over word embeddings (packed sequences, prefix-LM shift blocks)
- bytes decoder: causal transformer generating each word's bytes from the latent vector of the previous word

Only the transformer blocks of each GPTModel are used: embeddings are replaced by word embeddings / byte embeddings,
and output layers by the mapping layers. Unused HF modules are deleted after loading.
"""
import dataclasses
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from megatron.bridge import AutoBridge
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.module import MegatronModule
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from transformers import AutoConfig

PATCH_DIM = 16 * 16 * 3

register_allowed_target_prefix("welt")  # Checkpoints' run_config.yaml instantiate WeLTModelProvider

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


def safetensors_checkpoint(name_or_path: str) -> str:
    """Megatron-Bridge reads safetensors only, convert (once) checkpoints that only have pytorch_model.bin."""
    from huggingface_hub import list_repo_files
    from transformers import AutoModelForCausalLM

    files = os.listdir(name_or_path) if os.path.isdir(name_or_path) else list_repo_files(name_or_path)
    if any(f.endswith(".safetensors") for f in files):
        return name_or_path
    path = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                        "safetensors", os.path.abspath(name_or_path).lstrip("/") if os.path.isdir(name_or_path)
                        else name_or_path)
    if not os.path.isdir(path):
        AutoModelForCausalLM.from_pretrained(name_or_path).save_pretrained(path)
    return path


class ByteEmbedding(nn.Module):
    """Embedding table plus an additive (zero-initialized) projection of each token's 8 bits."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        num_embeddings, dim = weight.shape
        self.weight = nn.Parameter(weight.detach().clone())
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
        return cls(**fields | dict(bytes_encoder=provider(bytes_encoder),
                                   image_encoder=provider(image_encoder),
                                   bytes_decoder=provider(bytes_decoder),
                                   bytes_encoder_hf_path=pretrained(bytes_encoder),
                                   image_encoder_hf_path=pretrained(image_encoder),
                                   latent_transformer_hf_path=pretrained(latent_transformer),
                                   bytes_decoder_hf_path=pretrained(bytes_decoder)) | kwargs)

    def sub_providers(self):
        return [p for p in (self.bytes_encoder, self.image_encoder, self.bytes_decoder) if p is not None]

    def finalize(self):
        for sub in self.sub_providers():
            for name in SHARED_CONFIG_FIELDS:
                setattr(sub, name, getattr(self, name))
            sub.finalize()
        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "WeLTModel":
        assert self.pipeline_model_parallel_size == 1, "WeLT does not support pipeline parallelism"
        assert self.context_parallel_size == 1, "WeLT does not support context parallelism"
        return WeLTModel(self)


FLEX_BLOCK_SIZE = 128
compiled_flex_attention = torch.compile(flex_attention, dynamic=True)


def repeat_kv(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    """(B, kv_heads, S, D) -> (B, heads, S, D) for grouped query attention.
    FlexAttention's enable_gqa fails to compile for some small inputs."""
    return tensor.repeat_interleave(heads // tensor.size(1), dim=1)


def packed_block_mask(cu_seqlens: torch.Tensor, causal: bool) -> BlockMask:
    """FlexAttention block mask over packed (THD) sequences, built in O(tokens) from cu_seqlens.
    Each query block attends to the key blocks spanned by the sequences it overlaps."""
    lengths = cu_seqlens.diff().long()
    total = int(cu_seqlens[-1])
    sequence = torch.repeat_interleave(torch.arange(len(lengths), device=lengths.device), lengths)
    starts = torch.repeat_interleave(cu_seqlens[:-1].long(), lengths)
    ends = torch.repeat_interleave(cu_seqlens[1:].long(), lengths) - 1

    num_blocks = (total + FLEX_BLOCK_SIZE - 1) // FLEX_BLOCK_SIZE
    blocks = torch.arange(num_blocks, device=lengths.device)
    first_block = starts[blocks * FLEX_BLOCK_SIZE] // FLEX_BLOCK_SIZE
    last_tokens = ((blocks + 1) * FLEX_BLOCK_SIZE - 1).clamp(max=total - 1)
    last_block = blocks if causal else ends[last_tokens] // FLEX_BLOCK_SIZE
    kv_num_blocks = (last_block - first_block + 1).int()
    kv_indices = (first_block[:, None] + blocks[None, :]).clamp(max=num_blocks - 1).int()

    def mask_mod(batch, head, q_index, kv_index):
        same_sequence = sequence[q_index] == sequence[kv_index]
        return same_sequence & (kv_index <= q_index) if causal else same_sequence

    return BlockMask.from_kv_blocks(kv_num_blocks[None, None], kv_indices[None, None], BLOCK_SIZE=FLEX_BLOCK_SIZE,
                                    mask_mod=mask_mod, seq_lengths=(total, total))


class PackedAttention(nn.Module):
    """Core attention over packed (THD) short sequences, e.g. the bytes of each word, with FlexAttention.
    Varlen flash attention is ~2x slower on thousands of few-token sequences (its backward pads per sequence)."""

    def __init__(self, config, causal: bool):
        super().__init__()
        assert not config.attention_dropout, "Attention dropout is not supported"
        self.causal = causal
        self.softmax_scale = config.softmax_scale

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None, attention_bias=None,
                packed_seq_params: PackedSeqParams = None, **kwargs):
        """(T, heads, dim) packed query, key, value -> (T, heads, dim)"""
        # The mask is shared by all layers of the transformer, cached on its packed_seq_params
        cache_key = f"_welt_block_mask_{self.causal}"
        block_mask = getattr(packed_seq_params, cache_key, None)
        if block_mask is None:
            block_mask = packed_block_mask(packed_seq_params.cu_seqlens_q, self.causal)
            setattr(packed_seq_params, cache_key, block_mask)
        query, key, value = (t.transpose(0, 1).unsqueeze(0) for t in (query, key, value))
        key, value = repeat_kv(key, query.size(1)), repeat_kv(value, query.size(1))
        out = compiled_flex_attention(query, key, value, block_mask=block_mask, scale=self.softmax_scale)
        return out.squeeze(0).transpose(0, 1)


class MaskedAttention(nn.Module):
    """Core attention with an arbitrary (B, 1, S, S) mask (True = masked) in FlexAttention, e.g. the latent
    transformer's packed sequences with bidirectional shift blocks."""

    def __init__(self, config):
        super().__init__()
        assert not config.attention_dropout, "Attention dropout is not supported"
        self.softmax_scale = config.softmax_scale

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=None, attention_bias=None, **kwargs):
        """(S, B, heads, dim) query, key, value -> (S, B, heads * dim)"""
        # The mask is shared by all layers of the transformer, cached on it
        block_mask = getattr(attention_mask, "_welt_block_mask", None)
        if block_mask is None:
            allowed = ~attention_mask[:, 0]
            block_mask = create_block_mask(lambda b, h, q, kv: allowed[b, q, kv], allowed.size(0), None,
                                           allowed.size(1), allowed.size(2), device=allowed.device, _compile=True)
            attention_mask._welt_block_mask = block_mask
        query, key, value = (t.permute(1, 2, 0, 3) for t in (query, key, value))
        key, value = repeat_kv(key, query.size(1)), repeat_kv(value, query.size(1))
        out = compiled_flex_attention(query, key, value, block_mask=block_mask, scale=self.softmax_scale)
        return out.permute(2, 0, 1, 3).flatten(2)


def build_transformer(provider: GPTModelProvider, hf_path: str | None, attention: str,
                      keep_output_layer=False) -> tuple[GPTModel, torch.Tensor]:
    """Build a Megatron GPTModel, load HF weights, then drop the parts WeLT replaces.
    attention: "arbitrary" (an attention_mask, BSHD) or "causal"/"bidirectional" (packed short sequences, THD).
    Attention is replaced by FlexAttention.
    Returns the model, and its (vocab, hidden) input embeddings table, removed from it."""
    provider.share_embeddings_and_output_weights = False  # embeddings and output layers are replaced
    assert provider.position_embedding_type == "rope", "Only RoPE transformers are supported"
    assert provider.window_size is None, "Sliding window attention is not supported"
    assert not getattr(provider, "attn_logit_softcapping", None), "Attention logit softcapping is not supported"
    model = GPTModelProvider.provide(provider, pre_process=True, post_process=True)

    if hf_path is not None and provider.perform_initialization:  # Otherwise, weights come from a checkpoint
        # HF vocabularies differ from bytes, skip mismatched embedding & lm_head
        AutoBridge.from_hf_pretrained(safetensors_checkpoint(hf_path)).load_hf_weights(
            [model], allowed_mismatched_params=["embedding.word_embeddings.weight", "output_layer.weight"])

    for layer in model.decoder.layers:
        if attention == "arbitrary":
            layer.self_attention.core_attention = MaskedAttention(provider)
        else:
            layer.self_attention.core_attention = PackedAttention(provider, causal=attention == "causal")

    # The vocabulary is split across tensor parallel ranks
    embeddings = gather_from_sequence_parallel_region(model.embedding.word_embeddings.weight.detach(),
                                                      tensor_parallel_output_grad=False)
    del model.embedding
    if not keep_output_layer:
        del model.output_layer
        model.post_process = False  # Only affects the (unused) forward and output layer checkpointing
    return model, embeddings


def run_transformer(model: GPTModel, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """(B, S, H) -> (B, S, H), including the final layer norm. attention_mask: True = masked."""
    hidden = hidden.transpose(0, 1).contiguous()
    rotary_pos_emb = model.rotary_pos_emb(hidden.size(0))
    if model.config.sequence_parallel:  # Each tensor parallel rank holds a part of the sequence
        hidden = scatter_to_sequence_parallel_region(hidden)
    hidden = model.decoder(hidden_states=hidden, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
    if model.config.sequence_parallel:
        hidden = gather_from_sequence_parallel_region(hidden, tensor_parallel_output_grad=False)
    return hidden.transpose(0, 1)


def run_packed_transformer(model: GPTModel, hidden: torch.Tensor, mask: torch.Tensor,
                           gather: bool = True) -> torch.Tensor:
    """Runs the valid positions of (N, S, H) right-padded sequences, packed without padding (THD format).
    Returns (num_valid, H) outputs, including the final layer norm. With sequence parallelism and gather=False,
    returns this rank's part of the (padded) packed outputs instead."""
    lengths = mask.sum(dim=-1, dtype=torch.int32)
    hidden = hidden[mask]
    num_valid = len(hidden)
    max_length = mask.size(1)
    if model.config.sequence_parallel:
        # Splitting the packed tokens across ranks needs a multiple of their number: pad with a dummy sequence
        padding = -num_valid % model.config.tensor_model_parallel_size
        hidden = F.pad(hidden, (0, 0, 0, padding))
        lengths = F.pad(lengths, (0, 1), value=padding)
        max_length = max(max_length, padding)  # RoPE covers the dummy sequence too
    cu_seqlens = F.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0))
    params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
                             max_seqlen_q=max_length, max_seqlen_kv=max_length)
    rotary_pos_emb = model.rotary_pos_emb(max_length, packed_seq=True)
    hidden = hidden.unsqueeze(1)
    if model.config.sequence_parallel:
        hidden = scatter_to_sequence_parallel_region(hidden)
    hidden = model.decoder(hidden_states=hidden, attention_mask=None,
                           rotary_pos_emb=rotary_pos_emb, packed_seq_params=params)
    if not model.config.sequence_parallel:
        return hidden.squeeze(1)
    if not gather:
        return hidden
    return gather_from_sequence_parallel_region(hidden, tensor_parallel_output_grad=False)[:num_valid].squeeze(1)


class WordEncoder(MegatronModule):  # Its sharded_state_dict recurses into the (tensor parallel) transformer
    """Bidirectional transformer, whose first position output is the word embedding."""

    def __init__(self, provider: GPTModelProvider, hf_path: str | None, embed: type[nn.Module]):
        super().__init__(config=provider)
        self.transformer, embeddings = build_transformer(provider, hf_path, "bidirectional")
        self.embed = embed(embeddings)  # Initialized from the (possibly pretrained) input embeddings
        self.hidden_size = provider.hidden_size

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """inputs: (N, T, ...), mask: (N, T) True = valid, right padded -> (N, H)"""
        hidden = run_packed_transformer(self.transformer, self.embed(inputs), mask)
        first_positions = F.pad(mask.sum(dim=-1).cumsum(0), (1, 0))[:-1]
        return hidden[first_positions]


class PatchImageEncoder(WordEncoder):
    """Bidirectional transformer over each word image's 16x16 patches (NaViT-style, native sizes), and a CLS."""

    def __init__(self, provider: GPTModelProvider, hf_path: str | None):
        super().__init__(provider, hf_path, lambda embeddings: PatchEmbedding(embeddings.size(1)))

    def forward(self, patches: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
        """(N, P, 768) uint8 patches, (N, 2) patch rows and columns of each image -> (N, H)"""
        counts = shapes.prod(dim=-1)
        mask = torch.arange(patches.size(1) + 1, device=counts.device)[None, :] <= counts[:, None]  # With CLS
        return super().forward(patches, mask)


class PatchEmbedding(nn.Module):
    """uint8 16x16 RGB patches -> CLS + linear patch embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(PATCH_DIM, dim)
        self.cls = nn.Parameter(torch.randn(dim) * 0.02)
        nn.init.xavier_uniform_(self.proj.weight)  # Like PIXEL / ViT-MAE patch embeddings

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        embeds = self.proj(patches.to(self.proj.weight.dtype) / 127.5 - 1)
        return torch.cat([self.cls.expand(len(embeds), 1, -1), embeds], dim=1)


class WeLTModel(MegatronModule):
    def __init__(self, config: WeLTModelProvider):
        super().__init__(config=config)
        self.pre_process = self.post_process = True

        self.bytes_encoder = None
        if config.bytes_encoder is not None:
            config.bytes_encoder.vocab_size = config.num_tokens
            self.bytes_encoder = WordEncoder(config.bytes_encoder, config.bytes_encoder_hf_path, ByteEmbedding)

        self.image_encoder = None
        if config.image_encoder is not None:
            self.image_encoder = PatchImageEncoder(config.image_encoder, config.image_encoder_hf_path)

        self.latent_transformer, _ = build_transformer(config, config.latent_transformer_hf_path, "arbitrary")

        decoder_config = config.bytes_decoder
        decoder_config.vocab_size = config.num_tokens
        self.bytes_decoder, embeddings = build_transformer(decoder_config, config.bytes_decoder_hf_path, "causal",
                                                           keep_output_layer=True)
        self.bytes_decoder_embedding = ByteEmbedding(embeddings)

        # Mapping layers
        encoders = [e for e in (self.image_encoder, self.bytes_encoder) if e is not None]
        encoder_dim = sum(e.hidden_size for e in encoders)
        self.encoder_mapping = nn.Linear(encoder_dim, config.hidden_size)
        self.encoder_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.decoder_mapping = nn.Linear(config.hidden_size, decoder_config.hidden_size)
        self.decoder_norm = nn.RMSNorm(decoder_config.hidden_size, eps=1e-6)

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
                     input_patches_shape: torch.Tensor | None = None) -> torch.Tensor:
        """Word embeddings in the latent space: (B, L, T) bytes [+ (B, L, P, 768) patches] -> (B, L, H)"""
        B, L, T = input_ids.shape  # noqa: N806
        input_ids = input_ids.view(B * L, T)
        words_mask = input_attention_mask.view(B * L, T).bool()
        valid = words_mask[:, 0]  # Words have BOS, batch padding words do not

        # Encode each distinct word once (about a third of the words in a batch), its embedding is shared
        unique_words, inverse = torch.unique(input_ids[valid], dim=0, return_inverse=True)
        positions = torch.arange(len(inverse), device=inverse.device)
        first = positions.new_full((len(unique_words),), len(inverse)).scatter_reduce(0, inverse, positions, "amin")
        rows = valid.nonzero().squeeze(1)[first]

        embeds = []
        if self.image_encoder is not None:
            patches = input_patches.view(B * L, *input_patches.shape[2:])[rows]
            embeds.append(self.image_encoder(patches, input_patches_shape.view(B * L, 2)[rows]))
        if self.bytes_encoder is not None:
            embeds.append(self.bytes_encoder(input_ids[rows], words_mask[rows]))

        scale = self._modality_scale(len(embeds), input_ids.device)
        embeds = torch.cat([e * s for e, s in zip(embeds, scale, strict=True)], dim=-1)
        word_embeds = embeds.new_zeros(B * L, embeds.size(-1))
        word_embeds[valid] = embeds[inverse]
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
        # With sequence parallelism, the output layer gathers the sequence itself
        hidden = run_packed_transformer(self.bytes_decoder, embeds, mask, gather=False)
        logits, _ = self.bytes_decoder.output_layer(hidden)  # Vocabulary split across tensor parallel ranks
        return logits.reshape(-1, logits.size(-1))[:int(mask.sum())], mask  # Without sequence parallel padding

    def forward(self,
                input_ids: torch.Tensor,
                input_attention_mask: torch.Tensor,
                attention_mask: torch.Tensor,
                labels_input: torch.Tensor,
                labels_attention_mask: torch.Tensor,
                labels_output: torch.Tensor,
                input_patches: torch.Tensor | None = None,
                input_patches_shape: torch.Tensor | None = None):
        """
        Args:
            input_ids: (B, L, T) bytes of each word
            input_attention_mask: (B, L, T) attention within each word
            attention_mask: (B, 1, L, L) attention across words, True = attend
            labels_input: (B, L, T') bytes decoder inputs (next word, without EOS)
            labels_attention_mask: (B, L, T')
            labels_output: (B, L, T') bytes decoder targets (next word, without BOS)
            input_patches: (B, L, P, 768) uint8 patches of the rendered words
            input_patches_shape: (B, L, 2) rows and columns of patches of each word

        Returns:
            (N, T') per-byte losses and (N, T') per-byte correctness, for the N words with labels
        """
        word_embeds = self.encode_words(input_ids, input_attention_mask, input_patches, input_patches_shape)
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
        full_logits = gather_from_tensor_model_parallel_region(logits.detach())
        correct[mask] = full_logits.argmax(dim=-1) == packed_targets
        return losses[:, 1:], correct[:, 1:], labels_output
