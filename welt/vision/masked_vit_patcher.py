import collections.abc
import types
from functools import lru_cache
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
import transformers.models.vit.modeling_vit as hvit
from transformers.models.vit.modeling_vit import (
    ViTEmbeddings,
    ViTForImageClassification,
    ViTModel,
    ViTSelfAttention,
)


# --------------------------- Public API ---------------------------

def patch_vit_zero_pad_mask(model: ViTModel, pad_value: float = 0.0, tol: float = 0.0) -> None:
    """
    Fast-ish, training-safe padding invariance for ViT with strict equality:

    Guarantees:
      1) Padded patches never influence attention (pre-softmax logits masking).
      2) Positional encodings computed on each sample's tight (content-only) grid, then scattered.
      3) Final outputs at padded patches are exactly zeros (functional, CUDA-safe).

    Optimizations kept:
      - Zero-patch detection via max_pool2d (no unfold).
      - Per-(hs,ws) positional encodings with lightweight caching.
      - No (layers,heads,L,L) mask expansion; compact (B,1,L,L) broadcast mask.

    Assumptions:
      - Padding value ~ 0 (tune `pad_value`/`tol` if needed).
      - Content is a top-left rectangle (typical with right/bottom padding).
    """
    if getattr(model, "_zero_pad_mask_patched", False):
        return

    # ---- helpers ----
    def _patch_hw(vit_model: ViTModel) -> Tuple[int, int]:
        ps = vit_model.embeddings.patch_size
        if isinstance(ps, collections.abc.Iterable):
            ph, pw = int(ps[0]), int(ps[1])
        else:
            ph = pw = int(ps)
        return ph, pw

    def _valid_grid(vit_model: ViTModel, x: torch.Tensor) -> Tuple[torch.BoolTensor, int, int]:
        """
        Detect non-zero patches cheaply using pooling.

        Returns:
          valid: (B, Hpatch, Wpatch) bool
          Hpatch, Wpatch
        """
        B, C, H, W = x.shape
        ph, pw = _patch_hw(vit_model)
        Hpatch, Wpatch = H // ph, W // pw

        delta = (x - pad_value).abs() if pad_value != 0.0 else x.abs()
        delta = delta.amax(dim=1, keepdim=True)  # (B,1,H,W)
        win = F.max_pool2d(delta, kernel_size=(ph, pw), stride=(ph, pw))  # (B,1,Hpatch,Wpatch)
        valid = (win > tol).squeeze(1)  # (B,Hpatch,Wpatch)
        return valid, Hpatch, Wpatch

    # ---- per-sample positional encodings with tiny cache ----
    # We keep a per-model dict cache to avoid dtype/device headaches in functools.lru_cache.
    if not hasattr(model.embeddings, "_pos_tensor_cache"):
        model.embeddings._pos_tensor_cache = {}  # type: ignore[attr-defined]
    _pos_tensor_cache: dict = model.embeddings._pos_tensor_cache  # type: ignore[assignment]

    # Small metadata LRU to avoid repeated key construction (acts as a registry)
    @lru_cache(maxsize=256)
    def _pos_meta_key(hs: int, ws: int, D: int, ph: int, pw: int, dtype: torch.dtype, device_str: str):
        return (hs, ws, D, ph, pw, str(dtype), device_str)

    # ---- patch eager attention to support pre-softmax masking (strict equality) ----
    # We *don't* touch model.config._attn_implementation to avoid surprises.
    if not hasattr(hvit, "_orig_eager_attention_forward"):
        hvit._orig_eager_attention_forward = hvit.eager_attention_forward

        def _eager_attention_forward_premask(
            module: nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor | None,
            scaling: float,
            dropout: float = 0.0,
            **kwargs,
        ):
            # logits
            attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scaling  # (B,H,L,L)

            # If module provides a pre-softmax pair mask, apply it here (bool with True=keep).
            premask = getattr(module, "_pre_softmax_pair_mask", None)  # (B,1,L,L) or None
            if premask is not None:
                mask_bool = premask.bool() if premask.dtype != torch.bool else premask
                attn_scores = attn_scores.masked_fill(~mask_bool, torch.finfo(attn_scores.dtype).min)

            # Softmax
            attn_weights = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)

            # Back-compat: HF eager uses post-softmax multiply if a mask is passed in
            if premask is None and attention_mask is not None:
                attn_weights = attn_weights * attention_mask

            attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
            attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
            return attn_output, attn_weights

        hvit.eager_attention_forward = _eager_attention_forward_premask

    # ---- override embeddings.forward to use per-sample pos-enc (with caching) ----
    orig_embed_forward = model.embeddings.forward

    def embeddings_forward_with_invariant_pos(
        self: ViTEmbeddings,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        interpolate_pos_encoding: bool | None = None,
    ) -> torch.Tensor:
        B, C, H, W = pixel_values.shape
        ph, pw = _patch_hw(model)

        # Allow arbitrary size; we control positions
        patch_tokens = self.patch_embeddings(pixel_values, interpolate_pos_encoding=True)  # (B,N,D)
        N, D = patch_tokens.shape[1], patch_tokens.shape[2]

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(B, N, -1)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            patch_tokens = patch_tokens * (1.0 - mask) + mask_tokens * mask

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B,1,D)

        valid, Hpatch, Wpatch = _valid_grid(model, pixel_values)  # (B,Hpatch,Wpatch)
        row_has = valid.any(dim=2)
        col_has = valid.any(dim=1)

        def last_true_id(x: torch.Tensor) -> int:
            if not torch.count_nonzero(x):
                return 0
            idxs = torch.arange(x.numel(), device=x.device)
            return int((x.long() * idxs).max().item())

        pos_big = torch.zeros(B, Hpatch, Wpatch, D, device=pixel_values.device, dtype=patch_tokens.dtype)
        cls_pos_all = torch.empty(B, 1, D, device=pixel_values.device, dtype=patch_tokens.dtype)

        for b in range(B):
            hs_b = last_true_id(row_has[b]) + 1
            ws_b = last_true_id(col_has[b]) + 1
            key = _pos_meta_key(hs_b, ws_b, D, ph, pw, patch_tokens.dtype, str(pixel_values.device))
            entry = _pos_tensor_cache.get(key)
            if entry is None:
                dummy = torch.zeros(1, 1 + hs_b * ws_b, D, device=pixel_values.device, dtype=patch_tokens.dtype)
                pos_small_all = self.interpolate_pos_encoding(dummy, ph * hs_b, pw * ws_b)  # (1,1+hs*ws,D)
                cls_small = pos_small_all[:, :1, :].contiguous()
                patch_small = pos_small_all[:, 1:, :].reshape(1, hs_b, ws_b, D).contiguous()
                _pos_tensor_cache[key] = (cls_small, patch_small)
                entry = (cls_small, patch_small)
            cls_small, patch_small = entry
            cls_pos_all[b : b + 1] = cls_small
            pos_big[b, :hs_b, :ws_b, :] = patch_small[0]

        pos_big = pos_big.view(B, Hpatch * Wpatch, D)  # (B,N,D)
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)      # (B,1+N,D)
        pos_cat = torch.cat([cls_pos_all, pos_big], dim=1)         # (B,1+N,D)
        tokens = tokens + pos_cat
        tokens = self.dropout(tokens)
        return tokens

    model.embeddings.forward = types.MethodType(embeddings_forward_with_invariant_pos, model.embeddings)

    # ---- encoder wrapper: attach compact pair mask to every attention module for this call ----
    orig_enc_forward = model.encoder.forward

    def enc_forward_with_pairmask(self, hidden_states, *args, **kwargs):
        # Use compact (B,1,L,L) mask stored by model.forward
        pair_mask = getattr(model, "_pre_softmax_pair_mask", None)

        if pair_mask is not None:
            for lyr in self.layer:
                attn = lyr.attention.attention  # ViTSelfAttention
                attn._pre_softmax_pair_mask = pair_mask  # type: ignore[attr-defined]

        try:
            return orig_enc_forward(hidden_states, *args, **kwargs)
        finally:
            if pair_mask is not None:
                for lyr in self.layer:
                    attn = lyr.attention.attention
                    if hasattr(attn, "_pre_softmax_pair_mask"):
                        delattr(attn, "_pre_softmax_pair_mask")

    model.encoder.forward = types.MethodType(enc_forward_with_pairmask, model.encoder)

    # ---- model.forward wrapper: compute masks, call orig, functional zero-out ----
    orig_model_forward = model.forward

    def forward_with_masks(self: ViTModel, *args, **kwargs):
        pixel_values = kwargs.get("pixel_values", None)
        if pixel_values is None and len(args) > 0:
            pixel_values = args[0]

        # Build compact pairwise keep mask: (B,1,L,L) with CLS=True, valid=True, padded=False
        valid, Hpatch, Wpatch = _valid_grid(self, pixel_values) if pixel_values is not None else (None, None, None)
        if valid is not None:
            B = valid.size(0)
            valid_flat = valid.view(B, -1)  # (B,N)
            cls = torch.ones(B, 1, dtype=torch.bool, device=valid_flat.device)
            keep = torch.cat([cls, valid_flat], dim=1)  # (B,L)
            pair = (keep[:, None, :, None] & keep[:, None, None, :])  # (B,1,L,L)
            self._pre_softmax_pair_mask = pair.to(dtype=pixel_values.dtype)  # used by encoder/attention
        else:
            self._pre_softmax_pair_mask = None

        # Run original forward (our eager kernel will pre-mask logits)
        outputs = orig_model_forward(*args, **kwargs)

        # Post-mask: zero-out outputs (functional; CUDA-safe)
        if valid is not None and hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            B = valid.size(0)
            valid_flat = valid.view(B, -1)

            fully_zero = ~valid_flat.any(dim=1)                  # (B,)
            fully_zero_vec = fully_zero[:, None, None]           # (B,1,1)
            fully_zero_col = fully_zero[:, None]                 # (B,1)

            keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=valid_flat.device), valid_flat], dim=1)
            keep_f = keep[:, :, None]  # (B,L,1)

            lhs = outputs.last_hidden_state
            lhs = torch.where(fully_zero_vec, torch.zeros_like(lhs), lhs * keep_f.to(lhs.dtype))
            outputs.last_hidden_state = lhs

            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
                new_hs = []
                for t in outputs.hidden_states:
                    t_masked = torch.where(fully_zero_vec, torch.zeros_like(t), t * keep_f.to(t.dtype))
                    new_hs.append(t_masked)
                outputs.hidden_states = tuple(new_hs)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                po = outputs.pooler_output
                po = torch.where(fully_zero_col, torch.zeros_like(po), po)
                outputs.pooler_output = po

            if hasattr(outputs, "logits") and outputs.logits is not None:
                lg = outputs.logits
                lg = torch.where(fully_zero_col, torch.zeros_like(lg), lg)
                outputs.logits = lg

        # cleanup
        if hasattr(self, "_pre_softmax_pair_mask"):
            delattr(self, "_pre_softmax_pair_mask")

        return outputs

    model.forward = types.MethodType(forward_with_masks, model)
    # Force the eager attention path so our pre-softmax mask is actually used
    model.config._attn_implementation = "eager"
    model._zero_pad_mask_patched = True


def maybe_patch_vit_model(model: Union[ViTModel, ViTForImageClassification]) -> None:
    vit = model.vit if hasattr(model, "vit") else model
    if not getattr(vit, "_zero_pad_mask_patched", False):
        patch_vit_zero_pad_mask(vit)


def is_vit_model(model: nn.Module) -> bool:
    return isinstance(model, ViTModel) or isinstance(getattr(model, "vit", None), ViTModel)