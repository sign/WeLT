"""Image encoders from HuggingFace vision backbones (ViT, DeiT, DINOv2/v3, CLIP, SigLIP, ...), over rendered words."""
import inspect

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel

from welt.processor import PATCH_SIZE


def unpatchify(patches: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """(N, rows * cols, p*p*3) uint8 row-major patches -> (N, 3, rows * p, cols * p) images"""
    n = patches.size(0)
    images = patches[:, :rows * cols].reshape(n, rows, cols, PATCH_SIZE, PATCH_SIZE, 3)
    return images.permute(0, 5, 1, 3, 2, 4).reshape(n, 3, rows * PATCH_SIZE, cols * PATCH_SIZE)


def initialize_patch_embeddings(model: nn.Module):
    """Initialize patch embedding convolutions like linear layers (xavier uniform), as PIXEL and ViT-MAE do.
    https://github.com/xplip/pixel/blob/main/src/pixel/models/pixel/modeling_pixel.py#L573"""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight.data.view(module.weight.size(0), -1))


class HFImageEncoder(nn.Module):
    """Encodes each word image with a HF vision backbone, at the backbone's patch size, using its pooled output."""

    def __init__(self, name_or_path: str, pretrained: bool, trust_remote_code: bool = False):
        super().__init__()
        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        if pretrained:
            model = AutoModel.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        else:
            model = AutoModel.from_config(config, trust_remote_code=trust_remote_code)
        self.model = getattr(model, "vision_model", model)  # e.g. the vision tower of CLIP / SigLIP
        if not pretrained:
            initialize_patch_embeddings(self.model)
        vision_config = getattr(config, "vision_config", config)
        self.hidden_size = vision_config.hidden_size
        self.patch_size = vision_config.patch_size

        processor = AutoImageProcessor.from_pretrained(name_or_path, trust_remote_code=trust_remote_code)
        self.register_buffer("mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 3, 1, 1), persistent=False)
        parameters = inspect.signature(self.model.forward).parameters
        self.forward_kwargs = {"interpolate_pos_encoding": True} if "interpolate_pos_encoding" in parameters else {}

    def save_pretrained(self, path: str, name_or_path: str):
        self.model.save_pretrained(path)
        AutoImageProcessor.from_pretrained(name_or_path).save_pretrained(path)

    def forward(self, patches: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
        """(N, P, 768) uint8 patches, (N, 2) patch rows and columns of each image -> (N, H)"""
        embeds = None
        # Images of the same size are batched together
        unique_shapes, groups = torch.unique(shapes, dim=0, return_inverse=True)
        for group, (rows, cols) in enumerate(unique_shapes.tolist()):
            indices = (groups == group).nonzero().squeeze(1)
            images = unpatchify(patches[indices], rows, cols).float() / 255
            if self.patch_size != PATCH_SIZE:
                images = F.interpolate(images, size=(rows * self.patch_size, cols * self.patch_size), mode="bilinear")
            images = ((images - self.mean) / self.std).to(self.model.dtype)
            outputs = self.model(pixel_values=images, **self.forward_kwargs)
            pooled = getattr(outputs, "pooler_output", None)
            pooled = outputs.last_hidden_state[:, 0] if pooled is None else pooled
            if embeds is None:
                embeds = pooled.new_empty(len(patches), pooled.size(-1))
            embeds[indices] = pooled
        return embeds


def is_vision_model(config) -> bool:
    """Whether a HF config is a vision backbone (rather than a language model, used as a patch transformer)."""
    return hasattr(getattr(config, "vision_config", config), "patch_size")
