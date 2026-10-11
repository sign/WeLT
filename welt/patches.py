"""Words rendered as images, as 16x16 patches: from the renders (in the processor) to their embeddings (in the image
encoder, and for its vLLM engine in inference). Torch only."""
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

PATCH_SIZE = 16  # pixel_renderer renders lines of 16px height, widths rounded to 16px
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3
MAX_PATCH_POSITION = 256  # Rows and columns of patches beyond it share its position embedding


def patchify(image, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """(H, W, C) uint8 render -> (H/p * W/p, p*p*C) uint8 patches, row-major."""
    image = torch.from_numpy(image)
    h, w, c = image.shape
    patches = image.reshape(h // patch_size, patch_size, w // patch_size, patch_size, c)
    return patches.permute(0, 2, 1, 3, 4).reshape(-1, patch_size * patch_size * c)


def patch_positions(shapes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 2) rows and columns of patches of N images -> the row and column of each of their row-major patches,
    packed: two (total patches,) tensors, capped at MAX_PATCH_POSITION - 1."""
    counts = shapes.prod(dim=-1)
    within = torch.arange(int(counts.sum()), device=shapes.device) - torch.repeat_interleave(
        F.pad(counts.cumsum(0), (1, 0))[:-1], counts)
    columns = torch.repeat_interleave(shapes[:, 1], counts)
    return (within // columns).clamp(max=MAX_PATCH_POSITION - 1), (within % columns).clamp(max=MAX_PATCH_POSITION - 1)


class PatchEmbedding(nn.Module):
    """uint8 16x16 RGB patches -> CLS + linear patch embeddings, plus the embeddings of their row and column in the
    image (images of words in several rows, e.g. SignWriting, are 2D; the transformer only sees a 1D sequence)."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(PATCH_DIM, dim)
        self.cls = nn.Parameter(torch.randn(dim) * 0.02)
        self.rows = nn.Parameter(torch.randn(MAX_PATCH_POSITION, dim) * 0.02)
        self.columns = nn.Parameter(torch.randn(MAX_PATCH_POSITION, dim) * 0.02)
        nn.init.xavier_uniform_(self.proj.weight)  # Like PIXEL / ViT-MAE patch embeddings

    def forward(self, patches: torch.Tensor, shapes: torch.Tensor) -> torch.Tensor:
        """(total patches, 768) packed patches, (N, 2) rows and columns of patches of each image -> (total + N, H),
        each image's CLS followed by its patch embeddings"""
        rows, columns = patch_positions(shapes)
        embeds = self.proj(patches.to(self.proj.weight.dtype) / 127.5 - 1) + self.rows[rows] + self.columns[columns]
        lengths = shapes.prod(dim=-1) + 1
        is_cls = torch.zeros(len(embeds) + len(lengths), dtype=torch.bool, device=embeds.device)
        is_cls[F.pad(lengths.cumsum(0), (1, 0))[:-1]] = True
        hidden = embeds.new_empty(len(is_cls), embeds.size(-1))
        hidden[is_cls] = self.cls.to(embeds.dtype)
        hidden[~is_cls] = embeds
        return hidden
