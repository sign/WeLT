from itertools import chain

import torch
from transformers import ViTForImageClassification, ViTModel
from transformers.image_transforms import group_images_by_shape, reorder_images

from welt.collator import stack_pad_tensors
from welt.vision.navit import NaViTModel
from welt.vision.vision_utils import encode_images as utils_encode_images

# Define a type alias for image encoders, for type inference and clarity
# However, every image encoder should be supported
ImageEncoder = ViTModel | ViTForImageClassification | NaViTModel


def encode_padded_images(image_encoder: ImageEncoder,
                         input_images: torch.Tensor) -> torch.Tensor:
    """Image encoder should accept variable size images and return consistent embeddings."""
    B, L, *_ = input_images.shape  # noqa: N806

    linear_images = input_images.view(B * L, *input_images.shape[2:])
    embeds = encode_images_batch(image_encoder=image_encoder, images=linear_images)
    embeds = embeds.view(B, L, -1)
    return embeds


def encode_images(image_encoder: ImageEncoder,
                  input_images: torch.Tensor,
                  input_images_dimensions: torch.Tensor) -> torch.Tensor:
    """Image encoder should accept variable size images and return consistent embeddings."""
    B, L, *_ = input_images.shape  # noqa: N806

    # Recreate as list of lists if input is a nested tensor, cropping padding from each image
    nested_images = [
        [img[:, :h, :w] for img, (h, w) in zip(images, dims, strict=False) if h > 0 and w > 0]
        for images, dims in zip(input_images, input_images_dimensions, strict=False)
    ]

    # Flatten images
    all_images = list(chain.from_iterable(nested_images))

    # Re-arrange the embeddings to match the original batch structure
    embeddings = encode_images_group(image_encoder=image_encoder, images=all_images)

    # Restructure the flat embeddings back into the nested format
    embeds = embeddings.new_zeros((B, L, embeddings.size(-1)))

    lengths = torch.tensor([len(inner) for inner in nested_images], device=embeddings.device)
    row_idx = torch.repeat_interleave(torch.arange(B, device=embeddings.device), lengths)
    col_idx = torch.cat([torch.arange(n, device=embeddings.device) for n in lengths])

    # One fused write instead of a Python loop
    embeds.index_put_((row_idx, col_idx), embeddings, accumulate=False)

    return embeds


def encode_images_batch(image_encoder: ImageEncoder,
                        images: list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(images, list):
        images = stack_pad_tensors(images)

    # Encode images using the image encoder
    return utils_encode_images(image_encoder, images)


def encode_images_sequentially(image_encoder: ImageEncoder,
                               images: list[torch.Tensor]) -> torch.Tensor:
    encoded_images = [encode_images_batch(image_encoder, image.unsqueeze(0)) for image in images]
    return torch.cat(encoded_images, dim=0)


def encode_images_group(image_encoder: ImageEncoder,
                        images: list[torch.Tensor]) -> torch.Tensor:
    if isinstance(image_encoder, NaViTModel):
        # NaViT can handle variable size images natively!
        return utils_encode_images(image_encoder, images)

    grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=False)

    # Encode each group separately
    encoded_groups = {size: encode_images_batch(image_encoder=image_encoder, images=group)
                      for size, group in grouped_images.items()}

    # Re-arrange the encoded images to match the original order
    rearranged_images = reorder_images(encoded_groups, grouped_images_index)

    return torch.stack(rearranged_images)
