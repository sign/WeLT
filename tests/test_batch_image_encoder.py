from functools import cache

import pytest
import torch
from transformers import AutoConfig, AutoModel, DINOv3ConvNextConfig, DINOv3ViTConfig

from welt.collator import stack_pad_tensors_list
from welt.vision.batch_image_encoder import (
    encode_images,
    encode_images_batch,
    encode_images_group,
    encode_images_sequentially,
)
from welt.vision.navit import NaViTConfig
from welt.vision.vision_utils import image_encoder_size

MODELS = {
    "custom-navit": 512,
    "WinKawaks/vit-tiny-patch16-224": 192,
    "microsoft/swinv2-tiny-patch4-window16-256": 768,
    "google/vit-base-patch16-224": 768,
    "microsoft/resnet-18": 512,
    "apple/mobilevit-xx-small": 320,
    "facebook/dinov3-vits16-pretrain-lvd1689m": 384,
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m": 768,
}

MODEL_NAMES = list(MODELS.keys())


def images_dimensions(images: list[list[torch.Tensor]]) -> torch.Tensor:
    tensors = [
        [torch.tensor([img.shape[-2], img.shape[-1]], dtype=torch.long) for img in batch]
        for batch in images
    ]
    return stack_pad_tensors_list(tensors)


@cache
def image_encoder(model_name):
    if model_name == "custom-navit":
        config = NaViTConfig()
    elif model_name == "facebook/dinov3-vits16-pretrain-lvd1689m":
        # We only test randomly initialized models. Local configs keep this
        # architecture coverage independent of gated Hub access in fork CI.
        config = DINOv3ViTConfig(hidden_size=384, num_hidden_layers=2, num_attention_heads=6)
    elif model_name == "facebook/dinov3-convnext-tiny-pretrain-lvd1689m":
        config = DINOv3ConvNextConfig(hidden_sizes=[96, 192, 384, 768], depths=[1, 1, 1, 1])
    else:
        config = AutoConfig.from_pretrained(model_name)

    model = AutoModel.from_config(config)
    model.eval()
    return model


def create_random_image(height, width, channels=3):
    """Create a random image tensor."""
    return torch.randn(channels, height, width)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_image_encoder_size(model_name):
    """Test that image_encoder_size returns the expected hidden size for each model."""
    model = image_encoder(model_name)
    expected_size = MODELS[model_name]
    actual_size = image_encoder_size(model)
    assert actual_size == expected_size, f"Expected {expected_size}, got {actual_size} for {model_name}"


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_encode_images_single_image(model_name):
    """Test encode_images with a single image."""
    model = image_encoder(model_name)

    # Single batch with single image
    images = [[create_random_image(64, 64)]]
    dimensions = images_dimensions(images)
    images = stack_pad_tensors_list(images)

    embeddings = encode_images(model, images, dimensions)

    assert embeddings.shape == (1, 1, embeddings.shape[2])


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_encode_images_deterministic(model_name):
    model = image_encoder(model_name)

    images = [[create_random_image(64, 64)]]
    dimensions = images_dimensions(images)
    images = stack_pad_tensors_list(images)

    embeddings1 = encode_images(model, images, dimensions)
    embeddings2 = encode_images(model, images, dimensions)

    # Results should be identical
    assert torch.equal(embeddings1, embeddings2)


def test_encode_non_equal_lists():
    model = image_encoder("WinKawaks/vit-tiny-patch16-224")

    img = create_random_image(32, 32)
    images = [
        [img],
        [img, img]
    ]
    dimensions = images_dimensions(images)
    images = stack_pad_tensors_list(images)

    # Test encoding
    embedding = encode_images(model, images, dimensions)
    assert embedding.shape == (2, 2, 192)

    assert torch.equal(embedding[0][0], embedding[1][0])
    assert torch.equal(embedding[1][0], embedding[1][1])
    assert torch.equal(embedding[0][1], torch.zeros_like(embedding[0][1]))


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_encode_images_batched_or_sequential(model_name):
    """Make sure the batch implementation and the sequential implementation return the same result"""
    model = image_encoder(model_name)

    images = [create_random_image(64, 64), create_random_image(64, 64)]

    embeddings1 = encode_images_batch(model, images)
    embeddings2 = encode_images_sequentially(model, images)

    assert embeddings1.shape == embeddings2.shape
    assert torch.allclose(embeddings1, embeddings2, atol=1e-5)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_encode_images_group_or_sequential(model_name):
    """Make sure the batch implementation and the sequential implementation return the same result"""
    model = image_encoder(model_name)

    images = [
        create_random_image(64, 64),
        create_random_image(64, 32),
        create_random_image(64, 64)
    ]

    embeddings1 = encode_images_sequentially(model, images)
    embeddings2 = encode_images_group(model, images)

    assert embeddings1.shape == embeddings2.shape
    assert torch.allclose(embeddings1, embeddings2, atol=1e-5)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_encode_images_basic(model_name):
    """Test basic functionality of encode_images with different image sizes."""
    model = image_encoder(model_name)

    # Create images of different sizes
    images = [
        [create_random_image(32, 32), create_random_image(64, 64)],
        [create_random_image(32, 64), create_random_image(128, 32)]
    ]
    dimensions = images_dimensions(images)
    images = stack_pad_tensors_list(images)

    # Test encoding
    embeddings = encode_images(model, images, dimensions)

    # Check output shape
    assert embeddings.shape[0] == 2  # batch size
    assert embeddings.shape[1] == 2  # sequence length (number of images per batch)
    assert embeddings.shape[2] > 0  # embedding dimension


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("image_size", [(32, 32), (32, 64), (32, 128), (64, 128)])
def test_encode_images_different_models_and_sizes(model_name, image_size):
    """Test encode_images with different model types and image sizes."""
    model = image_encoder(model_name)
    height, width = image_size

    # Create test image of specified size
    images = [[create_random_image(height, width)]]
    dimensions = images_dimensions(images)
    images = stack_pad_tensors_list(images)

    embeddings = encode_images(model, images, dimensions)

    assert embeddings.shape[0] == 1  # batch size
    assert embeddings.shape[1] == 1  # sequence length
    assert embeddings.shape[2] > 0  # embedding dimension


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_encode_images_batch_vs_individual(model_name):
    """Test that batch processing gives same results as individual processing for different models."""
    model = image_encoder(model_name)

    # Create test images of different sizes
    img1 = create_random_image(32, 32)
    img2 = create_random_image(64, 64)
    img3 = create_random_image(32, 128)
    img4 = create_random_image(64, 128)

    # Batch processing - all images at once
    batch_images = [[img1, img2, img3, img4]]
    dimensions = images_dimensions(batch_images)
    batch_images = stack_pad_tensors_list(batch_images)

    batch_embeddings = encode_images(model, batch_images, dimensions)

    # Individual processing - each image separately
    individual_embeddings = []
    for img in [img1, img2, img3, img4]:
        sub_batch = [[img]]

        img_embeddings = encode_images(model,
                                       stack_pad_tensors_list(sub_batch),
                                       images_dimensions(sub_batch))
        individual_embeddings.append(img_embeddings.squeeze(0))

    pairs = zip(individual_embeddings, batch_embeddings[0], strict=False)
    for i, (individual_embedding, batch_embedding) in enumerate(pairs):
        print(individual_embedding.shape, batch_embedding.shape)
        # Compare results (allowing for small numerical differences)
        assert torch.allclose(batch_embedding, individual_embedding, atol=1e-3), \
            f"Batch vs individual results ({i}) differ for model {model_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
