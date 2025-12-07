# Vision

Working with Image encoder models is less standardized than working with text models.

In this repository, it is recommended to use, in order of preference:

1. [NaViT](https://github.com/lucidrains/vit-pytorch#navit) models, since they natively support variable-sized inputs.
2. Models that implement
   [ViTModel](https://github.com/huggingface/transformers/blob/main/src/transformers/models/vit/modeling_vit.py),
   since we have a patch to handle variable-sized inputs.
3. Any other image encoder model from `transformers`, in which we have utilities to group images by size and batch them,
   avoiding padding artifacts, with a lot of compute overhead.

### Vision Utils [./vision_utils.py](./vision_utils.py)

This utility is under a PR [#40587](https://github.com/huggingface/transformers/pull/40587) to `transformers`.

This utility addresses the lack of standardization across vision transformer models and image encoders.
Unlike text models, where core attributes like `hidden_size` or `vocab_size` are consistently exposed,
vision models vary widely in how they represent hidden dimensions, pooling strategies, and forward-pass arguments.
Some models store the hidden size in `config.hidden_size`, others in `vision_config.hidden_size`,
`neck_hidden_sizes`, or `hidden_sizes`.
Similarly, handling positional encoding interpolation or pooling hidden states differs across architectures.
Without a unified interface, downstream code must implement ad-hoc checks and case-by-case logic.
This module centralizes those edge cases into robust, cached utilities that reliably determine encoder dimensions,
construct forward arguments, and pool features, providing a consistent abstraction for working with
heterogeneous vision backbones.

### Batch Image Encoder [./batch_image_encoder.py](./batch_image_encoder.py)

This utility tackles the challenge that most vision transformer encoders can only process batches of
equally sized images, which makes handling variable-sized inputs difficult and error-prone.
Padding smaller images to match the largest one in a batch leads to wasted computation and, worse, changes the
encoder’s output in subtle ways since positional embeddings and convolutions can be affected by padded regions.

To resolve this, the utility reorganizes inputs by cropping them to their true dimensions,
grouping images of the same shape, and then running those groups through the encoder in efficient batches.
After encoding, the results are reordered and reassembled back into the original nested batch structure,
ensuring consistency and eliminating the artifacts introduced by padding.
This provides a reliable and scalable way to extract embeddings from heterogeneous image sets while still
leveraging batch parallelism where possible.
