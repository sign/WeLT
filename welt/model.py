import logging
import warnings
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    GenerationConfig,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.auto.auto_factory import _get_model_class
from utf8_tokenizer.embeddings import patch_embedding_layers
from utf8_tokenizer.tokenizer import UTF8Tokenizer
from words_segmentation.pretokenizer import WordStoppingCriteria

from welt.config import WordLatentTransformerConfig
from welt.noop import NoopConfig
from welt.processor import TextImageProcessor
from welt.vision.batch_image_encoder import encode_images
from welt.vision.vision_utils import image_encoder_size

logger = logging.getLogger(__name__)


def model_from_config(config: PretrainedConfig,
                      cls: type[PreTrainedModel],
                      dtype: torch.dtype = torch.float32,
                      load_pretrained: bool = False,
                      attn_implementation=None) -> PreTrainedModel:
    """Load pretrained model or initialize from config with new weights."""

    # Override attn_implementation if not supported
    if attn_implementation in ["flash_attention_2", "flash_attention_3"]:
        resolved_class = _get_model_class(config, cls._model_mapping)

        if not getattr(resolved_class, "_supports_flash_attn", False):
            print(f"Model {resolved_class.__name__} does not support flash_attention, using default attention.")
            attn_implementation = None

    if load_pretrained:
        name_or_path = getattr(config, "_name_or_path", "")
        if name_or_path:
            print(f"Loading pretrained model from {name_or_path}")
            return cls.from_pretrained(name_or_path,
                                       config=config,
                                       dtype=dtype,
                                       attn_implementation=attn_implementation)

    return cls.from_config(config, dtype=dtype, attn_implementation=None)


def set_module_trainable(module, trainable: bool = True):
    for p in module.parameters():
        p.requires_grad = trainable


class WordLatentTransformer(PreTrainedModel):
    config_class = WordLatentTransformerConfig

    _supports_flash_attn = True

    def __init__(self, config: WordLatentTransformerConfig,
                 load_pretrained: bool = False,
                 attn_implementation=None):
        super().__init__(config=config)

        is_image_encoder = not isinstance(config.image_encoder, NoopConfig)
        is_bytes_encoder = not isinstance(config.bytes_encoder, NoopConfig)

        assert is_bytes_encoder or is_image_encoder, \
            "At least one encoder must be provided"

        if not is_bytes_encoder or not is_image_encoder:
            warnings.warn("Image encoder and bytes encoder are not provided, setting modality_dropout to 0.0",
                          stacklevel=2)
            config.modality_dropout = 0.0

        # Image Encoder
        if is_image_encoder:
            self.image_encoder = model_from_config(config.image_encoder, AutoModel,
                                                   config.dtype, load_pretrained, attn_implementation)
            self.image_encoder_dim = image_encoder_size(self.image_encoder)
        else:
            self.image_encoder = None
            self.image_encoder_dim = 0

        # Bytes Encoder
        if is_bytes_encoder:
            self.bytes_encoder = model_from_config(config.bytes_encoder, AutoModelForMaskedLM,
                                                   config.dtype, load_pretrained, attn_implementation)
            self.bytes_encoder.resize_token_embeddings(config.num_tokens, pad_to_multiple_of=8)
            self.bytes_encoder.cls = self.bytes_encoder.decoder = torch.nn.Identity()  # delete the decoder head
            self.bytes_encoder.get_output_embeddings = lambda: None  # bytes encoder no longer has output embeddings
            self.bytes_encoder_dim = self.bytes_encoder.config.hidden_size
            patch_embedding_layers(self.bytes_encoder)
        else:
            self.bytes_encoder = None
            self.bytes_encoder_dim = 0

        # Latent Transformer
        # TODO: not passing attn_implementation since we have 4D attention masks which error.
        #       https://github.com/Dao-AILab/flash-attention/issues/1857
        self.latent_transformer = model_from_config(config.latent_transformer, AutoModelForCausalLM,
                                                    config.dtype, load_pretrained, None)
        self.latent_transformer.resize_token_embeddings(0, pad_to_multiple_of=1)
        model_dim = self.latent_transformer.config.hidden_size

        # Small Language Model (Bytes Decoder)
        # Note: The bytes decoder uses on-the-fly sequence packing to reduce padding waste.
        # Multiple short words are packed into single sequences before decoding, which can
        # significantly reduce computation (e.g., 58% fewer tokens in typical cases).
        self.bytes_decoder = model_from_config(config.bytes_decoder, AutoModelForCausalLM,
                                               config.dtype, load_pretrained, attn_implementation)
        self.bytes_decoder.resize_token_embeddings(config.num_tokens, pad_to_multiple_of=8)
        patch_embedding_layers(self.bytes_decoder)
        bytes_decoder_dim = self.bytes_decoder.config.hidden_size

        # Mapping layers
        encoder_dim = self.bytes_encoder_dim + self.image_encoder_dim
        self.encoder_mapping = nn.Linear(encoder_dim, model_dim, dtype=self.latent_transformer.dtype)
        self.decoder_mapping = nn.Linear(model_dim, bytes_decoder_dim, dtype=self.bytes_decoder.dtype)

        # Post init
        self.post_init()

    def _should_drop_modality(self):
        if not self.training or self.config.modality_dropout == 0:
            return False
        return torch.rand(1).item() < self.config.modality_dropout

    def encode_images(self,
                      input_images: torch.Tensor,
                      input_images_dimensions: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
        """
        Args:
            input_images: Tensor of nested images, where each inner list contains images for one sample
            input_images_dimensions: (BATCH, LENGTH, 2) - Original dimensions of each image (height, width)
            device: Device to move the tensors to
        Returns:
            torch.Tensor: (BATCH, LENGTH, HIDDEN_DIM) - Image embeddings
        """

        if self.image_encoder is None or self._should_drop_modality():
            # If image encoder is None, return zeros
            B, L, *_, = input_images.shape  # noqa: N806
            dtype = getattr(self.image_encoder, "dtype", self.latent_transformer.dtype)
            return torch.zeros((B, L, self.image_encoder_dim), device=device, dtype=dtype)

        return encode_images(self.image_encoder,
                             input_images=input_images,
                             input_images_dimensions=input_images_dimensions)

    def encode_texts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (BATCH, LENGTH, INPUT_TOKENS)
            attention_mask: (BATCH, LENGTH, INPUT_TOKENS)
        Returns:
            torch.Tensor: (BATCH, LENGTH, HIDDEN_DIM) - Text embeddings
        """
        B, L, T = input_ids.shape  # noqa: N806

        # If bytes encoder is None, return zeros
        if self.bytes_encoder is None or self._should_drop_modality():
            dtype = getattr(self.bytes_encoder, "dtype", self.latent_transformer.dtype)
            return torch.zeros(B, L, self.bytes_encoder_dim, device=input_ids.device, dtype=dtype)

        # Flatten batch and length dimensions
        input_ids = input_ids.view(B * L, T)
        attention_mask = attention_mask.view(B * L, T)
        # Encode texts using the bytes decoder as encoder
        text_outputs = self.bytes_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        text_embeds = text_outputs.hidden_states[-1]

        # Apply attention mask to text embeddings
        text_embeds *= attention_mask.unsqueeze(-1)

        # Pool sequence dimension weighted by attention mask
        sequence_lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
        text_embeds = text_embeds.sum(dim=1) / sequence_lengths
        return text_embeds.view(B, L, -1)

    def encode_input(self,
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     input_images: torch.Tensor,
                     input_images_dimensions: torch.Tensor):
        embeds = []
        if self.image_encoder_dim > 0:
            image_embeds = self.encode_images(input_images, input_images_dimensions, device=input_ids.device)
            logger.debug("Image embeddings shape: %s", image_embeds.shape)
            embeds.append(image_embeds)

        if self.bytes_encoder_dim > 0:
            text_embeds = self.encode_texts(input_ids, attention_mask)
            logger.debug("Text embeddings shape: %s", text_embeds.shape)
            embeds.append(text_embeds)

        assert len(embeds) > 0, "At least one type of encoder must be provided"

        concatenated_embeds = torch.cat(embeds, dim=-1)
        logger.debug("Concatenated embeddings shape: %s", concatenated_embeds.shape)

        # # For dropout, scale embedding by number of zeros in the concatenated embeddings
        # if len(embeds) > 1 and self.training and self.config.modality_dropout > 0:
        #     percent_zeros = (concatenated_embeds == 0).sum(dim=-1) / concatenated_embeds.numel()
        #     scale_factor = 1.0 / (1.0 - percent_zeros)
        #     concatenated_embeds *= scale_factor.clamp(min=1).unsqueeze(-1)

        return self.encoder_mapping(concatenated_embeds)

    def _num_words_per_datum(self, attention_mask: torch.Tensor) -> torch.Tensor:
        return attention_mask.sum(dim=-1).gt(0).sum(dim=-1)

    def forward(self,
                input_ids: torch.Tensor,
                input_attention_mask: torch.Tensor,
                attention_mask: torch.Tensor,
                input_images: torch.Tensor,
                input_images_dimensions: torch.Tensor,
                position_ids: torch.Tensor | None = None,
                labels_input: torch.Tensor | None = None,
                labels_attention_mask: torch.Tensor | None = None,
                labels_output: torch.Tensor | None = None):
        """
        Args:
            input_ids: (BATCH, LENGTH, INPUT_TOKENS)
            input_attention_mask: Attention within a word (BATCH, LENGTH, INPUT_TOKENS)
            attention_mask: Attention across words (BATCH, 1, LENGTH, LENGTH)
            input_images: (BATCH, LENGTH, CHANNELS, HEIGHT, WIDTH)
            input_images_dimensions: (BATCH, LENGTH, 2)
            position_ids: (BATCH, LENGTH) - Position IDs for latent transformer (useful for sequence packing)
            labels_input: (BATCH, LENGTH, OUTPUT_TOKENS) - Input tokens for bytes decoder
            labels_attention_mask: (BATCH, LENGTH, OUTPUT_TOKENS) - Attention mask for labels
            labels_output: (BATCH, LENGTH, OUTPUT_TOKENS) - Target tokens for language modeling
        """
        # Embed images and texts
        mapped_embeds = self.encode_input(input_ids, input_attention_mask, input_images, input_images_dimensions)

        # Process the sequence with the latent transformer
        latent_outputs = self.latent_transformer(
            inputs_embeds=mapped_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        latent_vectors = latent_outputs.hidden_states[-1]  # (B, L, hidden_dim)

        logger.debug("Latent vectors shape: %s", latent_vectors.shape)
        mapped_embeds = self.decoder_mapping(latent_vectors)
        logger.debug("Mapped embeddings shape: %s", mapped_embeds.shape)

        # Decode the latent vectors to bytes using parallel causal decoding
        logits = self.parallel_causal_decode(mapped_embeds, labels_input, labels_attention_mask)

        loss = None
        if labels_output is not None:
            # Flatten dimensions for cross entropy loss
            # logits: (B, L, T, vocab_size) -> (B*L*T, vocab_size)
            # labels_output: (B, L, T) -> (B*L*T,)
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = labels_output.reshape(-1)

            # Compute cross entropy loss
            loss = torch.nn.functional.cross_entropy(flat_logits, flat_labels, ignore_index=self.config.pad_token_id)

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=(mapped_embeds,),
            attentions=None
        )

    def _pack_sequences_for_decoding(self,
                                      latent_vectors_flat: torch.Tensor,
                                      target_embeds: torch.Tensor,
                                      target_mask_flat: torch.Tensor,
                                      max_packed_length: int) -> tuple:
        """
        Pack multiple short sequences into longer sequences to reduce padding waste.

        Args:
            latent_vectors_flat: (N, hidden_dim) - latent vectors for each word
            target_embeds: (N, T, embed_dim) - embeddings for target tokens
            target_mask_flat: (N, T) - attention mask for target tokens
            max_packed_length: maximum length for packed sequences

        Returns:
            tuple of:
                - packed_embeds: list of packed embedding tensors
                - packed_masks: list of packed attention masks
                - unpack_indices: list of (pack_idx, start_pos, end_pos) for unpacking
        """
        N, T, embed_dim = target_embeds.shape  # noqa: N806
        device = target_embeds.device

        # Calculate actual length of each sequence (including latent vector position)
        seq_lengths = target_mask_flat.sum(dim=1).long()  # (N,)
        # Add 1 for the latent vector position
        total_lengths = seq_lengths + 1  # (N,)

        packed_embeds = []
        packed_masks = []
        unpack_indices = []

        current_pack_embeds = []
        current_pack_masks = []
        current_length = 0

        for i in range(N):
            seq_len = total_lengths[i].item()
            
            # Skip empty sequences (though this should not happen in practice)
            if seq_len <= 0:
                continue

            # If adding this sequence would exceed max_packed_length, start a new pack
            if current_length > 0 and current_length + seq_len > max_packed_length:
                # Finalize current pack
                packed_embeds.append(torch.cat(current_pack_embeds, dim=0))
                packed_masks.append(torch.cat(current_pack_masks, dim=0))
                current_pack_embeds = []
                current_pack_masks = []
                current_length = 0

            # Add sequence to current pack
            # Get latent vector and target embeddings for this sequence
            latent_vec = latent_vectors_flat[i].unsqueeze(0)  # (1, hidden_dim)
            
            # Only add target embeddings if sequence has actual tokens
            if seq_lengths[i] > 0:
                seq_target_embeds = target_embeds[i, :seq_lengths[i]]  # (seq_len, embed_dim)
                seq_combined = torch.cat([latent_vec, seq_target_embeds], dim=0)  # (1 + seq_len, embed_dim)
            else:
                seq_combined = latent_vec  # Just the latent vector if no tokens

            # Create mask
            seq_mask = torch.ones(seq_len, device=device)

            # Record unpacking information
            start_pos = current_length
            end_pos = current_length + seq_len
            unpack_indices.append((len(packed_embeds), start_pos, end_pos))

            current_pack_embeds.append(seq_combined)
            current_pack_masks.append(seq_mask)
            current_length += seq_len

        # Finalize last pack
        if current_pack_embeds:
            packed_embeds.append(torch.cat(current_pack_embeds, dim=0))
            packed_masks.append(torch.cat(current_pack_masks, dim=0))

        return packed_embeds, packed_masks, unpack_indices

    def _unpack_logits(self,
                       packed_logits_list: list[torch.Tensor],
                       unpack_indices: list[tuple],
                       original_shape: tuple) -> torch.Tensor:
        """
        Unpack logits from packed sequences back to original shape.

        Args:
            packed_logits_list: list of logits tensors from packed sequences
            unpack_indices: list of (pack_idx, start_pos, end_pos) for unpacking
            original_shape: (N, T) shape to restore to

        Returns:
            torch.Tensor: (N, T, vocab_size) - unpacked logits
        """
        N, T = original_shape  # noqa: N806
        vocab_size = packed_logits_list[0].shape[-1]
        device = packed_logits_list[0].device

        # Initialize output tensor with zeros (will be ignored by loss due to padding)
        output_logits = torch.zeros(N, T, vocab_size, device=device, dtype=packed_logits_list[0].dtype)

        for i, (pack_idx, start_pos, end_pos) in enumerate(unpack_indices):
            packed_logits = packed_logits_list[pack_idx]
            # Extract logits for this sequence (skip first position which is latent vector)
            seq_logits = packed_logits[start_pos + 1:end_pos]  # (seq_len - 1, vocab_size)
            seq_len = seq_logits.shape[0]
            # Copy to output
            output_logits[i, :seq_len] = seq_logits

        return output_logits

    def parallel_causal_decode(self,
                               latent_vectors: torch.Tensor,
                               target_ids: torch.Tensor,
                               target_mask: torch.Tensor) -> torch.Tensor:
        """
        Parallel causal decoding with word-level vectors prepended to character sequences.
        Uses on-the-fly packing to reduce padding waste when decoding multiple short words.

        Args:
            latent_vectors: (B, L, hidden_dim) - latent representations for each word
            target_ids: (B, L, T) - target token IDs for each word
            target_mask: (B, L, T) - attention mask for target tokens

        Returns:
            torch.Tensor: (B, L, T, vocab_size) - logits for each token in each word
        """
        B, L, hidden_dim = latent_vectors.shape  # noqa: N806
        _, _, T = target_ids.shape  # noqa: N806

        # Step 1: Reshape inputs from [B, L, T] to [B*L, T]
        target_ids_flat = target_ids.view(B * L, T)  # [B*L, T]
        target_mask_flat = target_mask.view(B * L, T)  # [B*L, T]
        latent_vectors_flat = latent_vectors.view(B * L, hidden_dim)  # [B*L, hidden_dim]

        # Step 2: Get embeddings for target tokens
        embed_layer = self.bytes_decoder.get_input_embeddings()
        target_embeds = embed_layer(target_ids_flat)  # [B*L, T, embed_dim]

        # Step 3: Pack sequences to reduce padding
        # Maximum packed sequence length - conservative estimate to fit in memory
        # We use T as the baseline since that's the current max word length
        max_packed_length = T * 2  # Allow packing of ~2 average words per sequence

        packed_embeds, packed_masks, unpack_indices = self._pack_sequences_for_decoding(
            latent_vectors_flat, target_embeds, target_mask_flat, max_packed_length
        )

        # Step 4: Process each packed sequence through the decoder
        packed_logits_list = []
        for pack_embeds, pack_mask in zip(packed_embeds, packed_masks, strict=True):
            # Add batch dimension
            pack_embeds = pack_embeds.unsqueeze(0)  # (1, pack_len, embed_dim)
            pack_mask = pack_mask.unsqueeze(0)  # (1, pack_len)

            # Pass through bytes decoder
            outputs = self.bytes_decoder(
                inputs_embeds=pack_embeds,
                attention_mask=pack_mask,
                output_hidden_states=False
            )

            # Extract logits
            packed_logits_list.append(outputs.logits.squeeze(0))  # (pack_len, vocab_size)

        # Step 5: Unpack logits back to original shape
        char_logits = self._unpack_logits(packed_logits_list, unpack_indices, (B * L, T))

        # Step 6: Reshape back to [B, L, T, vocab_size]
        logits = char_logits.view(B, L, T, -1)

        return logits

    def freeze_pretrained_models(self):
        """Freeze everything, then enable just the requested submodules."""
        set_module_trainable(self, False)

        # Enable newly created embeddings, LM head, mapping layers
        if self.bytes_encoder is not None:
            set_module_trainable(self.bytes_encoder.get_input_embeddings())

        set_module_trainable(self.bytes_decoder.get_input_embeddings())
        set_module_trainable(self.bytes_decoder.get_output_embeddings())

        set_module_trainable(self.encoder_mapping)
        set_module_trainable(self.decoder_mapping)

    def unfreeze(self):
        """Unfreeze everything."""
        set_module_trainable(self, True)


class WordLatentTransformerForCausalLM(WordLatentTransformer, GenerationMixin):

    def _generate_latents(self, latent_past_key_values, encoded_input: torch.Tensor,
                          attention_mask: torch.Tensor,
                          num_words: torch.Tensor) -> tuple[torch.Tensor, Any, torch.Tensor]:
        latent_output = self.latent_transformer(
            inputs_embeds=encoded_input,
            attention_mask=attention_mask,
            # TODO: past_key_values can improve efficiency, but currently fails tests.
            # past_key_values=latent_past_key_values,
            use_cache=True,
            output_hidden_states=True
        )

        # Get the last latent state for the new word
        latents = latent_output.hidden_states[-1]  # (B, L, hidden_dim)

        assert len(latents.shape) == 3, "Latents should be of shape (B, L, latent_dim)"

        batch_indices = torch.arange(latents.size(0), device=latents.device)
        last_latents = latents[batch_indices, num_words - 1].unsqueeze(1)  # (B, 1, latent_dim)

        # Map latent state to bytes decoder dimension
        mapped_latent = self.decoder_mapping(last_latents)  # (B, 1, bytes_decoder_dim)

        # New attention mask after adding one more word
        B, h, L1, L2 = attention_mask.shape  # noqa: N806
        new_attention_mask = torch.zeros((B, h, L1 + 1, L2 + 1),
                                         device=attention_mask.device, dtype=attention_mask.dtype)
        new_attention_mask[:, :, :L1, :L2] = attention_mask
        for i, n in enumerate(num_words):
            new_attention_mask[i, :, n, :n + 1] = 1

        return new_attention_mask, latent_output.past_key_values, mapped_latent

    def _generate_word_bytes(
            self,
            latents: torch.Tensor,
            tokenizer: UTF8Tokenizer,
            bytes_generation_config: GenerationConfig | None = None,
            **bytes_generation_kwargs
    ) -> torch.Tensor:
        # Start generation with BOS token
        current_input_ids = torch.full((len(latents), 1), tokenizer.bos_token_id,
                                       dtype=torch.long, device=latents.device)
        current_input_embeds = self.bytes_decoder.get_input_embeddings()(current_input_ids)

        inputs_embeds = torch.cat([latents, current_input_embeds], dim=1)  # (B, 2, bytes_decoder_dim)

        # Generate bytes for this word
        return self.bytes_decoder.generate(
            inputs_embeds=inputs_embeds,
            generation_config=bytes_generation_config,
            tokenizer=tokenizer,
            stopping_criteria=[WordStoppingCriteria(tokenizer)],
            **bytes_generation_kwargs
        )

    def prepare_next_inputs(self, words: list[str],
                            words_indexes: torch.Tensor,
                            processor: TextImageProcessor,
                            encoded_input: torch.Tensor) -> torch.Tensor:
        tokenized_words = processor.tokenize_words(words, device=encoded_input.device)
        new_input_ids = tokenized_words.input_ids.unsqueeze(1)
        new_attention_mask = tokenized_words.attention_mask.unsqueeze(1)

        new_input_images, new_input_images_dimensions = processor.render_texts(words)
        new_input_images = new_input_images.unsqueeze(1)
        new_input_images_dimensions = new_input_images_dimensions.unsqueeze(1)

        new_encoded_input = self.encode_input(new_input_ids, new_attention_mask,
                                              new_input_images, new_input_images_dimensions)

        # Extend the encoded input with an empty tensor
        empty_word = torch.zeros_like(encoded_input[:, 0:1])
        encoded_input = torch.cat([encoded_input, empty_word], dim=1)

        # Apply the new words' indexes to the encoded input
        batch_indices = torch.arange(encoded_input.size(0), device=encoded_input.device)
        encoded_input[batch_indices, words_indexes] = new_encoded_input.squeeze(1)

        return encoded_input

    def _prep_bytes_generation_config(self,
                                      max_word_length: int,
                                      tokenizer: UTF8Tokenizer,
                                      bytes_generation_config: GenerationConfig | None = None) -> GenerationConfig:
        default_generation_config_args = dict(
            max_new_tokens=max_word_length,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if bytes_generation_config is None:
            return GenerationConfig(**default_generation_config_args)

        for key, value in default_generation_config_args.items():
            setattr(bytes_generation_config, key, value)

        return bytes_generation_config

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            input_attention_mask: torch.Tensor,
            input_images: torch.Tensor,
            input_images_dimensions: torch.Tensor,
            attention_mask: torch.Tensor,
            processor: TextImageProcessor,
            max_generated_words: int = 50,
            bytes_generation_config: GenerationConfig | None = None,
            **_unused_kwargs):
        """
        Generate text sequences using iterative latent-then-bytes generation.

        This method implements a two-level generation process:
        1. Latent transformer generates latent states (greedy only for now)
        2. Bytes decoder generates words using full HuggingFace generation capabilities

        In practice, this means:
        1. self.encode_input(input_ids, attention_mask, input_images)
        2. self.latent_transformer.generate(...) to get latent states
        3. self.bytes_decoder.generate(...) to generate a single word from latent states (sequence of bytes)
        4. Detokenize the generated bytes to get the final word
        5. Create a new input_ids tensor for the new bytes. new attention_mask,
            and an input_images tensor only for the new word (using render_texts())
        6. Call self.encode_input() again with the new input_ids, attention_mask, and input_images
        7. Concatenate the encoded input with the previous encoded input to form the
            new input for the next word generation
        8. Repeat from step 2 until max_generated_words is reached, or the generated word is just an EOS token.

        We utilize the HuggingFace generation cache to efficiently generate.

        Args:
            input_ids: Text input tokens (B, L, T) for encoding
            input_attention_mask: Attention within a word (B, L, T)
            input_images: (BATCH, LENGTH, CHANNELS, HEIGHT, WIDTH)
            input_images_dimensions: (BATCH, LENGTH, 2)
            attention_mask: Attention across words (BATCH, 1, LENGTH, LENGTH)
            processor: TextImageProcessor instance for tokenization and image processing
            max_generated_words: Maximum number of words to generate
            bytes_generation_config: Generation config for bytes_decoder
        """
        bytes_generation_config = self._prep_bytes_generation_config(processor.max_word_length,
                                                                     processor.tokenizer,
                                                                     bytes_generation_config)

        num_words = self._num_words_per_datum(input_attention_mask)

        # Track generated words per sample
        all_generated_words = [[] for _ in range(len(input_images))]

        # Step 1: Encode initial input
        encoded_input = self.encode_input(input_ids, input_attention_mask, input_images, input_images_dimensions)

        past_key_values = None  # Initialize past key values for caching

        # Main generation loop
        for _ in range(max_generated_words):
            # Step 2: Generate next latent state
            attention_mask, past_key_values, latents = self._generate_latents(past_key_values, encoded_input,
                                                                              attention_mask=attention_mask,
                                                                              num_words=num_words)

            # Step 3: Generate bytes for this word
            generated_bytes = self._generate_word_bytes(
                latents=latents,
                tokenizer=processor.tokenizer,
                bytes_generation_config=bytes_generation_config
            )

            # Step 4: Process generated bytes
            words = processor.tokenizer.batch_decode(generated_bytes, skip_special_tokens=True)
            batch_finished = all(len(word) == 0 for word in words)

            if batch_finished:
                break

            for word, generated_words in zip(words, all_generated_words, strict=False):
                if len(generated_words) == 0 or len(generated_words[-1]) > 0:
                    # Only add words if not EOS
                    generated_words.append(word)

            # Step 5, 6, 7: Create next inputs and encode them\
            encoded_input = self.prepare_next_inputs(words, num_words, processor, encoded_input)

            num_words += 1

        # Step 8: Return generated texts
        texts = ["".join(generated_words) for generated_words in all_generated_words]
        return texts


AutoConfig.register(WordLatentTransformerConfig.model_type, WordLatentTransformerConfig)
AutoModel.register(WordLatentTransformerConfig, WordLatentTransformer)
AutoModelForCausalLM.register(WordLatentTransformerConfig, WordLatentTransformerForCausalLM)
