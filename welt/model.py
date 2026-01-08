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
from utf8_tokenizer.logits_processor import UTF8ValidationLogitsProcessor
from utf8_tokenizer.tokenizer import UTF8Tokenizer
from words_segmentation.pretokenizer import WordStoppingCriteria, is_word_complete

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
    if attn_implementation is not None and attn_implementation.startswith("flash_attention"):
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
    _keys_to_ignore_on_load_missing = [
        # Layers we replace with Identity
        r"bytes_encoder\.cls.*",
        r"bytes_encoder\.decoder.*"
    ]

    def _initialize_missing_keys(self, missing_keys, is_quantized):
        """Override to prevent initialization of missing keys when loading from pretrained."""
        pass

    def __init__(self, config: WordLatentTransformerConfig,
                 load_pretrained: bool = False,
                 attn_implementation=None):
        super().__init__(config=config)

        is_image_encoder = not isinstance(config.image_encoder, NoopConfig)
        is_bytes_encoder = not isinstance(config.bytes_encoder, NoopConfig)

        assert is_bytes_encoder or is_image_encoder, \
            "At least one encoder must be provided"

        if not is_bytes_encoder or not is_image_encoder:
            warnings.warn("Either image encoder or bytes encoder is not provided, setting modality_dropout to 0.0",
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

        # Small Language Model
        self.bytes_decoder = model_from_config(config.bytes_decoder, AutoModelForCausalLM,
                                               config.dtype, load_pretrained, attn_implementation)
        self.bytes_decoder.resize_token_embeddings(config.num_tokens, pad_to_multiple_of=8)
        patch_embedding_layers(self.bytes_decoder)
        bytes_decoder_dim = self.bytes_decoder.config.hidden_size

        # Mapping layers
        encoder_dim = self.bytes_encoder_dim + self.image_encoder_dim
        self.encoder_mapping = nn.Linear(encoder_dim, model_dim, dtype=self.latent_transformer.dtype)

        # Set decoder_mapping as the LM head so we can use .logits directly
        decoder_mapping = nn.Linear(model_dim, bytes_decoder_dim, dtype=self.bytes_decoder.dtype)
        self.latent_transformer.set_output_embeddings(decoder_mapping)

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
            dtype = self.latent_transformer.dtype if self.image_encoder is None else self.image_encoder.dtype
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
            dtype = self.latent_transformer.dtype if self.bytes_encoder is None else self.bytes_encoder.dtype
            return torch.zeros(B, L, self.bytes_encoder_dim, device=input_ids.device, dtype=dtype)

        # Flatten batch and length dimensions
        input_ids = input_ids.view(B * L, T)
        attention_mask = attention_mask.view(B * L, T)
        # Encode texts using the bytes decoder as encoder
        if hasattr(self.bytes_encoder, "model"):
            # For models like ModernBertForMaskedLM
            text_outputs = self.bytes_encoder.model(input_ids=input_ids, attention_mask=attention_mask)
            text_embeds = text_outputs.last_hidden_state
        else:
            text_outputs = self.bytes_encoder(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              output_hidden_states=True)
            text_embeds = text_outputs.hidden_states[-1]

        # Use BOS token embedding as word embedding
        text_embeds = text_embeds[:, 0]
        return text_embeds.view(B, L, -1)

    def encode_input(self,
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     input_images: torch.Tensor,
                     input_images_dimensions: torch.Tensor):
        embeds = []
        if self.image_encoder_dim > 0:
            image_embeds = self.encode_images(input_images, input_images_dimensions, device=input_ids.device)
            embeds.append(image_embeds)

        if self.bytes_encoder_dim > 0:
            text_embeds = self.encode_texts(input_ids, attention_mask)
            embeds.append(text_embeds)

        assert len(embeds) > 0, "At least one type of encoder must be provided"

        concatenated_embeds = torch.cat(embeds, dim=-1)

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
        )
        # they aren't really "logits", since we replace the lm_head with decoder_mapping
        mapped_embeds = latent_outputs.logits

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

    def parallel_causal_decode(self,
                               latent_vectors: torch.Tensor,
                               target_ids: torch.Tensor,
                               target_mask: torch.Tensor) -> torch.Tensor:
        """
        Parallel causal decoding with word-level vectors prepended to character sequences.

        Args:
            latent_vectors: (B, L, hidden_dim) - latent representations for each word
            target_ids: (B, L, T) - target token IDs for each word
            target_mask: (B, L, T) - attention mask for target tokens

        Returns:
            torch.Tensor: (B, L, T, vocab_size) - logits for each token in each word
        """
        B, L, hidden_dim = latent_vectors.shape  # noqa: N806
        _, _, T = target_ids.shape  # noqa: N806

        # Step 1: Reshape target_ids from [B, L, T] to [B*L, T]
        target_ids_flat = target_ids.view(B * L, T)  # [B*L, T]
        target_mask_flat = target_mask.view(B * L, T)  # [B*L, T]

        # Step 2: Get embeddings for target tokens
        embed_layer = self.bytes_decoder.get_input_embeddings()
        target_embeds = embed_layer(target_ids_flat)  # [B*L, T, embed_dim]

        # Step 3: Each decoder uses only one latent vector (no history)
        # Decoder i uses latent_vectors[:, i]
        # Reshape from [B, L, hidden_dim] to [B*L, hidden_dim] then add sequence dimension
        latent_vectors_flat = latent_vectors.view(B * L, hidden_dim).unsqueeze(1)  # [B*L, 1, hidden_dim]

        # Step 4: Concatenate single latent vector with character embeddings
        # Each sequence gets only its corresponding latent vector prepended
        combined_embeds = torch.cat([latent_vectors_flat, target_embeds], dim=1)  # [B*L, 1+T, embed_dim]

        # Step 5: Create attention mask by padding target_mask with 1 on the left
        combined_mask = torch.nn.functional.pad(target_mask_flat, (1, 0), value=1)  # [B*L, 1+T]

        # Step 6: Pass through bytes decoder
        outputs = self.bytes_decoder(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            output_hidden_states=False
        )

        # Step 7: Extract character-level logits (skip the single latent position)
        all_logits = outputs.logits  # [B*L, 1+T, vocab_size]
        char_logits = all_logits[:, 1:]  # [B*L, T, vocab_size]

        # Step 8: Reshape back to [B, L, T, vocab_size]
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
        # decoder_mapping
        set_module_trainable(self.latent_transformer.get_output_embeddings())

    def unfreeze(self):
        """Unfreeze everything."""
        set_module_trainable(self, True)


class WordLatentTransformerForCausalLM(WordLatentTransformer, GenerationMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_decode = None
        self._compile_enabled = False

        self.logits_processor = UTF8ValidationLogitsProcessor()

    def _get_partial_word_prefix(
            self,
            input_ids: torch.Tensor,
            input_attention_mask: torch.Tensor,
            initial_num_words: torch.Tensor,
            tokenizer: UTF8Tokenizer,
    ) -> torch.Tensor | None:
        """
        Get prefix token IDs for partial (incomplete) last words.

        Returns:
            prefix_ids: (B, T) tensor if any words are incomplete, else None.
                       Complete words get [BOS, PAD, ...], incomplete get their tokens (without EOS).
        """
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)

        last_word_ids = input_ids[batch_indices, initial_num_words - 1]
        last_word_mask = input_attention_mask[batch_indices, initial_num_words - 1]

        last_words = [tokenizer.decode(ids[mask.bool()].tolist(), skip_special_tokens=True)
                      for ids, mask in zip(last_word_ids, last_word_mask, strict=True)]

        if all(is_word_complete(w) for w in last_words):
            return None

        print("Found partial words:", last_words, last_word_ids)
        print("input_ids", input_ids)

        prefix_ids = last_word_ids.clone()
        prefix_ids[prefix_ids == tokenizer.eos_token_id] = tokenizer.pad_token_id

        return prefix_ids

    def enable_backend_optimizations(self):
        """
        Enable PyTorch backend optimizations (TF32, Flash Attention, cudnn benchmark).

        These optimizations are safe for both training and inference and provide
        significant speedups (especially Flash Attention). They don't use torch.compile
        so they're compatible with Hugging Face Trainer and Accelerate.

        Call this before training or inference for best performance.
        """
        print("Enabling backend optimizations...")

        torch.set_float32_matmul_precision('high')  # Use TF32 for faster matmul

        if torch.cuda.is_available():
            # Enable global PyTorch optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Enable Flash Attention for scaled_dot_product (significant speedup)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(False)  # Force optimized paths only

            print("✓ Enabled cudnn benchmark, TF32, Flash Attention, and CUDA optimizations")
        else:
            print("✓ Enabled TF32 optimizations (CPU mode)")

    def enable_optimizations(self, compile_mode: str = "default"):
        """
        Enable performance optimizations for inference.

        WARNING: Do NOT use this during training with Hugging Face Trainer!
        The torch.compile calls conflict with Accelerate's model unwrapping.
        For training, use enable_backend_optimizations() instead.
        """
        print(f"Enabling optimizations (compile_mode={compile_mode})...")

        # First enable backend optimizations
        self.enable_backend_optimizations()

        if not torch.cuda.is_available():
            # Fix for CPU torch.compile inductor bug with missing variable declarations
            torch._dynamo.config.suppress_errors = True
            torch._inductor.config.cpp.simdlen = None  # Disable vectorization that causes the bug

        # Compile _decode (called repeatedly in generation loop)
        print("  Compiling _decode...")
        # Store original method before compilation
        original_decode = self.__class__._decode
        # Compile the unbound method and bind it to this instance
        compiled_decode = torch.compile(original_decode, mode=compile_mode)
        # Replace instance method with compiled version
        self._decode = compiled_decode.__get__(self, type(self))

        # Compile encoder_mapping (small but called frequently)
        print("  Compiling mapping layers...")
        self.encoder_mapping = torch.compile(self.encoder_mapping, mode=compile_mode)

        # Compile bytes_decoder forward (called in character generation loop)
        print("  Compiling bytes_decoder...")
        self.bytes_decoder.forward = torch.compile(self.bytes_decoder.forward, mode=compile_mode)

        print("  Compiling logits processor...")
        self.logits_processor = torch.compile(self.logits_processor, mode=compile_mode)

        # Compile bytes_encoder forward (called in _encode_words per word)
        if self.bytes_encoder is not None:
            print("  Compiling bytes_encoder...")
            self.bytes_encoder.forward = torch.compile(self.bytes_encoder.forward, mode=compile_mode)

        self._compile_enabled = True
        print("✓ Optimizations enabled")

    def _prefill(self, encoded_input: torch.Tensor, attention_mask: torch.Tensor,
                 num_words: torch.Tensor, position_ids: torch.Tensor | None = None) -> tuple[Any, torch.Tensor]:
        """
        Prefill stage: Process the full input sequence and build the KV-cache.

        Args:
            encoded_input: Full encoded input (B, L, hidden_dim)
            attention_mask: 4D attention mask (B, 1, L, L) with causal + padding masking
            num_words: Number of valid words per batch item (B,)
            position_ids: Optional position IDs (B, L) for handling variable-length sequences

        Returns:
            past_key_values: KV-cache for subsequent decode steps
            mapped_latent: Mapped latent state for the last word (B, 1, bytes_decoder_dim)
        """
        latent_output = self.latent_transformer(
            inputs_embeds=encoded_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )

        # LM head (decoder_mapping) is applied internally, output via .logits
        logits = latent_output.logits  # (B, L, bytes_decoder_dim)
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        mapped_latent = logits[batch_indices, num_words - 1].unsqueeze(1)  # (B, 1, bytes_decoder_dim)

        return latent_output.past_key_values, mapped_latent

    def _decode(self, past_key_values: Any, new_embedding: torch.Tensor,
                attention_mask: torch.Tensor, position_ids: torch.Tensor | None = None) -> tuple[Any, torch.Tensor]:
        """
        Decode stage: Process a single new token using the KV-cache.

        Args:
            past_key_values: KV-cache from prefill or previous decode step
            new_embedding: Embedding for the new token (B, 1, hidden_dim)
            attention_mask: 2D attention mask (B, seq_len) indicating which cached positions to attend to
            position_ids: Optional position IDs (B, 1) for the new tokens

        Returns:
            past_key_values: Updated KV-cache
            mapped_latent: Mapped latent state (B, 1, bytes_decoder_dim)
        """
        latent_output = self.latent_transformer(
            inputs_embeds=new_embedding,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
        )

        return latent_output.past_key_values, latent_output.logits

    def _generate_word_bytes(
            self,
            latents: torch.Tensor,
            tokenizer: UTF8Tokenizer,
            bos_embed: torch.Tensor,
            bytes_generation_config: GenerationConfig | None = None,
            stopping_criteria: list | None = None,
            prefix_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = latents.size(0)
        embed_layer = self.bytes_decoder.get_input_embeddings()

        if prefix_ids is not None:
            # Mid-word generation only supports batch_size=1 (checked in generate())
            # Trim prefix_ids to actual length (remove trailing PAD tokens)
            prefix_len = (prefix_ids[0] != tokenizer.pad_token_id).sum().item()
            trimmed_ids = prefix_ids[:, :prefix_len]

            prefix_embeds = embed_layer(trimmed_ids)
            inputs_embeds = torch.cat([latents, prefix_embeds], dim=1)
            attention_mask = None  # No padding, no mask needed
        else:
            bos_embeds = bos_embed.expand(batch_size, -1, -1)
            inputs_embeds = torch.cat([latents, bos_embeds], dim=1)
            attention_mask = None

        return self.bytes_decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=bytes_generation_config,
            tokenizer=tokenizer,
            logits_processor=[self.logits_processor],
            stopping_criteria=stopping_criteria,
        )

    def _encode_words(self, words: list[str], processor: TextImageProcessor, device: torch.device) -> torch.Tensor:
        """Encode words into embeddings for the next decode step."""
        tokenized_words = processor.tokenize_words(words, device=device)
        new_input_ids = tokenized_words.input_ids.unsqueeze(1)
        new_attention_mask = tokenized_words.attention_mask.unsqueeze(1)

        new_input_images, new_input_images_dimensions = processor.render_texts(words, device=device)
        new_input_images = new_input_images.unsqueeze(1)
        new_input_images_dimensions = new_input_images_dimensions.unsqueeze(1)

        return self.encode_input(new_input_ids, new_attention_mask,
                                 new_input_images, new_input_images_dimensions).squeeze(1)  # (B, hidden_dim)

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

    @torch.inference_mode()
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
        Generate text using prefill/decode with KV-cache.

        1. Prefill: encode input and process full sequence to build KV-cache
        2. Decode loop: for each word, encode it, run single decode step, generate bytes

        Args:
            input_ids: (B, L, T) - text input tokens
            input_attention_mask: (B, L, T) - attention within each word
            input_images: (B, L, C, H, W) - input images
            input_images_dimensions: (B, L, 2) - original image dimensions
            attention_mask: (B, 1, L, L) - causal attention across words
            processor: TextImageProcessor for tokenization and rendering
            max_generated_words: maximum words to generate
            bytes_generation_config: optional GenerationConfig for bytes_decoder
        """
        tokenizer = processor.tokenizer
        device = input_ids.device
        batch_size = len(input_images)

        bytes_generation_config = self._prep_bytes_generation_config(
            processor.max_word_length, tokenizer, bytes_generation_config)
        stopping_criteria = [WordStoppingCriteria(tokenizer)]
        bos_embed = self.bytes_decoder.get_input_embeddings()(
            torch.tensor([[tokenizer.bos_token_id]], device=device))

        # Prefill: encode input and build KV-cache
        initial_num_words = self._num_words_per_datum(input_attention_mask)
        encoded_input = self.encode_input(input_ids, input_attention_mask, input_images, input_images_dimensions)

        prefix_ids = self._get_partial_word_prefix(
                input_ids, input_attention_mask, initial_num_words, tokenizer)
        if prefix_ids is not None:
            if batch_size > 1:
                raise ValueError("Mid-word generation with prefix_ids is only supported for batch_size=1")
            # Exclude the partial word from prefill - we use it as generation prefix instead
            encoded_input = encoded_input[:, :-1]
            attention_mask = attention_mask[:, :, :-1, :-1]
            initial_num_words = initial_num_words - 1

        # Use default position_ids for prefill (sequential), attention mask handles padding
        max_initial = initial_num_words.max().item()
        past_key_values, latents = self._prefill(encoded_input, attention_mask, initial_num_words)

        # Pre-allocate decode attention mask (1s everywhere except padding positions)
        decode_mask_full = torch.ones((batch_size, max_initial + max_generated_words), device=device,
                                      dtype=attention_mask.dtype)
        positions = torch.arange(decode_mask_full.size(1), device=device)
        padding_mask = (positions >= initial_num_words.unsqueeze(1)) & (positions < max_initial)
        decode_mask_full.masked_fill_(padding_mask, 0)

        # Generation loop
        all_generated_words = [[] for _ in range(batch_size)]
        words = None

        for step_idx in range(max_generated_words):
            if words is not None:
                # Decode: encode new words and run single transformer step
                new_embedding = self._encode_words(words, processor, device).unsqueeze(1)
                decode_mask = decode_mask_full[:, :past_key_values.get_seq_length() + 1]

                # Compute position_ids for each batch item: continuing from their last valid position
                # position_id = initial_num_words + (step_idx - 1), since step 0 doesn't do decode
                decode_position_ids = (initial_num_words + step_idx - 1).unsqueeze(1)  # (B, 1)

                past_key_values, latents = self._decode(
                    past_key_values, new_embedding, decode_mask, decode_position_ids
                )

            # Generate bytes from latents
            generated_bytes = self._generate_word_bytes(
                latents, tokenizer, bos_embed, bytes_generation_config, stopping_criteria,
                prefix_ids=prefix_ids)
            prefix_ids = None  # Only use prefix for the first generated word
            words = tokenizer.batch_decode(generated_bytes, skip_special_tokens=True)

            if all(len(w) == 0 for w in words):
                break

            # Collect words (skip if previous word was empty = EOS for that sample)
            for word, collected in zip(words, all_generated_words, strict=False):
                if not collected or collected[-1]:
                    collected.append(word)

        return ["".join(words) for words in all_generated_words]


AutoConfig.register(WordLatentTransformerConfig.model_type, WordLatentTransformerConfig)
AutoModel.register(WordLatentTransformerConfig, WordLatentTransformer)
AutoModelForCausalLM.register(WordLatentTransformerConfig, WordLatentTransformerForCausalLM)
