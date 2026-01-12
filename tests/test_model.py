import tempfile

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_model, save_model
from transformers.modeling_outputs import CausalLMOutput
from utf8_tokenizer import CharacterCausalLMWrapper, CharacterEmbedding

from welt.model import WordLatentTransformer
from welt.model_utils import setup_model


def setup_tiny_model(
        image_encoder_name="WinKawaks/vit-tiny-patch16-224",
        bytes_encoder_name="prajjwal1/bert-tiny",
        latent_transformer_name="sbintuitions/tiny-lm",
        bytes_decoder_name="sign/utf8-lm-tiny",
        **kwargs):
    """Set up a tiny version of the WordLatentTransformer model for testing, the tinyer the better."""
    return setup_model(
        image_encoder_name=image_encoder_name,
        bytes_encoder_name=bytes_encoder_name,
        latent_transformer_name=latent_transformer_name,
        bytes_decoder_name=bytes_decoder_name,
        load_pretrained=False,
        max_word_length=32,
        **kwargs
    )


def make_dataset(texts: list[str]):
    """Create a dataset from a list of texts."""
    return Dataset.from_dict({"text": texts})


def dataset_to_batch(model, processor, collator, dataset):
    # Compute losses for each sequence - process entire batch at once
    device = next(model.parameters()).device
    dataset = dataset.with_transform(processor)
    batch = collator([dataset[i] for i in range(len(dataset))])
    # Move batch to the same device as the model
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def predict_dataset(texts: list[str], model, processor, collator):
    """Predict a dataset and return the logits."""
    dataset = make_dataset(texts)
    batch = dataset_to_batch(model, processor, collator, dataset)

    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    labels = batch['labels_output']

    output_per_text = {}
    losses = {}
    for i, text in enumerate(texts):
        # Compute cross entropy loss for this item
        item_loss = torch.nn.functional.cross_entropy(input=logits[i].reshape(-1, logits.size(-1)),
                                                      target=labels[i].reshape(-1),
                                                      ignore_index=processor.tokenizer.pad_token_type_id,
                                                      reduction='none')
        # Reduce loss based on ignore_index
        losses[text] = item_loss.sum().item() / (labels[i] != processor.tokenizer.pad_token_type_id).sum().item()
        print(f"Loss for '{text}': {losses[text]:.4f}")

        output_per_text[text] = CausalLMOutput(
            loss=item_loss,
            logits=logits[i],
            hidden_states=(outputs.hidden_states[0][i],),
            attentions=None
        )

    return losses, output_per_text


def test_attention_no_look_ahead():
    """Test that attention does not look ahead - causal masking is working correctly."""
    model, processor, collator = setup_model()
    model.eval()

    # Test sequences that share prefixes
    texts = ["a b c x y z", "a b d m"]

    _, outputs = predict_dataset(texts, model, processor, collator)
    for text in texts:
        print(f"Loss for '{text}':", outputs[text].loss.cpu().numpy())

    # Check that the first 4 tokens have identical losses
    for i in range(4):
        assert abs(outputs[texts[0]].loss[i] - outputs[texts[1]].loss[i]) < 1e-4, \
            f"Loss at position {i} should be identical: {outputs[texts[0]].loss[i]} vs {outputs[texts[1]].loss[i]}"


def test_attention_does_look_back():
    """Test that attention does look back - model uses previous context."""
    model, processor, collator = setup_model()
    model.eval()

    # Test sequences with shared suffix but different prefix
    texts = ["c b a", "d b a"]

    _, outputs = predict_dataset(texts, model, processor, collator)
    for text in texts:
        print(f"Loss for '{text}':", outputs[text].loss.cpu().numpy())

    # Check that ALL positions have different losses due to different context
    for i in range(7):  # Check all 7 positions (excluding padding)
        loss_diff = abs(outputs[texts[0]].loss[i] - outputs[texts[1]].loss[i])
        assert loss_diff > 1e-4, \
            (f"Loss at position {i} should be different due to context: "
             f"{outputs[texts[0]].loss[i]} vs {outputs[texts[1]].loss[i]} (diff: {loss_diff})")


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if torch.backends.mps.is_available():
    DEVICES.append("mps")


@pytest.mark.parametrize("device", DEVICES)
def test_multiple_texts_batch_not_nan(device):
    """Test that attention does look back - model uses previous context."""
    model, processor, collator = setup_model()
    # model.eval()

    # Move model to specified device
    model = model.to(torch.device(device))

    # Test sequences with shared suffix but different prefix
    texts = ["1", "1 2 3"]

    dataset = make_dataset(texts)
    batch = dataset_to_batch(model, processor, collator, dataset)

    # TODO: this fails on mps device, because of the attention mask
    #   ONLY when no_grad is used https://github.com/pytorch/pytorch/issues/167515
    with torch.no_grad():
        outputs = model(**batch)
    assert not torch.isnan(outputs.loss).any(), "Loss contains NaN values"


def test_loss_is_independent_of_batch():
    """Test that loss at first position is identical regardless of other items in batch."""
    model, processor, collator = setup_model()
    model.eval()

    batches = [
        # Run first batch with just "a"
        ["a"],
        # Run second batch with "a" and additional text
        ["a", "2 w"],
        # Run third batch with "a" and additional longer text
        ["a", "two words"],
    ]
    outputs = [predict_dataset(batch, model, processor, collator)[1] for batch in batches]

    # Get the loss for "a" from both batches
    losses = [outputs[i]["a"].loss[0].item() for i in range(len(outputs))]
    max_loss = max(losses)
    losses = [loss / max_loss for loss in losses]  # Normalize losses for comparison, across different models

    # Check that the loss at the first position (first token) is nearly identical
    # Note: losses[0] and losses[1] should be the same
    # since they're both predicting the same token with the same context
    # Small numerical differences are acceptable due to batching implementation details
    assert abs(losses[0] - losses[1]) < 1e-3, \
        f"Loss at first position should be nearly identical: {losses[0]} vs {losses[1]}"

    assert abs(losses[0] - losses[2]) < 1e-3, \
        f"Loss at first position should be nearly identical: {losses[0]} vs {losses[2]}"

    print(f"✓ Loss at first position is batch-independent: {losses[0]:.4f}")


def num_model_params(model):
    return sum(p.numel() for p in model.parameters())


def test_model_save_and_load_works():
    """Test that the model can be saved and loaded without issues."""
    model, processor, collator = setup_tiny_model()

    with tempfile.NamedTemporaryFile(suffix=".safetensors") as temp_file:
        original_num_parameters = num_model_params(model)
        save_model(model, temp_file.name)
        load_model(model, temp_file.name)
        loaded_num_parameters = num_model_params(model)
        assert original_num_parameters == loaded_num_parameters, \
            f"Number of parameters mismatch: {original_num_parameters:,} vs {loaded_num_parameters:,}"


def test_model_from_pretrained_works():
    """Test that the model can be saved and loaded without issues."""
    model, processor, collator = setup_tiny_model()

    with tempfile.TemporaryDirectory() as temp_dir:
        original_num_parameters = num_model_params(model)
        model.save_pretrained(save_directory=temp_dir, push_to_hub=False)

        new_model = WordLatentTransformer.from_pretrained(temp_dir)
        loaded_num_parameters = num_model_params(new_model)

        assert original_num_parameters == loaded_num_parameters, \
            f"Number of parameters mismatch: {original_num_parameters:,} vs {loaded_num_parameters:,}"


def test_freeze_unfreeze_model_works():
    """Test that freezing the model works correctly."""
    model, processor, collator = setup_tiny_model()

    model.freeze_pretrained_models()

    decoder_mapping = model.latent_transformer.get_output_embeddings()
    decoder_mapping_params = {param for _, param in decoder_mapping.named_parameters()}

    # All latent transformer parameters should be frozen except decoder mapping which is freshly initialized
    for name, param in model.latent_transformer.named_parameters():
        if param in decoder_mapping_params:
            continue
        assert not param.requires_grad, f"Parameter {name} should be frozen but is unfrozen."

    for layer in [model.encoder_mapping, model.encoder_norm, decoder_mapping, model.decoder_norm]:
        for name, param in layer.named_parameters():
            assert param.requires_grad, f"Parameter {name} should be unfrozen but is frozen."

    model.unfreeze()

    for name, param in model.named_parameters():
        assert param.requires_grad, f"Parameter {name} should be unfrozen but is frozen."


def test_model_from_pretrained_works_without_image_encoder():
    """Test that the model can be saved and loaded without issues."""
    model, processor, collator = setup_model(image_encoder_name=None)

    with tempfile.TemporaryDirectory() as temp_dir:
        original_num_parameters = num_model_params(model)
        model.save_pretrained(save_directory=temp_dir, push_to_hub=False)

        new_model = WordLatentTransformer.from_pretrained(temp_dir)
        loaded_num_parameters = num_model_params(new_model)

        assert original_num_parameters == loaded_num_parameters, \
            f"Number of parameters mismatch: {original_num_parameters:,} vs {loaded_num_parameters:,}"


def test_model_from_pretrained_works_without_bytes_encoder():
    """Test that the model can be saved and loaded without bytes encoder, checking actual values."""
    model, processor, collator = setup_model(bytes_encoder_name=None)
    model.eval()

    # Test with some sample data
    texts = ["hello world", "test"]

    # Get outputs before saving
    original_losses, original_outputs = predict_dataset(texts, model, processor, collator)

    with tempfile.TemporaryDirectory() as temp_dir:
        original_num_parameters = num_model_params(model)
        model.save_pretrained(save_directory=temp_dir, push_to_hub=False)

        new_model = WordLatentTransformer.from_pretrained(temp_dir)
        new_model.eval()
        loaded_num_parameters = num_model_params(new_model)

        assert original_num_parameters == loaded_num_parameters, \
            f"Number of parameters mismatch: {original_num_parameters:,} vs {loaded_num_parameters:,}"

        # Get outputs after loading
        loaded_losses, loaded_outputs = predict_dataset(texts, new_model, processor, collator)

        # Check that losses are identical
        for text in texts:
            original_loss = original_losses[text]
            loaded_loss = loaded_losses[text]
            assert abs(original_loss - loaded_loss) < 1e-6, \
                f"Loss mismatch for '{text}': {original_loss} vs {loaded_loss}"

        # Check that logits are identical
        for text in texts:
            original_logits = original_outputs[text].logits
            loaded_logits = loaded_outputs[text].logits
            max_diff = torch.max(torch.abs(original_logits - loaded_logits)).item()
            assert max_diff < 1e-5, \
                f"Logits mismatch for '{text}': max difference {max_diff}"

        print("✓ Model outputs are identical after save/load")


def test_utf32_encoding_wraps_utf8_decoder():
    """Test that UTF-32 encoding wraps a UTF-8 decoder with CharacterCausalLMWrapper."""
    model, processor, collator = setup_tiny_model(
        encoding="UTF-32",
        bytes_decoder_name="sign/utf8-lm-tiny",
    )

    # Check decoder is wrapped
    assert isinstance(model.bytes_decoder, CharacterCausalLMWrapper), \
        "UTF-32 encoding should wrap UTF-8 decoder with CharacterCausalLMWrapper"
    assert model.bytes_decoder.config.num_bytes == 4, \
        "UTF-32 wrapper should have num_bytes=4"

    # Check encoder has CharacterEmbedding
    encoder_embeddings = model.bytes_encoder.get_input_embeddings()
    assert isinstance(encoder_embeddings, CharacterEmbedding), \
        "UTF-32 encoding should use CharacterEmbedding for encoder"
    assert encoder_embeddings.num_bytes == 4, \
        "Encoder CharacterEmbedding should have num_bytes=4"

    # Test forward pass without labels
    model.eval()
    texts = ["hello world"]
    dataset = make_dataset(texts)
    batch = dataset_to_batch(model, processor, collator, dataset)
    # batch.pop("labels_output", None)

    with torch.no_grad():
        outputs = model(**batch)

    assert outputs.logits is not None, "Forward pass should produce logits"
    assert not torch.isnan(outputs.logits).any(), "Logits should not contain NaN"

    # Test generation
    batch = processor(["hello "], collated=True)
    generated = model.generate(**batch, processor=processor, max_generated_words=2)
    assert len(generated) == 1, "Should generate one output"
    assert isinstance(generated[0], str), "Generated output should be a string"


def test_utf32_encoding_with_utf32_decoder():
    """Test that UTF-32 encoding with already-wrapped decoder doesn't double-wrap."""
    model, processor, collator = setup_tiny_model(
        encoding="UTF-32",
        bytes_decoder_name="sign/utf32-lm-tiny",
    )

    assert isinstance(model.bytes_decoder, CharacterCausalLMWrapper), \
        "UTF-32 decoder should be a CharacterCausalLMWrapper"
    assert model.bytes_decoder.config.num_bytes == 4, \
        "UTF-32 wrapper should have num_bytes=4"
    assert not isinstance(model.bytes_decoder.model, CharacterCausalLMWrapper), \
        "Should not double-wrap the decoder"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
