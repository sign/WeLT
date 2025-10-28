import tempfile

import pytest
from transformers import Trainer, TrainingArguments
from trl import pack_dataset

from tests.test_model import make_dataset, predict_dataset, setup_tiny_model


# TODO: this training is flaky due to https://github.com/huggingface/transformers/issues/40219
def train_model(setup_function,
                num_epochs=10,
                train_texts=None,
                packing=False):
    model, processor, collator = setup_function()

    if train_texts is None:
        train_texts = ["a b", "b a", "a cat", "a dog"]

    train_dataset = make_dataset(train_texts)

    if packing:
        train_dataset = processor.pretokenize_dataset(train_dataset)
        train_dataset = pack_dataset(train_dataset, seq_length=7)

    train_dataset = train_dataset.with_transform(processor)

    # Setup training arguments with more epochs for overfitting
    training_args = TrainingArguments(
        output_dir=tempfile.mktemp(),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=len(train_texts),
        logging_steps=1,
        logging_strategy="steps",
        save_strategy="no",  # Disable saving to avoid shared tensor issues
        remove_unused_columns=False,
        dataloader_drop_last=False,
        warmup_steps=0,  # No warmup for immediate learning
        weight_decay=0.0,  # No regularization for overfitting
        learning_rate=1e-4,
        lr_scheduler_type="constant",  # Keep learning rate constant
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=processor,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    # Train the model
    print("Training model on texts:", train_texts)
    trainer.train()

    # Set to eval mode
    model.eval()

    return model, processor, collator


@pytest.fixture(scope="module")
def trained_models():
    """Train the model once and reuse for all tests."""
    num_epochs = 200
    return {
        "packed": train_model(setup_tiny_model, num_epochs=num_epochs, packing=True),
        "unpacked": train_model(setup_tiny_model, num_epochs=num_epochs, packing=False)
    }


@pytest.fixture
def model_configuration(request, trained_models):
    """Configure model based on the test parameter."""
    model_type, config_name = request.param
    model, processor, collator = trained_models[model_type]

    # Store original encoders
    original_bytes_encoder = model.bytes_encoder
    original_image_encoder = model.image_encoder

    # Apply configuration
    if config_name == "full_model":
        print("\n[Configuration: Full model with all encoders]")
        # No changes needed
    elif config_name == "no_bytes_encoder":
        print("\n[Configuration: Model without bytes encoder]")
        model.bytes_encoder = None
    elif config_name == "no_image_encoder":
        print("\n[Configuration: Model without image encoder]")
        model.image_encoder = None

    # Yield the configured model
    yield model, processor, collator

    # Restore original encoders after test
    model.bytes_encoder = original_bytes_encoder
    model.image_encoder = original_image_encoder


MODEL_CONFIGURATIONS = [
    # Packed setups
    ("packed", "full_model"),
    ("packed", "no_bytes_encoder"),
    ("packed", "no_image_encoder"),

    # Unpacked setups
    ("unpacked", "full_model"),
    ("unpacked", "no_bytes_encoder"),
    ("unpacked", "no_image_encoder"),
]

MODEL_IDs = [f"{config} / {model_type}" for config, model_type in MODEL_CONFIGURATIONS]
parameterization = pytest.mark.parametrize("model_configuration", MODEL_CONFIGURATIONS, indirect=True, ids=MODEL_IDs)


@parameterization
def test_character_level_conditioning(model_configuration):
    """Test 1: Character-level conditioning (a b vs a a, b a vs b b)"""
    model, processor, collator = model_configuration

    print("\n=== Test 1: Character-level conditioning ===")

    test_texts_char = ["a b", "b a", "a a", "b b"]
    losses, predictions = predict_dataset(test_texts_char, model, processor, collator)

    # Check conditioning: trained sequences should have lower loss
    assert losses['a b'] < losses['a a'], \
        f"'a b' should have lower loss than 'a a': {losses['a b']:.4f} vs {losses['a a']:.4f}"

    assert losses['b a'] < losses['b b'], \
        f"'b a' should have lower loss than 'b b': {losses['b a']:.4f} vs {losses['b b']:.4f}"

    print("✅ Character-level conditioning test passed!")


@parameterization
def test_word_level_conditioning(model_configuration):
    """Test 2: Word-level conditioning (a cat vs a dat, a dog vs a cog)"""
    model, processor, collator = model_configuration

    print("\n=== Test 2: Word-level conditioning ===")

    test_texts_word = ["a cat", "a dog", "a dat", "a cog", "a bat", "a fog"]
    losses, predictions = predict_dataset(test_texts_word, model, processor, collator)

    # Check conditioning: trained sequences should have lower loss
    assert losses['a cat'] < losses['a dat'], \
        f"'a cat' should have lower loss than 'a dat': {losses['a cat']:.4f} vs {losses['a dat']:.4f}"

    assert losses['a dog'] < losses['a cog'], \
        f"'a dog' should have lower loss than 'a cog': {losses['a dog']:.4f} vs {losses['a cog']:.4f}"

    print("✅ Word-level conditioning test passed!")


@parameterization
def test_byte_level_conditioning(model_configuration):
    """Test 3: Byte-level conditioning within words"""
    model, processor, collator = model_configuration

    print("\n=== Test 3: Byte-level conditioning within words ===")

    # For "a cat" and "a dog", after seeing "a c" or "a d", the model should be confident about the rest
    # This tests that the byte decoder is properly conditioned on previous bytes

    # Create a special test to check conditional probabilities
    test_conditional = ["a cat", "a cog", "a dog", "a dat"]
    losses, predictions = predict_dataset(test_conditional, model, processor, collator)

    # After 'a c', 'cat' should be more likely than 'cog'
    assert losses['a cat'] < losses['a cog'], \
        (f"'a cat' should have lower loss than 'a cog': "
         f"{losses['a cat']:.4f} vs {losses['a cog']:.4f}")

    # After 'a d', 'dog' should be more likely than 'dat'
    assert losses['a dog'] < losses['a dat'], \
        (f"'a dog' should have lower loss than 'a dat': "
         f"{losses['a dog']:.4f} vs {losses['a dat']:.4f}")

    print("✅ Byte-level conditioning test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
