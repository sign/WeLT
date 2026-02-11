from itertools import chain

import pytest
import torch
from transformers import AutoModelForCausalLM, GenerationConfig

from tests.test_model import dataset_to_batch, make_dataset, setup_tiny_model


@pytest.fixture(scope="module")
def generation_model_setup():
    """Setup the generation model with trained weights."""

    # Setup the base model
    model, processor, collator = setup_tiny_model()

    # Set to eval mode
    model.eval()

    return model, processor, collator


def predict_texts(texts: list[str], model, processor, collator):
    """Helper function to predict texts using the generation model."""

    # add trailing space to each text to ensure proper "word" boundary
    texts = [text.strip() + " " for text in texts]

    print("-" * 30)
    dataset = make_dataset(texts)
    batch = dataset_to_batch(model, processor, collator, dataset)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=batch["input_ids"],
            input_attention_mask=batch["input_attention_mask"],
            input_images=batch["input_images"],
            input_images_dimensions=batch["input_images_dimensions"],
            attention_mask=batch["attention_mask"],
            processor=processor,
            max_generated_words=5
        )

    for text, output in zip(texts, outputs, strict=False):
        print(f"Generated for '{text}': {output}")
    return outputs


def test_batch_interference(generation_model_setup):
    """Test that generation of a batch does not interfere between texts."""
    model, processor, collator = generation_model_setup

    print("\n=== Testing batch interference ===")
    batches = [
        ["a"],
        ["a", "two words"],
        ["a", "even three words"],
        ["a", "ü", "a_long_word"],
        ["a", "a"]
    ]
    outputs = [predict_texts(batch, model, processor, collator) for batch in batches]

    single = outputs[0][0]  # Single result for "a"
    print(f"Single result for 'a': '{outputs[0][0]}'")
    print(f"Batch 1 result for 'a': '{outputs[1][0]}'")
    print(f"Batch 2 result for 'a': '{outputs[2][0]}'")
    print(f"Batch 3 result for 'a': '{outputs[3][0]}'")
    print(f"Batch 4 result for 'a': '{outputs[4][0]}'")

    # All results for 'a' should be the same
    all_a_results = [outputs[0][0], outputs[1][0], outputs[2][0], outputs[3][0], outputs[4][0], outputs[4][1]]
    print(f"\nAll results for 'a': {all_a_results}")

    # Check that all occurrences of "a" generate the same output
    assert all(result == single for result in all_a_results), \
        f"Not all 'a' generations are equal. Expected all to be '{single}', but got: {all_a_results}"

    # Check that different inputs produce different outputs (model responds to input)
    # From batch 3: ["a", "b", "a_long_word"] - "a" and "b" should differ
    all_outputs = set(chain.from_iterable(outputs))
    assert len(all_outputs) > 1, \
        "All different input produced identical outputs. Model may not be responding to input."

    print("✅ Generation test passed - no batch interference detected")


def test_same_text_in_batch(generation_model_setup):
    """Test that same text in batch returns same output."""
    model, processor, collator = generation_model_setup

    print("\n=== Testing same text in batch returns same output ===")

    # Test with 4 times "a"
    batch_a = predict_texts(["a", "a", "a", "a"], model, processor, collator)
    print(f"\nBatch with 4x 'a': {batch_a}")
    assert all(
        result == batch_a[0] for result in batch_a), f"Same text 'a' in batch produced different outputs: {batch_a}"
    print("✅ All 'a' inputs in batch produced same output")

    # Test with 4 times "b"
    batch_b = predict_texts(["b", "b", "b", "b"], model, processor, collator)
    print(f"\nBatch with 4x 'b': {batch_b}")
    assert all(
        result == batch_b[0] for result in batch_b), f"Same text 'b' in batch produced different outputs: {batch_b}"
    print("✅ All 'b' inputs in batch produced same output")


def test_batch_vs_individual_different_lengths():
    """
    BUG from: https://github.com/sign/WeLT/issues/49

    Test that batch generation matches individual generation for texts of different lengths.

    This test uses the "sign/WeLT-string-repetition" model from HuggingFace and tests
    three texts of different lengths: "A B C D" (4 words), "E F G" (3 words), "H I" (2 words).

    The test should fail if there's a bug where batch processing doesn't properly handle
    texts of different lengths.
    """
    print("\n=== Testing batch vs individual generation for different lengths ===")

    # Load the trained model from HuggingFace
    print("Loading model from HuggingFace: sign/WeLT-string-repetition")
    model = AutoModelForCausalLM.from_pretrained("sign/WeLT-string-repetition", trust_remote_code=True)
    model.eval()

    # Get processor from setup_tiny_model with no image encoder
    print("Setting up processor")
    _, processor, collator = setup_tiny_model(image_encoder_name=None)

    # Test texts of different lengths using the string-repetition task format
    # Format: <text>\x0E<content>\x0F<repeat>
    texts = [
        "<text>\x0EA B C D\x0F<repeat> ",  # 4 words (longest)
        "<text>\x0EE F G\x0F<repeat> ",    # 3 words (medium)
        "<text>\x0EH I\x0F<repeat> ",      # 2 words (shortest)
    ]

    print(f"\nTest texts: {texts}")

    def generate_batch(batch):
        """Helper function to generate outputs for a batch."""
        with torch.no_grad():
            return model.generate(
                **batch,
                processor=processor,
                max_generated_words=10,
            )

    # Prepare batches ahead of time
    individual_batches = [dataset_to_batch(model, processor, collator, make_dataset([text])) for text in texts]
    combined_batch = dataset_to_batch(model, processor, collator, make_dataset(texts))

    # Generate each text individually
    print("\n--- Individual generation ---")
    individual_outputs = []
    for text, batch in zip(texts, individual_batches, strict=False):
        outputs = generate_batch(batch)
        individual_outputs.append(outputs[0])
        print(f"  '{text}' -> '{outputs[0]}'")

    # Generate all texts as a batch
    print("\n--- Batch generation ---")
    batch_outputs = generate_batch(combined_batch)
    for text, output in zip(texts, batch_outputs, strict=False):
        print(f"  '{text}' -> '{output}'")

    # Check that batch outputs match individual outputs
    print("\n--- Checking consistency ---")
    all_match = True
    for i, (text, individual, batch) in enumerate(zip(texts, individual_outputs, batch_outputs, strict=False)):
        match = individual == batch
        status = "✅" if match else "❌"
        print(f"{status} Text {i} ('{text}'):")
        print(f"    Individual: '{individual}'")
        print(f"    Batch:      '{batch}'")
        print(f"    Match:      {match}")

        if not match:
            all_match = False

    # This assertion should fail if there's a bug
    assert all_match, (
        "Batch generation does not match individual generation for texts of different lengths!\n"
        "Individual outputs: " + str(individual_outputs) + "\n"
        "Batch outputs:      " + str(batch_outputs)
    )

    print("\n✅ All outputs match - batch and individual generation are consistent!")


def test_mid_word_generation():
    """Test that the model can continue generating from a partial word."""
    from words_segmentation.pretokenizer import is_word_complete

    model = AutoModelForCausalLM.from_pretrained("sign/WeLT-string-repetition", trust_remote_code=True)
    model.eval()

    _, processor, _ = setup_tiny_model(image_encoder_name=None)

    # First, the real output
    text = "<text>\x0eHello\x0f<repeat> "
    words = processor.pretokenize(text)
    assert is_word_complete(words[-1])
    batch = processor([text], collated=True)
    output = model.generate(**batch, processor=processor, max_generated_words=2)[0]
    assert output == "Hello"

    # Then, test with a partial word
    text = "<text>\x0eHello\x0f<repeat> Hel"
    words = processor.pretokenize(text)
    assert not is_word_complete(words[-1])
    batch = processor([text], collated=True)
    output = model.generate(**batch, processor=processor, max_generated_words=2)[0]
    assert output == "lo"


def test_bytes_generation_config_max_new_tokens():
    """Test that bytes_generation_config max_new_tokens truncates per-word output."""
    model = AutoModelForCausalLM.from_pretrained("sign/WeLT-string-repetition", trust_remote_code=True)
    model.eval()

    _, processor, _ = setup_tiny_model(image_encoder_name=None)

    text = "<text>\x0eHello\x0f<repeat> "
    batch = processor([text], collated=True)

    # Normal generation should produce "Hello"
    normal_output = model.generate(**batch, processor=processor, max_generated_words=2)[0]
    assert normal_output == "Hello"

    # With max_new_tokens=2, per-word byte generation is limited to 2 bytes → "He"
    truncated_output = model.generate(
        **batch, processor=processor, max_generated_words=2,
        bytes_generation_config=GenerationConfig(max_new_tokens=2),
    )[0]
    assert len(truncated_output) < len(normal_output), \
        f"Expected truncated output shorter than '{normal_output}', got '{truncated_output}'"
    assert truncated_output == normal_output[:2], \
        f"Expected '{normal_output[:2]}', got '{truncated_output}'"


def test_bytes_generation_config_min_new_tokens():
    """Test that min_new_tokens forces per-word byte generation past the natural word boundary."""
    model = AutoModelForCausalLM.from_pretrained("sign/WeLT-string-repetition", trust_remote_code=True)
    model.eval()

    _, processor, _ = setup_tiny_model(image_encoder_name=None)

    text = "<text>\x0eHello\x0f<repeat> "
    batch = processor([text], collated=True)

    normal_output = model.generate(**batch, processor=processor, max_generated_words=2)[0]
    assert normal_output == "Hello"

    # min_new_tokens=10 forces at least 10 bytes per word, overriding WordStoppingCriteria
    min_output = model.generate(
        **batch, processor=processor, max_generated_words=2,
        bytes_generation_config=GenerationConfig(min_new_tokens=10),
    )[0]
    assert len(min_output) > len(normal_output), \
        f"Expected longer than '{normal_output}', got '{min_output}'"


def test_bytes_generation_config_exact_length():
    """Test that min_new_tokens=3 and max_new_tokens=3 forces exactly 3 bytes per word."""
    model = AutoModelForCausalLM.from_pretrained("sign/WeLT-string-repetition", trust_remote_code=True)
    model.eval()

    _, processor, _ = setup_tiny_model(image_encoder_name=None)

    text = "<text>\x0eHello\x0f<repeat> "
    batch = processor([text], collated=True)

    output = model.generate(
        **batch, processor=processor, max_generated_words=1,
        bytes_generation_config=GenerationConfig(min_new_tokens=3, max_new_tokens=3),
    )[0]
    assert len(output) == 3, \
        f"Expected exactly 3 chars, got '{output}' ({len(output)} chars)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
