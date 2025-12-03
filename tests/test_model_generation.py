import pytest
import torch

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

    print("-" * 30)
    dataset = make_dataset(texts)
    batch = dataset_to_batch(model, processor, collator, dataset)

    processor.max_word_length = 5

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
        ["a", "two words", ""],
        ["a", "even three words"],
        ["a", "b", "a_long_word"],
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
    result_a = outputs[3][0]
    result_b = outputs[3][1]
    assert result_a != result_b, \
        f"Different inputs 'a' and 'b' produced identical outputs: '{result_a}'. Model may not be responding to input."

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

    # Check that different inputs produce different outputs (model responds to input)
    assert batch_a[0] != batch_b[0], \
        f"Different inputs 'a' and 'b' produced identical outputs: '{batch_a[0]}'. Model may not be responding to input."
    print("✅ 'a' and 'b' produce different outputs - model responds to input")

    print("✅ Test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
