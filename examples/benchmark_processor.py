from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl import pack_dataset

from tests.test_model import setup_tiny_model


def collate_fn(batch):
    """Custom collate function to handle batches with variable-sized tensors."""
    # Don't collate, just return the list of examples
    return batch

if __name__ == '__main__':
    dataset = load_dataset("Helsinki-NLP/opus-100", "en-he", split="train")

    model, processor, collator = setup_tiny_model()

    # Convert dataset to text
    dataset = dataset.map(
        lambda batch: {"text": f"<en>\x0E{batch['translation']['en']}\x0F<he> {batch['translation']['he']}"},
        remove_columns=["translation"])
    dataset = processor.pretokenize_dataset(dataset)

    dataset = pack_dataset(dataset, seq_length=128)
    dataset = dataset.with_transform(processor)

    dataloader = DataLoader(dataset,
                            batch_size=2,
                            num_workers=2,
                            collate_fn=collator)

    for _ in tqdm(dataloader):
        pass
