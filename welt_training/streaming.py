from itertools import islice

import torch
from datasets import IterableDataset


class CustomIterableDataset(IterableDataset):
    """IterableDataset wrapper that supports with_transform and set_transform like regular Dataset."""

    def __init__(self, dataset: IterableDataset):
        self._dataset = dataset
        self._transform = None
        self._transforms = None

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __getitem__(self, key):
        raise TypeError("CustomIterableDataset does not support indexing. Use iteration instead.")

    def __iter__(self):
        for example in self._dataset:
            if self._transform is not None:
                batch = {k: [v] for k, v in example.items()}
                result = self._transform(batch)
                yield {k: v[0] if isinstance(v, list) and len(v) == 1 else v
                       for k, v in result.items()}
            else:
                yield example

    def set_transform(self, transform):
        self._transform = transform
        self._transforms = transform

    def with_transform(self, transform):
        new_dataset = CustomIterableDataset(self._dataset)
        new_dataset.set_transform(transform)
        return new_dataset

    def map(self, *args, **kwargs):
        return CustomIterableDataset(self._dataset.map(*args, **kwargs))

    def filter(self, *args, **kwargs):
        return CustomIterableDataset(self._dataset.filter(*args, **kwargs))

    def take(self, n):
        return CustomIterableDataset(self._dataset.take(n))


class TorchIterableAdapter(torch.utils.data.IterableDataset):
    """Expose HF iterables to DataLoader and optionally shard torch iterables by rank."""

    def __init__(self, dataset, rank=0, world_size=1):
        self._dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        for index, example in enumerate(self._dataset):
            if index % self.world_size == self.rank:
                yield example


def take_streaming_dataset(dataset, count):
    """Keep a fixed streaming subset that remains iterable on subsequent epochs.

    HF take() locks source order and newer datasets releases reject epoch
    reshuffling through that operation. A generator preserves the same subset
    without exposing the locked source to the outer dataset's epoch handling.
    """
    def examples():
        yield from islice(dataset, count)

    return IterableDataset.from_generator(examples, features=dataset.features)
