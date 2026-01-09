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
