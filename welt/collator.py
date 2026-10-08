import torch


def stack_pad_tensors(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Stack tensors of the same rank, right-padding each dimension to the largest size with zeros."""
    return torch.nested.nested_tensor(tensors).to_padded_tensor(0)


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: stack_pad_tensors([item[key] for item in batch]) for key in batch[0]}
