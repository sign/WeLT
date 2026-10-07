import torch


def stack_pad_tensors(tensors: list[torch.Tensor], pad_value=0) -> torch.Tensor:
    """Stack tensors of the same rank, right-padding each dimension to the largest size."""
    if len(tensors) == 1:
        return tensors[0].unsqueeze(0)
    return torch.nested.nested_tensor(tensors).to_padded_tensor(pad_value)


def collate_fn(batch: list[dict[str, torch.Tensor]], pad_value=0) -> dict[str, torch.Tensor]:
    return {key: stack_pad_tensors([item[key] for item in batch], pad_value=pad_value) for key in batch[0]}
