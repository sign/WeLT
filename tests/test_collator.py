import torch

from welt.collator import collate_fn, stack_pad_tensors


def test_stack_pad_tensors_pads_every_dimension():
    a = torch.ones(2, 3, dtype=torch.long)
    b = torch.ones(3, 1, dtype=torch.long) * 2
    stacked = stack_pad_tensors([a, b])
    assert stacked.shape == (2, 3, 3)
    assert torch.equal(stacked[0, :2], a)
    assert (stacked[0, 2] == 0).all()
    assert torch.equal(stacked[1, :, :1], b)
    assert (stacked[1, :, 1:] == 0).all()


def test_collate_fn_keeps_dtypes():
    batch = [{"mask": torch.tensor([True, False]), "ids": torch.tensor([1])},
             {"mask": torch.tensor([True]), "ids": torch.tensor([2, 3])}]
    collated = collate_fn(batch)
    assert collated["mask"].dtype == torch.bool
    assert torch.equal(collated["ids"], torch.tensor([[1, 0], [2, 3]]))
