"""Device selection utility."""

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available torch device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
