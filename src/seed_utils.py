import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class SeedConfig:
    """
    Centralized seed configuration for deterministic replay.

    Covers Python's random, NumPy, and PyTorch (when available).
    """

    base_seed: int = 42
    python_seed: Optional[int] = None
    numpy_seed: Optional[int] = None
    torch_seed: Optional[int] = None

    def resolved_python_seed(self) -> int:
        return self.python_seed if self.python_seed is not None else self.base_seed

    def resolved_numpy_seed(self) -> int:
        return self.numpy_seed if self.numpy_seed is not None else self.base_seed

    def resolved_torch_seed(self) -> int:
        return self.torch_seed if self.torch_seed is not None else self.base_seed


def set_global_seeds(config: SeedConfig) -> None:
    """
    Apply the configured seeds to all supported RNGs.

    This function should be called at the start of any experiment or test
    that we want to be reproducible.
    """
    seed_py = config.resolved_python_seed()
    seed_np = config.resolved_numpy_seed()

    random.seed(seed_py)
    np.random.seed(seed_np)

    if _TORCH_AVAILABLE:
        seed_torch = config.resolved_torch_seed()
        torch.manual_seed(seed_torch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_torch)


__all__ = ["SeedConfig", "set_global_seeds"]

