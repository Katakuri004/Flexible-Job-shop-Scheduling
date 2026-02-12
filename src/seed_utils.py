import random
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SeedConfig:
    """
    Centralized seed configuration for deterministic replay.

    For now we cover Python's `random` and NumPy. PyTorch and other
    libraries can be added here later when RL is introduced.
    """

    base_seed: int = 42
    python_seed: Optional[int] = None
    numpy_seed: Optional[int] = None

    def resolved_python_seed(self) -> int:
        return self.python_seed if self.python_seed is not None else self.base_seed

    def resolved_numpy_seed(self) -> int:
        return self.numpy_seed if self.numpy_seed is not None else self.base_seed


def set_global_seeds(config: SeedConfig) -> None:
    """
    Apply the configured seeds to all supported RNGs.

    This function should be called at the start of any experiment or test
    that we want to be reproducible.
    """

    random.seed(config.resolved_python_seed())
    np.random.seed(config.resolved_numpy_seed())


__all__ = ["SeedConfig", "set_global_seeds"]

