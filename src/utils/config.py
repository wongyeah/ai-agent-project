"""
Recursive config object: turns a nested dict (e.g. loaded from YAML) into
an object with attribute access, so `cfg.agent.search.debug_prob` works
instead of `cfg["agent"]["search"]["debug_prob"]`.
"""

import random

import numpy as np
import torch


class Config:
    """Recursive configuration class with dot-notation attribute access."""

    def __init__(self, dictionary: dict):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


def set_seed(seed: int = 531) -> None:
    """Set random seeds across python/numpy/torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
