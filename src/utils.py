"""
Shared utilities: YAML config loading, RNG seeding, project-root resolution.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml


def project_root() -> Path:
    """Return the repository root (the directory that contains 'src/'."""
    return Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """
    Resolve the data directory:
      1. $NEURIPS_DATA_DIR if set,
      2. else <project_root>/data.
    """
    env = os.environ.get("NEURIPS_DATA_DIR")
    return Path(env) if env else project_root() / "data"


def _to_namespace(obj: Any) -> Any:
    """Recursively convert nested dicts to SimpleNamespace for attribute access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(x) for x in obj]
    return obj


def load_config(path: str | Path) -> SimpleNamespace:
    """Load a YAML config file into a nested SimpleNamespace."""
    with open(path) as f:
        return _to_namespace(yaml.safe_load(f))


def set_seed(seed: int) -> None:
    """Seed Python, numpy, and torch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(arg: str | None = None) -> torch.device:
    """Resolve a torch.device, preferring CUDA when available."""
    if arg is not None:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total
