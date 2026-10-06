"""
Shared utilities: YAML config loading, RNG seeding, provenance records,
project-root resolution.
"""
from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    """
    Seed Python's and torch's global generators (CPU + CUDA).

    Used for training stages, where data shuffling, dropout and parameter
    initialisation draw from torch's global stream. Monte-Carlo estimation
    uses explicit per-image generators instead (see `make_generator`).
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_generator(seed: int, device: torch.device | str) -> torch.Generator:
    """A torch.Generator on `device`, seeded with `seed`."""
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=project_root(), capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def provenance(config_paths: dict[str, str | Path] | None = None,
               seed: int | None = None,
               argv: list[str] | None = None) -> dict:
    """
    Record what produced an output: git commit (and whether the tree was
    dirty), command line, seed, config file contents, and package versions.
    Stored inside every checkpoint and result file the stages write.
    """
    configs = {}
    for name, path in (config_paths or {}).items():
        if path is not None:
            configs[name] = {"path": str(path), "text": Path(path).read_text()}
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "argv": list(sys.argv if argv is None else argv),
        "seed": seed,
        "configs": configs,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": {p: _version(p) for p in
                     ("torch", "torchvision", "diffusers", "plenoptic", "numpy", "h5py")},
        "cuda": torch.version.cuda,
    }


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
