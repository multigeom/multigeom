"""
Configuration: loading YAML files, the default configurations shipped with the package,
and the paths of a project.

The default configurations (`multigeom/configs/*.yaml`) are package data, so they are
found wherever the package is installed: `load_config("estimator")` loads
`estimator.yaml` of the package, and `load_config("path/to/my.yaml")` loads a file of
your own. They are

    estimator   settings of the G(x) estimator (samples, timesteps, sigma, seed)
    diffusion   the conditional diffusion model and its training recipe
    inception, resnet50, onoff
                the reference encoders and the training of their readouts
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def package_root() -> Path:
    """The folder of the installed package (the one that contains this file)."""
    return Path(__file__).resolve().parent


def project_root() -> Path:
    """
    The working root of a project, where `runs/` and `data/` live by default:
    $MULTIGEOM_REPO if set, else the current directory. Run the pipeline, or a notebook,
    from the folder that holds `runs/` and `data/`.
    """
    env = os.environ.get("MULTIGEOM_REPO")
    return Path(env) if env else Path.cwd()


def data_dir() -> Path:
    """
    Resolve the data directory:
      1. $NEURIPS_DATA_DIR if set,
      2. else <project_root>/data.
    """
    env = os.environ.get("NEURIPS_DATA_DIR")
    return Path(env) if env else project_root() / "data"


def config_path(name_or_path: str | Path) -> Path:
    """
    The YAML file for a packaged config name ("estimator", "diffusion", "inception",
    ...) or for a path.
    """
    p = Path(name_or_path)
    if p.suffix == "" and not p.exists():
        packaged = CONFIG_DIR / f"{p.name}.yaml"
        if packaged.exists():
            return packaged
    return p


def _to_namespace(obj: Any) -> Any:
    """Recursively convert nested dicts to SimpleNamespace for attribute access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(x) for x in obj]
    return obj


def load_config(name_or_path: str | Path) -> SimpleNamespace:
    """
    Load a YAML config (a packaged name such as "estimator", or a file path) into a
    nested SimpleNamespace.
    """
    with open(config_path(name_or_path)) as f:
        return _to_namespace(yaml.safe_load(f))
