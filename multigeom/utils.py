"""
Shared utilities: RNG seeding, provenance records, device selection and a progress line. Configuration loading and
project paths are in `multigeom.config`.
"""
from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch

from .config import config_path, package_root, project_root


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
            resolved = config_path(path)
            configs[name] = {"path": str(resolved), "text": Path(resolved).read_text()}
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
                     ("multigeom", "torch", "torchvision", "diffusers", "plenoptic", "numpy", "h5py")},
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


def subprocess_env(extra: dict | None = None) -> dict:
    """
    The environment for a child process that runs `python -m multigeom...`: this package is importable in it even if it
    is not installed (a source checkout), and `extra` is added.
    """
    env = dict(os.environ)
    parent = str(package_root().parent)
    env["PYTHONPATH"] = os.pathsep.join([parent] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.update(extra or {})
    return env


class ProgressLine:
    """
    A one-line progress bar that is updated in place: in a notebook through `display(..., display_id=True)`
    (no ipywidgets needed, no extra lines), in a terminal with a carriage return. The text is tqdm's own
    format, e.g. `estimate G(x): 50%|█████     | 1/2 [00:15<00:15, 15.00s/img, 1 worker(s)]`.
    """

    def __init__(self, total: int, initial: int = 0, desc: str = "estimate G(x)", unit: str = "img",
                 notebook: bool | None = None):
        self.total, self.initial, self.desc, self.unit = total, initial, desc, unit
        self.t0, self._last, self._handle = time.time(), None, None
        if notebook is None:                                   # detect a Jupyter kernel
            try:
                from IPython import get_ipython
                ip = get_ipython()
                notebook = ip is not None and getattr(ip, "kernel", None) is not None
            except ImportError:
                notebook = False
        self._notebook = notebook
        self.update(initial)

    def text(self, n: int, postfix: str = "") -> str:
        """The progress line for `n` finished items, in tqdm's format."""
        from tqdm import tqdm
        return tqdm.format_meter(n, self.total, time.time() - self.t0, initial=self.initial, prefix=self.desc,
                                 unit=self.unit, postfix=postfix, ncols=100)

    def update(self, n: int, postfix: str = "") -> None:
        """Show `n` finished items; does nothing if neither `n` nor `postfix` changed."""
        if (n, postfix) == self._last:
            return                                  # nothing but the clock changed: leave the line alone
        self._last = (n, postfix)
        self._write(self.text(n, postfix))

    def _write(self, text: str) -> None:
        if self._notebook:
            from IPython.display import Pretty, display
            if self._handle is None:
                self._handle = display(Pretty(text), display_id=True)     # Pretty: plain text, without quotes
            else:
                self._handle.update(Pretty(text))
        else:
            sys.stdout.write("\r" + text)
            sys.stdout.flush()

    def close(self) -> None:
        """End the line (terminal only)."""
        if not self._notebook:
            sys.stdout.write("\n")
            sys.stdout.flush()
