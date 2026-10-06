"""Shared test helpers: repository on sys.path, stage-module loader, tiny configs."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# diffusers imports transformers, which imports TensorFlow/JAX when installed;
# that is slow and can hang. The pipeline is torch-only.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def load_stage(filename: str):
    """Import stages/<filename> (file names start with digits, so not importable by name)."""
    path = REPO / "stages" / filename
    spec = importlib.util.spec_from_file_location(f"stage_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def linear_beta_alphas_cumprod(T: int = 1000, beta_start: float = 1e-4,
                               beta_end: float = 0.02) -> torch.Tensor:
    """ᾱ_t = Π_{s≤t} (1 − β_s) for the linear β schedule of Ho et al. (float64)."""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    return torch.cumprod(1.0 - betas, dim=0)


def estimator_cfg(n_samples: int, batch_size: int, start: int, stop: int, step: int,
                  sigma: float, top_k: int) -> SimpleNamespace:
    """Namespace shaped like configs/estimator.yaml."""
    return SimpleNamespace(
        top_k=top_k,
        monte_carlo=SimpleNamespace(
            n_samples=n_samples, batch_size=batch_size,
            timestep_grid=SimpleNamespace(start=start, stop=stop, step=step),
            observation_noise_sigma=sigma,
        ),
    )


@pytest.fixture
def repo_root() -> Path:
    return REPO


def write_fake_tvsd(tvsd_dir: Path, n_train: int = 9, n_test: int = 4, n_trials: int = 6,
                    n_channels: int = 1024, seed: int = 0, transpose: bool = False) -> dict:
    """
    Write a synthetic MATLAB-v7.3-like TVSD pair with h5py, laid out like the
    real files as h5py reads them:

      monkeyN_THINGS_normMUA.mat
        train_MUA      [n_train, n_channels]   value = 1000·image + channel
        test_MUA       [n_test,  n_channels]   trial average of test_MUA_reps
        test_MUA_reps  [n_trials, n_test, n_channels]
        oracle         [n_channels, 1]         random permutation of distinct values
      things_imgs.mat
        train_imgs/things_path, test_imgs/things_path: [n, 1] object references
        to [len, 1] uint16 character arrays under '#refs#' (Windows separators).

    `transpose=True` stores the response matrices as [n_channels, n_images]
    (the MATLAB orientation) to test the layout check.
    """
    import h5py
    import numpy as np

    tvsd_dir = Path(tvsd_dir)
    tvsd_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    train = (1000.0 * np.arange(n_train)[:, None] + np.arange(n_channels)[None, :])
    test_mean = rng.standard_normal((n_test, n_channels))
    noise = rng.standard_normal((n_trials, n_test, n_channels))
    noise -= noise.mean(axis=0, keepdims=True)
    reps = test_mean[None] + noise
    oracle = rng.permutation(n_channels).astype(np.float64) / n_channels
    with h5py.File(tvsd_dir / "monkeyN_THINGS_normMUA.mat", "w") as f:
        f["train_MUA"] = train.T if transpose else train
        f["test_MUA"] = test_mean.T if transpose else test_mean
        f["test_MUA_reps"] = reps
        f["oracle"] = oracle[:, None]
    paths = {
        "train": [f"cat{i:02d}\\cat{i:02d}_{i:02d}s.jpg" for i in range(n_train)],
        "test": [f"dog{i:02d}\\dog{i:02d}_{i:02d}b.jpg" for i in range(n_test)],
    }
    with h5py.File(tvsd_dir / "things_imgs.mat", "w") as f:
        refs_grp = f.create_group("#refs#")
        for split, plist in paths.items():
            refs = []
            for i, p in enumerate(plist):
                chars = np.array([ord(ch) for ch in p], dtype=np.uint16)[:, None]
                refs.append(refs_grp.create_dataset(f"{split}_{i}", data=chars).ref)
            g = f.create_group(f"{split}_imgs")
            g.create_dataset("things_path", data=np.array(refs, dtype=h5py.ref_dtype)[:, None])
    return {"train": train, "test_mean": test_mean, "reps": reps, "oracle": oracle,
            "paths": paths}
