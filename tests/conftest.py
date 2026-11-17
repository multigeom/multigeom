"""
Shared test helpers: the package and the repository on sys.path, synthetic data, tiny
configs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

# diffusers imports transformers, which imports TensorFlow/JAX when installed;
# that is slow and can hang. The pipeline is torch-only.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # the package, also without installing
    sys.path.insert(0, str(REPO))


def linear_beta_alphas_cumprod(
    T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    """ᾱ_t = Π_{s≤t} (1 − β_s) for the linear β schedule of Ho et al. (float64)."""
    betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
    return torch.cumprod(1.0 - betas, dim=0)


def estimator_cfg(
    n_samples: int,
    batch_size: int,
    start: int,
    stop: int,
    step: int,
    sigma: float,
    top_k: int,
) -> SimpleNamespace:
    """Namespace shaped like the packaged estimator config."""
    return SimpleNamespace(
        top_k=top_k,
        monte_carlo=SimpleNamespace(
            n_samples=n_samples,
            batch_size=batch_size,
            timestep_grid=SimpleNamespace(start=start, stop=stop, step=step),
            observation_noise_sigma=sigma,
        ),
    )


@pytest.fixture
def repo_root() -> Path:
    return REPO


def write_fake_tvsd(
    tvsd_dir: Path,
    n_train: int = 9,
    n_test: int = 4,
    n_trials: int = 6,
    n_channels: int = 1024,
    seed: int = 0,
    transpose: bool = False,
) -> dict:
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
    train = 1000.0 * np.arange(n_train)[:, None] + np.arange(n_channels)[None, :]
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
            g.create_dataset(
                "things_path", data=np.array(refs, dtype=h5py.ref_dtype)[:, None]
            )
    return {
        "train": train,
        "test_mean": test_mean,
        "reps": reps,
        "oracle": oracle,
        "paths": paths,
    }


# ── mock diffusion networks with known posterior means (stimulus features, spectrum) ──
MOCK_N_NEURONS = 3  # the size of their null token


class GaussianPriorUNet(nn.Module):
    """
    ε(x_t, t, r) of the posterior mean of a zero-mean Gaussian prior of covariance
    C = U diag(c) Uᵀ, x̂(y) = U diag(c / (c + t)) Uᵀ y; records the responses it is
    given.
    """

    n_neurons = MOCK_N_NEURONS

    def __init__(self, alphas_cumprod: torch.Tensor, U: torch.Tensor, c: torch.Tensor):
        super().__init__()
        self.register_buffer("ab", alphas_cumprod.float())
        self.register_buffer("U", U.float())
        self.register_buffer("c", c.float())
        self.responses: list[torch.Tensor] = []

    def forward(self, x_t, t, r):
        self.responses.append(r.detach().clone())
        ab = self.ab[t].view(-1, 1)
        noise_variance = (1 - ab) / ab
        y = x_t.flatten(1) / ab.sqrt()
        gains = self.c / (self.c + noise_variance)  # [B, P]
        x_hat = ((y @ self.U) * gains) @ self.U.T
        eps = (x_t.flatten(1) - ab.sqrt() * x_hat) / (1 - ab).sqrt()
        return eps.view_as(x_t)


class ConstantMeanUNet(nn.Module):
    """ε of a posterior mean that ignores x_t: x̂ = 0, so ε = x_t / √(1 − ᾱ)."""

    n_neurons = MOCK_N_NEURONS

    def __init__(self, alphas_cumprod: torch.Tensor):
        super().__init__()
        self.register_buffer("ab", alphas_cumprod.float())

    def forward(self, x_t, t, r):
        return x_t / (1 - self.ab[t]).sqrt().view(-1, 1, 1, 1)


class NoNoiseUNet(nn.Module):
    """ε = 0: the posterior mean is the noisy image itself, x̂ = y, so M = I."""

    n_neurons = MOCK_N_NEURONS

    def forward(self, x_t, t, r):
        return torch.zeros_like(x_t)


def gaussian_prior(n_pixels: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A random orthogonal basis U [P, P] and a spectrum c [P] decaying by halves from 2
    (float64): the covariance C = U diag(c) Uᵀ of a Gaussian prior for the mocks above.
    """
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn(n_pixels, n_pixels, generator=generator, dtype=torch.float64)
    U, _ = torch.linalg.qr(basis)
    c = 2.0 * 0.5 ** torch.arange(n_pixels, dtype=torch.float64)
    return U, c


class LinearGaussianUNet(nn.Module):
    """
    ε(x_t, t, r) of the exact posterior means of a linear-Gaussian model: prior
    x ~ N(0, C), responses r = A x + σ ξ, noisy image y = x_t / √ᾱ (variance-exploding).
    The null token gives E[x | y]; a response gives E[x | y, r].
    """

    def __init__(self, alphas_cumprod, prior, readout, sigma: float):
        super().__init__()
        self.register_buffer("ab", alphas_cumprod.double())
        self.register_buffer("prior_precision", torch.linalg.inv(prior.double()))
        self.register_buffer("readout", readout.double())
        self.sigma = float(sigma)
        self.n_neurons = readout.shape[0]

    def forward(self, x_t, t, r):
        ab = self.ab[t].view(-1, 1)
        noise_variance = (1 - ab) / ab
        y = x_t.flatten(1).double() / ab.sqrt()

        # the posterior precision and evidence: the prior and y, plus r unless null
        has_response = r.any(dim=1).double().view(-1, 1)
        identity = torch.eye(y.shape[1], dtype=torch.float64)
        likelihood = self.readout.T @ self.readout / self.sigma**2
        precision = self.prior_precision + identity / noise_variance.view(-1, 1, 1)
        precision = precision + has_response.view(-1, 1, 1) * likelihood
        evidence = y / noise_variance
        evidence = evidence + has_response * (r.double() @ self.readout) / self.sigma**2

        # the posterior mean, and the noise prediction Tweedie's formula turns into it
        x_hat = torch.linalg.solve(precision, evidence.unsqueeze(-1)).squeeze(-1)
        eps = (x_t.flatten(1).double() - ab.sqrt() * x_hat) / (1 - ab).sqrt()
        return eps.view_as(x_t).to(x_t.dtype)


class LinearEncoder(nn.Module):
    """The responses A x of the linear-Gaussian model, without their noise."""

    def __init__(self, readout):
        super().__init__()
        self.register_buffer("readout", readout.float())
        self.n_neurons = readout.shape[0]

    def forward(self, images):
        return images.flatten(1) @ self.readout.T
