"""
Synthetic neural data with a known ground truth, to learn the tools on and to test them.

`GaborTeacher` is a model of V1-like neurons with known receptive fields: each neuron has a Gabor filter (position,
size, orientation, spatial frequency, phase) and is either a *simple cell* (rectified filter output) or a *complex cell*
(energy: the output of the filter and of its quadrature pair, combined). It follows the encoder contract, so its Fisher
information J(x) is a ground truth for the geometry of an encoder fitted to its responses.

    teacher = GaborTeacher(n_neurons=50, seed=0).calibrate(images)       # responses of order 1
    noisy, clean = make_synthetic_neural_data(teacher, images, noise_sigma=0.5, seed=1)
    ceiling = noise_ceiling(clean, noise_sigma=0.5)                       # best correlation any model can reach

`make_synthetic_experiment` does all of it in one call, on TinyImageNet images, and splits the images for fitting and testing.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn


class GaborTeacher(nn.Module):
    """
    Ground-truth encoder of V1-like neurons. Input [B, 1, size, size] in [-1, 1], output [B, n_neurons].

    Half of the neurons (`complex_fraction`) are complex cells, the others simple cells. The parameters are drawn once
    from `seed`: centres inside the image, filter widths of 0.1 to 0.25 of the image, orientations over the half circle,
    3 to 8 cycles per image, random phases. `gain` scales the responses; `calibrate(images)` sets it so that the typical
    response has unit standard deviation (responses of order 1, as the observation noise assumes).
    """

    def __init__(self, n_neurons: int = 50, image_size: int = 64, seed: int = 0, complex_fraction: float = 1.0,
                 width_range: tuple[float, float] = (0.10, 0.25), cycles_range: tuple[float, float] = (3, 8)):
        super().__init__()
        self.n_neurons = n_neurons
        self.image_size = image_size
        g = np.random.default_rng(seed)
        centre = g.uniform(-0.5, 0.5, size=(n_neurons, 2))              # (x, y) in units of the half image width
        width = g.uniform(*width_range, size=n_neurons) * 2               # Gaussian envelope sigma, in half image widths
        theta = g.uniform(0, math.pi, size=n_neurons)
        cycles = g.uniform(*cycles_range, size=n_neurons)                         # cycles per image
        phase = g.uniform(0, 2 * math.pi, size=n_neurons)
        is_complex = np.zeros(n_neurons, dtype=bool)
        is_complex[g.permutation(n_neurons)[: int(round(complex_fraction * n_neurons))]] = True
        self.params = {"centre": centre, "width": width, "theta": theta, "cycles": cycles, "phase": phase,
                       "is_complex": is_complex}

        ax = np.linspace(-1, 1, image_size)
        xx, yy = np.meshgrid(ax, ax)                                      # image coordinates in [-1, 1]
        filters = np.zeros((n_neurons, 2, image_size, image_size), dtype=np.float32)
        for n in range(n_neurons):
            dx, dy = xx - centre[n, 0], yy - centre[n, 1]
            u = dx * math.cos(theta[n]) + dy * math.sin(theta[n])        # along the carrier
            envelope = np.exp(-(dx ** 2 + dy ** 2) / (2 * width[n] ** 2))
            carrier = 2 * math.pi * cycles[n] * u / 2                     # cycles per image: the image spans 2 units
            even, odd = envelope * np.cos(carrier + phase[n]), envelope * np.sin(carrier + phase[n])
            f = np.stack([even, odd])
            filters[n] = f / (np.linalg.norm(f[0]) + 1e-8)               # unit-norm even filter
        self.register_buffer("filters", torch.from_numpy(filters))
        self.register_buffer("is_complex", torch.from_numpy(is_complex))
        self.register_buffer("gain", torch.ones(n_neurons))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Responses [B, n_neurons] to the images [B, 1, size, size]."""
        flat = x.flatten(1)                                               # [B, P]
        w = self.filters.flatten(2)                                       # [N, 2, P]
        out = torch.einsum("bp,nkp->bnk", flat, w)                        # [B, N, 2]
        simple = torch.relu(out[..., 0])
        energy = torch.sqrt(out[..., 0] ** 2 + out[..., 1] ** 2 + 1e-8)
        return self.gain * torch.where(self.is_complex, energy, simple)

    @torch.no_grad()
    def calibrate(self, images: torch.Tensor, target_std: float = 1.0) -> "GaborTeacher":
        """Set the gain of every neuron so that its responses to `images` have standard deviation `target_std`."""
        self.gain.fill_(1.0)
        r = self(images)
        self.gain.copy_(target_std / r.std(dim=0).clamp_min(1e-6))
        return self


@torch.no_grad()
def make_synthetic_neural_data(teacher: nn.Module, images: torch.Tensor, noise_sigma: float = 0.5, seed: int = 0,
                               batch_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single-trial responses of the teacher: `(noisy, clean)`, both [N, n_neurons], where
    `noisy = clean + noise_sigma * N(0, 1)` is what an experiment would record and `clean` is the ground truth.
    """
    was_training = teacher.training
    teacher.eval()
    clean = torch.cat([teacher(images[i: i + batch_size]) for i in range(0, len(images), batch_size)], dim=0).cpu()
    teacher.train(was_training)
    gen = torch.Generator().manual_seed(seed)
    noisy = clean + noise_sigma * torch.randn(clean.shape, generator=gen)
    return noisy, clean


def noise_ceiling(clean: torch.Tensor, noise_sigma: float) -> np.ndarray:
    """
    Per neuron, the correlation between the noisy responses and the clean ones that a model predicting the clean
    responses exactly would reach on the noisy data: sqrt(var / (var + sigma^2)). No model can do better on average.
    """
    var = clean.var(dim=0, unbiased=False).numpy()
    return np.sqrt(var / (var + noise_sigma ** 2))


class SyntheticExperiment(NamedTuple):
    """
    A synthetic experiment. `x_*` are the images [N, 1, 64, 64] in [-1, 1]; `r_*` the recorded (noisy) responses and `c_*`
    the clean responses, the ground truth, both [N, n_neurons]; `train` is for fitting and `test` for scoring. `teacher` is the
    `GaborTeacher` that made the responses, and `ceiling` the per-neuron noise ceiling of the test responses.
    """
    teacher: GaborTeacher
    x_train: torch.Tensor
    x_test: torch.Tensor
    r_train: torch.Tensor
    r_test: torch.Tensor
    c_train: torch.Tensor
    c_test: torch.Tensor
    ceiling: np.ndarray


def make_synthetic_experiment(
    n_neurons: int = 50,
    noise_sigma: float = 0.22,
    n_train: int = 6000,
    n_test: int = 2000,
    seed: int = 0,
    noise_seed: int = 1,
    calibration_images: int = 2000,
    data_directory: str | Path | None = None,
    download: bool = True,
) -> SyntheticExperiment:
    """
    Simulate an experiment that shows TinyImageNet images to V1-like neurons.

    The first `n_train + n_test` validation images are shown to a `GaborTeacher` of `n_neurons` neurons, calibrated so that
    its responses have unit scale; the recorded responses are `clean + noise_sigma * N(0, 1)`. The first `n_train` images
    are for fitting, the others for testing. TinyImageNet is downloaded the first time (240 MB) unless `download` is False.

    Args:
        n_neurons: neurons to simulate.
        noise_sigma: standard deviation of the trial-to-trial noise (the paper measures 0.22 in V1).
        n_train, n_test: images (with their responses) for fitting and for testing.
        seed: seed of the neurons' filters. noise_seed: seed of the noise.
        calibration_images: number of the first images used to set the scale of the responses.
        data_directory: the folder that contains `tiny-imagenet-200` (default: `data_dir()`).
        download: download TinyImageNet if it is missing.
    """
    from ..config import data_dir
    from .download import download_tinyimagenet
    from .tinyimagenet import TinyImageNetDataset

    data_directory = Path(data_directory) if data_directory else data_dir()
    if download:
        download_tinyimagenet(data_directory)
    dataset = TinyImageNetDataset("val", data_directory / "tiny-imagenet-200")
    n = n_train + n_test
    if n > len(dataset):
        raise ValueError(f"n_train + n_test = {n} images asked, the validation set has {len(dataset)}")
    images = torch.stack([dataset[i] for i in range(n)])
    teacher = GaborTeacher(n_neurons, seed=seed).calibrate(images[:calibration_images])
    noisy, clean = make_synthetic_neural_data(teacher, images, noise_sigma=noise_sigma, seed=noise_seed)
    return SyntheticExperiment(
        teacher=teacher, x_train=images[:n_train], x_test=images[n_train:], r_train=noisy[:n_train], r_test=noisy[n_train:],
        c_train=clean[:n_train], c_test=clean[n_train:], ceiling=noise_ceiling(clean, noise_sigma),
    )
