"""
Linear–nonlinear–Poisson (LN) models of retinal ganglion cells, fitted to recorded
spike counts, and their mosaics.

A cell: a difference-of-Gaussians receptive field w (a centre and an opposite-sign
surround) applied to the stimulus minus its grey level, then a sigmoid,

    λ(x) = r_max σ(⟨w, x − grey⟩ + b),

the expected spike count of the response window, with Poisson counts. Generalised from
the LN-sigmoid encoder of proj_zsmultigeom (`encoders/lnp.py`: Gaussian receptive fields
on a grid, a sigmoid, Poisson counts) with a surround and fitted parameters.

A mosaic: one fitted cell copied on a hexagonal lattice (200 µm apart, the measured
spacing of these cells), as open-retina's mosaics of the Lorenzi models. Its Fisher
information about the image is exact: each cell's is rank one, g_n g_nᵀ with
g_n = λ_n'/√λ_n w_n (`compute_fisher_per_cell` with `count_noise()`).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from ..geometry.noise import PoissonNoise


def get_difference_of_gaussians(
    size: int,
    center: torch.Tensor,
    log_sigmas: torch.Tensor,
    surround: torch.Tensor,
    amplitude: torch.Tensor,
) -> torch.Tensor:
    """
    The receptive field [size, size]: amplitude (G_c − surround G_s), with G_c and G_s
    Gaussians of unit sum centred at `center` (row, column, in pixels) of widths
    exp(log_sigmas) (the surround's at least the centre's).
    """
    rows = torch.arange(size, dtype=center.dtype, device=center.device)
    d2 = (rows[:, None] - center[0]) ** 2 + (rows[None, :] - center[1]) ** 2
    sigma_c = log_sigmas[0].exp()
    sigma_s = sigma_c + log_sigmas[1].exp()
    gauss_c = torch.exp(-d2 / (2 * sigma_c**2)) / (2 * math.pi * sigma_c**2)
    gauss_s = torch.exp(-d2 / (2 * sigma_s**2)) / (2 * math.pi * sigma_s**2)
    return amplitude * (gauss_c - surround * gauss_s)


class LNCell(nn.Module):
    """
    One LN cell on `size` x `size` stimuli (pixels of its experiment) with grey level
    `grey`. Parameters: the receptive field's centre (row, column), log widths (centre,
    and surround minus centre), surround weight in [0, 1] (through a sigmoid),
    amplitude (its sign is the cell's polarity), and the sigmoid's bias and log
    maximum. forward(x [B, size, size] or [B, 1, size, size]) → λ [B].
    """

    def __init__(self, size: int, grey: float, center=None, amplitude: float = 1.0):
        super().__init__()
        self.size, self.grey = int(size), float(grey)
        center = center if center is not None else (size / 2 - 0.5, size / 2 - 0.5)
        self.center = nn.Parameter(torch.tensor(center, dtype=torch.float32))
        self.log_sigmas = nn.Parameter(torch.tensor([1.0, 1.0]))
        self.surround_logit = nn.Parameter(torch.tensor(0.0))
        self.amplitude = nn.Parameter(torch.tensor(float(amplitude)))
        self.bias = nn.Parameter(torch.tensor(-1.0))
        self.log_max = nn.Parameter(torch.tensor(1.0))

    def receptive_field(self) -> torch.Tensor:
        """The receptive field w [size, size]."""
        return get_difference_of_gaussians(
            self.size,
            self.center,
            self.log_sigmas,
            torch.sigmoid(self.surround_logit),
            self.amplitude,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The expected spike counts λ [B] of the stimuli `x`."""
        x = x.reshape(len(x), -1) - self.grey
        drive = x @ self.receptive_field().reshape(-1) + self.bias
        return self.log_max.exp() * torch.sigmoid(drive)

    def describe(self) -> dict:
        """The fitted parameters, in pixels and spike counts."""
        sigma_c = float(self.log_sigmas[0].exp())
        return {
            "center": [round(float(v), 2) for v in self.center],
            "sigma_center_px": round(sigma_c, 2),
            "sigma_surround_px": round(sigma_c + float(self.log_sigmas[1].exp()), 2),
            "surround_weight": round(float(torch.sigmoid(self.surround_logit)), 3),
            "polarity": "ON" if float(self.amplitude) > 0 else "OFF",
            "max_count": round(float(self.log_max.exp()), 2),
        }


def fit_ln_cell(
    stimuli: torch.Tensor,
    counts: torch.Tensor,
    grey: float,
    steps: int = 3000,
    lr: float = 0.03,
    device: torch.device | None = None,
) -> tuple[LNCell, list]:
    """
    Fit an LN cell to the spike counts `counts` [N] of the stimuli `stimuli` [N, H, W],
    by maximising the Poisson likelihood with Adam (full batch). The receptive field
    starts at the peak of the spike-triggered average, with its sign. Returns the cell
    (on the CPU) and the loss at each step.
    """
    device = device or torch.device("cpu")
    x = stimuli.float().to(device)
    y = counts.float().to(device)

    # the spike-triggered average, smoothed: its peak and sign start the fit
    centred = x - grey
    sta = (y[:, None, None] * centred).sum(0) / y.sum() - centred.mean(0)
    smooth = torch.nn.functional.avg_pool2d(sta[None, None], 3, 1, 1)[0, 0]
    peak = int(smooth.abs().argmax())
    row, column = divmod(peak, x.shape[-1])
    sign = 1.0 if float(smooth.reshape(-1)[peak]) > 0 else -1.0
    cell = LNCell(x.shape[-1], grey, (float(row), float(column)), sign).to(device)
    with torch.no_grad():
        cell.log_max.fill_(math.log(float(y.max()) + 1.0))
        cell.bias.fill_(float(torch.logit(y.mean() / cell.log_max.exp())))

    # Poisson negative log-likelihood (without the constant log n!)
    optimizer = torch.optim.Adam(cell.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        rates = cell(x).clamp(min=1e-8)
        loss = (rates - y * rates.log()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    return cell.cpu().eval(), losses


def get_hex_lattice(size: int, spacing: float) -> np.ndarray:
    """The points [n, 2] (row, column, in pixels) of a hexagonal lattice of nearest-
    neighbour distance `spacing` inside a `size` x `size` image, centred on it."""
    row_step = spacing * math.sqrt(3) / 2
    n_rows = int((size - 1) // row_step) + 1
    n_columns = int((size - 1) // spacing) + 1
    top = (size - 1 - (n_rows - 1) * row_step) / 2
    points = []
    for i in range(n_rows):
        shift = spacing / 2 if i % 2 else 0.0
        left = (size - 1 - (n_columns - 1) * spacing) / 2 + shift
        for j in range(n_columns):
            column = left + j * spacing
            if column <= size - 1:
                points.append((top + i * row_step, column))
    return np.array(points)


class LNMosaic(nn.Module):
    """
    The LN cell `cell` copied at every point of a hexagonal lattice `spacing_um` apart
    on a `size` x `size` image of `microns_per_px` µm per pixel (the cell's own), with
    the cell's receptive field relative to its centre. forward(x [B, 1, size, size])
    → λ [B, n_neurons]. Fixed (no parameters to train).
    """

    def __init__(
        self,
        cell: LNCell,
        size: int = 120,
        microns_per_px: float = 21.0,
        spacing_um: float = 200.0,
    ):
        super().__init__()
        self.centers = get_hex_lattice(size, spacing_um / microns_per_px)
        log_sigmas = cell.log_sigmas.detach()
        surround = torch.sigmoid(cell.surround_logit.detach())
        fields = [
            get_difference_of_gaussians(
                size,
                torch.tensor(c, dtype=torch.float32),
                log_sigmas,
                surround,
                cell.amplitude.detach(),
            ).reshape(-1)
            for c in self.centers
        ]
        self.register_buffer("fields", torch.stack(fields))
        self.grey = cell.grey
        self.bias = float(cell.bias)
        self.max_count = float(cell.log_max.exp())

    @property
    def n_neurons(self) -> int:
        """The number of cells of the mosaic."""
        return len(self.fields)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The expected spike counts λ [B, n_neurons] of the images `x`."""
        drive = (x.reshape(len(x), -1) - self.grey) @ self.fields.T + self.bias
        return self.max_count * torch.sigmoid(drive)


def count_noise(n_neurons: int) -> PoissonNoise:
    """The Poisson noise of spike counts whose expectation is the LN rate λ (exposure
    1, responses not standardised), for `compute_fisher_per_cell`."""
    return PoissonNoise(1.0, torch.zeros(n_neurons), torch.ones(n_neurons))
