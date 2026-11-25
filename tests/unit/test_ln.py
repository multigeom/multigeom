"""
Tests for multigeom.encoders.ln: LN (linear–nonlinear–Poisson) cells fitted to spike
counts, and their mosaics.

Analytic references:
  - λ = r_max σ(⟨w, x − grey⟩ + b) has the gradient λ' w with λ' = r_max σ (1 − σ), so
    the Poisson Fisher information of cell n is g_n g_nᵀ with g_n = λ_n' w_n / √λ_n.
  - A hexagonal lattice of spacing d has every point's nearest neighbour at distance d.
  - Counts drawn from a known LN cell determine its parameters: a fit to enough of them
    recovers them.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from multigeom.encoders.ln import (
    LNCell,
    LNMosaic,
    count_noise,
    fit_ln_cell,
    get_hex_lattice,
)
from multigeom.geometry import compute_fisher_per_cell


def make_cell(size: int = 30) -> LNCell:
    """An OFF cell with a surround, its parameters set by hand."""
    cell = LNCell(size, grey=0.1, center=(12.0, 17.0), amplitude=-30.0)
    with torch.no_grad():
        cell.log_sigmas.copy_(torch.tensor([math.log(2.0), math.log(3.0)]))
        cell.surround_logit.fill_(0.0)
        cell.bias.fill_(-0.5)
        cell.log_max.fill_(math.log(6.0))
    return cell.eval()


def test_hex_lattice_has_the_spacing_and_stays_inside():
    """
    Verify that every point of the hexagonal lattice lies inside the image and has its
    nearest neighbour at the requested spacing.
    """
    points = get_hex_lattice(120, 9.5)
    assert ((points >= 0) & (points <= 119)).all()
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert np.allclose(distances.min(axis=1), 9.5)


def test_fisher_per_cell_of_a_mosaic_is_the_analytic_one():
    """
    Verify that the Fisher information per cell of an LN mosaic, computed from the
    encoder's Jacobian with Poisson counts, is the analytic g_n = λ_n' w_n / √λ_n.
    """
    mosaic = LNMosaic(make_cell(), size=40, microns_per_px=21.0)
    image = torch.randn(1, 1, 40, 40, generator=torch.Generator().manual_seed(0))
    fisher = compute_fisher_per_cell(
        image, mosaic, 1.0, torch.device("cpu"), noise=count_noise(mosaic.n_neurons)
    )

    # the analytic patterns
    drive = (image.reshape(1, -1) - mosaic.grey) @ mosaic.fields.T + mosaic.bias
    sig = torch.sigmoid(drive)[0].double()
    rate = mosaic.max_count * sig
    slope = mosaic.max_count * sig * (1 - sig)
    expected = (slope / rate.sqrt())[:, None] * mosaic.fields.double()
    assert np.allclose(fisher["cells"], expected.numpy(), rtol=1e-4, atol=1e-8)


def test_fit_recovers_the_parameters_of_a_known_cell():
    """
    Verify that fitting an LN cell to Poisson counts drawn from a known one, on 4,000
    smoothed noise images, recovers its centre (within 0.5 pixel), its centre width
    (within 15%) and its polarity.
    """
    true = make_cell()
    generator = torch.Generator().manual_seed(1)
    noise = torch.randn(4000, 1, 30, 30, generator=generator)
    stimuli = torch.nn.functional.avg_pool2d(noise, 3, 1, 1)[:, 0] * 2.0
    with torch.no_grad():
        counts = torch.poisson(true(stimuli), generator=generator)

    cell, losses = fit_ln_cell(stimuli, counts, grey=0.1, steps=3000)
    found, known = cell.describe(), true.describe()
    assert np.allclose(found["center"], known["center"], atol=0.5)
    assert abs(found["sigma_center_px"] / known["sigma_center_px"] - 1) < 0.15
    assert found["polarity"] == known["polarity"]
    assert losses[-1] < losses[0]
