"""
Tests for multigeom.geometry.env: the stimulus features
M_t^env(x) = (1/t) E[Cov(X | x_t)] at each noise level, from Jacobian-vector products of
the unconditional model.

Analytic reference. For a zero-mean Gaussian prior with covariance C = U diag(c) Uᵀ, the
posterior mean is linear, x̂(y) = K_t y with K_t = C (C + t I)⁻¹ in variance-exploding
coordinates y = x_t / √ᾱ, t = (1 − ᾱ)/ᾱ, so M_t^env = K_t at every image and every noisy
image: eigenvalues c_i / (c_i + t), eigenvectors the columns of U. The central finite
differences are exact for a linear map, so every row is Δ = K_t ξ exactly; with N
probes the singular values of D are those of K_t up to the Monte-Carlo error of
ΞᵀΞ/N ≈ I, about sqrt(P / N) / 2 ≈ 3% for the sizes used here; the tolerance is 10%.

Two limits are exact row by row: a posterior mean that ignores x_t gives M = 0 (every
row zero), and the identity (x̂ = y, a network that predicts no noise) gives M = I
(every row its probe).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("diffusers")

from multigeom.geometry.env import JacobianProbe, estimate_env_per_scale  # noqa: E402
from multigeom.geometry.estimator import _build_log_tau_grid  # noqa: E402
from conftest import (  # noqa: E402
    ConstantMeanUNet,
    GaussianPriorUNet,
    NoNoiseUNet,
    estimator_cfg,
    gaussian_prior,
    linear_beta_alphas_cumprod,
)

SIZE = 4  # a 4 x 4 image: P = 16 pixels
P = SIZE * SIZE
GRID = list(range(100, 901, 200))  # five noise levels


def _alphas32():
    return linear_beta_alphas_cumprod().float()


def _cfg(n_samples: int = 2000, batch_size: int = 500, top_k: int = 4):
    return estimator_cfg(
        n_samples=n_samples,
        batch_size=batch_size,
        start=GRID[0],
        stop=GRID[-1],
        step=GRID[1] - GRID[0],
        sigma=0.3,
        top_k=top_k,
    )


def _image(seed: int = 1):
    return torch.from_numpy(
        np.random.default_rng(seed).uniform(-1, 1, (1, 1, SIZE, SIZE))
    ).float()


# ── the Gaussian prior, against the closed form ────────────────────────────
@pytest.fixture(scope="module")
def gaussian_estimate():
    ab = _alphas32()
    U, c = gaussian_prior(P)
    unet = GaussianPriorUNet(ab, U, c)
    out = estimate_env_per_scale(
        _image(),
        unet,
        ab,
        _cfg(),
        torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
        n_noisy=2,
    )
    noise_variance = (1 - linear_beta_alphas_cumprod()[GRID]) / (
        linear_beta_alphas_cumprod()[GRID]
    )
    expected = c[None, :] / (c[None, :] + noise_variance[:, None])  # [J, P]
    return out, U, expected, unet


def test_grid_and_shapes(gaussian_estimate):
    out, *_ = gaussian_estimate
    torch.testing.assert_close(out.log_tau, _build_log_tau_grid(GRID, _alphas32()))
    assert tuple(out.eigenvalues.shape) == (len(GRID), 4)
    assert tuple(out.eigenvectors.shape) == (len(GRID), 4, P)


def test_eigenvalues_are_c_over_c_plus_t_not_squared(gaussian_estimate):
    out, _, expected, _ = gaussian_estimate
    np.testing.assert_allclose(
        out.eigenvalues.double().numpy(), expected[:, :4].numpy(), rtol=0.10
    )


def test_eigenvectors_span_the_leading_columns_of_u(gaussian_estimate):
    out, U, *_ = gaussian_estimate
    # the top-4 subspace at each level: (1/4) ‖V Uᵀ‖² over the 4 leading columns
    overlap = (out.eigenvectors.double() @ U[:, :4]).pow(2).sum(dim=(1, 2)) / 4
    assert bool((overlap > 0.95).all()), overlap


def test_each_eigenvector_matches_its_own_prior_direction(gaussian_estimate):
    """
    Verify that, for the Gaussian prior, each of the four leading eigenvectors of the
    stimulus features M_t^env(x) points along its own direction of image variation,
    the column u_i of U (|cos| above 0.95), in the right order, at every noise level
    where each of its eigenvalues c_i / (c_i + t) differs from its neighbours by at
    least 20%. Where two are closer, the Monte-Carlo error can mix their eigenvectors,
    and only the subspace they span is defined (see the test above).
    """
    out, U, expected, _ = gaussian_estimate

    # the levels where the five leading expected eigenvalues are at least 20% apart
    gaps = (expected[:, :4] - expected[:, 1:5]) / expected[:, :4]
    separated = (gaps >= 0.2).all(dim=1)
    assert int(separated.sum()) >= 3

    # |cos| between each eigenvector and its own direction u_i, at those levels
    cosines = (out.eigenvectors.double() * U[:, :4].T).sum(dim=2).abs()
    assert bool((cosines[separated] > 0.95).all()), cosines


def test_only_the_null_token_is_used(gaussian_estimate):
    *_, unet = gaussian_estimate
    responses = torch.cat(unet.responses)
    assert responses.shape[1] == unet.n_neurons and not bool(responses.any())


# ── exact limits, row by row ───────────────────────────────────────────────
def _sampler(unet, generator):
    ab = _alphas32()
    return JacobianProbe(
        image=_image(),
        model=unet,
        alphas_cumprod=ab,
        cfg=_cfg(),
        device=torch.device("cpu"),
        generator=generator,
        n_noisy=3,
        eta=0.02,
    )


def _noisy(t_j: int, batch: int = 6):
    ab = _alphas32()
    z = torch.randn(batch, 1, SIZE, SIZE, generator=torch.Generator().manual_seed(9))
    return ab[t_j].sqrt() * _image() + (1 - ab[t_j]).sqrt() * z


@pytest.mark.parametrize("t_j", GRID)
def test_a_posterior_mean_that_ignores_x_t_gives_zero_rows(t_j):
    rows = _sampler(ConstantMeanUNet(_alphas32()), torch.Generator().manual_seed(3))
    delta = rows.sample_rows(_noisy(t_j), torch.full((6,), t_j), t_j)
    # zero up to the rounding of the finite differences (the probes have norm ~4)
    assert float(delta.abs().max()) < 1e-3


@pytest.mark.parametrize("t_j", GRID)
def test_the_identity_gives_rows_equal_to_their_probes(t_j):
    # the probes are the first draw of a row: the same generator state gives them back
    generator = torch.Generator().manual_seed(4)
    probes = torch.randn(
        6, 1, SIZE, SIZE, generator=torch.Generator().set_state(generator.get_state())
    )
    delta = _sampler(NoNoiseUNet(), generator).sample_rows(
        _noisy(t_j), torch.full((6,), t_j), t_j
    )
    assert torch.equal(delta, probes.flatten(1))


def test_the_identity_has_singular_values_near_one_with_many_probes():
    out = estimate_env_per_scale(
        _image(),
        NoNoiseUNet(),
        _alphas32(),
        _cfg(n_samples=4000, batch_size=1000),
        torch.device("cpu"),
        generator=torch.Generator().manual_seed(5),
        n_noisy=1,
    )
    # M = I: the singular values of Ξ / √N, close to 1 when N >> P
    np.testing.assert_allclose(out.eigenvalues.numpy(), 1.0, atol=0.15)


# ── arguments ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "kwargs, image",
    [
        ({"n_noisy": 0}, _image()),
        ({"eta": 0.0}, _image()),
        ({}, torch.zeros(2, 1, SIZE, SIZE)),
    ],
)
def test_bad_arguments_raise(kwargs, image):
    with pytest.raises(ValueError):
        estimate_env_per_scale(
            image, NoNoiseUNet(), _alphas32(), _cfg(), torch.device("cpu"), **kwargs
        )
