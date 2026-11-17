"""
Tests for multigeom.geometry.spectrum: the envelope spectrum
ρ^env(τ) = E[(1/t) ‖x̂(x_t) − x‖²], from the null token of a diffusion network.

Analytic references:
  - Gaussian prior x ~ N(0, C), C = U diag(c) Uᵀ, with the exact posterior mean: over
    images drawn from the prior, the curve is Σ_i c_i / (c_i + t) at every level (the
    error of the Wiener filter, which is the posterior mean here). Monte-Carlo error
    about 2% with 4000 noisy images per level; the tolerance is 10%.
  - x̂ = 0 (a posterior mean that ignores x_t): (1/t) ‖x‖² for each image, exactly,
    whatever the noise.
  - x̂ = y (a network that predicts no noise): ‖ε‖², whose mean is the number of
    pixels P at every level.

The coding spectrum, on a linear-Gaussian model (prior N(0, C), responses
r = A x + σ ξ, exact posterior means, Σ_t and Σ_t,r the posterior covariances given y
and given y and r):
  - over images drawn from the prior, the error without the responses is tr Σ_t / t,
    with them tr Σ_t,r / t, and ρ = (tr Σ_t − tr Σ_t,r) / t, both from the difference
    of the errors (eq. 7) and from what the responses change;
  - the curve of compute_g_per_scale, whose responses are drawn given the denoised
    image with spread σ, is (σ² / t) tr(L Lᵀ), L = Σ_t,r Aᵀ / σ², below ρ;
  - ½ ∫ ρ d log t over the DDPM steps is I(R; X) = ½ log det(I + A C Aᵀ / σ²).
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("diffusers")

from multigeom.geometry.estimator import _build_log_tau_grid  # noqa: E402
from multigeom.geometry.spectrum import (  # noqa: E402
    estimate_coding_spectrum,
    estimate_env_spectrum,
)
from conftest import (  # noqa: E402
    ConstantMeanUNet,
    GaussianPriorUNet,
    LinearEncoder,
    LinearGaussianUNet,
    NoNoiseUNet,
    gaussian_prior,
    linear_beta_alphas_cumprod,
)

SIZE = 4  # a 4 x 4 image: P = 16 pixels
P = SIZE * SIZE
STEPS = [1, 10, 100, 400, 900, 999]  # from the finest noise level to the coarsest


def _alphas32():
    return linear_beta_alphas_cumprod().float()


def _noise_variances():
    # t from the float32 ᾱ that the estimator uses (1 − ᾱ is exact in float32 near 1)
    ab = _alphas32()[STEPS].double()
    return (1 - ab) / ab


def _uniform_images(n: int, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(n, 1, SIZE, SIZE, generator=generator) * 2 - 1


def _spectrum(images, unet, n_noise=8, seed=0):
    return estimate_env_spectrum(
        images,
        unet,
        _alphas32(),
        STEPS,
        torch.device("cpu"),
        n_noise=n_noise,
        batch_size=1000,
        generator=torch.Generator().manual_seed(seed),
    )


def test_grid_and_shapes():
    out = _spectrum(_uniform_images(5), NoNoiseUNet(), n_noise=2)
    torch.testing.assert_close(out.log_tau, _build_log_tau_grid(STEPS, _alphas32()))
    assert tuple(out.values.shape) == (5, len(STEPS))
    torch.testing.assert_close(out.mean, out.values.mean(dim=0))


def test_gaussian_prior_gives_the_wiener_curve():
    # 500 images drawn from the prior N(0, U diag(c) Uᵀ), 8 noisy versions each
    U, c = gaussian_prior(P)
    generator = torch.Generator().manual_seed(3)
    z = torch.randn(500, P, generator=generator, dtype=torch.float64)
    images = ((z * c.sqrt()) @ U.T).float().view(500, 1, SIZE, SIZE)
    out = _spectrum(images, GaussianPriorUNet(_alphas32(), U, c))

    # Σ_i c_i / (c_i + t) at each level
    expected = (c[None, :] / (c[None, :] + _noise_variances()[:, None])).sum(dim=1)
    torch.testing.assert_close(out.mean.double(), expected, rtol=0.10, atol=0)


def test_a_posterior_mean_that_ignores_x_t_gives_the_image_energy_over_t():
    # x̂ = 0: (1/t) ‖x‖² for each image at each level, whatever the noise
    images = _uniform_images(7)
    out = _spectrum(images, ConstantMeanUNet(_alphas32()), n_noise=3)
    energy = images.flatten(1).double().pow(2).sum(dim=1)
    expected = energy[:, None] / _noise_variances()[None, :]
    torch.testing.assert_close(out.values.double(), expected, rtol=1e-4, atol=0)


def test_no_denoising_gives_the_number_of_pixels():
    # x̂ = y: the error is the noise itself, ‖ε‖², of mean P
    out = _spectrum(_uniform_images(500), NoNoiseUNet())
    torch.testing.assert_close(
        out.mean, torch.full((len(STEPS),), float(P)), rtol=0.05, atol=0
    )


@pytest.mark.parametrize(
    "images, kwargs",
    [
        (torch.zeros(1, SIZE, SIZE), {}),
        (torch.zeros(2, 1, SIZE, SIZE), {"n_noise": 0}),
        (torch.zeros(2, 1, SIZE, SIZE), {"batch_size": 0}),
    ],
)
def test_bad_arguments_raise(images, kwargs):
    with pytest.raises(ValueError):
        estimate_env_spectrum(
            images, NoNoiseUNet(), _alphas32(), STEPS, torch.device("cpu"), **kwargs
        )


# ── the coding spectrum, on a linear-Gaussian model ─────────────────────────
CODING_STEPS = [10, 100, 300, 600, 900]


def _linear_gaussian():
    """The prior covariance C [P, P], the readout A [3, P] and the response noise σ."""
    U, c = gaussian_prior(P)
    generator = torch.Generator().manual_seed(4)
    readout = 0.5 * torch.randn(3, P, generator=generator, dtype=torch.float64)
    return U @ torch.diag(c) @ U.T, readout, 0.3


def _closed_form(prior, readout, sigma, t):
    """The mean terms of the coding spectrum at the noise variance t, in closed form."""
    identity = torch.eye(P, dtype=torch.float64)
    posterior = torch.linalg.inv(torch.linalg.inv(prior) + identity / t)
    likelihood = readout.T @ readout / sigma**2
    posterior_r = torch.linalg.inv(torch.linalg.inv(prior) + identity / t + likelihood)
    gain = posterior_r @ readout.T / sigma**2
    return {
        "env": float(torch.trace(posterior)) / t,
        "rho": float(torch.trace(posterior) - torch.trace(posterior_r)) / t,
        "sampled": sigma**2 * float(torch.trace(gain @ gain.T)) / t,
    }


def _coding_spectrum(steps, n_images, n_noise):
    """The coding spectrum of images drawn from the linear-Gaussian prior."""
    prior, readout, sigma = _linear_gaussian()
    generator = torch.Generator().manual_seed(5)
    z = torch.randn(n_images, P, generator=generator, dtype=torch.float64)
    images = (z @ torch.linalg.cholesky(prior).T).float().view(n_images, 1, SIZE, SIZE)
    return estimate_coding_spectrum(
        images,
        LinearEncoder(readout),
        LinearGaussianUNet(_alphas32(), prior, readout, sigma),
        _alphas32(),
        steps,
        torch.device("cpu"),
        sigma=sigma,
        n_noise=n_noise,
        batch_size=1000,
        generator=generator,
    )


@pytest.fixture(scope="module")
def coding_spectrum():
    return _coding_spectrum(CODING_STEPS, n_images=1000, n_noise=4)


def test_coding_spectrum_matches_the_closed_form(coding_spectrum):
    prior, readout, sigma = _linear_gaussian()
    ab = _alphas32()[CODING_STEPS].double()
    expected = [_closed_form(prior, readout, sigma, float(t)) for t in (1 - ab) / ab]
    rows = {
        name: torch.tensor([e[name] for e in expected], dtype=torch.float64)
        for name in expected[0]
    }

    # the error without the responses, ρ both ways, and the curve of compute_g_per_scale
    out = coding_spectrum
    torch.testing.assert_close(
        out.env.mean(dim=0).double(), rows["env"], rtol=0.05, atol=0
    )
    torch.testing.assert_close(out.rho.double(), rows["rho"], rtol=0.15, atol=0)
    torch.testing.assert_close(out.rho_gain.double(), rows["rho"], rtol=0.10, atol=0)
    torch.testing.assert_close(
        out.rho_sampled.double(), rows["sampled"], rtol=0.10, atol=0
    )

    # drawing the responses given the denoised image lowers the curve
    assert bool((rows["sampled"] < rows["rho"]).all())


def test_half_the_integral_of_rho_is_the_mutual_information():
    # 40 noise levels from the finest DDPM step to the coarsest (the grid alone gives
    # the information to 0.02%), and 2000 images: their sampling error is about 1%
    steps = sorted({int(round(10**e)) for e in torch.linspace(0, 2.9996, 40).tolist()})
    out = _coding_spectrum(steps, n_images=2000, n_noise=2)

    # ½ ∫ ρ d log t, trapezoids, against ½ log det(I + A C Aᵀ / σ²)
    prior, readout, sigma = _linear_gaussian()
    gram = torch.eye(3, dtype=torch.float64) + readout @ prior @ readout.T / sigma**2
    information = 0.5 * float(torch.logdet(gram))
    integral = 0.5 * float(torch.trapezoid(out.rho_gain.double(), out.log_tau.double()))
    assert integral == pytest.approx(information, rel=0.03)


def test_coding_spectrum_bad_arguments_raise():
    prior, readout, sigma = _linear_gaussian()
    unet = LinearGaussianUNet(_alphas32(), prior, readout, sigma)
    with pytest.raises(ValueError):
        estimate_coding_spectrum(
            torch.zeros(1, SIZE, SIZE),
            LinearEncoder(readout),
            unet,
            _alphas32(),
            CODING_STEPS,
            torch.device("cpu"),
            sigma=sigma,
        )
