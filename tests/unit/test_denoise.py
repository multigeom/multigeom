"""
Tests for multigeom.diffusion.denoise: the denoised image of a diffusion model,
x̂(x_t) = (x_t − √(1 − ᾱ) ε̂) / √ᾱ.

Analytic reference: x_t = √ᾱ x + √(1 − ᾱ) ε, so a network that predicts the true noise
ε gives x̂ = x exactly, at every step.
"""

from __future__ import annotations

import torch

from multigeom.diffusion import compute_denoised, sample_images


class NoiseOracle(torch.nn.Module):
    """A network that returns a stored noise, and records the responses it is given."""

    n_neurons = 3

    def __init__(self, noise: torch.Tensor):
        super().__init__()
        self.noise = noise
        self.seen = None

    def forward(self, x_t, t, r):
        self.seen = (t, r)
        return self.noise


def make_noisy(t_j: int, scale: float = 1.0):
    """Images x (scaled by `scale`), noises ε, the schedule, and x_t at step t_j."""
    generator = torch.Generator().manual_seed(0)
    x = scale * torch.randn(4, 1, 8, 8, generator=generator, dtype=torch.float64)
    eps = torch.randn(4, 1, 8, 8, generator=generator, dtype=torch.float64)
    alphas_cumprod = torch.linspace(0.999, 0.001, 1000, dtype=torch.float64)
    ab = alphas_cumprod[t_j]
    return x, eps, alphas_cumprod, ab.sqrt() * x + (1 - ab).sqrt() * eps


def test_denoised_image_is_the_clean_image_for_a_perfect_noise_prediction():
    """Verify that x̂ = x, unclamped, at a fine and a coarse step, when the network
    predicts the true noise, and that ε̂ is returned as predicted."""
    for t_j in (10, 900):
        x, eps, alphas_cumprod, x_t = make_noisy(t_j, scale=3.0)
        x_hat, eps_hat = compute_denoised(NoiseOracle(eps), x_t, t_j, alphas_cumprod)
        assert torch.allclose(x_hat, x, atol=1e-10)
        assert torch.equal(eps_hat, eps)
        assert x_hat.abs().max() > 1


def test_denoised_image_uses_the_null_token_unless_responses_are_given():
    """Verify that the network gets zero responses of its width and the step as a long
    tensor by default, and the given responses otherwise."""
    x, eps, alphas_cumprod, x_t = make_noisy(500)
    unet = NoiseOracle(eps)
    compute_denoised(unet, x_t, 500, alphas_cumprod)
    steps, responses = unet.seen
    assert steps.dtype == torch.long and torch.equal(steps, torch.full((4,), 500))
    assert torch.equal(responses, torch.zeros(4, 3))
    given = torch.ones(4, 3)
    compute_denoised(unet, x_t, 500, alphas_cumprod, given)
    assert unet.seen[1] is given


def test_clamp_clips_the_denoised_image_to_the_canonical_range():
    """Verify that clamp=True clips x̂ to [-1, 1] and leaves the values inside it."""
    x, eps, alphas_cumprod, x_t = make_noisy(10, scale=3.0)
    x_hat, _ = compute_denoised(NoiseOracle(eps), x_t, 10, alphas_cumprod, clamp=True)
    assert x_hat.min() >= -1 and x_hat.max() <= 1
    inside = x.abs() < 0.999
    assert torch.allclose(x_hat[inside], x[inside], atol=1e-10)


class GaussianDenoiser(torch.nn.Module):
    """
    The exact noise prediction for images drawn from N(0, c I): given x_t, the posterior
    mean of the noise is √(1 − ᾱ) x_t / (ᾱ c + 1 − ᾱ).
    """

    n_neurons = 3

    def __init__(self, variance: float, alphas_cumprod: torch.Tensor):
        super().__init__()
        self.variance = variance
        self.alphas_cumprod = alphas_cumprod

    def forward(self, x_t, t, r):
        ab = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return (1 - ab).sqrt() * x_t / (ab * self.variance + 1 - ab)


def test_sample_images_draws_from_the_prior_of_an_exact_denoiser():
    """
    Verify that the DDPM sampler, driven by the exact denoiser of a Gaussian prior
    N(0, c I) with c = 0.3, generates images whose pixel variance is c within 5% and
    whose pixels are uncorrelated with their neighbours (|correlation| < 0.05): the
    sampler draws from the prior its network has learned.
    """
    # the schedule of the models (linear β from 1e-4 to 0.02 over 1000 steps)
    betas = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64)
    alphas_cumprod = torch.cumprod(1 - betas, dim=0).float()
    unet = GaussianDenoiser(0.3, alphas_cumprod)

    # 64 images of 16 x 16, seeded
    generator = torch.Generator().manual_seed(0)
    images = sample_images(
        unet, alphas_cumprod, (64, 1, 16, 16), torch.device("cpu"), generator
    )

    # their variance, and the correlation of horizontally neighbouring pixels
    variance = float(images.var())
    left, right = images[..., :-1].flatten(), images[..., 1:].flatten()
    correlation = float(torch.corrcoef(torch.stack([left, right]))[0, 1])
    assert abs(variance / 0.3 - 1) < 0.05, variance
    assert abs(correlation) < 0.05, correlation


def test_sample_images_keeps_its_trajectory_without_changing_the_samples():
    """
    Verify that sample_images with `keep` returns the same samples as without it, the
    kept steps from the coarsest to the finest, and at step 0 a denoised image x̂(x_0)
    within 0.01 of the sample: the last guess of the model is the image it generates.
    """
    # the exact denoiser of N(0, 0.3 I), on the schedule of the models
    betas = torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64)
    alphas_cumprod = torch.cumprod(1 - betas, dim=0).float()
    unet = GaussianDenoiser(0.3, alphas_cumprod)
    shape, device = (4, 1, 8, 8), torch.device("cpu")

    # the same seed, without and with the trajectory
    plain = sample_images(
        unet, alphas_cumprod, shape, device, torch.Generator().manual_seed(1)
    )
    images, trajectory = sample_images(
        unet,
        alphas_cumprod,
        shape,
        device,
        torch.Generator().manual_seed(1),
        keep=[0, 500, 900],
    )
    assert torch.equal(plain, images)
    assert list(trajectory) == [900, 500, 0]
    assert float((trajectory[0][1] - images).abs().max()) < 0.01
