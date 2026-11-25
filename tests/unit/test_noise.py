"""
Check the noise models of the responses (`multigeom.geometry.noise`): the Gaussian one
draws as the conditional-pair estimator did before it existed, the Poisson one has the
moments of spike counts, and the Poisson Fisher information has its closed form.
"""

from __future__ import annotations

import numpy as np
import torch

from multigeom.geometry.fisher import fisher_eigendecomposition
from multigeom.geometry.noise import GaussianNoise, PoissonNoise


def test_gaussian_noise_draws_like_the_former_estimator():
    """
    Verify that two draws of `GaussianNoise(σ)` from one generator equal those the
    conditional-pair estimator made before the noise models existed (two standard normal
    draws of the shape of the responses, then the responses plus σ times each), so the
    estimates of V1 and V4 are unchanged.
    """
    # the responses of 5 images and 7 neurons
    mean = torch.randn(5, 7, generator=torch.Generator().manual_seed(1))
    noise = GaussianNoise(0.22)

    # with the noise model
    generator = torch.Generator().manual_seed(0)
    r1, r2 = noise.sample(mean, generator), noise.sample(mean, generator)

    # as before: both standard normal draws first, then the responses
    generator = torch.Generator().manual_seed(0)
    xi1 = torch.randn(mean.shape, generator=generator)
    xi2 = torch.randn(mean.shape, generator=generator)
    torch.testing.assert_close(r1, mean + 0.22 * xi1, rtol=0, atol=0)
    torch.testing.assert_close(r2, mean + 0.22 * xi2, rtol=0, atol=0)


def test_poisson_noise_has_the_moments_of_spike_counts():
    """
    Verify that `PoissonNoise` with exposure g draws responses whose mean is the rate λ
    and whose variance is λ / g (counts n ~ Poisson(g λ), response n / g), before the
    standardisation (here mean 0 and standard deviation 1), within Monte-Carlo error.
    """
    # 3 neurons, rates 0.05 to 2, exposure 14, no standardisation
    rates = torch.tensor([0.05, 0.5, 2.0], dtype=torch.float64)
    noise = PoissonNoise(14.0, torch.zeros(3), torch.ones(3))

    # 200,000 draws of each neuron
    generator = torch.Generator().manual_seed(0)
    draws = noise.sample(rates.expand(200_000, 3).clone(), generator)

    # mean λ and variance λ / g, within 2% (the standard error is below 0.5%)
    np.testing.assert_allclose(draws.mean(0).numpy(), rates.numpy(), rtol=0.02)
    np.testing.assert_allclose(draws.var(0).numpy(), (rates / 14.0).numpy(), rtol=0.02)


def test_poisson_fisher_of_an_exponential_encoder_has_its_closed_form():
    """
    Verify that the Poisson Fisher information of an encoder of rates λ = exp(A x) is
    J(x) = g Σ_n λ_n a_n a_nᵀ (the gradient λ_n a_n of each rate, squared and divided by
    λ_n), by comparing the eigenvalues and the leading eigenvector of
    `fisher_eigendecomposition` with those of that matrix.
    """

    class ExponentialEncoder(torch.nn.Module):
        """Rates exp(A x) of 4 neurons, for images of 6 x 6 pixels."""

        def __init__(self, weight: torch.Tensor):
            super().__init__()
            self.weight = torch.nn.Parameter(weight, requires_grad=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.exp(x.flatten(1) @ self.weight.T)

    # the encoder, an image, and the noise of exposure 10
    generator = torch.Generator().manual_seed(0)
    weight = 0.3 * torch.randn(4, 36, generator=generator)
    image = 0.5 * torch.randn(1, 1, 6, 6, generator=generator)
    noise = PoissonNoise(10.0, torch.zeros(4), torch.ones(4))
    out = fisher_eigendecomposition(
        image, ExponentialEncoder(weight), sigma=1.0, validate=False, noise=noise
    )

    # the closed form: g Σ_n λ_n a_n a_nᵀ
    rates = torch.exp(image.flatten(1) @ weight.T)[0].double()
    matrix = 10.0 * (weight.double().T * rates) @ weight.double()
    values, vectors = torch.linalg.eigh(matrix)
    np.testing.assert_allclose(out.eigenvalues, values.flip(0)[:4], rtol=1e-4)
    cosine = (out.eigenvectors[0].double() @ vectors[:, -1]).abs()
    assert float(cosine) > 0.9999
