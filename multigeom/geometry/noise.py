"""
The noise of the responses given the image, p(r | x): it sets how the conditional-pair
estimator of G(x) draws responses, and the weights of the Fisher information J(x).

    GaussianNoise(sigma)
        r = f(x) + σ ξ (the encoders of the paper); J(x) = (1/σ²) J_fᵀ J_f
    PoissonNoise(gain, resp_mean, resp_std)
        spike counts n ~ Poisson(g λ(x)) with λ = f(x) the rate, r = n / g, standardised
        as the diffusion model saw it in training (the retina models);
        J(x) = g J_fᵀ diag(1/λ) J_f

Both give `sample(rates, generator)`, the responses the diffusion model is conditioned
on, and `fisher_weights(rates)`, the weight of each neuron's row of J_f so that J(x) is
the Gram matrix of the weighted rows.
"""

from __future__ import annotations

import torch


class GaussianNoise:
    """Gaussian noise of standard deviation `sigma` around the encoder's output."""

    def __init__(self, sigma: float):
        self.sigma = float(sigma)

    def sample(
        self, rates: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Responses r = f(x) + σ ξ, from the encoder's output `rates` [B, N]."""
        xi = torch.randn(
            rates.shape, device=rates.device, dtype=rates.dtype, generator=generator
        )
        return rates + self.sigma * xi

    def fisher_weights(self, rates: torch.Tensor) -> torch.Tensor:
        """The weight 1/σ of each neuron's gradient, [N] for `rates` [N]."""
        return torch.full_like(rates, 1.0 / self.sigma)


class PoissonNoise:
    """
    Poisson spike counts with exposure `gain` (g, the time window over the Fano factor),
    their rate given by the encoder's output; the response is the count over g,
    standardised with the response statistics of the training images (`resp_mean`,
    `resp_std` [N]), as the diffusion model was conditioned in training.
    """

    def __init__(self, gain: float, resp_mean: torch.Tensor, resp_std: torch.Tensor):
        self.gain = float(gain)
        self.resp_mean = resp_mean.detach().clone()
        self.resp_std = resp_std.detach().clamp(min=1e-6).clone()

    def sample(
        self, rates: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """
        Standardised responses (n / g − mean) / std with n ~ Poisson(g λ), from the
        rates λ = `rates` [B, N].
        """
        # the spike counts of the window, then the response on the scale of the rates
        counts = torch.poisson((self.gain * rates).clamp(min=0.0), generator=generator)
        responses = counts / self.gain

        # standardised as in training
        mean = self.resp_mean.to(rates.device, rates.dtype)
        std = self.resp_std.to(rates.device, rates.dtype)
        return (responses - mean) / std

    def fisher_weights(self, rates: torch.Tensor) -> torch.Tensor:
        """The weight √(g / λ_n) of each neuron's gradient, [N] for `rates` [N]."""
        return (self.gain / rates.clamp(min=1e-8)).sqrt()
