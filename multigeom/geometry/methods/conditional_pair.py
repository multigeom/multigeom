"""
The conditional-pair method of G(x) (Algorithm 1 of the paper, lines 7-15): the rows of
the matrix D are the differences of two conditional noise predictions,

    Δ = ε_θ(x_t, t, r_1) − ε_θ(x_t, t, r_2),   r_i ~ p(r | x̂(x_t)),

with x̂ the posterior mean given by Tweedie's formula and the null-token pass of the
network, and p(r | x) Gaussian, N(f_φ(x), σ² I), unless `cfg.monte_carlo.noise` gives
another noise (`multigeom.geometry.noise`, e.g. Poisson for the retina models). It
needs the conditional diffusion model of the encoder
(`multigeom.diffusion.NeuralConditionedUNet`).
"""

from __future__ import annotations

import torch

from ...diffusion.denoise import compute_denoised
from ..noise import GaussianNoise


class ConditionalPair:
    """
    Rows of D from a pair of conditional noise predictions.

    Each row is the difference of two predictions, so its covariance is twice that of
    one prediction: `row_normalization = 2` puts the factor 1/2 of Eq. (eq:cc) in the
    weight of the row.
    """

    row_normalization = 2.0

    def __init__(
        self, *, image, encoder, model, alphas_cumprod, cfg, device, generator=None
    ):
        self.encoder = encoder
        self.unet = model
        self.alphas_cumprod = alphas_cumprod
        self.device = device
        self.generator = generator
        self.sigma = float(cfg.monte_carlo.observation_noise_sigma)

        # the noise of the responses: Gaussian with σ, unless the config gives another
        self.noise = getattr(cfg.monte_carlo, "noise", None) or GaussianNoise(
            self.sigma
        )

    def sample_rows(self, x_t: torch.Tensor, t: torch.Tensor, t_j: int) -> torch.Tensor:
        """
        Rows Δ [B, P] for the noisy images `x_t` [B, 1, H, W] at the DDPM step `t_j`
        (`t` is the [B] tensor of that step).
        """
        # Line 7-9: unconditional posterior mean via Tweedie + null-token pass (a null
        # token for each noisy image, so batches can have any size)
        null_token = torch.zeros(len(x_t), self.encoder.n_neurons, device=self.device)
        x_hat, _ = compute_denoised(
            self.unet, x_t, t_j, self.alphas_cumprod, null_token, clamp=True
        )

        # Line 10-12: sample r_1, r_2 ~ p(r | x̂), N(f_φ(x̂), σ² I) by default
        mu = self.encoder(x_hat)  # [B, n_neurons]
        r1 = self.noise.sample(mu, self.generator)
        r2 = self.noise.sample(mu, self.generator)

        # Line 13-14: two conditional forward passes, take the ε-difference
        eps1 = self.unet(x_t, t, r1)
        eps2 = self.unet(x_t, t, r2)
        return (eps1 - eps2).reshape(x_t.shape[0], -1)  # [B, P]
