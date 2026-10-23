"""
The conditional-pair method of G(x) (Algorithm 1 of the paper, lines 7-15): the rows of
the matrix D are the differences of two conditional noise predictions,

    Δ = ε_θ(x_t, t, r_1) − ε_θ(x_t, t, r_2),   r_i ~ N(f_φ(x̂(x_t)), σ² I),

with x̂ the posterior mean given by Tweedie's formula and the null-token pass of the
network. It needs the conditional diffusion model of the encoder
(`multigeom.diffusion.NeuralConditionedUNet`).
"""

from __future__ import annotations

import torch


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
        batch_size = int(cfg.monte_carlo.batch_size)
        self.null_token = torch.zeros(batch_size, encoder.n_neurons, device=device)

    def sample_rows(self, x_t: torch.Tensor, t: torch.Tensor, t_j: int) -> torch.Tensor:
        """
        Rows Δ [B, P] for the noisy images `x_t` [B, 1, H, W] at the DDPM step `t_j`
        (`t` is the [B] tensor of that step).
        """
        sqrt_ab = float(self.alphas_cumprod[t_j].sqrt())
        sqrt_1m = float((1.0 - self.alphas_cumprod[t_j]).sqrt())

        # Line 7-9: unconditional posterior mean via Tweedie + null-token pass
        eps_u = self.unet(x_t, t, self.null_token)
        x_hat = (x_t - sqrt_1m * eps_u) / sqrt_ab
        x_hat = x_hat.clamp(-1.0, 1.0)

        # Line 10-12: sample r_1, r_2 ~ N(f_φ(x̂), σ² I)
        mu = self.encoder(x_hat)  # [B, n_neurons]
        xi1 = torch.randn(
            mu.shape, device=self.device, dtype=mu.dtype, generator=self.generator
        )
        xi2 = torch.randn(
            mu.shape, device=self.device, dtype=mu.dtype, generator=self.generator
        )
        r1 = mu + self.sigma * xi1
        r2 = mu + self.sigma * xi2

        # Line 13-14: two conditional forward passes, take the ε-difference
        eps1 = self.unet(x_t, t, r1)
        eps2 = self.unet(x_t, t, r2)
        return (eps1 - eps2).reshape(x_t.shape[0], -1)  # [B, P]
