"""
The stimulus features M_t^env(x) at each noise level: the posterior covariance of the
image given a noisy version of it, per unit of noise variance,

    M_t^env(x) = (1/t) E_{x_t|x}[Cov(X | x_t)] = E_{x_t|x}[∂x̂(x_t)/∂x_t]

(second-order Tweedie formula, with x_t = x + √t ε in variance-exploding coordinates and
x̂(x_t) = E[x | x_t] the posterior mean of the unconditional model). It is estimated from
the null token of the conditional diffusion model, ε_θ(x_t, t, 0), so it depends on the
statistics of the images only, not on the responses: it is the stimulus counterpart of
the integrand M_t(x) of G(x) (`estimate_g_per_scale`), and M_t(x) ≤ M_t^env(x) for
exact posteriors.

The Jacobian is never formed (modified Algorithm 1 of the note on the scale-resolved
decomposition of G(x)). Each row of D is a Jacobian-vector product on a random probe
ξ_s ~ N(0, I), by central finite differences of the noise prediction, averaged over N_x
noisy images x_t^(i) of the image:

    Δ_s = ξ_s − (1/(2η)) (1/N_x) Σ_i [ε(x_t^(i) + √ᾱ h ξ_s, t, 0)
                                       − ε(x_t^(i) − √ᾱ h ξ_s, t, 0)]  ≈  M_t^env ξ_s,

with h = η √t, so that √ᾱ h = η √(1 − ᾱ) in the coordinates of the network. D has the
rows Δ_s / √N of one noise level, so DᵀD ≈ M², and the SVD of D gives the eigenvectors
of M (the right singular vectors) and its eigenvalues (the singular values, not
squared).

Where M is nearly flat (fine noise levels, where many eigenvalues are close to 1), the
estimate needs many probes: with N probes of P pixels, M = I gives singular values near
√(P/N) instead of 1, and leading eigenvectors that are arbitrary.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from .estimator import (
    PerScaleOutput,
    _build_log_tau_grid,
    _check_inputs,
    _draw_rows,
    _level_row_scales,
    _per_scale_output,
    _timestep_grid,
)


class JacobianProbe:
    """
    Rows of D for M_t^env(x): Jacobian-vector products of the posterior mean of the
    unconditional model (the null token of the diffusion model) on random probes, by
    central finite differences, averaged over `n_noisy` noisy images per probe. A row
    is Δ = M ξ for its probe ξ, so the rows are scaled by 1/√N only (row normalisation
    1). It has the interface of the methods of `multigeom.geometry.methods`, but it is
    not one of them: its rows do not add up to G(x).
    """

    row_normalization = 1.0

    def __init__(
        self,
        *,
        image: torch.Tensor,
        model: torch.nn.Module,
        alphas_cumprod: torch.Tensor,
        cfg: SimpleNamespace,
        device: torch.device,
        generator: torch.Generator | None = None,
        n_noisy: int = 8,
        eta: float = 0.02,
    ):
        # the clean image [1, 1, H, W] on the device, and the network ε_θ
        self.image = image
        self.unet = model

        # the noise schedule, the device and the random draws
        self.alphas_cumprod = alphas_cumprod
        self.device = device
        self.generator = generator

        # the noisy images per probe, and the step of the differences in units of √t
        self.n_noisy = int(n_noisy)
        self.eta = float(eta)

    def sample_rows(self, x_t: torch.Tensor, t: torch.Tensor, t_j: int) -> torch.Tensor:
        """
        Rows Δ [B, P] at the DDPM step `t_j`, one per noisy image of `x_t` [B, 1, H, W],
        which is the first of the `n_noisy` noisy images of its probe (`t`, the [B]
        tensor of the step, is not needed).
        """
        # the weights of the clean image and of the noise at this level
        sqrt_ab = float(self.alphas_cumprod[t_j].sqrt())
        sqrt_1m = float((1.0 - self.alphas_cumprod[t_j]).sqrt())
        batch, shape = x_t.shape[0], x_t.shape[1:]
        draw = {"device": self.device, "dtype": x_t.dtype, "generator": self.generator}

        # one probe ξ per row
        probes = torch.randn(x_t.shape, **draw)

        # n_noisy noisy images per probe [B, n_noisy, 1, H, W]: x_t and n_noisy − 1 more
        noise = torch.randn((batch, self.n_noisy - 1, *shape), **draw)
        others = sqrt_ab * self.image.unsqueeze(1) + sqrt_1m * noise
        noisy = torch.cat([x_t.unsqueeze(1), others], dim=1)

        # the step √ᾱ h ξ = η √(1 − ᾱ) ξ, the same for the noisy images of a probe
        step = self.eta * sqrt_1m * probes.unsqueeze(1)

        # the noise predictions of the null token at x_t ± step, in one forward pass
        inputs = torch.cat([noisy + step, noisy - step]).flatten(0, 1)
        steps = torch.full((len(inputs),), t_j, device=self.device, dtype=torch.long)
        null = torch.zeros(len(inputs), self.unet.n_neurons, device=self.device)
        eps = self.unet(inputs, steps, null).reshape(2, batch, self.n_noisy, -1)

        # Δ = ξ − (ε₊ − ε₋) / (2η), the differences averaged over the noisy images
        jacobian_product = (eps[0] - eps[1]).mean(dim=1) / (2 * self.eta)
        return probes.flatten(1) - jacobian_product


@torch.no_grad()
def estimate_env_per_scale(
    image: torch.Tensor,
    unet: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
    generator: torch.Generator | None = None,
    n_noisy: int = 8,
    eta: float = 0.02,
) -> PerScaleOutput:
    """
    Compute the top-k eigenpairs of the stimulus features
    M_t^env(x) = (1/t) E[Cov(X | x_t)] at each noise level of the grid of `cfg`, from
    the null token of the diffusion model `unet` (see the module):
    `cfg.monte_carlo.n_samples` probes per level, in batches of
    `cfg.monte_carlo.batch_size`, each averaged over `n_noisy` noisy images, with the
    finite-difference step η √t (`eta`); `cfg.top_k` eigenpairs per level, whose
    eigenvalues are the singular values of D. A batch makes 2 · batch_size · n_noisy
    forward passes at once: lower the batch size if the device runs out of memory.

    No encoder is needed: M_t^env does not depend on the responses. The other arguments
    are those of `estimate_g_per_scale`.

    Returns:
        PerScaleOutput with `log_tau` [J], `eigenvalues` [J, k] and `eigenvectors`
        [J, k, P].
    """
    # the settings of the probes, the image and the settings of the estimator
    if n_noisy < 1:
        raise ValueError(f"n_noisy must be at least 1, got {n_noisy}")
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")
    _check_inputs(image, cfg)

    # inference mode, and the clean image on the device of the network
    unet.eval()
    image = image.to(device)

    # the rows Δ_s = M ξ_s, from probes on the null token
    sampler = JacobianProbe(
        image=image,
        model=unet,
        alphas_cumprod=alphas_cumprod,
        cfg=cfg,
        device=device,
        generator=generator,
        n_noisy=n_noisy,
        eta=eta,
    )

    # the noise levels, and the rows of each scaled by 1/√N (no weight w_j)
    log_tau = _build_log_tau_grid(_timestep_grid(cfg), alphas_cumprod)
    row_scales = _level_row_scales(sampler, cfg, len(log_tau))

    # one SVD per level; its singular values are the eigenvalues of M (DᵀD ≈ M²)
    levels = _draw_rows(
        image, sampler, alphas_cumprod, cfg, row_scales, device, generator
    )
    return _per_scale_output(log_tau, levels, int(cfg.top_k), sampler, squared=False)
