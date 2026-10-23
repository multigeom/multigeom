"""
Conditional-pair estimator of the multi-scale Fisher information matrix
G(x) — Algorithm 1 of the paper.

Given a clean image x ∈ [-1, 1]^{1×64×64}, an encoder f_φ : ℝ^{1×64×64} → ℝ^N,
and a conditional diffusion network ε_θ(x_t, t, r) trained on f_φ-generated
responses with classifier-free guidance, the estimator returns the top-k
eigenpairs of

    G(x) = ∫_0^∞ [φ_τ * J_τ](x) dτ              (paper Eq. (eq:Gdef))

via Monte-Carlo integration of the line element of Eq. (eq:line) along the
log-SNR axis log τ = log[(1−ᾱ_t)/ᾱ_t]. The integrand reduces to squared
noise-prediction differences after Tweedie's identity and the change of
variable dτ = τ d(log τ) cancel against the 1/τ² factor that Tweedie
contributes (Appendix C.2).

Algorithm 1 (paper) — implemented here verbatim:

    Inputs:  x ∈ [-1, 1]^{64×64}, ε_θ, f_φ, ᾱ_t schedule, grid {t_j}, σ, N_MC, B
    Output:  top-k eigenpairs of G(x) ≈ Dᵀ D

    1. log τ_j ← log[(1−ᾱ_{t_j})/ᾱ_{t_j}] for j = 1,...,J
    2. w_j     ← ½ (log τ_{j+1} − log τ_{j−1})        (half-weights at endpoints)
    3. for j = 1,...,J:
    4.   for n = 1,...,N_MC in batches of B:
    5.     z ~ N(0, I)                                        [B,1,64,64]
    6.     x_t ← √ᾱ_{t_j} x + √(1−ᾱ_{t_j}) z                  [B,1,64,64]
    7.     ε^u  ← ε_θ(x_t, t_j, 0)                            null-token pass
    8.     x̂   ← (x_t − √(1−ᾱ_{t_j}) ε^u) / √ᾱ_{t_j}                            Tweedie
    9.     x̂   ← clip(x̂, −1, 1)
   10.     μ   ← f_φ(x̂)                                                         [B,N]
   11.     ξ_1, ξ_2 ~ N(0, I)                                                   [B,N]
   12.     r_1, r_2 ← μ + σ ξ_1, μ + σ ξ_2                    Eq. (eq:obs-model)
   13.     ε^c_1, ε^c_2 ← ε_θ(x_t, t_j, r_1), ε_θ(x_t, t_j, r_2)
   14.     Δ ← flatten(ε^c_1 − ε^c_2)                                           [B, 64²]
   15.     append √(w_j / (2 N_MC)) · Δ to D
   16. Stack rows → D ∈ ℝ^{(J·N_MC) × 64²}
   17. (U, S, Vᵀ) ← SVD(D)
   18. return  S²[:k]  and rows of Vᵀ[:k]            top-k eigenpairs of G(x) ≈ DᵀD

This is the **conditional-pair** estimator (Eq. (eq:cc)). It is *not* the
unconditional-conditional estimator (which has a per-image-dependent bias
documented in Appendix C.2 of the paper).

The skeleton (lines 1-2, 3-6, 15-18) is shared by all methods and lives in `estimate_g`;
lines 7-15, which produce the rows of D, are a method of `multigeom.geometry.methods`
(`conditional_pair` by default). `conditional_pair_estimate` is `estimate_g` with that
method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import torch

from ..encoders.contract import ensure_valid
from .methods import DEFAULT_METHOD, get_method

if TYPE_CHECKING:  # the estimator itself needs no diffusers: the UNet is only annotated
    from ..diffusion.model import NeuralConditionedUNet


@dataclass
class EstimatorOutput:
    """
    Top eigenpairs of G(x) returned by `compute_g`: eigenvalues [k] and eigenvectors [k,
    P], largest first. `diagnostics` is what the method reports about its run (a dict,
    or None: the conditional-pair method reports nothing).
    """

    eigenvalues: torch.Tensor  # [k]   top-k eigenvalues of G(x), descending
    eigenvectors: torch.Tensor  # [k, P] rows are top-k eigenvectors of G(x)
    diagnostics: dict | None = None  # reported by the method, if it has any


def _build_log_tau_grid(
    timestep_grid: Iterable[int], alphas_cumprod: torch.Tensor
) -> torch.Tensor:
    """log τ_j = log[(1 − ᾱ_{t_j}) / ᾱ_{t_j}], evaluated at integer timesteps."""
    grid_t = torch.tensor(
        list(timestep_grid), dtype=torch.long, device=alphas_cumprod.device
    )
    a_bar = alphas_cumprod[grid_t]
    return torch.log((1.0 - a_bar) / a_bar)


def _trapezoid_weights(log_tau: torch.Tensor) -> torch.Tensor:
    """Centered trapezoid weights with half-weights at endpoints."""
    J = log_tau.shape[0]
    w = torch.zeros_like(log_tau)
    w[0] = 0.5 * (log_tau[1] - log_tau[0])
    w[-1] = 0.5 * (log_tau[-1] - log_tau[-2])
    if J > 2:
        w[1:-1] = 0.5 * (log_tau[2:] - log_tau[:-2])
    return w


def resolve_method(cfg, method: str | None = None) -> str:
    """
    The name of the method of G(x) to use: `method` if given, else the `estimator` entry
    of the estimator config `cfg`, else "conditional_pair".
    """
    return method or getattr(cfg, "estimator", None) or DEFAULT_METHOD


@torch.no_grad()
def estimate_g(
    image: torch.Tensor,
    encoder,
    unet,
    alphas_cumprod: torch.Tensor,
    cfg,
    device: torch.device,
    generator: torch.Generator | None = None,
    validate: bool = False,
    method: str | None = None,
) -> EstimatorOutput:
    """
    Compute top-k eigenpairs of G(x) for a single image: the shared skeleton of
    Algorithm 1, with the rows of D supplied by a method of
    `multigeom.geometry.methods`.

    Args: image: [1, 1, 64, 64] in [-1, 1] (canonical pipeline input). encoder: f_φ —
    built by `multigeom.encoders.build_encoder`. Must accept `[B, 1, 64, 64]` in [-1, 1]
    and return `[B, n_neurons]`. unet: the diffusion model that the method needs (for
    `conditional_pair`, the trained NeuralConditionedUNet ε_θ). alphas_cumprod:
    scheduler.alphas_cumprod, shape [T] on device. cfg: estimator config namespace (see
    configs/estimator.yaml). device: torch device. generator: optional torch.Generator
    on `device` for all Monte-Carlo draws. Stage 4 seeds one per image so each image's
    estimate is reproducible on its own. validate: check the encoder against the
    contract of `multigeom.encoders.contract` first (it runs the encoder a few times, so
    mocks that record their calls should leave this off). Raises EncoderContractError.
    Off by default here; the Fisher baseline validates by default. method: the name of
    a registered method (`multigeom.geometry.methods.available_methods()`); by default
    the `estimator` entry of `cfg`, else "conditional_pair".

    Returns:
        EstimatorOutput with `eigenvalues` [k] and `eigenvectors` [k, P].
    """
    method_cls = get_method(resolve_method(cfg, method))
    if validate:
        ensure_valid(
            encoder,
            device=device,
            image_shape=tuple(image.shape[1:]),
            sigma=float(cfg.monte_carlo.observation_noise_sigma),
            purpose="inference",
        )
    encoder.eval()
    unet.eval()

    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError(
            f"Expected single-image batch [1, C, H, W], got {tuple(image.shape)}"
        )
    image = image.to(device)
    H, W = image.shape[-2:]
    P = H * W

    # ── Line 1-2: log-τ grid + trapezoid weights ─────────────────────────
    grid_t = list(
        range(
            cfg.monte_carlo.timestep_grid.start,
            cfg.monte_carlo.timestep_grid.stop + 1,
            cfg.monte_carlo.timestep_grid.step,
        )
    )
    log_tau = _build_log_tau_grid(grid_t, alphas_cumprod)
    w = _trapezoid_weights(log_tau)  # [J]
    J = len(grid_t)

    N_MC = int(cfg.monte_carlo.n_samples)
    B = int(cfg.monte_carlo.batch_size)
    if N_MC % B != 0:
        raise ValueError(f"N_MC={N_MC} must be divisible by batch_size={B}")
    n_batches_per_t = N_MC // B
    sampler = method_cls(
        image=image,
        encoder=encoder,
        model=unet,
        alphas_cumprod=alphas_cumprod,
        cfg=cfg,
        device=device,
        generator=generator,
    )

    # Preallocate D matrix on CPU (fits comfortably: J·N_MC × 4096 × 4 B).
    total_rows = J * N_MC
    D = torch.empty(total_rows, P, dtype=torch.float32)
    write_idx = 0

    # ── Lines 3-15: Monte-Carlo loop ─────────────────────────────────────
    for j, t_j in enumerate(grid_t):
        s_j = float((w[j] / (sampler.row_normalization * N_MC)).sqrt())
        sqrt_ab = float(alphas_cumprod[t_j].sqrt())
        sqrt_1m = float((1.0 - alphas_cumprod[t_j]).sqrt())
        t_tensor = torch.full((B,), t_j, device=device, dtype=torch.long)

        for _ in range(n_batches_per_t):
            # Line 5-6: forward diffusion from clean image
            z = torch.randn(B, 1, H, W, device=device, generator=generator)
            x_t = sqrt_ab * image + sqrt_1m * z

            # Lines 7-14: the rows of D, from the method
            delta = sampler.sample_rows(x_t, t_tensor, t_j)  # [B, P]
            D[write_idx : write_idx + B] = (s_j * delta).cpu()
            write_idx += B

    assert write_idx == total_rows, f"row mismatch: {write_idx} vs {total_rows}"

    # ── Lines 16-18: thin SVD → top-k eigenpairs of G ≈ DᵀD ──────────────
    # Thin SVD of D ∈ ℝ^{(J·N_MC) × P}; with the paper's settings
    # J·N_MC = 24·200 = 4800 rows and P = 64² = 4096 columns.
    U, S, Vh = torch.linalg.svd(D, full_matrices=False)
    eigenvalues = S**2  # G ≈ DᵀD has eigenvalues = singular values of D, squared
    eigenvectors = Vh  # rows of Vh are right singular vectors of D = eigvecs of DᵀD

    k = int(cfg.top_k)
    report = getattr(sampler, "diagnostics", None)
    return EstimatorOutput(
        eigenvalues=eigenvalues[:k].clone(),
        eigenvectors=eigenvectors[:k].clone(),
        diagnostics=report() if callable(report) else None,
    )


def conditional_pair_estimate(
    image: torch.Tensor,
    encoder,
    unet: NeuralConditionedUNet,
    alphas_cumprod: torch.Tensor,
    cfg,
    device: torch.device,
    generator: torch.Generator | None = None,
    validate: bool = False,
) -> EstimatorOutput:
    """
    Top-k eigenpairs of G(x) with the conditional-pair estimator (Algorithm 1):
    `estimate_g` with `method="conditional_pair"`, whatever the `estimator` entry of
    `cfg`. Arguments and result as in `estimate_g`.
    """
    return estimate_g(
        image,
        encoder,
        unet,
        alphas_cumprod,
        cfg,
        device,
        generator=generator,
        validate=validate,
        method="conditional_pair",
    )
