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

`estimate_g_per_scale` runs the same Monte-Carlo loop but keeps the noise levels apart:
G(x) = ∫ M_τ(x) d log τ, and at each τ_j of the grid it returns the top eigenpairs of
the integrand

    M_τ(x) = ½ E[ (1/τ) (x̂(x_τ, r) − x̂(x_τ, r′)) (x̂(x_τ, r) − x̂(x_τ, r′))ᵀ ]

from the N_MC rows drawn at τ_j alone, without the weight w_j (line 15 appends
√(1 / (2 N_MC)) · Δ, and lines 16-18 are done once per noise level). The 1/τ cancels
the √τ that Tweedie's formula puts in front of ε, so the rows are the same differences
of noise predictions as for G(x).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterable, Iterator

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


@dataclass
class PerScaleOutput:
    """
    Top eigenpairs of M_τ(x), the integrand of G(x) = ∫ M_τ(x) d log τ, at each noise
    level τ_j of the grid, returned by `compute_g_per_scale`: log τ_j [J] (ascending),
    and at each τ_j the eigenvalues [J, k] and eigenvectors [J, k, P], largest first.
    `diagnostics` is what the method reports about its run (a dict, or None).
    """

    log_tau: torch.Tensor  # [J]       log of the noise variance τ_j, ascending
    eigenvalues: torch.Tensor  # [J, k]    top-k eigenvalues of M_τ(x) at each τ_j
    eigenvectors: torch.Tensor  # [J, k, P] rows are its top-k eigenvectors
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


def _timestep_grid(cfg: SimpleNamespace) -> list[int]:
    """The DDPM steps t_1, ..., t_J of the estimator config `cfg`, stop included."""
    # the grid of the config, evenly spaced on the DDPM index
    grid = cfg.monte_carlo.timestep_grid

    # the steps from start to stop (included)
    return list(range(grid.start, grid.stop + 1, grid.step))


def _check_inputs(image: torch.Tensor, cfg: SimpleNamespace) -> None:
    """
    Raise a ValueError unless `image` is a single image [1, C, H, W] and the number of
    Monte-Carlo samples N_MC is a multiple of the batch size of `cfg`.
    """
    # one image at a time
    if image.dim() != 4 or image.shape[0] != 1:
        raise ValueError(
            f"Expected single-image batch [1, C, H, W], got {tuple(image.shape)}"
        )

    # the samples of each noise level are drawn in whole batches
    n_samples, batch_size = cfg.monte_carlo.n_samples, cfg.monte_carlo.batch_size
    if int(n_samples) % int(batch_size) != 0:
        raise ValueError(
            f"N_MC={n_samples} must be divisible by batch_size={batch_size}"
        )


def _build_sampler(
    image: torch.Tensor,
    encoder: torch.nn.Module,
    unet: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
    generator: torch.Generator | None,
    validate: bool,
    method: str | None,
) -> tuple[torch.Tensor, Any]:
    """
    Check the arguments of `estimate_g` (described there) and set up the method that
    draws the rows of D: returns the image on `device` and the method's sampler.
    """
    # the image and the settings must fit the estimator
    _check_inputs(image, cfg)

    # the class of the method that draws the rows of D (lines 7-14)
    method_cls = get_method(resolve_method(cfg, method))

    # optionally, check that the encoder follows the contract (this runs the encoder)
    if validate:
        ensure_valid(
            encoder,
            device=device,
            image_shape=tuple(image.shape[1:]),
            sigma=float(cfg.monte_carlo.observation_noise_sigma),
            purpose="inference",
        )

    # inference mode: no dropout in the encoder or the network
    encoder.eval()
    unet.eval()

    # the clean image x, on the device of the networks
    image = image.to(device)

    # the method, set up for this image
    sampler = method_cls(
        image=image,
        encoder=encoder,
        model=unet,
        alphas_cumprod=alphas_cumprod,
        cfg=cfg,
        device=device,
        generator=generator,
    )
    return image, sampler


def _draw_rows(
    image: torch.Tensor,
    sampler: Any,
    alphas_cumprod: torch.Tensor,
    cfg: SimpleNamespace,
    row_scales: list[float],
    device: torch.device,
    generator: torch.Generator | None,
) -> Iterator[torch.Tensor]:
    """
    Lines 3-15 of Algorithm 1, one noise level at a time: for each DDPM step t_j of the
    grid, the N_MC rows of D drawn at t_j ([N_MC, P], on the device of the method), each
    multiplied by `row_scales[j]`.
    """
    # the number of samples per noise level, drawn in batches
    n_samples = int(cfg.monte_carlo.n_samples)
    batch_size = int(cfg.monte_carlo.batch_size)

    # the height and width of the image
    height, width = image.shape[-2:]

    # line 3: one noise level t_j at a time, with the scale of its rows
    for t_j, scale in zip(_timestep_grid(cfg), row_scales):
        # the weights of the clean image and of the noise at this level
        sqrt_ab = float(alphas_cumprod[t_j].sqrt())
        sqrt_1m = float((1.0 - alphas_cumprod[t_j]).sqrt())

        # the step t_j, once for each image of a batch
        t_tensor = torch.full((batch_size,), t_j, device=device, dtype=torch.long)

        # line 4: the N_MC samples of this level, in batches
        rows = []
        for _ in range(n_samples // batch_size):
            # lines 5-6: noisy images x_t = √ᾱ x + √(1 − ᾱ) z (forward diffusion)
            z = torch.randn(
                batch_size, 1, height, width, device=device, generator=generator
            )
            x_t = sqrt_ab * image + sqrt_1m * z

            # lines 7-15: the rows of D, from the method, scaled
            rows.append(scale * sampler.sample_rows(x_t, t_tensor, t_j))

        # the N_MC rows of this level
        yield torch.cat(rows)


def _top_eigenpairs(
    rows: torch.Tensor, k: int, squared: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Lines 17-18 of Algorithm 1: the top-k eigenpairs from the thin SVD of the rows D, as
    eigenvalues [k] and eigenvectors [k, P] (the right singular vectors), largest first.
    The eigenvalues are those of DᵀD, the squared singular values (`squared`), or the
    singular values themselves, those of M when the rows of D are M ξ for random probes
    ξ (DᵀD ≈ M², as for M_t^env in `multigeom.geometry.env`).
    """
    # thin SVD D = U S Vᵀ, so DᵀD = V S² Vᵀ: eigenvalues S², eigenvectors the rows of Vᵀ
    _, singular_values, right_vectors = torch.linalg.svd(rows, full_matrices=False)

    # S² or S, the k largest, as tensors of their own (not views of the decomposition)
    values = singular_values**2 if squared else singular_values
    return values[:k].clone(), right_vectors[:k].clone()


def _level_row_scales(sampler: Any, cfg: SimpleNamespace, n_levels: int) -> list[float]:
    """
    The scale of the rows of each of `n_levels` noise levels when they are not weighted
    by w_j: √(1 / (norm · N_MC)), with norm the `row_normalization` of the method.
    """
    # the number of rows of a level, times the normalisation of the method
    n_rows = sampler.row_normalization * int(cfg.monte_carlo.n_samples)
    return [n_rows**-0.5] * n_levels


def _per_scale_output(
    log_tau: torch.Tensor,
    levels: Iterator[torch.Tensor],
    k: int,
    sampler: Any,
    squared: bool = True,
) -> PerScaleOutput:
    """
    The top-k eigenpairs of the rows of each noise level (`levels`, [N_MC, P] each, as
    `_draw_rows` yields them), stacked along a first axis into a PerScaleOutput on the
    CPU, with what the method reports once all its rows are drawn.
    """
    # one SVD per level
    pairs = [_top_eigenpairs(rows, k, squared) for rows in levels]

    # the eigenpairs of all the levels, stacked, and the method's report
    return PerScaleOutput(
        log_tau=log_tau.cpu(),
        eigenvalues=torch.stack([values for values, _ in pairs]).cpu(),
        eigenvectors=torch.stack([vectors for _, vectors in pairs]).cpu(),
        diagnostics=_diagnostics(sampler),
    )


def _diagnostics(sampler: Any) -> dict | None:
    """What the method reports about its run, or None if it reports nothing."""
    # a method may define diagnostics(); the conditional-pair method does not
    report = getattr(sampler, "diagnostics", None)
    return report() if callable(report) else None


@torch.no_grad()
def estimate_g(
    image: torch.Tensor,
    encoder: torch.nn.Module,
    unet: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    cfg: SimpleNamespace,
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
    # the image on the device, and the method that draws the rows of D
    image, sampler = _build_sampler(
        image, encoder, unet, alphas_cumprod, cfg, device, generator, validate, method
    )

    # ── Line 1-2: log-τ grid + trapezoid weights ─────────────────────────
    log_tau = _build_log_tau_grid(_timestep_grid(cfg), alphas_cumprod)
    w = _trapezoid_weights(log_tau)  # [J]

    # line 15: the rows of level j are weighted by √(w_j / (2 N_MC))
    n_rows = sampler.row_normalization * int(cfg.monte_carlo.n_samples)
    row_scales = [float((w_j / n_rows).sqrt()) for w_j in w]

    # ── Lines 3-16: Monte-Carlo loop, all the rows stacked into D on the CPU ──
    # (J·N_MC × P × 4 B: 4800 × 4096 × 4 B = 79 MB with the paper's settings)
    levels = _draw_rows(
        image, sampler, alphas_cumprod, cfg, row_scales, device, generator
    )
    D = torch.cat([rows.cpu() for rows in levels]).float()

    # ── Lines 17-18: thin SVD → top-k eigenpairs of G ≈ DᵀD ──────────────
    eigenvalues, eigenvectors = _top_eigenpairs(D, int(cfg.top_k))
    return EstimatorOutput(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        diagnostics=_diagnostics(sampler),
    )


@torch.no_grad()
def estimate_g_per_scale(
    image: torch.Tensor,
    encoder: torch.nn.Module,
    unet: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
    generator: torch.Generator | None = None,
    validate: bool = False,
    method: str | None = None,
) -> PerScaleOutput:
    """
    Compute the top-k eigenpairs of M_τ(x), the integrand of G(x) = ∫ M_τ(x) d log τ, at
    each noise level τ_j of the grid of `cfg`: the Monte-Carlo loop of `estimate_g`,
    with one SVD of the N_MC rows of each noise level, without the weight w_j, in place
    of one SVD of all the rows. Arguments as in `estimate_g`; `cfg.top_k` eigenpairs are
    kept at each noise level.

    With the same `generator` the rows are those of `estimate_g`, so Σ_j w_j M_τ_j(x) is
    its G(x). Each M_τ(x) rests on N_MC rows only, J times fewer than G(x), so its
    eigenvectors are noisier: raise `cfg.monte_carlo.n_samples` for stabler ones.

    Returns:
        PerScaleOutput with `log_tau` [J], `eigenvalues` [J, k] and `eigenvectors`
        [J, k, P].
    """
    # the image on the device, and the method that draws the rows of D
    image, sampler = _build_sampler(
        image, encoder, unet, alphas_cumprod, cfg, device, generator, validate, method
    )

    # line 1: the noise levels, as log τ_j
    log_tau = _build_log_tau_grid(_timestep_grid(cfg), alphas_cumprod)

    # line 15 without w_j: every row is weighted by √(1 / (2 N_MC))
    row_scales = _level_row_scales(sampler, cfg, len(log_tau))

    # lines 3-18, one noise level at a time: the top-k eigenpairs of M_τ_j ≈ D_jᵀ D_j
    levels = _draw_rows(
        image, sampler, alphas_cumprod, cfg, row_scales, device, generator
    )
    return _per_scale_output(log_tau, levels, int(cfg.top_k), sampler)


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
