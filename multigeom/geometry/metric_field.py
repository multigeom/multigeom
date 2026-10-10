"""
Local metric field on a two-dimensional slice of image space (paper Fig. 5).

At a reference image x₀ let v₁, v₂ be the two leading eigenvectors of a
metric M(x₀) (M = G, the multi-scale Fisher, or M = J, the Fisher
information) of the reference encoder, with eigenvalues λ₁ ≥ λ₂. The
perturbed images are

    x(a, b) = clip(x₀ + a ε₁ v₁ + b ε₂ v₂, −1, 1),     a, b ∈ {−h, …, h},

on an n × n grid (n = 2h + 1). At every grid point the metric of each
encoder is projected onto the fixed reference plane V = [v₁, v₂] ∈ ℝ^{P×2}:

    M₂(a, b) = Vᵀ M(x(a, b)) V ∈ ℝ^{2×2},

evaluated from the truncated eigendecomposition M ≈ U diag(λ) Uᵀ returned by
`multigeom.geometry.estimator` / `multigeom.geometry.fisher`, i.e. M₂ = (VᵀU) diag(λ) (VᵀU)ᵀ. No P × P
matrix is formed.

Step rule. The paper scales the perturbations "by the principal eigenvalues
at the reference image". The default rule is therefore

    ε_i = step_size · √λ_i          ("sqrt_lambda"),

so the grid extends further along the direction the metric weights more. The
alternative "fixed" rule uses ε₁ = ε₂ = step_size (pixel-space units; v_i
have unit Euclidean norm, so a·ε₁ is the Euclidean length of the step).

Ellipse convention. A 2 × 2 PSD matrix M₂ with eigenpairs (μ_max, u_max),
(μ_min, u_min) is drawn as the image of the unit circle under M₂^{1/2}:

    width  = 2 √μ_max,   height = 2 √μ_min,
    angle  = direction of u_max in degrees, measured from the v₁ axis.

The long axis points along the most informative direction of the plane and
its length grows with the information. This is the convention of the original
analysis, and it keeps a small projected metric (e.g. a model whose leading
directions lie outside the reference plane) visible as a small ellipse rather
than an unbounded one. It is the inverse of the iso-distance ellipse
{u : uᵀ M₂ u = 1}, whose semi-axes are 1/√μ.

Everything here is CPU/GPU agnostic and has no dependency on `diffusers`.
"""
from __future__ import annotations

import math

import numpy as np
import torch

STEP_RULES = ("sqrt_lambda", "fixed")


def grid_coefficients(grid_size: int) -> np.ndarray:
    """Integer grid coefficients {−h, …, h} for an odd `grid_size` = 2h + 1."""
    if grid_size < 1 or grid_size % 2 == 0:
        raise ValueError(f"grid_size must be a positive odd integer, got {grid_size}")
    h = grid_size // 2
    return np.arange(-h, h + 1)


def step_sizes(lam1: float, lam2: float, rule: str, step_size: float) -> tuple[float, float]:
    """(ε₁, ε₂) for the given step rule (see module docstring)."""
    if rule == "sqrt_lambda":
        return step_size * math.sqrt(max(lam1, 0.0)), step_size * math.sqrt(max(lam2, 0.0))
    if rule == "fixed":
        return float(step_size), float(step_size)
    raise ValueError(f"Unknown step rule {rule!r}; expected one of {STEP_RULES}")


def canonical_sign(v: torch.Tensor) -> torch.Tensor:
    """Flip `v` so that its largest-magnitude entry is positive (eigenvector
    signs are arbitrary; this makes the grid orientation reproducible)."""
    return v if v[v.abs().argmax()] >= 0 else -v


def perturbation_grid(
    x0: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    eps1: float,
    eps2: float,
    coeffs: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Images x(a, b) = clip(x₀ + a ε₁ v₁ + b ε₂ v₂, −1, 1) on the grid.

    Args:
        x0: [1, 1, H, W] reference image in [−1, 1].
        v1, v2: [H·W] unit-norm directions.
        coeffs: integer grid coefficients (from `grid_coefficients`).

    Returns:
        images [n, n, H, W] with images[i, j] = x(coeffs[i], coeffs[j]), and
        clamp_fraction [n, n], the fraction of pixels the clip changed.
    """
    H, W = x0.shape[-2:]
    base = x0.reshape(H, W)
    d1 = v1.reshape(H, W).to(base)
    d2 = v2.reshape(H, W).to(base)
    n = len(coeffs)
    images = torch.empty(n, n, H, W, dtype=base.dtype, device=base.device)
    clamp_fraction = torch.empty(n, n)
    for i, a in enumerate(coeffs):
        for j, b in enumerate(coeffs):
            raw = base + float(a) * eps1 * d1 + float(b) * eps2 * d2
            images[i, j] = raw.clamp(-1.0, 1.0)
            clamp_fraction[i, j] = float(((raw < -1.0) | (raw > 1.0)).float().mean())
    return images, clamp_fraction


def project_metric(eigvals: torch.Tensor, eigvecs: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    M₂ = Vᵀ (U diag(λ) Uᵀ) V for a truncated eigendecomposition.

    Args:
        eigvals: [k] eigenvalues λ.
        eigvecs: [k, P] eigenvectors as rows (the package convention).
        V: [2, P] basis vectors as rows.

    Returns:
        [2, 2] symmetric float64 tensor.
    """
    proj = V.to(torch.float64) @ eigvecs.to(V.device, torch.float64).T       # [2, k]
    lam = eigvals.to(V.device, torch.float64).clamp(min=0)
    M2 = (proj * lam.unsqueeze(0)) @ proj.T
    return 0.5 * (M2 + M2.T)


def ellipse_params(M2: np.ndarray | torch.Tensor) -> tuple[float, float, float]:
    """(width, height, angle in degrees) of the M₂^{1/2} ellipse (module docstring)."""
    M2 = np.asarray(M2, dtype=np.float64)
    M2 = 0.5 * (M2 + M2.T)
    mu, U = np.linalg.eigh(M2)                      # ascending
    mu = np.clip(mu, 0.0, None)
    width = 2.0 * math.sqrt(mu[1])
    height = 2.0 * math.sqrt(mu[0])
    angle = math.degrees(math.atan2(U[1, 1], U[0, 1]))
    return width, height, angle


def ellipse_grid(matrices: np.ndarray | torch.Tensor) -> np.ndarray:
    """Apply `ellipse_params` over an [n, n, 2, 2] grid → [n, n, 3]."""
    matrices = np.asarray(matrices, dtype=np.float64)
    n1, n2 = matrices.shape[:2]
    out = np.empty((n1, n2, 3))
    for i in range(n1):
        for j in range(n2):
            out[i, j] = ellipse_params(matrices[i, j])
    return out


def point_seed(seed: int, img_idx: int, i: int, j: int, grid_size: int) -> int:
    """
    Deterministic seed for grid point (i, j) of image `img_idx`. Offset by
    10⁶ from the per-image seeds `seed + img_idx` of Stage 4, so the
    Monte-Carlo draws at grid points are distinct from those at x₀. Both
    encoders use the same seed at a grid point (common random numbers).
    """
    return int(seed) + 1_000_000 + (int(img_idx) * grid_size + i) * grid_size + j
