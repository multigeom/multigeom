"""
Cross-encoder similarity measures for the multi-scale Fisher metric G(x)
and the classical Fisher J(x).

Two measures, both used in the paper:

  1. **Subspace alignment** — angle-only, eigenvalue-blind.
       𝒜_k(A, B) = (1/k) ‖V_Aᵀ V_B‖_F²
                 = (1/k) Σ_i cos² θ_i        (paper Eq. (eq:subspace-alignment))

  2. **Bures–Wasserstein distance** — sensitive to both orientation and scale.
       d_BW²(A, B) = tr(A) + tr(B) − 2 tr[(A^{1/2} B A^{1/2})^{1/2}]   (Eq. (eq:bures))

       In practice we evaluate the cross term on the truncated eigendecomposition
       A ≈ V_A Λ_A V_Aᵀ, B ≈ V_B Λ_B V_Bᵀ:
           tr[(Λ_A^{1/2} C Λ_B Cᵀ Λ_A^{1/2})^{1/2}] = Σ_i √μ_i
       where C = V_Aᵀ V_B ∈ ℝ^{k × k} and {μ_i} are the eigenvalues of
       M = Λ_A^{1/2} C Λ_B Cᵀ Λ_A^{1/2}. We symmetrise M before
       diagonalisation to guard against numerical asymmetry.

Both functions take two truncated eigendecompositions in the convention
returned by `src.estimator.conditional_pair_estimate` and
`src.fisher.fisher_eigendecomposition`: eigenvectors as **rows** of a
[k, P] matrix, eigenvalues as a [k] vector, both descending.
"""
from __future__ import annotations

import torch


def _maybe_truncate(eigvals: torch.Tensor, eigvecs: torch.Tensor, top_k: int | None):
    """Restrict to the top-`top_k` rows if requested."""
    if top_k is None:
        return eigvals, eigvecs
    return eigvals[:top_k], eigvecs[:top_k]


def subspace_alignment(
    eigvecs_A: torch.Tensor,
    eigvecs_B: torch.Tensor,
    top_k: int | None = None,
) -> float:
    """
    Compute 𝒜_k = (1/k) ‖V_Aᵀ V_B‖_F² where V_A, V_B are the top-k
    eigenvectors of two PSD matrices.

    Args:
        eigvecs_A, eigvecs_B: [k, P] tensors with eigenvectors as rows.
        top_k: if given, restrict both inputs to their top-`top_k` rows.

    Returns:
        Scalar in [0, 1]. 1.0 ↔ identical subspaces.
    """
    if eigvecs_A.dim() != 2 or eigvecs_B.dim() != 2:
        raise ValueError("eigvecs_A and eigvecs_B must be 2-D [k, P]")
    if top_k is not None:
        eigvecs_A = eigvecs_A[:top_k]
        eigvecs_B = eigvecs_B[:top_k]
    if eigvecs_A.shape[0] != eigvecs_B.shape[0]:
        raise ValueError(
            f"Mismatched k: {eigvecs_A.shape[0]} vs {eigvecs_B.shape[0]}; "
            f"slice both to the same top_k first."
        )
    k = eigvecs_A.shape[0]
    overlap = eigvecs_A @ eigvecs_B.T              # [k, k]
    return float((overlap.norm(p="fro") ** 2) / k)


def bures_wasserstein(
    eigvals_A: torch.Tensor,
    eigvecs_A: torch.Tensor,
    eigvals_B: torch.Tensor,
    eigvecs_B: torch.Tensor,
    top_k: int | None = None,
) -> float:
    """
    Compute the Bures–Wasserstein distance d_BW(A, B) on the cone of PSD
    matrices, using truncated eigendecompositions.

    Args:
        eigvals_*: [k] tensors of eigenvalues (descending order).
        eigvecs_*: [k, P] tensors of eigenvectors as rows.
        top_k: if given, restrict to top-`top_k` eigenpairs of each.

    Returns:
        d_BW(A, B) ≥ 0. Returns 0 iff A = B.
    """
    eigvals_A, eigvecs_A = _maybe_truncate(eigvals_A, eigvecs_A, top_k)
    eigvals_B, eigvecs_B = _maybe_truncate(eigvals_B, eigvecs_B, top_k)
    if eigvals_A.shape != eigvals_B.shape:
        raise ValueError(
            f"Mismatched k: {tuple(eigvals_A.shape)} vs {tuple(eigvals_B.shape)}"
        )

    # Cast to float64 for numerical stability of the matrix square root.
    eigvals_A = eigvals_A.to(torch.float64)
    eigvals_B = eigvals_B.to(torch.float64)
    eigvecs_A = eigvecs_A.to(torch.float64)
    eigvecs_B = eigvecs_B.to(torch.float64)

    # Truncated eigendecomposition convention here: rows of eigvecs_* are
    # eigenvectors. Then A ≈ V_Aᵀ Λ_A V_A; the orientation flips compared to
    # the columns convention, but the cross term is unchanged because
    # V_Aᵀ V_B and (V_Aᵀ)ᵀ (V_Bᵀ) = V_A V_Bᵀ both have the same singular values.
    C = eigvecs_A @ eigvecs_B.T                                  # [k, k]
    sqrt_LA = eigvals_A.clamp(min=0).sqrt()                      # [k]
    LB = eigvals_B.clamp(min=0)                                  # [k]
    # M = Λ_A^{1/2} C Λ_B Cᵀ Λ_A^{1/2}, computed without forming the diagonals
    M = (sqrt_LA.unsqueeze(1) * C) * LB.unsqueeze(0)             # = Λ_A^{1/2} C Λ_B
    M = M @ C.T                                                  # × Cᵀ
    M = M * sqrt_LA.unsqueeze(0)                                 # × Λ_A^{1/2}
    M = 0.5 * (M + M.T)                                          # symmetrise

    mu = torch.linalg.eigvalsh(M)                                # real eigenvalues
    cross_term = mu.clamp(min=0).sqrt().sum()

    tr_A = eigvals_A.clamp(min=0).sum()
    tr_B = eigvals_B.clamp(min=0).sum()
    d_sq = (tr_A + tr_B - 2.0 * cross_term).clamp(min=0)
    return float(d_sq.sqrt())
