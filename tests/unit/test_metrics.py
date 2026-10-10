"""
Tests for multigeom.metrics: subspace alignment and Bures-Wasserstein distance.

Expected values come from (i) analytic cases (identical / rotated / orthogonal
subspaces; commuting PSD matrices, for which d_BW² = Σ(√a_i − √b_i)²),
(ii) principal angles from scipy.linalg.subspace_angles, and (iii) a dense
P×P evaluation of d_BW² = tr A + tr B − 2 tr (A^{1/2} B A^{1/2})^{1/2} with
matrix square roots computed independently of the k×k reduction in src.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
import torch

from multigeom.geometry.metrics import bures_wasserstein, bures_wasserstein_from_overlap, subspace_alignment


def _orthonormal_rows(k: int, P: int, seed: int) -> torch.Tensor:
    """[k, P] with orthonormal rows (eigenvectors-as-rows convention)."""
    g = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(g.standard_normal((P, k)))
    return torch.from_numpy(Q.T.copy())


def _dense(vals: torch.Tensor, vecs: torch.Tensor) -> np.ndarray:
    """Rank-k reconstruction V Λ Vᵀ with vecs as rows: vecsᵀ diag(vals) vecs."""
    V = vecs.double().numpy()
    return V.T @ np.diag(vals.double().numpy()) @ V


def _psd_sqrt(M: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root via eigh (well defined for singular matrices)."""
    w, U = np.linalg.eigh(0.5 * (M + M.T))
    return (U * np.sqrt(np.clip(w, 0, None))) @ U.T


def _bw_dense(A: np.ndarray, B: np.ndarray, sqrt=_psd_sqrt) -> float:
    rA = sqrt(A)
    cross = np.trace(sqrt(rA @ B @ rA)).real
    return float(np.sqrt(max(np.trace(A) + np.trace(B) - 2 * cross, 0.0)))


# ── subspace alignment ─────────────────────────────────────────────────────
def test_alignment_identical_subspaces_is_one():
    V = _orthonormal_rows(5, 40, seed=0)
    assert subspace_alignment(V, V) == pytest.approx(1.0, abs=1e-6)


def test_alignment_invariant_to_rotation_within_subspace():
    V = _orthonormal_rows(5, 40, seed=1)
    R, _ = np.linalg.qr(np.random.default_rng(2).standard_normal((5, 5)))
    W = torch.from_numpy(R) @ V                       # same span, different basis
    assert subspace_alignment(V, W) == pytest.approx(1.0, abs=1e-6)


def test_alignment_orthogonal_subspaces_is_zero():
    Q = _orthonormal_rows(8, 40, seed=3)
    assert subspace_alignment(Q[:4], Q[4:]) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_alignment_equals_mean_cos2_principal_angles(seed):
    A = _orthonormal_rows(6, 30, seed=seed)
    B = _orthonormal_rows(6, 30, seed=seed + 100)
    theta = scipy.linalg.subspace_angles(A.numpy().T, B.numpy().T)
    assert subspace_alignment(A, B) == pytest.approx(np.mean(np.cos(theta) ** 2), abs=1e-6)


def test_alignment_top_k_uses_leading_rows():
    A = _orthonormal_rows(6, 30, seed=7)
    B = _orthonormal_rows(6, 30, seed=8)
    theta = scipy.linalg.subspace_angles(A[:3].numpy().T, B[:3].numpy().T)
    assert subspace_alignment(A, B, top_k=3) == pytest.approx(np.mean(np.cos(theta) ** 2),
                                                              abs=1e-6)


def test_alignment_rejects_mismatched_k():
    with pytest.raises(ValueError):
        subspace_alignment(_orthonormal_rows(3, 10, 0), _orthonormal_rows(4, 10, 1))


# ── Bures-Wasserstein ──────────────────────────────────────────────────────
def test_bw_of_matrix_with_itself_is_zero():
    V = _orthonormal_rows(5, 30, seed=10)
    lam = torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5], dtype=torch.float64)
    assert bures_wasserstein(lam, V, lam, V) == pytest.approx(0.0, abs=1e-6)


def test_bw_commuting_matrices_closed_form():
    """Shared eigenvectors: d_BW² = Σ_i (√a_i − √b_i)²."""
    V = _orthonormal_rows(4, 25, seed=11)
    a = torch.tensor([9.0, 4.0, 1.0, 0.25], dtype=torch.float64)
    b = torch.tensor([4.0, 4.0, 0.36, 1.0], dtype=torch.float64)
    expected = np.sqrt(np.sum((np.sqrt(a.numpy()) - np.sqrt(b.numpy())) ** 2))
    assert bures_wasserstein(a, V, b, V) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("seed", [12, 13, 14])
def test_bw_low_rank_matches_dense_reference(seed):
    """Random rank-k PSD matrices in ℝ^{P×P} with different column spaces."""
    k, P = 5, 20
    rng = np.random.default_rng(seed)
    VA, VB = _orthonormal_rows(k, P, seed), _orthonormal_rows(k, P, seed + 50)
    la = torch.from_numpy(np.sort(rng.uniform(0.1, 10, k))[::-1].copy())
    lb = torch.from_numpy(np.sort(rng.uniform(0.1, 10, k))[::-1].copy())
    expected = _bw_dense(_dense(la, VA), _dense(lb, VB))
    assert bures_wasserstein(la, VA, lb, VB) == pytest.approx(expected, rel=1e-7, abs=1e-9)


def test_bw_full_rank_matches_scipy_sqrtm():
    """k = P: both matrices are positive definite and scipy.linalg.sqrtm is well conditioned."""
    P = 8
    rng = np.random.default_rng(15)
    VA, VB = _orthonormal_rows(P, P, 15), _orthonormal_rows(P, P, 16)
    la = torch.from_numpy(np.sort(rng.uniform(0.5, 4, P))[::-1].copy())
    lb = torch.from_numpy(np.sort(rng.uniform(0.5, 4, P))[::-1].copy())
    expected = _bw_dense(_dense(la, VA), _dense(lb, VB), sqrt=scipy.linalg.sqrtm)
    assert bures_wasserstein(la, VA, lb, VB) == pytest.approx(expected, rel=1e-7)


def test_bw_is_symmetric_and_satisfies_triangle_inequality():
    k, P = 4, 15
    rng = np.random.default_rng(17)
    mats = [(torch.from_numpy(np.sort(rng.uniform(0.1, 5, k))[::-1].copy()),
             _orthonormal_rows(k, P, 17 + i)) for i in range(3)]
    d = lambda i, j: bures_wasserstein(*mats[i], *mats[j])  # noqa: E731
    assert d(0, 1) == pytest.approx(d(1, 0), rel=1e-9)
    assert d(0, 2) <= d(0, 1) + d(1, 2) + 1e-9


def test_bw_from_overlap_consistent_with_bw_at_every_k():
    K, P = 6, 30
    rng = np.random.default_rng(18)
    VA, VB = _orthonormal_rows(K, P, 18), _orthonormal_rows(K, P, 19)
    la = torch.from_numpy(np.sort(rng.uniform(0.1, 5, K))[::-1].copy())
    lb = torch.from_numpy(np.sort(rng.uniform(0.1, 5, K))[::-1].copy())
    C = VA @ VB.T
    for k in range(1, K + 1):
        assert bures_wasserstein_from_overlap(la[:k], lb[:k], C[:k, :k]) == pytest.approx(
            bures_wasserstein(la, VA, lb, VB, top_k=k), rel=1e-12)
        assert bures_wasserstein(la, VA, lb, VB, top_k=k) == pytest.approx(
            _bw_dense(_dense(la[:k], VA[:k]), _dense(lb[:k], VB[:k])), rel=1e-7, abs=1e-9)


def test_bw_rejects_mismatched_k():
    V = _orthonormal_rows(4, 10, 0)
    with pytest.raises(ValueError):
        bures_wasserstein(torch.ones(4), V, torch.ones(3), V[:3])
