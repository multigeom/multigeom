"""
Tests for multigeom.fisher: J(x) = (1/σ²) J_fᵀ J_f via reverse-mode autograd.

For an encoder whose readout ∘ backbone is linear, f(x) = W g(x) + b with g the
input conversion, J_f = W · ∂g/∂x and the eigenpairs of J(x) are known in
closed form: with the identity conversion they are those of WᵀW/σ²; with the
[-1, 1] → [0, 1] conversion of the ResNet/OnOff encoders (∂g/∂x = 1/2) they are
those of WᵀW/(4σ²). The reference uses numpy.linalg.eigh on the dense P×P
matrix.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from multigeom.geometry.fisher import fisher_eigendecomposition

H = W = 6
P = H * W


class LinearEncoder(nn.Module):
    """Mock exposing the interface multigeom.fisher uses: _to_backbone_input, backbone, readout."""

    def __init__(self, Wmat: torch.Tensor, to_unit_range: bool = False):
        super().__init__()
        self.n_neurons = Wmat.shape[0]
        self.to_unit_range = to_unit_range
        self.backbone = nn.Flatten()
        self.readout = nn.Linear(P, self.n_neurons)
        with torch.no_grad():
            self.readout.weight.copy_(Wmat)
            self.readout.bias.normal_()

    def _to_backbone_input(self, x):
        # Same conversion as multigeom.encoders._BaseEncoder for backbones in [0, 1].
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0) if self.to_unit_range else x


def _reference(Wmat: np.ndarray, scale: float, k: int):
    vals, vecs = np.linalg.eigh(scale * Wmat.T @ Wmat)
    order = np.argsort(vals)[::-1][:k]
    return vals[order], vecs[:, order]


@pytest.mark.parametrize("to_unit_range, jac", [(False, 1.0), (True, 0.5)])
def test_fisher_eigenpairs_of_linear_encoder(to_unit_range, jac):
    torch.manual_seed(0)
    n, sigma = 5, 0.22
    Wmat = torch.randn(n, P)
    enc = LinearEncoder(Wmat, to_unit_range)
    # Include saturated pixels (exactly ±1): the clamp must pass the gradient there.
    x = torch.rand(1, 1, H, W) * 2 - 1
    x[0, 0, 0, :3] = torch.tensor([-1.0, 1.0, -1.0])
    out = fisher_eigendecomposition(x, enc, sigma=sigma, top_k=n)

    ref_vals, ref_vecs = _reference(Wmat.double().numpy(), jac ** 2 / sigma ** 2, n)
    np.testing.assert_allclose(out.eigenvalues.double().numpy(), ref_vals, rtol=1e-4)
    cos = np.abs(np.einsum("kp,pk->k", out.eigenvectors.double().numpy(), ref_vecs))
    np.testing.assert_allclose(cos, 1.0, atol=1e-4)


def test_fisher_top_k_truncates_and_caps_at_n_neurons():
    torch.manual_seed(1)
    enc = LinearEncoder(torch.randn(4, P))
    x = torch.zeros(1, 1, H, W)
    assert fisher_eigendecomposition(x, enc, sigma=1.0, top_k=2).eigenvalues.shape == (2,)
    out = fisher_eigendecomposition(x, enc, sigma=1.0, top_k=50)
    assert out.eigenvalues.shape == (4,) and out.eigenvectors.shape == (4, P)


def test_fisher_rejects_batch():
    enc = LinearEncoder(torch.randn(3, P))
    with pytest.raises(ValueError):
        fisher_eigendecomposition(torch.zeros(2, 1, H, W), enc, sigma=1.0)


def test_fisher_works_inside_no_grad_context():
    torch.manual_seed(2)
    Wmat = torch.randn(3, P)
    with torch.no_grad():
        out = fisher_eigendecomposition(torch.zeros(1, 1, H, W), LinearEncoder(Wmat),
                                        sigma=1.0, top_k=3)
    ref_vals, _ = _reference(Wmat.double().numpy(), 1.0, 3)
    np.testing.assert_allclose(out.eigenvalues.double().numpy(), ref_vals, rtol=1e-4)
