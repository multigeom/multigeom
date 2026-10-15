"""
Checks of the 2 x 2 projection and the ellipse convention of
multigeom/geometry/metric_field.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from multigeom.geometry.metric_field import ellipse_params, project_metric

P = 64 * 64


def _orthonormal_rows(k: int, g: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(P, k, generator=g))
    return q.T.contiguous()  # [k, P]


def test_project_metric_recovers_eigenvalues_in_own_basis():
    g = torch.Generator().manual_seed(1)
    vecs = _orthonormal_rows(5, g)
    lam = torch.tensor([4.0, 1.0, 0.5, 0.2, 0.1])
    M2 = project_metric(lam, vecs, vecs[:2])
    assert torch.allclose(
        M2, torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64)), atol=1e-6
    )
    # A plane orthogonal to every eigenvector sees no metric.
    q, _ = torch.linalg.qr(torch.cat([vecs.T, torch.randn(P, 2, generator=g)], dim=1))
    assert torch.allclose(
        project_metric(lam, vecs, q[:, 5:].T),
        torch.zeros(2, 2, dtype=torch.float64),
        atol=1e-6,
    )


def test_ellipse_axes_are_twice_root_eigenvalues():
    w, h, ang = ellipse_params(np.diag([4.0, 1.0]))
    assert (w, h) == pytest.approx((4.0, 2.0))
    assert abs(np.sin(np.radians(ang))) == pytest.approx(0.0, abs=1e-9)
    w, h, ang = ellipse_params(np.diag([1.0, 9.0]))  # major axis along e2
    assert (w, h) == pytest.approx((6.0, 2.0))
    assert abs(np.cos(np.radians(ang))) == pytest.approx(0.0, abs=1e-9)
