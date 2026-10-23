"""
Tests of the methods of G(x): the registry, and the skeleton of `estimate_g` that all
methods share (log-τ grid, weights, Monte-Carlo loop, matrix D, SVD).

A method that returns constant rows has a known answer. With rows Δ = 1 ∈ ℝ^P, the rows
of D are s_j · 1 with s_j² = w_j / (norm · N_MC), so

    DᵀD = Σ_j N_MC s_j² 1 1ᵀ = (Σ_j w_j / norm) 1 1ᵀ,

whose only non-zero eigenvalue is (W / norm) · P, with W = Σ_j w_j = log τ_J − log τ_1
and eigenvector 1/√P. This checks the weights and the normalisation of the skeleton
independently of any diffusion model.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import estimator_cfg, linear_beta_alphas_cumprod
from multigeom.geometry import available_methods, estimate_g, register_method
from multigeom.geometry.estimator import (
    _build_log_tau_grid,
    conditional_pair_estimate,
    resolve_method,
)
from multigeom.geometry.methods import _METHODS, get_method

H = W_IMG = 4
P = H * W_IMG
GRID = (100, 900, 100)


class ConstantRows:
    """A method whose rows are all ones (a known answer)."""

    row_normalization = 1.0

    def __init__(self, **kwargs):
        pass

    def sample_rows(self, x_t, t, t_j):
        return torch.ones(x_t.shape[0], x_t[0].numel())


class HalfWeightRows(ConstantRows):
    """The same rows, normalised like the conditional-pair estimator."""

    row_normalization = 2.0


class TinyEncoder(nn.Module):
    n_neurons = 3

    def forward(self, x):
        return x.flatten(1).mean(1, keepdim=True) * torch.ones(1, self.n_neurons)


class TinyUNet(nn.Module):
    def forward(self, x_t, t, r):
        return torch.tanh(x_t) + 0.1 * r.sum(1).view(-1, 1, 1, 1)


@pytest.fixture
def registered():
    """Register the fake methods for one test, and remove them afterwards."""
    register_method("constant_rows", ConstantRows)
    register_method("half_weight_rows", HalfWeightRows)
    yield
    _METHODS.pop("constant_rows", None)
    _METHODS.pop("half_weight_rows", None)


def _cfg(**kwargs):
    cfg = estimator_cfg(
        n_samples=20,
        batch_size=10,
        start=GRID[0],
        stop=GRID[1],
        step=GRID[2],
        sigma=0.3,
        top_k=3,
    )
    for key, val in kwargs.items():
        setattr(cfg, key, val)
    return cfg


def _run(method=None, cfg=None):
    ab = linear_beta_alphas_cumprod().float()
    image = torch.linspace(-1, 1, P).view(1, 1, H, W_IMG)
    return estimate_g(
        image,
        TinyEncoder(),
        TinyUNet(),
        ab,
        cfg or _cfg(),
        torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
        method=method,
    )


def _total_weight():
    ab = linear_beta_alphas_cumprod().float()
    log_tau = _build_log_tau_grid(range(GRID[0], GRID[1] + 1, GRID[2]), ab)
    return float(log_tau[-1] - log_tau[0])


# ── The registry ─────────────────────────────────────────────────────────
def test_conditional_pair_is_registered_by_default():
    assert "conditional_pair" in available_methods()
    assert resolve_method(_cfg()) == "conditional_pair"


def test_unknown_method_names_the_valid_ones():
    with pytest.raises(ValueError, match="conditional_pair"):
        get_method("zeroshot_typo")
    with pytest.raises(ValueError, match="unknown method"):
        _run(method="zeroshot_typo")


def test_registering_twice_needs_overwrite(registered):
    with pytest.raises(ValueError, match="already registered"):
        register_method("constant_rows", ConstantRows)
    register_method("constant_rows", HalfWeightRows, overwrite=True)
    assert get_method("constant_rows") is HalfWeightRows


def test_method_comes_from_the_argument_or_the_config(registered):
    assert resolve_method(_cfg(estimator="constant_rows")) == "constant_rows"
    assert resolve_method(_cfg(estimator="constant_rows"), "half_weight_rows") == (
        "half_weight_rows"
    )


# ── The skeleton, with a method of known answer ──────────────────────────
def test_skeleton_weights_and_normalisation(registered):
    total = _total_weight()
    out = _run(method="constant_rows")
    assert float(out.eigenvalues[0]) == pytest.approx(total * P, rel=1e-5)
    assert float(out.eigenvalues[1]) == pytest.approx(0.0, abs=1e-4)
    vec = out.eigenvectors[0]
    assert torch.allclose(vec.abs(), torch.full((P,), P**-0.5), atol=1e-5)
    half = _run(method="half_weight_rows")
    assert float(half.eigenvalues[0]) == pytest.approx(total * P / 2, rel=1e-5)


def test_method_chosen_by_the_config_entry(registered):
    by_cfg = _run(cfg=_cfg(estimator="constant_rows"))
    by_arg = _run(method="constant_rows")
    assert torch.equal(by_cfg.eigenvalues, by_arg.eigenvalues)


# ── Backward compatibility ───────────────────────────────────────────────
def test_default_method_is_the_conditional_pair_estimator(registered):
    ab = linear_beta_alphas_cumprod().float()
    image = torch.linspace(-1, 1, P).view(1, 1, H, W_IMG)

    def run(fn, cfg):
        return fn(
            image,
            TinyEncoder(),
            TinyUNet(),
            ab,
            cfg,
            torch.device("cpu"),
            generator=torch.Generator().manual_seed(5),
        )

    default = run(estimate_g, _cfg())
    legacy = run(conditional_pair_estimate, _cfg())
    assert torch.equal(default.eigenvalues, legacy.eigenvalues)
    assert torch.equal(default.eigenvectors, legacy.eigenvectors)
    # `conditional_pair_estimate` ignores the `estimator` entry of the config
    other = run(conditional_pair_estimate, _cfg(estimator="constant_rows"))
    assert torch.equal(other.eigenvalues, legacy.eigenvalues)


def test_a_method_can_report_diagnostics(registered):
    class Reporting(ConstantRows):
        def diagnostics(self):
            return {"rows": 1}

    register_method("reporting", Reporting)
    try:
        assert _run(method="reporting").diagnostics == {"rows": 1}
        assert _run(method="constant_rows").diagnostics is None
    finally:
        _METHODS.pop("reporting", None)
