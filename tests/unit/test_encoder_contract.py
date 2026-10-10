"""
Tests for src.encoder_contract: the rules an encoder must satisfy (R1-R9), the check that enforces them, and
its use by the Fisher baseline (generic path through the public forward) and, opt-in, the G(x) estimator.
A good toy encoder passes; one deliberately bad encoder per rule is rejected with that rule's id.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoder_contract import EncoderContractError, ensure_valid, validate_encoder
from src.encoders import _BaseEncoder
from src.fisher import fisher_eigendecomposition

H = W = 8
P = H * W
SHAPE = (1, H, W)


class Toy(nn.Module):
    """A fitted linear model with 7 neurons; responses of order 1 on [-1, 1] images."""

    def __init__(self, n=7, scale=None):
        super().__init__()
        torch.manual_seed(0)
        self.n_neurons = n
        self.lin = nn.Linear(P, n)
        with torch.no_grad():
            self.lin.weight.mul_((3.0 / P ** 0.5) if scale is None else scale)

    def forward(self, x):
        return self.lin(x.flatten(1))


def check(enc, **kw):
    return validate_encoder(enc, image_shape=SHAPE, **kw)


# ── a good encoder passes ───────────────────────────────────────────────────
def test_good_encoder_passes_with_no_warning():
    report = check(Toy())
    assert report.n_neurons == 7 and not report.warnings and 0.1 < report.response_std < 10


def test_reference_encoder_structure_passes_all_rules():
    class TinyBackbone(nn.Module):
        out_channels, out_spatial = 4, 4

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3, padding=1)
            for p in self.parameters():
                p.requires_grad = False

        def forward(self, x):
            return F.adaptive_avg_pool2d(torch.relu(self.conv(x)), 4)

    class Ref(_BaseEncoder):
        backbone_input_channels = 1
        backbone_input_range = (0.0, 1.0)

        def __init__(self):
            super().__init__(TinyBackbone(), n_neurons=5, dropout=0.25)

    enc = Ref()
    report = check(enc)                                    # R5 included: the forward is differentiable
    assert report.n_neurons == 5
    x = torch.rand(3, *SHAPE) * 2 - 1
    with torch.no_grad():
        assert torch.equal(enc.eval()(x), enc.readout(enc.features(x)))      # same numbers as before the change


# ── one bad encoder per rule ────────────────────────────────────────────────
def rule_of(enc, **kw):
    with pytest.raises(EncoderContractError) as e:
        check(enc, **kw)
    return e.value.rule


def test_R1_not_a_module():
    assert rule_of(lambda x: x.flatten(1)[:, :3]) == "R1"


def test_R2_forward_fails_on_canonical_input():
    class ThreeChannels(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 2, 3)

        def forward(self, x):
            return self.conv(x).flatten(1)
    assert rule_of(ThreeChannels()) == "R2"


@pytest.mark.parametrize("make", [
    lambda x: x.flatten(1).sum(1),                          # 1-D output
    lambda x: x.flatten(1)[:, :3] * float("nan"),           # NaN
    lambda x: x.flatten(1)[:, :3].long(),                   # not floating point
    lambda x: x.flatten(1)[:, : x.shape[0]],                # width depends on the batch size
])
def test_R3_bad_output(make):
    class Bad(nn.Module):
        def forward(self, x):
            return make(x)
    assert rule_of(Bad()) == "R3"


def test_R3_declared_n_neurons_must_match():
    enc = Toy()
    enc.n_neurons = 9
    assert rule_of(enc) == "R3"


def test_R4_noise_at_inference():
    class Noisy(Toy):
        def forward(self, x):
            return super().forward(x) + 0.1 * torch.randn(x.shape[0], self.n_neurons)
    assert rule_of(Noisy()) == "R4"


def test_R4_dropout_forced_on():
    class Dropout(Toy):
        def forward(self, x):
            return F.dropout(super().forward(x), p=0.5, training=True)
    assert rule_of(Dropout()) == "R4"


@pytest.mark.parametrize("break_grad", [lambda y: y.detach(), None])
def test_R5_no_gradient_through_forward(break_grad):
    class Cut(Toy):
        def forward(self, x):
            if break_grad is None:
                with torch.no_grad():
                    return super().forward(x)
            return break_grad(super().forward(x))
    assert rule_of(Cut()) == "R5"


def test_R5_zero_gradient():
    class Flat(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 3) + 0 * x.flatten(1)[:, :3]
    assert rule_of(Flat()) == "R5"


def test_R6_batch_statistics():
    class BatchNorm(Toy):
        def forward(self, x):
            y = super().forward(x)
            return (y - y.mean(0, keepdim=True)) / (y.std(0, keepdim=True, correction=0) + 1e-3)   # train-mode batch norm
    assert rule_of(BatchNorm()) == "R6"


def test_R7_modifies_its_input():
    class InPlace(Toy):
        def forward(self, x):
            x.mul_(0.5)
            return super().forward(x)
    assert rule_of(InPlace()) == "R7"


# ── warnings (R8, R9) ───────────────────────────────────────────────────────
def test_R8_constant_output_warns():
    class Constant(nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 3) + 0 * x.flatten(1)[:, :3]
    r = check(Constant(), purpose="training")
    assert any(w.startswith("R8") for w in r.warnings)


@pytest.mark.parametrize("scale", [1e-4, 1e3])
def test_R9_response_scale_warns(scale):
    r = check(Toy(scale=scale))
    assert any(w.startswith("R9") for w in r.warnings)


# ── purpose: training needs fewer rules ─────────────────────────────────────
def test_training_does_not_need_gradients_or_pure_inputs():
    class NoGrad(Toy):
        def forward(self, x):
            with torch.no_grad():
                return super().forward(x)
    class InPlace(Toy):
        def forward(self, x):
            x.mul_(0.5)
            return super().forward(x)
    assert check(NoGrad(), purpose="training").n_neurons == 7             # training does not differentiate
    assert rule_of(NoGrad(), purpose="inference") == "R5"
    assert rule_of(InPlace(), purpose="inference") == "R7"
    assert rule_of(InPlace(), purpose="training") == "R4"                # not checked as R7, but the second call differs
    with pytest.raises(ValueError):
        check(Toy(), purpose="nonsense")


def test_ensure_valid_checks_once_per_encoder():
    calls = []

    class Counting(Toy):
        def forward(self, x):
            calls.append(1)
            return super().forward(x)
    enc = Counting()
    ensure_valid(enc, image_shape=SHAPE)
    n = len(calls)
    ensure_valid(enc, image_shape=SHAPE)
    assert n > 0 and len(calls) == n


# ── Fisher baseline through the public forward ──────────────────────────────
def test_fisher_of_a_user_encoder_matches_the_analytic_jacobian():
    sigma = 0.22
    enc = Toy()                                              # no private attributes: generic path
    x = torch.rand(1, *SHAPE) * 2 - 1
    out = fisher_eigendecomposition(x, enc, sigma=sigma, top_k=7)
    A = enc.lin.weight.detach().double().numpy()
    ref_vals = np.sort(np.linalg.eigvalsh(A.T @ A / sigma ** 2))[::-1][:7]
    np.testing.assert_allclose(out.eigenvalues.double().numpy(), ref_vals, rtol=1e-4)
    vecs = out.eigenvectors.double().numpy()                 # rows span the row space of A
    np.testing.assert_allclose(vecs @ vecs.T, np.eye(7), atol=1e-5)
    assert np.allclose(vecs @ (np.eye(P) - np.linalg.pinv(A) @ A) @ vecs.T, 0, atol=1e-5)


def test_fisher_rejects_a_bad_encoder():
    class Cut(Toy):
        def forward(self, x):
            with torch.no_grad():
                return super().forward(x)
    x = torch.rand(1, *SHAPE) * 2 - 1
    with pytest.raises(EncoderContractError) as e:
        fisher_eigendecomposition(x, Cut(), sigma=0.22)
    assert e.value.rule == "R5"


def test_estimator_validate_flag_rejects_a_bad_encoder():
    pytest.importorskip("diffusers")
    from conftest import estimator_cfg
    from src.estimator import conditional_pair_estimate

    class Cut(Toy):
        def forward(self, x):
            with torch.no_grad():
                return super().forward(x)
    cfg = estimator_cfg(n_samples=4, batch_size=2, start=40, stop=80, step=40, sigma=0.22, top_k=2)
    with pytest.raises(EncoderContractError):
        conditional_pair_estimate(torch.zeros(1, *SHAPE), Cut(), None, torch.ones(1000), cfg,
                                  torch.device("cpu"), validate=True)
