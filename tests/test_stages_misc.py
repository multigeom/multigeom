"""Small helpers of stages 01 and 03, checked against independent references."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from conftest import load_stage


@pytest.fixture(scope="module")
def stage01():
    return load_stage("01_train_encoder.py")


def test_correlation_per_neuron_matches_numpy(stage01):
    g = torch.Generator().manual_seed(0)
    target = torch.randn(200, 4, generator=g)
    pred = 0.7 * target + 0.5 * torch.randn(200, 4, generator=g)
    got = stage01.correlation_per_neuron(pred, target)
    expected = [np.corrcoef(pred[:, n].numpy(), target[:, n].numpy())[0, 1] for n in range(4)]
    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_correlation_per_neuron_constant_prediction_is_zero(stage01):
    target = torch.randn(50, 2)
    pred = torch.stack([torch.ones(50), torch.randn(50)], dim=1)
    assert stage01.correlation_per_neuron(pred, target)[0] == 0.0


def test_cosine_with_warmup_schedule():
    pytest.importorskip("diffusers")
    stage03 = load_stage("03_train_diffusion.py")
    f = stage03.cosine_with_warmup
    assert f(0, 500, 10_500) == 0.0
    assert f(250, 500, 10_500) == pytest.approx(0.5)
    assert f(500, 500, 10_500) == pytest.approx(1.0)
    assert f(5_500, 500, 10_500) == pytest.approx(0.5)            # half-way through decay
    assert f(10_500, 500, 10_500) == pytest.approx(0.0, abs=1e-12)
    s = 2_000
    assert f(s, 500, 10_500) == pytest.approx(0.5 * (1 + math.cos(math.pi * 1_500 / 10_000)))
