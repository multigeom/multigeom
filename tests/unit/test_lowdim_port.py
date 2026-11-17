"""
multigeom.lowdim reproduces, on the same grids, the values of an independent NumPy
implementation of the same quadrature (the reference the module was ported from).

The values in data/lowdim_reference.json were computed once with that implementation:
the bell-shaped neuron exp(-x^2 / 2) with a uniform prior on [-3, 3] (61 stimuli, 26
blur levels from 1e-3 to 1e2), and a neuron with two bumps exp(-(x + 3)^2 / 2) +
exp(-(x - 3)^2 / 2) with the prior N(6, 8^2) on [-24, 36] (121 stimuli, 21 blur levels
from 1e-2 to 1e3), sigma = 0.1. The grids, the prior and the tuning curve are stored
with the values, so both implementations see the same inputs. The per-scale terms are
stored at every fifth blur level.

Tolerance: 1e-12 relative to the largest value. Both implementations do the same
float64 operations in a different order, which leaves differences of a few 1e-15.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from multigeom import lowdim

REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "lowdim_reference.json").read_text()
)
SIGMA = REFERENCE["sigma"]
RTOL = 1e-12


def assert_same(actual, expected):
    expected = torch.as_tensor(expected, dtype=torch.float64)
    assert actual.shape == expected.shape
    scale = expected.abs().max()
    assert ((actual - expected).abs().max() / scale).item() < RTOL


@pytest.fixture(params=["bell", "two_bumps"])
def case(request):
    ref = REFERENCE[request.param]
    inputs = {
        k: torch.tensor(ref[k], dtype=torch.float64) for k in ("x", "tuning", "prior")
    }
    return ref, inputs


def test_multiscale_information_matches_the_reference(case):
    ref, code = case
    t = torch.tensor(ref["t"], dtype=torch.float64)
    out = lowdim.compute_g(**code, sigma=SIGMA, t=t, keep_terms=True)
    assert_same(out.fisher, ref["fisher"])
    assert_same(out.local_information, ref["local_information"])
    assert_same(out.terms[:: ref["terms_every"]], ref["terms"])


def test_information_measures_match_the_reference(case):
    ref, code = case
    mi = lowdim.mutual_information(**code, sigma=SIGMA)
    assert abs(mi.item() - ref["mutual_information"]) < RTOL * ref["mutual_information"]
    assert_same(lowdim.specific_surprise(**code, sigma=SIGMA), ref["specific_surprise"])
    assert_same(
        lowdim.stimulus_specific_information(**code, sigma=SIGMA),
        ref["stimulus_specific_information"],
    )


def test_blurred_code_matches_the_reference(case):
    ref, code = case
    out = lowdim.blurred_code(
        **code,
        sigma=SIGMA,
        t=ref["blurred_t"],
        query=torch.tensor(ref["blurred_query"], dtype=torch.float64),
    )
    assert_same(out.mean, ref["blurred_mean"])
    assert_same(out.std, ref["blurred_std"])
    assert_same(out.fisher, ref["blurred_fisher"])
