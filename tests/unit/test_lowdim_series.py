"""
Three neurons with sigma = 0.1 and a uniform prior on [-3, 3]: the bell-shaped
f(x) = exp(-x^2 / 2), its monotonic twin m(x) = f(x) for x < 0 and 2 - f(x) for
x >= 0, and the low-gain neuron m(x) / 2.

The reference values were computed with an independent NumPy implementation of the
same quadrature on converged grids (1201 stimuli, blur levels 0 and 100 per decade
from 1e-6 to 1e4). Doubling both changes the distances by less than 2e-3 and the
mutual information by less than 2e-4 bits. The Fisher distances also have a closed
form: the total variation of f over [-3, 3] over sigma, 2 (1 - exp(-4.5)) / 0.1 for
the bell-shaped neuron and its twin.

The fast tests use a coarse grid (241 stimuli, 20 blur levels per decade) and the
tolerances given with each assertion. The slow tests use the converged grids and
reproduce the reference values to near machine precision.
"""

from __future__ import annotations

import math

import pytest
import torch

from multigeom import lowdim

SIGMA = 0.1
D_J_BELL = 2 * (1 - math.exp(-4.5)) / SIGMA

# Converged-grid values: mutual information in bits, Fisher and G distances from -3
# to 3, G at -1 and at 0 (nats), and the prior average of the local information (bits).
CONVERGED = {
    "bell": dict(
        mi=1.6028540303522467,
        d_j=19.777695070016843,
        d_g=8.77812361827345,
        g_m1=3.285511894345366,
        g_0=1.4475723940028873,
        avg_local=1.6020649490297463,
    ),
    "mono": dict(
        mi=2.4192181122515897,
        d_j=19.777820069235585,
        d_g=10.833865392513925,
        g_m1=4.588595404650895,
        g_0=3.4823464683018264,
        avg_local=2.4191162801963775,
    ),
    "low": dict(
        mi=1.6615842578148128,
        d_j=9.888910034617792,
        d_g=9.001969829016822,
        g_m1=3.0493282192682174,
        g_0=2.5643966702710337,
        avg_local=1.6614216402213848,
    ),
}


def three_neurons(nx):
    x = torch.linspace(-3.0, 3.0, nx, dtype=torch.float64)
    prior = torch.ones_like(x)
    prior = prior / (prior.sum() * (x[1] - x[0]))
    bell = torch.exp(-(x**2) / 2)
    mono = torch.where(x < 0, bell, 2 - bell)
    return x, prior, {"bell": bell, "mono": mono, "low": 0.5 * mono}


def series(nx, per_decade):
    x, prior, neurons = three_neurons(nx)
    t = lowdim.blur_levels(1e-6, 1e4, per_decade)
    values = {}
    for name, f in neurons.items():
        out = lowdim.compute_g(x, f, prior, SIGMA, t)
        k_m1, k_0 = (int((x - x0).abs().argmin()) for x0 in (-1.0, 0.0))
        values[name] = dict(
            mi=lowdim.to_bits(lowdim.mutual_information(x, f, prior, SIGMA)).item(),
            d_j=lowdim.distance(x, out.fisher, -3.0, 3.0).item(),
            d_g=lowdim.distance(x, out.g, -3.0, 3.0).item(),
            g_m1=out.g[k_m1].item(),
            g_0=out.g[k_0].item(),
            avg_local=lowdim.to_bits(
                (prior * out.local_information).sum() * (x[1] - x[0])
            ).item(),
        )
    return values


@pytest.fixture(scope="module")
def coarse():
    return series(nx=241, per_decade=20)


@pytest.mark.parametrize("name", ["bell", "mono", "low"])
def test_mutual_information_on_a_coarse_grid(coarse, name):
    # 1.1e-3 bits or less from the converged value on 241 stimuli.
    assert coarse[name]["mi"] == pytest.approx(CONVERGED[name]["mi"], abs=3e-3)


@pytest.mark.parametrize(
    "name, exact", [("bell", D_J_BELL), ("mono", D_J_BELL), ("low", D_J_BELL / 2)]
)
def test_fisher_distance_matches_the_closed_form(coarse, name, exact):
    # Central differences and the trapezoid rule on 241 stimuli: 3e-3 or less.
    assert coarse[name]["d_j"] == pytest.approx(exact, abs=1e-2)


@pytest.mark.parametrize("name", ["bell", "mono", "low"])
def test_g_distance_on_a_coarse_grid(coarse, name):
    # 0.018 to 0.023 above the converged values with 20 blur levels per decade (the
    # trapezoid rule over t), 0.010 to 0.014 with 40.
    assert coarse[name]["d_g"] == pytest.approx(CONVERGED[name]["d_g"], abs=0.05)


def test_g_distances_order_the_three_neurons(coarse):
    # The bell-shaped neuron and its twin have the same Fisher distance, but the
    # twin separates more stimuli in G, and the low-gain twin lies in between.
    assert coarse["bell"]["d_g"] < coarse["low"]["d_g"] < coarse["mono"]["d_g"]


@pytest.mark.slow
def test_converged_grids_reproduce_the_reference_values():
    values = series(nx=1201, per_decade=100)
    for name, expected in CONVERGED.items():
        for key, value in expected.items():
            # Same grids and same operations as the reference: 1e-14 in the
            # comparison of test_lowdim_port.py, so 1e-9 leaves a wide margin.
            assert values[name][key] == pytest.approx(value, rel=1e-9), (name, key)
