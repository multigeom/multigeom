"""
multigeom.lowdim for one neuron and a one-dimensional stimulus: known answers, limits,
invariances, the identity I(R; X) = E_x[i(x)], input checks, dtype and device.

Known answers. A Gaussian density with standard deviation s has the entropy
(1/2) ln(2 pi e s^2). The linear code f(x) = a x with the prior N(0, s^2) has
I(R; X) = (1/2) ln(1 + s^2 a^2 / sigma^2) and the constant G(x) = ln(1 + s^2 a^2 /
sigma^2) (the one-dimensional case of the closed form of the linear Gaussian code).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from multigeom import lowdim

A, S, SIGMA = 1.0, 1.0, 0.5  # linear code: G = ln 5, I = ln 5 / 2


def linear_code(nx: int = 161):
    x = torch.linspace(-8 * S, 8 * S, nx, dtype=torch.float64)
    prior = torch.exp(-0.5 * (x / S) ** 2)
    return x, A * x, prior / (prior.sum() * (x[1] - x[0]))


def bell_code(nx: int = 121):
    x = torch.linspace(-3.0, 3.0, nx, dtype=torch.float64)
    prior = torch.full_like(x, 1.0 / ((x[1] - x[0]).item() * nx))
    return x, torch.exp(-(x**2) / 2), prior


def test_entropy_of_a_gaussian_density():
    x = torch.linspace(-15.0, 15.0, 3001, dtype=torch.float64)
    p = torch.exp(-0.5 * (x / 1.5) ** 2) / (math.sqrt(2 * math.pi) * 1.5)
    expected = 0.5 * math.log(2 * math.pi * math.e * 1.5**2)
    # Riemann sum of a smooth density, cut 10 standard deviations out: below 1e-12.
    assert lowdim.entropy(p, (x[1] - x[0]).item()).item() == pytest.approx(
        expected, abs=1e-10
    )


def test_bits_are_nats_over_ln_2():
    assert lowdim.to_bits(math.log(2.0)) == pytest.approx(1.0, abs=1e-15)


def test_posterior_is_a_density_over_the_stimulus_grid():
    x, f, prior = bell_code()
    post = lowdim.posterior(x, f, prior, 0.1, torch.tensor([0.2, 0.7, 1.0]))
    assert post.shape == (121, 3)
    sums = post.sum(0) * (x[1] - x[0])
    torch.testing.assert_close(sums, torch.ones(3, dtype=torch.float64))


def test_mutual_information_of_the_linear_code():
    x, f, prior = linear_code()
    expected = 0.5 * math.log(1 + S**2 * A**2 / SIGMA**2)
    # Riemann sums of Gaussians on grids of 161 and 361 points: 1e-14 in practice.
    mi = lowdim.mutual_information(x, f, prior, SIGMA)
    assert mi.item() == pytest.approx(expected, abs=1e-9)


def test_stimulus_specific_measures_average_to_the_mutual_information():
    x, f, prior = bell_code()
    dx = (x[1] - x[0]).item()
    mi = lowdim.mutual_information(x, f, prior, 0.1)
    for measure in (lowdim.specific_surprise, lowdim.stimulus_specific_information):
        # Rearrangements of the same double sum, exact up to the mass of p(r | x)
        # outside the response grid (beyond 6 sigma, 2e-9): 2e-10 here.
        average = (prior * measure(x, f, prior, 0.1)).sum() * dx
        assert average.item() == pytest.approx(mi.item(), rel=1e-8)


@pytest.mark.parametrize("per_decade, atol", [(20, 6e-3), (40, 1.5e-3)])
def test_g_of_the_linear_code_is_constant(per_decade, atol):
    # The error is the trapezoid rule over the blur levels: 3.5e-3 with 20 levels
    # per decade and 8e-4 with 40 (about 1 / per_decade^2). Tolerances: about twice
    # that. Points within 2 prior standard deviations, away from the grid ends.
    x, f, prior = linear_code()
    out = lowdim.compute_g(
        x, f, prior, SIGMA, lowdim.blur_levels(per_decade=per_decade)
    )
    inside = x.abs() <= 2 * S
    error = (out.g[inside] - math.log(1 + S**2 * A**2 / SIGMA**2)).abs().max()
    assert error.item() < atol


def test_prior_average_of_the_local_information_is_the_mutual_information():
    x = torch.linspace(-3.0, 3.0, 241, dtype=torch.float64)
    f = torch.exp(-(x**2) / 2)
    prior = torch.full_like(x, 1.0 / 6.0)
    prior = prior / (prior.sum() * (x[1] - x[0]))
    out = lowdim.compute_g(x, f, prior, 0.1, lowdim.blur_levels(per_decade=40))
    average = (prior * out.local_information).sum() * (x[1] - x[0])
    mi = lowdim.mutual_information(x, f, prior, 0.1)
    # 1e-4 bits apart on this grid. Tolerance 2e-3 bits: on the converged grids the
    # two still differ by 8e-4 bits (blur levels truncated at 1e4, grid of x_t).
    assert abs(lowdim.to_bits(average - mi).item()) < 2e-3


def test_per_scale_term_at_t_0_is_the_fisher_information():
    x, f, prior = bell_code()
    out = lowdim.compute_g(
        x, f, prior, 0.1, lowdim.blur_levels(1e-3, 1e1, 2), keep_terms=True
    )
    assert out.terms.shape == (len(out.t), len(x))
    torch.testing.assert_close(out.terms[0], out.fisher, rtol=0, atol=0)
    torch.testing.assert_close(
        out.fisher, (torch.gradient(f, spacing=(x[1] - x[0]).item())[0] / 0.1) ** 2
    )
    torch.testing.assert_close(out.local_information, out.g / 2)


def test_terms_are_kept_only_when_asked():
    x, f, prior = bell_code()
    assert (
        lowdim.compute_g(x, f, prior, 0.1, lowdim.blur_levels(1e-2, 1e1, 2)).terms
        is None
    )


def test_doubling_sigma_divides_the_fisher_information_by_4():
    x, f, _ = bell_code()
    torch.testing.assert_close(
        lowdim.compute_fisher(x, f, 0.2), lowdim.compute_fisher(x, f, 0.1) / 4
    )


def test_change_of_stimulus_unit():
    # y = c x: the prior density becomes p / c and the blur levels c^2 t. G has no
    # unit, J is in 1 / unit^2, so G is unchanged, G distances scale by c and Fisher
    # distances are unchanged. Same operations up to rounding: tolerance 1e-10.
    c = 2.5
    x, f, prior = bell_code()
    t = lowdim.blur_levels(1e-4, 1e3, 5)
    out = lowdim.compute_g(x, f, prior, 0.1, t)
    scaled = lowdim.compute_g(c * x, f, prior / c, 0.1, c**2 * t)
    torch.testing.assert_close(scaled.g, out.g, rtol=1e-10, atol=0)
    d_g = lowdim.distance(x, out.g, -3.0, 3.0)
    d_j = lowdim.distance(x, out.fisher, -3.0, 3.0)
    scaled_d_g = lowdim.distance(c * x, scaled.g, -3.0 * c, 3.0 * c)
    scaled_d_j = lowdim.distance(c * x, scaled.fisher, -3.0 * c, 3.0 * c)
    assert scaled_d_g.item() == pytest.approx(c * d_g.item(), rel=1e-10)
    assert scaled_d_j.item() == pytest.approx(d_j.item(), rel=1e-10)


def test_distance_of_a_constant_metric():
    x = torch.linspace(0.0, 2.0, 11, dtype=torch.float64)
    metric = torch.full_like(x, 9.0)
    torch.testing.assert_close(lowdim.cumulative_distance(x, metric), 3 * x)
    d = lowdim.distance(x, metric, torch.tensor([0.0, 0.5]), torch.tensor([2.0, 0.1]))
    torch.testing.assert_close(d, torch.tensor([6.0, 1.2], dtype=torch.float64))


def test_blurred_code_at_t_0_is_the_code():
    x, f, prior = bell_code()
    out = lowdim.blurred_code(x, f, prior, 0.1, 0.0)
    torch.testing.assert_close(out.mean, f)
    torch.testing.assert_close(out.std, torch.full_like(x, 0.1))
    torch.testing.assert_close(out.fisher, lowdim.compute_fisher(x, f, 0.1))


def test_blurred_code_of_the_linear_code():
    # x | x_t is Gaussian with mean m x_t and variance m t, m = s^2 / (s^2 + t), so
    # r | x_t has mean a m x_t, variance sigma^2 + a^2 m t, and J_t = (a m)^2 / var.
    x, f, prior = linear_code(321)
    t = 0.5
    query = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64)
    out = lowdim.blurred_code(x, f, prior, SIGMA, t, query=query)
    m = S**2 / (S**2 + t)
    var = SIGMA**2 + A**2 * m * t
    torch.testing.assert_close(out.mean, A * m * query, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out.std, torch.full_like(query, math.sqrt(var)))
    # J_t from central differences of ln p_t on a grid of step sqrt(t) / 12: 1e-3.
    expected = torch.full_like(query, (A * m) ** 2 / var)
    torch.testing.assert_close(out.fisher, expected, rtol=1e-3, atol=0)


def test_numpy_inputs_give_float64_and_float32_stays_float32():
    x, f, prior = bell_code(61)
    t = lowdim.blur_levels(1e-2, 1e1, 2)
    out = lowdim.compute_g(x.numpy(), f.numpy(), prior.numpy(), 0.1, t.numpy())
    assert out.g.dtype == torch.float64 and out.g.device.type == "cpu"
    single = lowdim.compute_g(x.float(), f.float(), prior.float(), 0.1, t.float())
    assert single.g.dtype == torch.float32
    torch.testing.assert_close(single.g.double(), out.g, rtol=1e-3, atol=1e-4)
    assert isinstance(lowdim.mutual_information(x, f, prior, 0.1), torch.Tensor)


def test_inputs_are_not_modified():
    x, f, prior = bell_code(61)
    copies = [a.clone() for a in (x, f, prior)]
    lowdim.compute_g(x, f, prior, 0.1, lowdim.blur_levels(1e-2, 1e1, 2))
    for a, b in zip((x, f, prior), copies):
        assert torch.equal(a, b)


def test_a_non_uniform_grid_is_refused():
    x = torch.tensor(np.geomspace(1.0, 10.0, 50))
    with pytest.raises(ValueError, match="uniform"):
        lowdim.compute_fisher(x, x, 0.1)


def test_a_tuning_curve_off_the_grid_is_refused():
    x, f, prior = bell_code()
    with pytest.raises(ValueError, match="tuning"):
        lowdim.mutual_information(x, f[:-1], prior, 0.1)


def test_blur_levels_must_start_at_0():
    x, f, prior = bell_code()
    with pytest.raises(ValueError, match="t\\[0\\] = 0"):
        lowdim.compute_g(x, f, prior, 0.1, torch.logspace(-2, 1, 5))


def test_a_negative_metric_is_refused():
    x = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="non-negative"):
        lowdim.cumulative_distance(x, x - 0.5)


def test_blur_levels_are_zero_then_log_spaced():
    t = lowdim.blur_levels(1e-6, 1e4, 100)
    assert t.dtype == torch.float64 and t[0] == 0 and len(t) == 1001
    np.testing.assert_allclose(t[1:].numpy(), np.logspace(-6, 4, 1000), rtol=1e-13)


@pytest.mark.gpu
def test_compute_g_on_cuda_matches_the_cpu():
    x, f, prior = bell_code()
    t = lowdim.blur_levels(1e-4, 1e3, 10)
    cpu = lowdim.compute_g(x, f, prior, 0.1, t, keep_terms=True)
    gpu = lowdim.compute_g(x.cuda(), f.cuda(), prior.cuda(), 0.1, t.cuda())
    assert gpu.g.device.type == "cuda" and gpu.g.dtype == torch.float64
    # Same float64 operations, reduced in a different order on the GPU.
    torch.testing.assert_close(gpu.g.cpu(), cpu.g, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(gpu.fisher.cpu(), cpu.fisher, rtol=1e-12, atol=0)
