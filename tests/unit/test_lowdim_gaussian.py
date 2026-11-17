"""
The linear Gaussian code in closed form, and the nearest-neighbour estimate of the
mutual information.

The closed form G = logm(Sigma^-1 + J) - logm(Sigma^-1) is checked against the
quadrature G = int_0^inf J_t dt of `gaussian_blurred_fisher`: the trapezoid rule in
ln t on [1e-8, 1e8] plus 1e-8 J for [0, 1e-8]. The error of that rule is set by the
cut at 1e-8 and falls a hundredfold when the range widens to [1e-10, 1e10] (a check
made once while writing the module), so the closed form is tested at two rule sizes.
"""

from __future__ import annotations

import math

import pytest
import torch

from multigeom import lowdim


def quadrature_g(prior_cov, fisher, log10_range, n):
    t = torch.logspace(-log10_range, log10_range, n, dtype=torch.float64)
    fisher_t = lowdim.gaussian_blurred_fisher(prior_cov, fisher, t)
    integral = torch.trapezoid(fisher_t * t[:, None, None], t.log(), dim=0)
    return integral + t[0] * torch.as_tensor(fisher, dtype=torch.float64)


def random_case(d, generator):
    b = torch.randn(d, d, generator=generator, dtype=torch.float64)
    prior_cov = b @ b.T + 0.05 * torch.eye(d, dtype=torch.float64)
    b = torch.randn(d, d, generator=generator, dtype=torch.float64)
    return prior_cov, 3 * b @ b.T


@pytest.mark.parametrize("d", [2, 3, 4])
def test_closed_form_agrees_with_quadrature_without_commuting(d):
    generator = torch.Generator().manual_seed(d)
    for _ in range(3):
        prior_cov, fisher = random_case(d, generator)
        assert (prior_cov @ fisher - fisher @ prior_cov).abs().max() > 0.1
        g = lowdim.gaussian_g(prior_cov, fisher)
        coarse = (quadrature_g(prior_cov, fisher, 8, 4001) - g).abs().max().item()
        fine = (quadrature_g(prior_cov, fisher, 10, 16001) - g).abs().max().item()
        # 2e-7 or less with the first rule, about 100 times less with the second.
        # A wrong closed form would leave the same error with both rules.
        assert coarse < 1e-6
        assert fine < coarse / 30


def test_commuting_case_is_the_sum_of_one_dimensional_terms():
    s = j = torch.tensor([4.0, 0.25], dtype=torch.float64)
    g = lowdim.gaussian_g(torch.diag(s), torch.diag(j))
    torch.testing.assert_close(g, torch.diag(torch.log1p(s * j)))


def test_information_is_half_the_trace_of_g():
    generator = torch.Generator().manual_seed(7)
    prior_cov, fisher = random_case(3, generator)
    half_trace = 0.5 * torch.trace(lowdim.gaussian_g(prior_cov, fisher))
    torch.testing.assert_close(
        lowdim.gaussian_information(prior_cov, fisher), half_trace
    )


def test_information_of_the_example_rotated_by_90_degrees():
    # Prior variances (4, 0.25), Fisher eigenvalues (0.25, 4): det(I + Sigma J) = 4.
    info = lowdim.gaussian_information([[4.0, 0.0], [0.0, 0.25]], [[0.25, 0], [0, 4]])
    assert info.item() == pytest.approx(math.log(2.0), abs=1e-14)


def test_fisher_from_a_jacobian():
    a = torch.tensor([[1.0, 2.0], [0.0, 3.0], [1.0, -1.0]], dtype=torch.float64)
    expected = torch.tensor([[2.0, 1.0], [1.0, 14.0]], dtype=torch.float64) / 0.25
    torch.testing.assert_close(lowdim.fisher_from_jacobian(a, 0.5), expected)
    batch = lowdim.fisher_from_jacobian(torch.stack([a, 2 * a]), 0.5)
    torch.testing.assert_close(batch, torch.stack([expected, 4 * expected]))


def test_nearest_neighbour_estimate_of_the_linear_gaussian_code():
    # Prior N(0, diag(4, 0.25)) and J = diag(4, 0.25): G = diag(ln 17, ln 1.0625)
    # and I = 1.447 nats. With 4000 samples the estimate has a standard deviation of
    # about 0.016 nats and a bias below 0.01 nats (measured over many repetitions
    # with an independent implementation). Tolerance 0.07 nats: four standard
    # deviations plus the bias, for this fixed generator.
    sd = torch.tensor([2.0, 0.5], dtype=torch.float64)
    prior_cov, fisher = torch.diag(sd**2), torch.diag(sd**2)
    generator = torch.Generator().manual_seed(0)
    samples = torch.randn(4000, 2, generator=generator, dtype=torch.float64) * sd
    g = lowdim.gaussian_g(prior_cov, fisher)
    estimate = lowdim.nearest_neighbour_information(samples, g, chunk_size=1000)
    exact = lowdim.gaussian_information(prior_cov, fisher)
    assert estimate.item() == pytest.approx(exact.item(), abs=0.07)
    per_sample = lowdim.nearest_neighbour_information(samples, g.expand(4000, 2, 2))
    torch.testing.assert_close(per_sample, estimate)


def test_nearest_neighbour_estimate_with_g_the_identity_is_d_over_2():
    generator = torch.Generator().manual_seed(1)
    samples = torch.randn(300, 3, generator=generator, dtype=torch.float64)
    estimate = lowdim.nearest_neighbour_information(samples, torch.eye(3))
    assert estimate.item() == pytest.approx(1.5, abs=1e-12)


def test_nearest_neighbour_directions_are_unit_vectors_to_the_neighbour():
    samples = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0], [5.0, 7.0]])
    u = lowdim.nearest_neighbour_directions(samples, chunk_size=3)
    expected = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    torch.testing.assert_close(u, expected)
