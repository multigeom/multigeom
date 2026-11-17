"""
The linear Gaussian code in any dimension, in closed form (Azeglio, Laquitaine,
Ferrari, Chalk, NeurIPS 2026, arXiv 2605.06304).

The stimulus has the prior N(mu, Sigma) in d dimensions, and the code r = A x + noise
with noise covariance sigma^2 I has the Fisher matrix J = A^T A / sigma^2, the same at
every stimulus. At the blur level t, with M_t = Sigma (Sigma + t I)^-1 and
C_t = Sigma - M_t Sigma the posterior covariance of x given x_t,

    J_t = M_t^T J (I + C_t J)^-1 M_t,

and the multi-scale Fisher information G = int_0^inf J_t dt is the same at every
stimulus:

    G = logm(Sigma^-1 + J) - logm(Sigma^-1).

Derivation. Write P = Sigma^-1 and s = 1 / t. Then C_t = (P + s I)^-1 and
M_t = s C_t. The identity C - (C^-1 + J)^-1 = C J (I + C J)^-1 C turns the formula
for J_t into

    J_t = s^2 [(P + s I)^-1 - (P + J + s I)^-1].

With dt = -ds / s^2,

    G = int_0^inf [(P + s I)^-1 - (P + J + s I)^-1] ds.

For a symmetric positive definite matrix A,

    logm(A) = int_0^inf [(1 + s)^-1 I - (A + s I)^-1] ds,

which is the scalar integral ln a = int_0^inf [1 / (1 + s) - 1 / (a + s)] ds applied
to each eigenvalue of A. Subtracting this representation for A = P from the one for
A = P + J gives the closed form. No step needs Sigma and J to commute. For integral
representations of the matrix logarithm see N. J. Higham, Functions of Matrices:
Theory and Computation, SIAM (2008), chapter 11.

The paper states the commuting case: when Sigma and J commute,
G = sum_i ln(1 + lambda_i j_i) e_i e_i^T with lambda_i and j_i their eigenvalues along
the shared eigenvectors e_i. The trace of G is ln det(I + Sigma J), twice the mutual
information I(R; X) in nats; I(R; X) = (1/2) ln det(I + Sigma J) is the mutual
information of the Gaussian channel (T. M. Cover and J. A. Thomas, Elements of
Information Theory, 2nd edition, Wiley, 2006, chapter 9). The closed form of G agrees
with the trapezoid rule for the t integral at the precision of the rule, also when
Sigma and J do not commute (see the unit tests).
"""

from __future__ import annotations

import torch

from ._grid import as_tensors


def _logm_spd(m: torch.Tensor) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(m)
    return (vectors * values.log().unsqueeze(-2)) @ vectors.mT


def _symmetric(m: torch.Tensor) -> torch.Tensor:
    return 0.5 * (m + m.mT)


def gaussian_blurred_fisher(prior_cov, fisher, t) -> torch.Tensor:
    """
    The Fisher information J_t of the blurred linear Gaussian code at blur levels t.

    Args:
        prior_cov: the prior covariance Sigma [d, d], positive definite.
        fisher: the Fisher matrix J [d, d], positive semi-definite.
        t: the blur levels [n_t] (variances).

    Returns:
        J_t [n_t, d, d].
    """
    prior_cov, fisher, t = as_tensors(prior_cov, fisher, t)
    eye = torch.eye(prior_cov.shape[-1], dtype=t.dtype, device=t.device)
    blurred = prior_cov + t.reshape(-1, 1, 1) * eye
    gain = torch.linalg.solve(blurred, prior_cov).mT  # M_t = Sigma (Sigma + t I)^-1
    cov_t = prior_cov - gain @ prior_cov
    return _symmetric(gain.mT @ fisher @ torch.linalg.solve(eye + cov_t @ fisher, gain))


def gaussian_g(prior_cov, fisher) -> torch.Tensor:
    """
    The multi-scale Fisher information G = logm(Sigma^-1 + J) - logm(Sigma^-1) of the
    linear Gaussian code.

    Args:
        prior_cov: the prior covariance Sigma [d, d], positive definite.
        fisher: the Fisher matrix J [d, d], positive semi-definite.

    Returns:
        G [d, d], in nats (an isotropic rescaling of the stimulus leaves it unchanged).
    """
    prior_cov, fisher = as_tensors(prior_cov, fisher)
    precision = _symmetric(torch.linalg.inv(prior_cov))
    return _symmetric(_logm_spd(precision + fisher) - _logm_spd(precision))


def gaussian_information(prior_cov, fisher) -> torch.Tensor:
    """
    The mutual information I(R; X) = (1/2) ln det(I + Sigma J) of the linear Gaussian
    code, in nats.

    Args:
        prior_cov: the prior covariance Sigma [d, d], positive definite.
        fisher: the Fisher matrix J [d, d], positive semi-definite.

    Returns:
        I(R; X), a scalar tensor.
    """
    prior_cov, fisher = as_tensors(prior_cov, fisher)
    eye = torch.eye(prior_cov.shape[-1], dtype=prior_cov.dtype, device=prior_cov.device)
    return 0.5 * torch.linalg.slogdet(eye + prior_cov @ fisher).logabsdet
