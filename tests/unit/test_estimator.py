"""
Tests for multigeom.estimator (Algorithm 1, conditional-pair estimator of G(x)).

Analytic reference. Take a mock score network whose conditional branch is
affine in the response, ε(x_t, t, r) = A r + h(x_t) (A ∈ ℝ^{P×n}), and whose
null-token branch is arbitrary. Algorithm 1 appends the rows
s_j (ε(r_1) − ε(r_2)) = s_j σ A (ξ_1 − ξ_2), s_j² = w_j / (2 N_MC), so

    E[DᵀD] = Σ_j N_MC · (w_j / (2 N_MC)) · σ² A E[(ξ_1−ξ_2)(ξ_1−ξ_2)ᵀ] Aᵀ
           = σ² (Σ_j w_j) A Aᵀ,

whatever h, the null branch and the encoder are. With A = U diag(s) Rᵀ
(U orthonormal columns) the eigenvalues are σ² W s_i², W = Σ_j w_j =
log τ_J − log τ_1, with eigenvectors the columns of U, and every further
eigenvalue is exactly zero. The Monte-Carlo relative error of the eigenvalues
is about sqrt(2/(J·N_MC)) ≈ 2.4% for the sizes used here; the tolerance is 10%.

The null branch is built so that Tweedie's formula returns a known image c:
ε_null = (x_t − √ᾱ_t c) / √(1−ᾱ_t) ⇒ x̂ = c, which the mock encoder records
(after the clip to [−1, 1]).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("diffusers")

from multigeom.geometry.estimator import (  # noqa: E402
    _build_log_tau_grid,
    _trapezoid_weights,
    conditional_pair_estimate,
    estimate_g,
    estimate_g_per_scale,
)
from conftest import estimator_cfg, linear_beta_alphas_cumprod  # noqa: E402

H = W = 8
P = H * W
N_NEURONS = 3
SIGMA = 0.3


class AffineUNet(nn.Module):
    """
    ε(x_t,t,r) = A r + h(x_t) for r ≠ 0; Tweedie-consistent null branch returning x̂ =
    c.
    """

    def __init__(self, A: torch.Tensor, c: torch.Tensor, alphas_cumprod: torch.Tensor):
        super().__init__()
        self.register_buffer("A", A.float())  # [P, n]
        self.register_buffer("c", c.float())  # [1, 1, H, W]
        self.register_buffer("ab", alphas_cumprod.float())
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x_t, t, r):
        ab = self.ab[t].view(-1, 1, 1, 1)
        eps_null = (x_t - ab.sqrt() * self.c) / (1 - ab).sqrt()
        eps_cond = (r @ self.A.T).view_as(x_t) + 5.0 * torch.sin(3.0 * x_t)
        is_null = (r == 0).all(dim=1).view(-1, 1, 1, 1)
        return torch.where(is_null, eps_null, eps_cond)


class RecordingEncoder(nn.Module):
    """Returns a response that depends on the input; records every input it sees."""

    def __init__(self, n_neurons: int = N_NEURONS):
        super().__init__()
        self.n_neurons = n_neurons
        self.inputs: list[torch.Tensor] = []
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        m = x.flatten(1).mean(dim=1, keepdim=True)
        return 2.0 + m * torch.arange(1, self.n_neurons + 1, dtype=x.dtype)


def _affine_problem(seed: int = 0, singular_values=(3.0, 2.0, 1.0)):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((P, N_NEURONS)))
    R, _ = np.linalg.qr(rng.standard_normal((N_NEURONS, N_NEURONS)))
    s = np.array(singular_values)
    A = torch.from_numpy(U @ np.diag(s) @ R.T)
    return A, torch.from_numpy(U), s


def _alphas32():
    return linear_beta_alphas_cumprod().float()


# ── grid and weights ───────────────────────────────────────────────────────
def test_log_tau_grid_matches_linear_beta_closed_form():
    from diffusers import DDPMScheduler

    sched = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=0.02
    )
    grid = list(range(40, 961, 40))
    got = _build_log_tau_grid(grid, sched.alphas_cumprod).double()
    ab = linear_beta_alphas_cumprod()[grid]
    expected = torch.log((1 - ab) / ab)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
    assert torch.all(got[1:] > got[:-1])  # τ increases with t


def test_trapezoid_weights_half_weights_and_sum():
    lt = torch.tensor([-3.0, -1.0, 0.5, 2.5, 3.0], dtype=torch.float64)
    w = _trapezoid_weights(lt)
    assert w[0].item() == pytest.approx(0.5 * (lt[1] - lt[0]).item())
    assert w[-1].item() == pytest.approx(0.5 * (lt[-1] - lt[-2]).item())
    assert w[2].item() == pytest.approx(0.5 * (lt[3] - lt[1]).item())
    assert w.sum().item() == pytest.approx((lt[-1] - lt[0]).item())


def test_trapezoid_weights_reproduce_numpy_trapezoid_rule():
    lt = torch.sort(torch.rand(11, dtype=torch.float64) * 10 - 5).values
    f = torch.cos(lt) + lt**2
    w = _trapezoid_weights(lt)
    trap = getattr(np, "trapezoid", None) or np.trapz
    assert float((w * f).sum()) == pytest.approx(
        float(trap(f.numpy(), lt.numpy())), rel=1e-12
    )


# ── Algorithm 1 against the analytic expectation ───────────────────────────
@pytest.fixture(scope="module")
def affine_estimate():
    A, U, s = _affine_problem(seed=0)
    rng = np.random.default_rng(1)
    c = torch.from_numpy(rng.uniform(-1.5, 1.5, (1, 1, H, W))).float()
    ab = _alphas32()
    unet, enc = AffineUNet(A, c, ab), RecordingEncoder()
    cfg = estimator_cfg(
        n_samples=400,
        batch_size=100,
        start=100,
        stop=900,
        step=100,
        sigma=SIGMA,
        top_k=5,
    )
    image = torch.from_numpy(rng.uniform(-1, 1, (1, 1, H, W))).float()
    out = conditional_pair_estimate(
        image,
        enc,
        unet,
        ab,
        cfg,
        torch.device("cpu"),
        generator=torch.Generator().manual_seed(123),
    )
    grid = list(range(100, 901, 100))
    lt = torch.log(
        (1 - linear_beta_alphas_cumprod()[grid]) / linear_beta_alphas_cumprod()[grid]
    )
    W_sum = float(lt[-1] - lt[0])
    return out, U, s, W_sum, c, enc


def test_cc_eigenvalues_match_sigma2_W_s2(affine_estimate):
    out, _, s, W_sum, _, _ = affine_estimate
    expected = SIGMA**2 * W_sum * s**2
    np.testing.assert_allclose(
        out.eigenvalues[:3].double().numpy(), expected, rtol=0.10
    )


def test_cc_eigenvalues_beyond_rank_vanish(affine_estimate):
    out, *_ = affine_estimate
    assert float(out.eigenvalues[3:].max()) < 1e-5 * float(out.eigenvalues[0])


def test_cc_eigenvectors_match_left_singular_vectors(affine_estimate):
    out, U, *_ = affine_estimate
    V = out.eigenvectors[:3].double()  # rows
    cos = (V @ U).abs().diagonal()  # matched by eigenvalue order
    assert torch.all(cos > 0.98), cos
    # The span is exact (no Monte-Carlo error): D's rows lie in span(U).
    proj = (V @ U).pow(2).sum() / 3
    assert float(proj) == pytest.approx(1.0, abs=1e-5)


def test_cc_eigenvectors_orthonormal(affine_estimate):
    out, *_ = affine_estimate
    V = out.eigenvectors.double()
    torch.testing.assert_close(
        V @ V.T, torch.eye(V.shape[0], dtype=torch.float64), atol=1e-5, rtol=0
    )


def test_encoder_receives_clipped_tweedie_denoised_image(affine_estimate):
    *_, c, enc = affine_estimate
    expected = c.clamp(-1, 1)
    seen = torch.cat(enc.inputs)
    assert seen.shape[1:] == (1, H, W)
    torch.testing.assert_close(seen, expected.expand_as(seen), atol=2e-3, rtol=0)


# ── seeding and argument checks ────────────────────────────────────────────
def _small_run(image, seed, global_seed):
    A, _, _ = _affine_problem(seed=2)
    ab = _alphas32()
    unet, enc = AffineUNet(A, torch.zeros(1, 1, H, W), ab), RecordingEncoder()
    cfg = estimator_cfg(
        n_samples=20, batch_size=10, start=200, stop=600, step=200, sigma=SIGMA, top_k=4
    )
    torch.manual_seed(global_seed)  # must not matter
    return conditional_pair_estimate(
        image,
        enc,
        unet,
        ab,
        cfg,
        torch.device("cpu"),
        generator=torch.Generator().manual_seed(seed),
    )


def test_same_seed_gives_identical_output_regardless_of_global_rng():
    img = torch.rand(1, 1, H, W) * 2 - 1
    a, b = _small_run(img, 7, global_seed=0), _small_run(img, 7, global_seed=999)
    assert torch.equal(a.eigenvalues, b.eigenvalues)
    assert torch.equal(a.eigenvectors, b.eigenvectors)


def test_different_seed_gives_different_output():
    img = torch.rand(1, 1, H, W) * 2 - 1
    assert not torch.equal(
        _small_run(img, 7, 0).eigenvalues, _small_run(img, 8, 0).eigenvalues
    )


def test_per_image_estimate_independent_of_processing_order():
    imgs = [torch.rand(1, 1, H, W) * 2 - 1 for _ in range(2)]
    forward = [_small_run(imgs[i], 100 + i, 0) for i in (0, 1)]
    backward = {i: _small_run(imgs[i], 100 + i, 0) for i in (1, 0)}
    for i in (0, 1):
        assert torch.equal(forward[i].eigenvalues, backward[i].eigenvalues)


def test_raises_when_n_mc_not_divisible_by_batch():
    A, _, _ = _affine_problem()
    ab = _alphas32()
    cfg = estimator_cfg(
        n_samples=25, batch_size=10, start=100, stop=300, step=100, sigma=SIGMA, top_k=2
    )
    with pytest.raises(ValueError):
        conditional_pair_estimate(
            torch.zeros(1, 1, H, W),
            RecordingEncoder(),
            AffineUNet(A, torch.zeros(1, 1, H, W), ab),
            ab,
            cfg,
            torch.device("cpu"),
        )


def test_raises_on_multi_image_batch():
    A, _, _ = _affine_problem()
    ab = _alphas32()
    cfg = estimator_cfg(
        n_samples=10, batch_size=10, start=100, stop=300, step=100, sigma=SIGMA, top_k=2
    )
    with pytest.raises(ValueError):
        conditional_pair_estimate(
            torch.zeros(2, 1, H, W),
            RecordingEncoder(),
            AffineUNet(A, torch.zeros(1, 1, H, W), ab),
            ab,
            cfg,
            torch.device("cpu"),
        )


# ── M_τ(x) at each noise level (estimate_g_per_scale) ─────────────────────
# Without the weights w_j, the rows of one noise level give M_τ = σ² A Aᵀ, whatever τ:
# its eigenvalues are σ² s_i² at every level, and its eigenvectors span the columns of
# U. The Monte-Carlo relative error of one level's eigenvalues is about sqrt(2/N_MC) ≈
# 3% with 2000 samples; the tolerance is 15%.
PER_SCALE_GRID = list(range(100, 901, 200))  # five noise levels


@pytest.fixture(scope="module")
def affine_per_scale():
    # the affine problem: mock network, recording encoder, clean image
    A, U, s = _affine_problem(seed=0)
    rng = np.random.default_rng(1)
    c = torch.from_numpy(rng.uniform(-1.5, 1.5, (1, 1, H, W))).float()
    image = torch.from_numpy(rng.uniform(-1, 1, (1, 1, H, W))).float()
    ab = _alphas32()

    # 2000 samples at each of the five noise levels, five eigenpairs kept
    cfg = estimator_cfg(
        n_samples=2000,
        batch_size=500,
        start=100,
        stop=900,
        step=200,
        sigma=SIGMA,
        top_k=5,
    )

    # M_τ(x) at each level, and G(x) from the same random draws (same seed)
    args = (
        image,
        RecordingEncoder(),
        AffineUNet(A, c, ab),
        ab,
        cfg,
        torch.device("cpu"),
    )
    per_scale = estimate_g_per_scale(
        *args, generator=torch.Generator().manual_seed(123)
    )
    g = estimate_g(*args, generator=torch.Generator().manual_seed(123))
    return per_scale, g, U, s, ab


def test_per_scale_grid_and_shapes(affine_per_scale):
    out, _, _, _, ab = affine_per_scale

    # one log τ_j per noise level, as for G(x)
    torch.testing.assert_close(out.log_tau, _build_log_tau_grid(PER_SCALE_GRID, ab))

    # five eigenpairs at each of the five levels
    assert tuple(out.eigenvalues.shape) == (len(PER_SCALE_GRID), 5)
    assert tuple(out.eigenvectors.shape) == (len(PER_SCALE_GRID), 5, P)


def test_per_scale_eigenvalues_match_sigma2_s2_at_every_level(affine_per_scale):
    out, _, _, s, _ = affine_per_scale

    # σ² s_i² at every level: no weight w_j
    expected = np.broadcast_to(SIGMA**2 * s**2, (len(PER_SCALE_GRID), 3))
    got = out.eigenvalues[:, :3].double().numpy()
    np.testing.assert_allclose(got, expected, rtol=0.15)


def test_per_scale_eigenvectors_span_the_columns_of_u(affine_per_scale):
    out, _, U, _, _ = affine_per_scale

    # the fraction of the three leading eigenvectors that lies in span(U), per level
    inside = (out.eigenvectors[:, :3].double() @ U).pow(2).sum(dim=(1, 2)) / 3

    # exact: the rows of D lie in span(U), there is no Monte-Carlo error here
    torch.testing.assert_close(inside, torch.ones_like(inside), atol=1e-5, rtol=0)


def test_weighted_sum_over_levels_is_g(affine_per_scale):
    out, g, _, _, _ = affine_per_scale

    # M_τ_j from its eigenpairs, at each level j (rank 3, so five eigenpairs are all)
    values, vectors = out.eigenvalues.double(), out.eigenvectors.double()
    m_levels = torch.einsum("jk,jkp,jkq->jpq", values, vectors, vectors)

    # Σ_j w_j M_τ_j, with the trapezoid weights of G(x)
    summed = torch.einsum(
        "j,jpq->pq", _trapezoid_weights(out.log_tau.double()), m_levels
    )

    # G(x) from its eigenpairs: the same random draws, so the same matrix
    g_vectors = g.eigenvectors.double()
    g_matrix = g_vectors.T @ torch.diag(g.eigenvalues.double()) @ g_vectors
    tolerance = 1e-5 * float(g.eigenvalues[0])
    torch.testing.assert_close(summed, g_matrix, atol=tolerance, rtol=0)


def test_per_scale_raises_on_multi_image_batch():
    # the same checks as estimate_g: one image at a time
    A, _, _ = _affine_problem()
    ab = _alphas32()
    cfg = estimator_cfg(
        n_samples=10, batch_size=10, start=100, stop=300, step=100, sigma=SIGMA, top_k=2
    )
    with pytest.raises(ValueError):
        estimate_g_per_scale(
            torch.zeros(2, 1, H, W),
            RecordingEncoder(),
            AffineUNet(A, torch.zeros(1, 1, H, W), ab),
            ab,
            cfg,
            torch.device("cpu"),
        )
