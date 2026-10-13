"""
Tests for the supporting analyses (multigeom.fit and multigeom.pipeline.estimator_variance).

calibrate_sigma: per-trial residuals of ±a_c (alternating over an even
number of trials) have standard deviation exactly a_c, so the expected median
is known without re-implementing the computation.

estimator_variance (UC form): with the affine mock score network of
test_estimator.py and a constant encoder μ_0, the UC row is
√(w_j/N_MC)·(σ A ξ − b) where b = ε(x_t,t,0) − (A μ_0 + h) is the mismatch of
the null-token prediction. Hence E[D_UCᵀ D_UC] = W (σ² A Aᵀ + b bᵀ),
W = Σ_j w_j. With b = 0 this equals the CC expectation σ² W A Aᵀ (the scaling
derived in the module docstring); with b ⟂ range(A) an extra eigenvalue W‖b‖²
appears along b.

layer_sweep: the truncated backbone is compared with the torchvision forward
order applied by hand, and with the src backbones (pretrained); the readout
trainer must recover a planted Gaussian readout.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import estimator_cfg, linear_beta_alphas_cumprod, write_fake_tvsd
from multigeom.fit.tvsd import AREA_CHANNELS, load_tvsd

pytestmark = pytest.mark.filterwarnings("ignore:`product` is deprecated:DeprecationWarning")


# ── calibrate_sigma ────────────────────────────────────────────────────────
def test_calibrate_sigma_recovers_planted_trial_std(tmp_path):
    from multigeom.fit.calibrate_sigma import main

    n_trials, n_test = 6, 4
    write_fake_tvsd(tmp_path, n_test=n_test, n_trials=n_trials)
    rng = np.random.default_rng(3)
    a = rng.uniform(0.1, 0.5, 1024)                       # per-channel residual std
    mean = rng.standard_normal((n_test, 1024))
    sign = np.where(np.arange(n_trials) % 2 == 0, 1.0, -1.0)[:, None, None]
    reps = mean[None] + sign * a[None, None, :]
    with h5py.File(tmp_path / "monkeyN_THINGS_normMUA.mat", "r+") as f:
        del f["test_MUA_reps"], f["test_MUA"]
        f["test_MUA_reps"] = reps
        f["test_MUA"] = mean

    out = main(["--tvsd-dir", str(tmp_path), "--n-neurons", "7",
                "--out", str(tmp_path / "sigma.json")])
    saved = json.loads((tmp_path / "sigma.json").read_text())
    assert saved["areas"].keys() == out["areas"].keys() == {"V1", "V4"}
    assert out["max_abs_trial_mean_minus_test_MUA"] < 1e-12
    for area, (lo, hi) in AREA_CHANNELS.items():
        top = load_tvsd(area, n_neurons=7, tvsd_dir=tmp_path)["neuron_indices"] + lo
        r = out["areas"][area]
        assert r["top_channel_indices_global"] == top.tolist()
        assert r["sigma_median_top_channels"] == pytest.approx(np.median(a[top]), rel=1e-12)
        assert r["sigma_median_all_area_channels"] == pytest.approx(
            np.median(a[lo:hi + 1]), rel=1e-12)
    assert "git_commit" in saved["provenance"]


# ── estimator_variance ─────────────────────────────────────────────────────
H = W = 8
P = H * W
N_NEURONS = 3
SIGMA = 0.3
MU0 = torch.tensor([0.5, -1.0, 2.0])


class MismatchUNet(nn.Module):
    """ε(r) = A r + h(x_t) for r ≠ 0; null branch A μ_0 + h(x_t) + b."""

    def __init__(self, A, b):
        super().__init__()
        self.register_buffer("A", A.float())
        self.register_buffer("b", b.float().view(1, 1, H, W))
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x_t, t, r):
        h = 5.0 * torch.sin(3.0 * x_t)
        cond = (r @ self.A.T).view_as(x_t) + h
        null = (MU0 @ self.A.T).view(1, 1, H, W) + h + self.b
        return torch.where((r == 0).all(dim=1).view(-1, 1, 1, 1), null, cond)


class ConstantEncoder(nn.Module):
    n_neurons = N_NEURONS

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return MU0.expand(x.shape[0], -1).clone()


def _problem(beta: float):
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((P, N_NEURONS + 1)))
    U, bdir = Q[:, :N_NEURONS], Q[:, N_NEURONS]
    R, _ = np.linalg.qr(rng.standard_normal((N_NEURONS, N_NEURONS)))
    s = np.array([3.0, 2.0, 1.0])
    A = torch.from_numpy(U @ np.diag(s) @ R.T)
    return A, torch.from_numpy(beta * bdir), torch.from_numpy(U), torch.from_numpy(bdir), s


def _grid_W(start, stop, step):
    ab = linear_beta_alphas_cumprod()[list(range(start, stop + 1, step))]
    lt = torch.log((1 - ab) / ab)
    return float(lt[-1] - lt[0])


def test_uc_without_mismatch_matches_cc_expectation():
    pytest.importorskip("diffusers")
    from multigeom.pipeline.estimator_variance import unconditional_conditional_estimate

    A, b, U, _, s = _problem(beta=0.0)
    cfg = estimator_cfg(400, 100, 100, 900, 100, SIGMA, top_k=4)
    out = unconditional_conditional_estimate(
        torch.zeros(1, 1, H, W), ConstantEncoder(), MismatchUNet(A, b),
        linear_beta_alphas_cumprod().float(), cfg, torch.device("cpu"),
        generator=torch.Generator().manual_seed(0))
    expected = SIGMA ** 2 * _grid_W(100, 900, 100) * s ** 2
    np.testing.assert_allclose(out.eigenvalues[:3].double().numpy(), expected, rtol=0.10)
    assert float(out.eigenvalues[3]) < 1e-5 * float(out.eigenvalues[0])
    assert float(((out.eigenvectors[:3].double() @ U) ** 2).sum() / 3) == pytest.approx(1, abs=1e-5)


def test_uc_mismatch_adds_W_b2_along_b():
    pytest.importorskip("diffusers")
    from multigeom.pipeline.estimator_variance import unconditional_conditional_estimate

    beta = 4.0
    A, b, _, bdir, s = _problem(beta=beta)
    cfg = estimator_cfg(400, 100, 100, 900, 100, SIGMA, top_k=4)
    out = unconditional_conditional_estimate(
        torch.zeros(1, 1, H, W), ConstantEncoder(), MismatchUNet(A, b),
        linear_beta_alphas_cumprod().float(), cfg, torch.device("cpu"),
        generator=torch.Generator().manual_seed(1))
    Wsum = _grid_W(100, 900, 100)
    assert float(out.eigenvalues[0]) == pytest.approx(Wsum * beta ** 2, rel=0.05)
    assert abs(float(out.eigenvectors[0].double() @ bdir)) > 0.99
    np.testing.assert_allclose(out.eigenvalues[1:4].double().numpy(),
                               SIGMA ** 2 * Wsum * s ** 2, rtol=0.10)


def test_variance_study_ratio_reflects_mismatch():
    pytest.importorskip("diffusers")
    from multigeom.pipeline.estimator_variance import run_variance_study

    beta = 4.0
    A, b, *_ = _problem(beta=beta)
    cfg = estimator_cfg(100, 50, 200, 800, 200, SIGMA, top_k=5)
    res = run_variance_study(torch.zeros(2, 1, H, W), ConstantEncoder(), MismatchUNet(A, b),
                             linear_beta_alphas_cumprod().float(), cfg, torch.device("cpu"),
                             n_runs=3, top_k=3, n_mc_uc=150, log=lambda *a: None)
    assert res["n_mc_cc"] == 100 and res["n_mc_uc"] == 150
    assert res["unet_passes_per_timestep"] == {"CC": 300, "UC": 300}
    assert len(res["per_image"]) == 2
    expected_ratio = beta ** 2 / (SIGMA ** 2 * 9.0)          # W‖b‖² / (σ² W s_1²)
    for p in res["per_image"]:
        assert np.array(p["eigenvalues_CC"]).shape == (3, 3)
        assert p["ratio_top_eigenvalue_UC_over_CC"] == pytest.approx(expected_ratio, rel=0.2)
    json.dumps(res)                                           # serialisable


def test_matched_pass_count():
    from multigeom.pipeline.estimator_variance import uc_samples_for_matched_passes

    assert uc_samples_for_matched_passes(200, 25) == 300      # 3·200 = 2·300
    assert uc_samples_for_matched_passes(100, 50) == 150


# ── layer_sweep ────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def nets():
    from multigeom.fit.layer_sweep import build_torchvision_net

    torch.manual_seed(0)
    return {bb: build_torchvision_net(bb, pretrained=False).eval()
            for bb in ("inception", "resnet50")}


def test_truncated_inception_matches_manual_forward(nets):
    from multigeom.fit.layer_sweep import TruncatedBackbone

    net = nets["inception"]
    tb = TruncatedBackbone("inception", "Conv2d_4a_3x3", net=net)
    x = torch.rand(2, 3, 64, 64) * 2 - 1
    with torch.no_grad():
        h = net.Conv2d_2b_3x3(net.Conv2d_2a_3x3(net.Conv2d_1a_3x3(x)))
        h = net.Conv2d_4a_3x3(net.Conv2d_3b_1x1(net.maxpool1(h)))
        torch.testing.assert_close(tb(x), h)
    assert (tb.out_channels, tb.out_spatial) == (192, 12)
    assert not any(p.requires_grad for p in tb.parameters())


def test_truncated_inception_skips_aux_and_rejects_collapsed_maps(nets):
    from multigeom.fit.layer_sweep import TruncatedBackbone

    tb = TruncatedBackbone("inception", "Mixed_6e", net=nets["inception"])
    assert "AuxLogits" not in dict(tb.features.named_children())
    assert (tb.out_channels, tb.out_spatial) == (768, 2)
    with pytest.raises(RuntimeError):
        TruncatedBackbone("inception", "Mixed_7a", net=nets["inception"])
    with pytest.raises(ValueError):
        TruncatedBackbone("inception", "Mixed_9z", net=nets["inception"])


def test_truncated_resnet_matches_manual_forward(nets):
    from multigeom.fit.layer_sweep import TruncatedBackbone
    from multigeom.encoders import ResNet50Backbone

    net = nets["resnet50"]
    x = torch.rand(2, 3, 64, 64)
    mean = torch.tensor(ResNet50Backbone.IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(ResNet50Backbone.IMAGENET_STD).view(1, 3, 1, 1)
    with torch.no_grad():
        stem = net.relu(net.bn1(net.conv1((x - mean) / std)))
        torch.testing.assert_close(TruncatedBackbone("resnet50", "conv1", net=net)(x), stem)
        h = net.layer2[0](net.layer1(net.maxpool(stem)))
        torch.testing.assert_close(TruncatedBackbone("resnet50", "layer2.0", net=net)(x), h)
    tb = TruncatedBackbone("resnet50", "layer1.0", net=net)
    assert (tb.out_channels, tb.out_spatial) == (256, 16)


def test_sweep_encoder_input_convention(nets):
    from multigeom.fit.layer_sweep import SweepEncoder
    from multigeom.encoders import InceptionEncoder, ResNet50Encoder

    for bb, ref in (("inception", InceptionEncoder), ("resnet50", ResNet50Encoder)):
        enc = SweepEncoder(bb, "Conv2d_2a_3x3" if bb == "inception" else "maxpool", 5, 0.25,
                           net=nets[bb])
        assert enc.backbone_input_range == ref.backbone_input_range
        assert enc.backbone_input_channels == ref.backbone_input_channels
        enc.train()
        assert not enc.backbone.training and enc.readout.training


def test_the_sweep_fits_readouts_with_the_shared_loop_on_cached_cpu_features():
    from multigeom.encoders import GaussianReadout
    from multigeom.encoders.fitting import fit_readout_on_features

    torch.manual_seed(0)
    N, C, S, n = 600, 4, 6, 3
    truth = GaussianReadout(n, C, S, dropout=0.0).eval()
    with torch.no_grad():
        truth.feature_weights.normal_()
    feats = torch.randn(N, C, S, S)
    with torch.no_grad():
        resp = truth(feats) + 0.05 * torch.randn(N, n)
    model = GaussianReadout(n, C, S, dropout=0.0)
    report = fit_readout_on_features(model, feats, resp, epochs=30, lr=3e-2, weight_decay=0.0, batch_size=64,
                                     val_fraction=0.1, seed=0, device="cpu", log=None)
    assert report.best_val_corr > 0.9 and 1 <= report.best_epoch <= 30


@pytest.mark.slow
def test_sweep_end_to_end_random_weights():
    from multigeom.fit.layer_sweep import sweep

    g = torch.Generator().manual_seed(0)
    images = {"train": torch.rand(40, 1, 64, 64, generator=g) * 2 - 1,
              "test": torch.rand(12, 1, 64, 64, generator=g) * 2 - 1}
    responses = {k: torch.randn(len(v), 3, generator=g) for k, v in images.items()}
    cfg = SimpleNamespace(lr=3e-4, weight_decay=1e-4, epochs=2, batch_size=16,
                          validation_fraction=0.25)
    res = sweep(["Conv2d_4a_3x3", "Mixed_7a"], "inception", images, responses,
                np.full(3, 0.9), cfg, n_neurons=3, dropout=0.25, device=torch.device("cpu"),
                seed=0, pretrained=False, log=lambda *a: None)
    assert "skipped" in res["Mixed_7a"]
    r = res["Conv2d_4a_3x3"]
    assert r["feature_shape"] == [192, 12, 12]
    assert r["readout_parameters"] == 3 * (2 + 1 + 192 + 1)
    assert np.isfinite(r["test_corr"]) and len(r["per_neuron_test_corr"]) == 3


@pytest.mark.pretrained
@pytest.mark.slow
@pytest.mark.parametrize("backbone, layer", [("inception", "Conv2d_4a_3x3"),
                                             ("resnet50", "layer1.0")])
def test_truncated_backbone_matches_src_backbone(backbone, layer):
    from multigeom.fit.layer_sweep import TruncatedBackbone
    from multigeom.encoders import InceptionBackbone, ResNet50Backbone

    ref = (InceptionBackbone if backbone == "inception" else ResNet50Backbone)(layer=layer)
    tb = TruncatedBackbone(backbone, layer)
    x = torch.rand(2, 3, 64, 64) * (2 if backbone == "inception" else 1) - (
        1 if backbone == "inception" else 0)
    with torch.no_grad():
        torch.testing.assert_close(tb(x), ref(x))
