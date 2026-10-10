"""Small helpers of stages 01 and 03, checked against independent references."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import importlib


@pytest.fixture(scope="module")
def stage01():
    return importlib.import_module("multigeom.fit.train_encoder")


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
    stage03 = importlib.import_module("multigeom.diffusion.train")
    f = stage03.cosine_with_warmup
    assert f(0, 500, 10_500) == 0.0
    assert f(250, 500, 10_500) == pytest.approx(0.5)
    assert f(500, 500, 10_500) == pytest.approx(1.0)
    assert f(5_500, 500, 10_500) == pytest.approx(0.5)            # half-way through decay
    assert f(10_500, 500, 10_500) == pytest.approx(0.0, abs=1e-12)
    s = 2_000
    assert f(s, 500, 10_500) == pytest.approx(0.5 * (1 + math.cos(math.pi * 1_500 / 10_000)))


# ── stage 01 --evaluate-only ───────────────────────────────────────────────
def test_test_set_fields_match_independent_computation(stage01):
    g = torch.Generator().manual_seed(1)
    resp = torch.randn(40, 5, generator=g)
    pred = 0.6 * resp + 0.8 * torch.randn(40, 5, generator=g)
    oracle = np.array([0.9, 0.8, 0.95, 0.7, 0.85])
    f = stage01.test_set_fields(pred, resp, oracle)
    r = np.array([np.corrcoef(pred[:, n].numpy(), resp[:, n].numpy())[0, 1] for n in range(5)])
    np.testing.assert_allclose(f["per_neuron_test_corr"], r, rtol=1e-6)
    np.testing.assert_allclose(f["per_neuron_test_corr_noise_corrected"], r / oracle, rtol=1e-6)
    assert f["test_corr"] == pytest.approx(r.mean())
    assert f["test_corr_noise_corrected"] == pytest.approx((r / oracle).mean())
    torch.testing.assert_close(f["test_predictions"], pred)


def test_evaluate_only_adds_test_set_fields(stage01, tmp_path, monkeypatch):
    """Run stage 01 --evaluate-only on synthetic TVSD files and images with a small
    stand-in encoder; the stored predictions must equal the encoder applied to the
    test images, and the weights must be unchanged."""
    import sys
    import torch.nn as nn
    from PIL import Image

    from conftest import write_fake_tvsd
    from multigeom.data import CANONICAL_TRANSFORM
    from multigeom.fit.tvsd import load_tvsd
    from multigeom.encoders import _BaseEncoder, _FrozenBackbone

    class SmallBackbone(_FrozenBackbone):
        out_channels, out_spatial = 2, 16

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 2, kernel_size=4, stride=4)
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

        def forward(self, x):
            return self.conv(x)

    class SmallEncoder(_BaseEncoder):
        backbone_input_channels, backbone_input_range = 3, (-1.0, 1.0)

    def build(cfg):
        torch.manual_seed(0)
        return SmallEncoder(SmallBackbone(), n_neurons=50, dropout=0.25)

    data = tmp_path / "data"
    fake = write_fake_tvsd(data / "tvsd", n_train=3, n_test=5)
    rng = np.random.default_rng(0)
    for p in fake["paths"]["test"]:
        path = data / "things_images" / "object_images" / p.replace("\\", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)).save(path)

    cfg = tmp_path / "enc.yaml"
    cfg.write_text("encoder:\n  name: inception\n  n_neurons: 50\n  dropout: 0.25\n"
                   "train:\n  seed: 0\n")
    enc = build(None)
    with torch.no_grad():
        enc.readout.feature_weights.normal_()
    ckpt_in = tmp_path / "legacy" / "best_model.pt"
    ckpt_in.parent.mkdir()
    torch.save({"model_state_dict": enc.state_dict(), "val_corr": 0.5}, ckpt_in)

    monkeypatch.setattr(stage01, "build_encoder", build)
    monkeypatch.setattr(stage01, "preextract", _preextract_without_workers(stage01))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "01_train_encoder.py", "--config", str(cfg), "--area", "V1", "--device", "cpu",
        "--data-dir", str(data), "--out-dir", str(out),
        "--evaluate-only", "--encoder-ckpt", str(ckpt_in)])
    stage01.main()

    got = torch.load(out / "best_model.pt", weights_only=False)
    tvsd = load_tvsd("V1", 50, tvsd_dir=data / "tvsd")
    images = torch.stack([CANONICAL_TRANSFORM(Image.open(
        data / "things_images" / "object_images" / p).convert("RGB")) for p in tvsd["test_paths"]])
    enc.eval()
    with torch.no_grad():
        expected = enc(images)
    torch.testing.assert_close(got["test_predictions"], expected)
    torch.testing.assert_close(got["test_responses"], tvsd["test_responses"])
    assert got["val_corr"] == 0.5
    for k, v in enc.state_dict().items():
        torch.testing.assert_close(got["model_state_dict"][k], v)
    assert got["per_neuron_test_corr"].shape == (50,)
    np.testing.assert_array_equal(got["neuron_indices"], tvsd["neuron_indices"])


def _preextract_without_workers(stage01):
    """stage01.preextract with a single-process DataLoader (no worker processes in tests)."""
    from torch.utils.data import DataLoader
    from multigeom.fit.tvsd import TVSDDataset

    @torch.no_grad()
    def preextract(encoder, paths, responses, images_root, transform, device, batch_size=64):
        loader = DataLoader(TVSDDataset(responses, paths, images_root, transform=transform),
                            batch_size=batch_size, shuffle=False, num_workers=0)
        feats, resp = [], []
        for img, r in loader:
            feats.append(encoder.features(img.to(device)).cpu())
            resp.append(r)
        return torch.cat(feats), torch.cat(resp)

    return preextract
