"""Tests for NeuralEncoder (predict, fit_readout, save, load, load_encoder), the readouts and the registries."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

import multigeom
from multigeom import validate_encoder
from multigeom.encoders import (
    BACKBONES, READOUTS, FactorizedReadout, GaussianReadout, LinearReadout, NeuralEncoder, SmallCNNBackbone,
    load_encoder, register_backbone,
)
from multigeom.encoders.fitting import correlation_per_neuron, fit_readout_on_features
from multigeom.encoders.reference import REFERENCE_ENCODERS, _BaseEncoder

N = 6


def tiny_encoder(readout="gaussian", channels=(4, 4, 4), n_neurons=N):
    torch.manual_seed(0)
    return NeuralEncoder(SmallCNNBackbone(channels), n_neurons, readout=readout)


def images(n=24, seed=0):
    """Smooth random images in [-1, 1] (their features differ from image to image, unlike white noise)."""
    g = torch.Generator().manual_seed(seed)
    low = torch.randn(n, 1, 6, 6, generator=g)
    return torch.nn.functional.interpolate(low, size=64, mode="bicubic", align_corners=False).clamp(-1, 1)


# ── readouts ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name, cls", [("gaussian", GaussianReadout), ("linear", LinearReadout),
                                       ("factorized", FactorizedReadout)])
def test_readouts_are_registered_and_map_features_to_neurons(name, cls):
    assert READOUTS[name] is cls
    readout = cls(n_neurons=5, n_channels=3, spatial=8, dropout=0.0)
    assert readout(torch.randn(2, 3, 8, 8)).shape == (2, 5)


def test_linear_readout_is_a_linear_map_of_the_spatial_mean():
    r = LinearReadout(4, 3, 8, dropout=0.0).eval()
    f = torch.randn(2, 3, 8, 8)
    torch.testing.assert_close(r(f), f.mean((-2, -1)) @ r.linear.weight.T + r.linear.bias)


def test_factorized_readout_window_sums_to_one_and_counts_parameters():
    r = FactorizedReadout(4, 3, 8)
    attn = torch.softmax(r.spatial_logits.flatten(1), 1)
    torch.testing.assert_close(attn.sum(1), torch.ones(4))
    assert sum(p.numel() for p in r.parameters()) == 4 * (8 * 8 + 3 + 1)


def test_unknown_readout_name_lists_the_available_ones():
    with pytest.raises(ValueError, match="gaussian"):
        NeuralEncoder(SmallCNNBackbone((4, 4, 4)), 3, readout="nope")


# ── the encoder ─────────────────────────────────────────────────────────────
def test_encoder_follows_the_contract_and_predict_matches_forward():
    enc = tiny_encoder()
    report = validate_encoder(enc, sigma=0.22)
    assert report.n_neurons == N
    x = images(10)
    enc.eval()
    with torch.no_grad():
        direct = enc(x)
    torch.testing.assert_close(enc.predict(x, batch_size=3), direct)


def test_predict_accepts_a_dataset_and_restores_the_mode():
    enc = tiny_encoder()
    enc.train()

    class DS(torch.utils.data.Dataset):
        def __len__(self): return 7
        def __getitem__(self, i): return images(1, seed=i)[0], i

    out = enc.predict(DS(), batch_size=4)
    assert out.shape == (7, N) and enc.training


def test_summary_names_the_parts():
    text = tiny_encoder().summary()
    assert "SmallCNNBackbone" in text and "GaussianReadout" in text and f"{N} neurons" in text


# ── fitting ─────────────────────────────────────────────────────────────────
def _teacher_data(enc_features, n=400, seed=1):
    """Responses that a Gaussian readout on `enc_features` can produce exactly, plus a little noise."""
    g = torch.Generator().manual_seed(seed)
    true = GaussianReadout(N, enc_features.shape[1], enc_features.shape[-1], dropout=0.0)
    with torch.no_grad():
        true.mu.copy_(torch.rand(N, 2, generator=g) - 0.5)
        true.feature_weights.copy_(torch.randn(N, enc_features.shape[1], generator=g))
        return true(enc_features), true


def test_fit_readout_recovers_responses_generated_by_a_readout():
    enc = tiny_encoder()
    x = images(400)
    enc.backbone.train()
    for _ in range(40):                                  # BatchNorm statistics of this data: features of unit scale
        enc.backbone(x)
    enc.backbone.eval()
    feats = enc.features(x)
    r, _ = _teacher_data(feats)
    report = enc.fit_readout(x, r, epochs=150, lr=3e-2, batch_size=64, verbose=False)
    pred = enc.predict(x)
    assert correlation_per_neuron(pred, r).mean() > 0.9
    assert report.best_epoch >= 1 and report.n_train + report.n_val == 400 and report.best_val_corr > 0.85
    assert all(not p.requires_grad for p in enc.backbone.parameters())      # fitting froze the backbone
    assert not enc.backbone.training                                       # and keeps its BatchNorm in eval mode


def test_fit_readout_keeps_the_backbone_weights_unchanged():
    enc = tiny_encoder()
    before = {k: v.clone() for k, v in enc.backbone.state_dict().items()}
    x = images(60)
    enc.fit_readout(x, torch.randn(60, N), epochs=3, verbose=False)
    for k, v in enc.backbone.state_dict().items():
        torch.testing.assert_close(v, before[k])


def test_fit_readout_is_reproducible_and_validates_shapes():
    x, r = images(80), torch.randn(80, N)
    a, b = tiny_encoder(), tiny_encoder()
    ra = a.fit_readout(x, r, epochs=5, seed=3, verbose=False)
    rb = b.fit_readout(x, r, epochs=5, seed=3, verbose=False)
    assert ra.history["val_corr"] == rb.history["val_corr"]
    with pytest.raises(ValueError, match="responses should be"):
        a.fit_readout(x, torch.randn(80, N + 1), verbose=False)


def test_fit_returns_the_best_epoch_readout():
    torch.manual_seed(0)
    feats = torch.randn(200, 4, 8, 8)
    resp = torch.randn(200, N)                                  # noise: validation correlation fluctuates
    readout = GaussianReadout(N, 4, 8, dropout=0.0)
    report = fit_readout_on_features(readout, feats, resp, epochs=8, batch_size=50, lr=1e-2, val_fraction=0.25,
                                     seed=0, log=None)
    perm = torch.randperm(200, generator=torch.Generator().manual_seed(0))
    val = perm[-50:]
    with torch.no_grad():
        got = correlation_per_neuron(readout.eval()(feats[val]), resp[val]).mean()
    assert got == pytest.approx(max(report.history["val_corr"]), abs=1e-6)
    assert report.history["val_corr"][report.best_epoch - 1] == pytest.approx(max(report.history["val_corr"]))


# ── save, load ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("readout", ["gaussian", "linear", "factorized"])
def test_save_and_load_round_trip(tmp_path, readout):
    enc = tiny_encoder(readout)
    enc.fit_readout(images(40), torch.randn(40, N), epochs=2, verbose=False)
    path = enc.save(tmp_path / "enc.pt", area="V1", note="hello")
    loaded = NeuralEncoder.load(path)
    x = images(5, seed=9)
    torch.testing.assert_close(loaded.predict(x), enc.predict(x))
    assert loaded.config == enc.config
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["metadata"] == {"area": "V1", "note": "hello"} and ckpt["format_version"] == 1
    assert loaded.metadata == {"area": "V1", "note": "hello"}
    assert ckpt["multigeom_version"] == multigeom.__version__


def test_load_checkpoint_into_an_existing_encoder(tmp_path):
    a, b = tiny_encoder(), tiny_encoder()
    b.readout.feature_weights.data.add_(1.0)
    a.save(tmp_path / "a.pt")
    b.load_checkpoint(tmp_path / "a.pt")
    x = images(4)
    torch.testing.assert_close(a.predict(x), b.predict(x))
    other = tiny_encoder(channels=(4, 4, 8))
    with pytest.raises(RuntimeError):
        other.load_checkpoint(tmp_path / "a.pt")


def test_an_unregistered_backbone_cannot_describe_itself():
    class Mine(nn.Module):
        out_channels, out_spatial = 2, 4
        def forward(self, x): return torch.zeros(len(x), 2, 4, 4)

    enc = NeuralEncoder(Mine(), 3)
    with pytest.raises(ValueError, match="register_backbone"):
        enc.save("unused.pt")


def test_a_registered_custom_backbone_round_trips(tmp_path):
    @register_backbone("test_flat")
    class Flat(nn.Module):
        out_spatial = 4

        def __init__(self, channels: int = 3):
            super().__init__()
            self.out_channels = channels
            self.conv = nn.Conv2d(1, channels, 16, stride=16)

        def forward(self, x):
            return self.conv(x)

    try:
        enc = NeuralEncoder(Flat(5), 4, readout="linear")
        assert enc.config["backbone"] == {"name": "test_flat", "kwargs": {"channels": 5}}
        loaded = NeuralEncoder.load(enc.save(tmp_path / "flat.pt"))
        x = images(3)
        torch.testing.assert_close(loaded.predict(x), enc.predict(x))
        with pytest.raises(ValueError, match="already registered"):
            register_backbone("test_flat")(type("Other", (nn.Module,), {}))
    finally:
        BACKBONES.pop("test_flat", None)


def test_loading_a_checkpoint_of_the_paper_format(tmp_path, monkeypatch):
    """The paper's checkpoints say only {"encoder": name, "area": ...}: the architecture comes from the reference name."""
    class TinyBackbone(nn.Module):
        out_channels, out_spatial = 4, 8
        def __init__(self, layer="x"):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, stride=8, padding=1)
        def forward(self, x): return self.conv(x)

    class TinyRef(_BaseEncoder):
        backbone_input_channels, backbone_input_range, reference_name = 3, (-1.0, 1.0), "tinyref"
        def __init__(self, n_neurons=50, layer="x", dropout=0.25):
            super().__init__(TinyBackbone(layer), n_neurons, dropout, layer=layer)

    monkeypatch.setitem(REFERENCE_ENCODERS, "tinyref", TinyRef)
    src = TinyRef(n_neurons=5)
    legacy = {"model_state_dict": src.state_dict(), "config": {"area": "V1", "encoder": "tinyref"}, "val_corr": 0.5}
    torch.save(legacy, tmp_path / "best_model.pt")
    loaded = load_encoder(tmp_path / "best_model.pt")
    x = images(3)
    torch.testing.assert_close(loaded.predict(x), src.predict(x))
    assert loaded.config["reference"] == "tinyref" and loaded.n_neurons == 5
    assert loaded.metadata == {"area": "V1", "val_corr": 0.5}
    new = tmp_path / "new.pt"
    src.save(new)
    torch.testing.assert_close(NeuralEncoder.load(new).predict(x), src.predict(x))


def test_a_checkpoint_without_architecture_is_explained(tmp_path):
    torch.save({"model_state_dict": {}}, tmp_path / "x.pt")
    with pytest.raises(ValueError, match="load_checkpoint"):
        NeuralEncoder.load(tmp_path / "x.pt")


# ── load_encoder ────────────────────────────────────────────────────────────
def test_load_encoder_downloads_the_checkpoint_of_a_name_and_area(tmp_path, monkeypatch):
    enc = tiny_encoder()
    path = enc.save(tmp_path / "ckpt.pt")
    calls = {}

    def fake_download(repo_id, filename, revision=None):
        calls.update(repo_id=repo_id, filename=filename, revision=revision)
        return str(path)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    loaded = load_encoder("inception", area="V4", revision="abc")
    assert calls == {"repo_id": "steevelaquitaine/multigeom", "filename": "encoders/inception/V4/best_model.pt",
                     "revision": "abc"}
    torch.testing.assert_close(loaded.predict(images(2)), enc.predict(images(2)))


def test_load_encoder_needs_an_area_for_a_name_and_a_file_for_a_path(tmp_path):
    with pytest.raises(ValueError, match="area"):
        load_encoder("inception")
    with pytest.raises(FileNotFoundError):
        load_encoder(tmp_path / "missing.pt")


def test_top_level_exports_are_lazy():
    assert multigeom.NeuralEncoder is NeuralEncoder and multigeom.load_encoder is load_encoder
    assert "NeuralEncoder" in multigeom.__all__


@pytest.mark.pretrained
def test_the_paper_checkpoint_loads_by_reference_name_and_predicts():
    from multigeom.encoders import InceptionEncoder
    enc = InceptionEncoder(n_neurons=3)
    ckpt = {"model_state_dict": enc.state_dict(), "config": {"area": "V1", "encoder": "inception"}}
    assert type(NeuralEncoder.from_config({"reference": "inception", "n_neurons": 3})) is InceptionEncoder
    assert ckpt["config"]["encoder"] == "inception"


def _original_stage1_loop(readout, feats, resp, epochs, lr, wd, batch_size, val_fraction, seed):
    """The loop of the paper's stage 1 as it was written before it became a shared function (for comparison)."""
    import torch.nn.functional as F
    n = len(feats)
    n_val = int(round(val_fraction * n))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    tr_idx, val_idx = perm[: n - n_val], perm[n - n_val:]
    opt = torch.optim.AdamW(readout.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state = -float("inf"), None
    for _ in range(epochs):
        readout.train()
        p = tr_idx[torch.randperm(len(tr_idx))]
        for i in range(0, len(p), batch_size):
            idx = p[i: i + batch_size]
            loss = F.mse_loss(readout(feats[idx]), resp[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0)
            opt.step()
        readout.eval()
        with torch.no_grad():
            corr = correlation_per_neuron(readout(feats[val_idx]), resp[val_idx]).mean()
        sched.step()
        if corr > best:
            best, best_state = corr, {k: v.detach().cpu().clone() for k, v in readout.state_dict().items()}
    readout.load_state_dict(best_state)


def test_the_shared_fitting_loop_reproduces_the_original_stage1_loop():
    feats, resp = torch.randn(120, 4, 8, 8), torch.randn(120, N)
    a, b = GaussianReadout(N, 4, 8), GaussianReadout(N, 4, 8)
    b.load_state_dict(a.state_dict())
    torch.manual_seed(5)
    fit_readout_on_features(a, feats, resp, epochs=6, lr=1e-2, weight_decay=1e-4, batch_size=32, val_fraction=0.1,
                            seed=2, log=None)
    torch.manual_seed(5)
    _original_stage1_loop(b, feats, resp, 6, 1e-2, 1e-4, 32, 0.1, 2)
    for (k, va), vb in zip(a.state_dict().items(), b.state_dict().values()):
        torch.testing.assert_close(va, vb, msg=k)


def test_a_class_defined_again_replaces_its_registration():
    """Running again the notebook cell that defines a registered class must not fail; another class with the name still does."""
    from multigeom.encoders.registry import BACKBONES, register_backbone

    def define():
        @register_backbone("redefined_backbone")
        class Redefined(torch.nn.Module):
            out_channels, out_spatial = 1, 1
        return Redefined

    first, second = define(), define()
    try:
        assert BACKBONES["redefined_backbone"] is second and first is not second
        with pytest.raises(ValueError, match="already registered"):
            @register_backbone("redefined_backbone")
            class Other(torch.nn.Module):
                out_channels, out_spatial = 1, 1
    finally:
        BACKBONES.pop("redefined_backbone", None)
