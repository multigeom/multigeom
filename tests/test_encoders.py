"""
Tests for src.encoders.

Fast tests use a tiny fake backbone subclassing `_FrozenBackbone`, so no
pretrained weights are downloaded. Tests that build the real torchvision /
plenoptic backbones are marked `pretrained`.

Expected values: the readout of Eq. (eq:readout) evaluated by explicit sums
over (c, h, w); attention normalisation and peak location from the Gaussian
definition; parameter count 50·(2 + 1 + C + 1) from the paper; feature-map
shapes from the torchvision architectures at 64×64 input (paper App. C).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.encoders import (
    GaussianReadout, InceptionEncoder, OnOffEncoder, ResNet50Encoder, _BaseEncoder,
    _FrozenBackbone,
)


class TinyBackbone(_FrozenBackbone):
    out_channels = 4
    out_spatial = 8

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=3, stride=8, padding=1)
        self.bn = nn.BatchNorm2d(4)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        return self.bn(self.conv(x))


def _stub_encoder(reference_cls):
    """An encoder with a tiny backbone and the input convention of `reference_cls`."""
    cls = type(f"Stub{reference_cls.__name__}", (_BaseEncoder,), {
        "backbone_input_channels": reference_cls.backbone_input_channels,
        "backbone_input_range": reference_cls.backbone_input_range,
    })
    return cls(TinyBackbone(reference_cls.backbone_input_channels), n_neurons=7, dropout=0.25)


# ── Gaussian readout ───────────────────────────────────────────────────────
def test_attention_sums_to_one_per_neuron():
    torch.manual_seed(0)
    ro = GaussianReadout(n_neurons=6, n_channels=3, spatial=9)
    a = ro._attention()
    assert a.shape == (6, 9, 9)
    torch.testing.assert_close(a.sum(dim=(1, 2)), torch.ones(6), atol=1e-5, rtol=0)
    assert torch.all(a >= 0)


def test_attention_peaks_at_mu():
    """μ = (x, y) in [-1, 1] grid coordinates; x indexes the last (width) axis."""
    S = 11
    ro = GaussianReadout(n_neurons=3, n_channels=1, spatial=S)
    grid = torch.linspace(-1, 1, S)
    targets = [(2, 8), (5, 5), (9, 1)]                     # (ix, iy)
    with torch.no_grad():
        ro.mu.copy_(torch.tensor([[grid[ix], grid[iy]] for ix, iy in targets]))
        ro.sigma.fill_(0.2)
    a = ro._attention()
    for n, (ix, iy) in enumerate(targets):
        h, w = divmod(int(a[n].argmax()), S)
        assert (h, w) == (iy, ix)


def test_readout_matches_explicit_sum():
    torch.manual_seed(1)
    N, C, S, B = 4, 3, 5, 2
    ro = GaussianReadout(N, C, S).eval()                    # eval: dropout off
    with torch.no_grad():
        ro.bias.normal_()
        ro.feature_weights.normal_()
    feats = torch.randn(B, C, S, S)
    a = ro._attention()
    expected = torch.empty(B, N)
    for b in range(B):
        for n in range(N):
            acc = 0.0
            for c in range(C):
                acc += ro.feature_weights[n, c] * (a[n] * feats[b, c]).sum()
            expected[b, n] = acc + ro.bias[n]
    torch.testing.assert_close(ro(feats), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("C, expected", [(192, 9_800), (256, 13_000), (2, 300)])
def test_readout_parameter_count(C, expected):
    ro = GaussianReadout(n_neurons=50, n_channels=C, spatial=4)
    assert sum(p.numel() for p in ro.parameters()) == 50 * (2 + 1 + C + 1) == expected


# ── frozen backbone ────────────────────────────────────────────────────────
def test_frozen_backbone_stays_in_eval_after_encoder_train():
    enc = _stub_encoder(InceptionEncoder)
    enc.train()
    assert enc.readout.training
    assert not enc.backbone.training
    assert not any(m.training for m in enc.backbone.modules())
    stats = enc.backbone.bn.running_mean.clone()
    enc(torch.rand(3, 1, 64, 64) * 2 - 1)
    torch.testing.assert_close(enc.backbone.bn.running_mean, stats)


def test_backbone_parameters_frozen_and_only_readout_trainable():
    enc = _stub_encoder(ResNet50Encoder)
    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("readout.") for n in trainable)


# ── input conversion ───────────────────────────────────────────────────────
@pytest.mark.parametrize("ref, channels, lo, hi", [
    (InceptionEncoder, 3, -1.0, 1.0),
    (ResNet50Encoder, 3, 0.0, 1.0),
    (OnOffEncoder, 1, 0.0, 1.0),
])
def test_to_backbone_input_range_and_channels(ref, channels, lo, hi):
    enc = _stub_encoder(ref)
    x = torch.linspace(-1, 1, 64 * 64).view(1, 1, 64, 64)
    y = enc._to_backbone_input(x)
    assert y.shape == (1, channels, 64, 64)
    assert y.min().item() == pytest.approx(lo) and y.max().item() == pytest.approx(hi)
    # affine map of the canonical image, identical across replicated channels
    expected = x if lo == -1.0 else (x + 1) / 2
    for c in range(channels):
        torch.testing.assert_close(y[:, c:c + 1], expected)


def test_forward_output_shape():
    enc = _stub_encoder(OnOffEncoder).eval()
    assert enc(torch.zeros(5, 1, 64, 64)).shape == (5, 7)


# ── real backbones (download weights) ──────────────────────────────────────
@pytest.mark.pretrained
@pytest.mark.slow
@pytest.mark.parametrize("cls, shape", [
    (InceptionEncoder, (192, 12, 12)),
    (ResNet50Encoder, (256, 16, 16)),
])
def test_pretrained_backbone_feature_shapes(cls, shape):
    enc = cls(n_neurons=50).eval()
    assert enc.backbone.out_channels == shape[0] and enc.backbone.out_spatial == shape[1]
    assert enc.features(torch.zeros(2, 1, 64, 64)).shape == (2, *shape)
    enc.train()
    assert not enc.backbone.training


@pytest.mark.pretrained
@pytest.mark.slow
def test_pretrained_onoff_feature_shape():
    pytest.importorskip("plenoptic")
    enc = OnOffEncoder(n_neurons=50).eval()
    assert enc.features(torch.zeros(2, 1, 64, 64)).shape == (2, 2, 64, 64)
