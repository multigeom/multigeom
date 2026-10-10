"""
Image-to-response encoders f_φ : x ↦ r̂ used throughout the paper.

Each encoder is a frozen feature-extracting backbone followed by a per-neuron
Gaussian spatial-attention readout (the same readout structure across all
three encoders, as discussed in Appendix C / D of the paper).

The canonical input is a [B, 1, 64, 64] tensor in [-1, 1] (matching the
diffusion-process input convention).  Each encoder rescales / replicates
channels internally to its backbone's expected input format.

  - InceptionEncoder: [-1, 1] → 3-channel → InceptionV3 → readout
                      (torchvision's Inception weights expect [-1, 1] input)
  - ResNet50Encoder:  [-1, 1] → 3-channel → ImageNet mean/std
                      standardisation → ResNet50 → readout
  - OnOffEncoder:     [-1, 1] → [0, 1] → 1-channel → plenoptic OnOff → readout

Trainable parameters live only in the readout. Backbone parameters are frozen
(requires_grad = False), so `forward` builds no graph for them unless the input
requires grad, in which case gradients with respect to the input image flow
(the encoder contract of `multigeom.encoder`, rule R5).

Readout parameter counts (n_neurons = 50):
  - Inception (C=192): 50 * (2 + 1 + 192 + 1) =  9,800
  - ResNet50  (C=256): 50 * (2 + 1 + 256 + 1) = 13,000
  - OnOff     (C=2):   50 * (2 + 1 +   2 + 1) =    300
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import InceptionBackbone, OnOffBackbone, ResNet50Backbone
from .readout import GaussianReadout


class _BaseEncoder(nn.Module):
    """
    Generic encoder = (frozen backbone) ∘ (normalise input) followed by
    a Gaussian spatial-attention readout.

    Subclasses set ``backbone_input_channels`` and ``backbone_input_range``;
    the canonical input format is always [B, 1, 64, 64] in [-1, 1].
    """

    backbone: nn.Module
    backbone_input_channels: int
    backbone_input_range: tuple[float, float]

    def __init__(self, backbone: nn.Module, n_neurons: int, dropout: float):
        super().__init__()
        self.backbone = backbone
        self.n_neurons = n_neurons
        self.readout = GaussianReadout(
            n_neurons=n_neurons,
            n_channels=backbone.out_channels,
            spatial=backbone.out_spatial,
            dropout=dropout,
        )

    def _to_backbone_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert canonical [B, 1, 64, 64] in [-1, 1] to the backbone's format."""
        if self.backbone_input_range == (0.0, 1.0):
            x = (x + 1.0) / 2.0
            x = x.clamp(0.0, 1.0)
        if self.backbone_input_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, 1, 64, 64] in [-1, 1] (canonical pipeline input).
        returns : [B, n_neurons]
        """
        x_bb = self._to_backbone_input(x)
        # No explicit torch.no_grad(): the backbone is frozen, so no graph is built for it unless the input
        # requires grad, and then the gradient w.r.t. the input flows (the encoder contract, rule R5).
        features = self.backbone(x_bb)
        return self.readout(features)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the frozen backbone features only (no readout). Useful for caching."""
        with torch.no_grad():
            return self.backbone(self._to_backbone_input(x))


class InceptionEncoder(_BaseEncoder):
    backbone_input_channels = 3
    backbone_input_range = (-1.0, 1.0)

    def __init__(self, n_neurons: int = 50, layer: str = "Conv2d_4a_3x3", dropout: float = 0.25):
        super().__init__(InceptionBackbone(layer=layer), n_neurons, dropout)


class ResNet50Encoder(_BaseEncoder):
    backbone_input_channels = 3
    backbone_input_range = (-1.0, 1.0)

    def __init__(self, n_neurons: int = 50, layer: str = "layer1.0", dropout: float = 0.25):
        super().__init__(ResNet50Backbone(layer=layer), n_neurons, dropout)


class OnOffEncoder(_BaseEncoder):
    backbone_input_channels = 1
    backbone_input_range = (0.0, 1.0)

    def __init__(self, n_neurons: int = 50, kernel_size: int = 31, dropout: float = 0.25):
        super().__init__(OnOffBackbone(kernel_size=kernel_size), n_neurons, dropout)


# ────────────────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────────────────
def build_encoder(cfg) -> _BaseEncoder:
    """
    Construct an encoder from a config namespace. Used by every stage script
    so the same `configs/<encoder>.yaml` controls every appearance of the
    encoder in the pipeline.
    """
    name = cfg.encoder.name.lower()
    if name == "inception":
        return InceptionEncoder(
            n_neurons=cfg.encoder.n_neurons,
            layer=cfg.encoder.layer,
            dropout=cfg.encoder.dropout,
        )
    if name == "resnet50":
        return ResNet50Encoder(
            n_neurons=cfg.encoder.n_neurons,
            layer=cfg.encoder.layer,
            dropout=cfg.encoder.dropout,
        )
    if name == "onoff":
        return OnOffEncoder(
            n_neurons=cfg.encoder.n_neurons,
            kernel_size=cfg.encoder.kernel_size,
            dropout=cfg.encoder.dropout,
        )
    raise ValueError(f"Unknown encoder name: {name!r}")
