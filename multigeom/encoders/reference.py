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
(the encoder contract of `multigeom.encoders.contract`, rule R5).

Readout parameter counts (n_neurons = 50):
  - Inception (C=192): 50 * (2 + 1 + 192 + 1) =  9,800
  - ResNet50  (C=256): 50 * (2 + 1 + 256 + 1) = 13,000
  - OnOff     (C=2):   50 * (2 + 1 +   2 + 1) =    300
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import InceptionBackbone, OnOffBackbone, ResNet50Backbone
from .neural import NeuralEncoder


class _BaseEncoder(NeuralEncoder):
    """
    Generic encoder = (frozen backbone) ∘ (normalise input) followed by
    a Gaussian spatial-attention readout.

    Subclasses set ``backbone_input_channels`` and ``backbone_input_range``;
    the canonical input format is always [B, 1, 64, 64] in [-1, 1].
    """

    backbone: nn.Module
    backbone_input_channels: int
    backbone_input_range: tuple[float, float]
    reference_name: str

    def __init__(self, backbone: nn.Module, n_neurons: int, dropout: float, **options):
        super().__init__(backbone, n_neurons, readout="gaussian", dropout=dropout)
        self._reference_options = {"dropout": dropout, **options}

    def _to_backbone_input(self, x: torch.Tensor) -> torch.Tensor:
        """Convert canonical [B, 1, 64, 64] in [-1, 1] to the backbone's format."""
        if self.backbone_input_range == (0.0, 1.0):
            x = (x + 1.0) / 2.0
            x = x.clamp(0.0, 1.0)
        if self.backbone_input_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        return x

    _prepare = _to_backbone_input

    @property
    def config(self) -> dict:
        """The architecture, by reference-encoder name (see `NeuralEncoder.config`)."""
        return {
            "reference": self.reference_name,
            "n_neurons": self.n_neurons,
            **self._reference_options,
        }


class InceptionEncoder(_BaseEncoder):
    """
    Encoder of the paper: frozen InceptionV3 backbone (truncated at `layer`) and a
    Gaussian readout.
    """

    backbone_input_channels = 3
    backbone_input_range = (-1.0, 1.0)
    reference_name = "inception"

    def __init__(
        self, n_neurons: int = 50, layer: str = "Conv2d_4a_3x3", dropout: float = 0.25
    ):
        super().__init__(
            InceptionBackbone(layer=layer), n_neurons, dropout, layer=layer
        )


class ResNet50Encoder(_BaseEncoder):
    """
    Encoder of the paper: frozen ResNet-50 backbone (truncated at `layer`) and a
    Gaussian readout.
    """

    backbone_input_channels = 3
    backbone_input_range = (-1.0, 1.0)
    reference_name = "resnet50"

    def __init__(
        self, n_neurons: int = 50, layer: str = "layer1.0", dropout: float = 0.25
    ):
        super().__init__(ResNet50Backbone(layer=layer), n_neurons, dropout, layer=layer)


class OnOffEncoder(_BaseEncoder):
    """
    Encoder of the paper: frozen ON-OFF front-end (plenoptic) and a Gaussian readout.
    """

    backbone_input_channels = 1
    backbone_input_range = (0.0, 1.0)
    reference_name = "onoff"

    def __init__(
        self, n_neurons: int = 50, kernel_size: int = 31, dropout: float = 0.25
    ):
        super().__init__(
            OnOffBackbone(kernel_size=kernel_size),
            n_neurons,
            dropout,
            kernel_size=kernel_size,
        )


REFERENCE_ENCODERS = {
    "inception": InceptionEncoder,
    "resnet50": ResNet50Encoder,
    "onoff": OnOffEncoder,
}


def build_reference(config: dict) -> _BaseEncoder:
    """
    A reference encoder (fresh readout) from its description, `{"reference":
    "inception", "n_neurons": 50, ...}`.
    """
    config = dict(config)
    name = str(config.pop("reference")).lower()
    if name not in REFERENCE_ENCODERS:
        raise ValueError(
            (
                f"unknown reference encoder {name!r}; available: "
                f"{sorted(REFERENCE_ENCODERS)}"
            )
        )
    return REFERENCE_ENCODERS[name](**config)


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
