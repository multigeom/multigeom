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
(requires_grad = False). `forward` runs the backbone under `torch.no_grad()`
for speed; `src.fisher` calls the backbone directly so that gradients with
respect to the input image can flow.

Readout parameter counts (n_neurons = 50):
  - Inception (C=192): 50 * (2 + 1 + 192 + 1) =  9,800
  - ResNet50  (C=256): 50 * (2 + 1 + 256 + 1) = 13,000
  - OnOff     (C=2):   50 * (2 + 1 +   2 + 1) =    300
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import inception_v3, resnet50, Inception_V3_Weights, ResNet50_Weights


# ────────────────────────────────────────────────────────────────────────────
# Gaussian spatial-attention readout (shared across encoders)
# ────────────────────────────────────────────────────────────────────────────
class GaussianReadout(nn.Module):
    """
    Per-neuron Gaussian spatial-attention readout (Eq. (eq:readout) in the
    paper):

        r̂_n(x) = Σ_c w_{n,c} · Σ_{h,w} a_n(h,w) · φ_c(x)_{h,w} + b_n,
        a_n(h,w) ∝ exp( -‖(h,w) - μ_n‖² / (2 σ_n²) ),

    with μ_n ∈ ℝ², σ_n > 0 in normalised [-1, 1] grid coordinates, and
    w_n ∈ ℝ^C, b_n ∈ ℝ.

    Trainable parameters: 50 * (2 + 1 + C + 1).
    """

    def __init__(self, n_neurons: int, n_channels: int, spatial: int, dropout: float = 0.25):
        super().__init__()
        self.n_neurons = n_neurons
        self.mu = nn.Parameter(torch.randn(n_neurons, 2) * 0.3)
        self.sigma = nn.Parameter(torch.ones(n_neurons) * 0.5)
        self.feature_weights = nn.Parameter(torch.randn(n_neurons, n_channels) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_neurons))
        self.dropout = nn.Dropout(p=dropout)

        grid = torch.linspace(-1, 1, spatial)
        self.register_buffer("grid_x", grid)
        self.register_buffer("grid_y", grid)

    def _attention(self) -> torch.Tensor:
        """Compute per-neuron attention masks. Returns [N, S, S], rows sum to 1."""
        gx = self.grid_x.unsqueeze(0)                       # [1, S]
        gy = self.grid_y.unsqueeze(0)                       # [1, S]
        mu_x = self.mu[:, 0:1]                              # [N, 1]
        mu_y = self.mu[:, 1:2]                              # [N, 1]
        sigma = self.sigma.unsqueeze(1).clamp(min=0.05)     # [N, 1]

        dist_x = (gx - mu_x) ** 2                           # [N, S]
        dist_y = (gy - mu_y) ** 2                           # [N, S]
        attn = torch.exp(
            -(dist_y.unsqueeze(2) + dist_x.unsqueeze(1)) / (2 * sigma.unsqueeze(1) ** 2)
        )
        return attn / (attn.sum(dim=(1, 2), keepdim=True) + 1e-8)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features : [B, C, S, S]
        returns  : [B, N]
        """
        features = self.dropout(features)
        attn = self._attention()                                  # [N, S, S]
        feat_exp = features.unsqueeze(1)                          # [B, 1, C, S, S]
        attn_exp = attn.unsqueeze(0).unsqueeze(2)                 # [1, N, 1, S, S]
        pooled = (feat_exp * attn_exp).sum(dim=(-2, -1))          # [B, N, C]
        weighted = (pooled * self.feature_weights.unsqueeze(0))   # [B, N, C]
        return weighted.sum(dim=-1) + self.bias                   # [B, N]


# ────────────────────────────────────────────────────────────────────────────
# Backbones
# ────────────────────────────────────────────────────────────────────────────
class _FrozenBackbone(nn.Module):
    """Backbone that stays in eval mode (BatchNorm statistics fixed) even when
    a parent module is switched to training mode."""

    def train(self, mode: bool = True):  # noqa: A003 (shadows nn.Module.train)
        return super().train(False)


class InceptionBackbone(_FrozenBackbone):
    """
    Frozen InceptionV3 truncated at `Conv2d_4a_3x3`.

    Expects 3-channel input in [-1, 1] (after the encoder's input normalisation).
    Produces a [B, 192, 12, 12] feature map for 64x64 input.
    """

    out_channels = 192
    out_spatial = 12

    def __init__(self, layer: str = "Conv2d_4a_3x3"):
        super().__init__()
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        # Build a sequential of the modules up to and including `layer`.
        self.features = nn.Sequential()
        for name, mod in net.named_children():
            self.features.add_module(name, mod)
            if name == layer:
                break
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 64, 64] in [-1, 1] (Inception accepts ImageNet-normalised
        # inputs at any reasonable resolution; the 64x64 case is undocumented
        # but well-defined and is the choice made in the paper for parity
        # with the diffusion model).
        return self.features(x)


class ResNet50Backbone(_FrozenBackbone):
    """
    Frozen ResNet-50 truncated at `layer1.0`.

    Standardises its 3-channel input with the ImageNet mean and standard
    deviation internally (paper App. C.3). The encoder passes the canonical
    [-1, 1] image, as in the paper (images are rescaled to [-1, 1] for both
    deep encoders) and in the original training code, so the released ResNet
    encoders reproduce their original responses exactly. Produces
    [B, 256, 16, 16] for 64x64 input.
    """

    out_channels = 256
    out_spatial = 16
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, layer: str = "layer1.0"):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential()
        # ResNet's children up to the requested layer.
        target_block, target_idx = layer.split(".")
        for name, mod in net.named_children():
            if name == target_block:
                # Only include modules up to target_idx within this block
                sub_seq = nn.Sequential()
                for sub_name, sub_mod in mod.named_children():
                    sub_seq.add_module(sub_name, sub_mod)
                    if sub_name == target_idx:
                        break
                self.features.add_module(name, sub_seq)
                break
            self.features.add_module(name, mod)
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("norm_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features((x - self.norm_mean) / self.norm_std)


class OnOffBackbone(_FrozenBackbone):
    """
    Frozen Berardino-2017 ON-OFF front-end from the `plenoptic` library.

    Expects single-channel luminance input in [0, 1]. Produces [B, 2, 64, 64].
    The 12 internal scalars (luminance/contrast gain, centre/surround radii,
    amplitude ratio) are loaded from the pretrained Berardino weights and
    held fixed throughout training.
    """

    out_channels = 2
    out_spatial = 64

    def __init__(self, kernel_size: int = 31):
        super().__init__()
        try:
            from plenoptic.simulate.models.frontend import OnOff   # plenoptic 1.x
        except ImportError:
            from plenoptic.models import OnOff                     # plenoptic 2.x
        self.frontend = OnOff(kernel_size=kernel_size, pretrained=True, cache_filt=True)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        # With cache_filt=True plenoptic registers its filters as `_filt`
        # buffers on the first forward pass. Run one now so that a freshly
        # built encoder has the same state_dict keys as a trained checkpoint.
        with torch.no_grad():
            self.frontend(torch.zeros(1, 1, 64, 64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.frontend(x)


# ────────────────────────────────────────────────────────────────────────────
# Encoders (backbone + readout, with input normalisation)
# ────────────────────────────────────────────────────────────────────────────
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
        with torch.no_grad():
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
