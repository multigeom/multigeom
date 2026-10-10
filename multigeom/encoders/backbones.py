"""
Frozen feature extractors of the reference encoders: InceptionV3 and ResNet-50 (ImageNet weights, torchvision) and
the ON-OFF front-end of Berardino et al. (2017) (plenoptic), truncated at a chosen layer.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import inception_v3, resnet50, Inception_V3_Weights, ResNet50_Weights


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


