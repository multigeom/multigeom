"""
Reference encoders (extra `[vision]`): frozen InceptionV3 / ResNet-50 / ON-OFF backbones with a per-neuron Gaussian
spatial-attention readout, as used in the paper, and `build_encoder` to create one from a config.

Any other fitted PyTorch model can be used instead: see `multigeom.encoder` for the contract it must satisfy.
"""
from .backbones import InceptionBackbone, OnOffBackbone, ResNet50Backbone, _FrozenBackbone
from .readout import GaussianReadout
from .reference import InceptionEncoder, OnOffEncoder, ResNet50Encoder, _BaseEncoder, build_encoder

__all__ = [
    "GaussianReadout", "InceptionBackbone", "ResNet50Backbone", "OnOffBackbone",
    "InceptionEncoder", "ResNet50Encoder", "OnOffEncoder", "build_encoder",
]
