"""
Encoders, the models that predict neural responses from images: a backbone followed by a readout fitted to neural data.

* `NeuralEncoder` and its methods `predict`, `fit_readout`, `save`, `load`; `load_encoder` to get an encoder of the paper;
* readouts (`GaussianReadout`, `LinearReadout`, `FactorizedReadout`) and the small trainable `SmallCNNBackbone`;
* `register_backbone` and `register_readout` to add your own;
* the reference encoders of the paper (frozen Inception, ResNet-50 or ON-OFF backbone, Gaussian readout) and `build_encoder`.

Any other fitted PyTorch model can be used instead: see `multigeom.encoder` for the contract it must satisfy.
"""
from ..encoder import EncoderContractError, EncoderReport, validate_encoder
from .backbones import InceptionBackbone, OnOffBackbone, ResNet50Backbone, SmallCNNBackbone, _FrozenBackbone
from .fitting import FitReport, correlation_per_neuron, fit_readout_on_features
from .neural import NeuralEncoder, load_encoder
from .readout import FactorizedReadout, GaussianReadout, LinearReadout
from .reference import InceptionEncoder, OnOffEncoder, ResNet50Encoder, _BaseEncoder, build_encoder, build_reference
from .registry import BACKBONES, READOUTS, register_backbone, register_readout

__all__ = [
    "NeuralEncoder", "load_encoder", "validate_encoder", "EncoderContractError", "EncoderReport", "FitReport",
    "GaussianReadout", "LinearReadout", "FactorizedReadout", "SmallCNNBackbone",
    "register_backbone", "register_readout", "BACKBONES", "READOUTS",
    "InceptionBackbone", "ResNet50Backbone", "OnOffBackbone",
    "InceptionEncoder", "ResNet50Encoder", "OnOffEncoder", "build_encoder",
]
