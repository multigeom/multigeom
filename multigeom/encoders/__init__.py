"""
Encoders, the models that predict neural responses from images: a backbone followed by a readout fitted to neural data.

* `NeuralEncoder` and its methods `predict`, `fit_readout`, `save`, `load`; `load_encoder` to get an encoder of the paper;
* readouts (`GaussianReadout`, `LinearReadout`, `FactorizedReadout`) and the small trainable `SmallCNNBackbone`;
* `register_backbone` and `register_readout` to add your own;
* the reference encoders of the paper (frozen Inception, ResNet-50 or ON-OFF backbone, Gaussian readout) and `build_encoder`.

Any other fitted PyTorch model can be used instead: see `multigeom.encoders.contract` for the contract it must satisfy.
"""
from .contract import EncoderContractError, EncoderReport, validate_encoder
from .registry import BACKBONES, READOUTS, register_backbone, register_readout

__all__ = [
    "NeuralEncoder", "load_encoder", "validate_encoder", "EncoderContractError", "EncoderReport", "FitReport",
    "GaussianReadout", "LinearReadout", "FactorizedReadout", "SmallCNNBackbone",
    "register_backbone", "register_readout", "BACKBONES", "READOUTS",
    "InceptionBackbone", "ResNet50Backbone", "OnOffBackbone",
    "InceptionEncoder", "ResNet50Encoder", "OnOffEncoder", "build_encoder",
]

_LAZY = {                                    # torchvision is imported only when the encoders themselves are asked for
    "backbones": ("InceptionBackbone", "OnOffBackbone", "ResNet50Backbone", "SmallCNNBackbone", "_FrozenBackbone"),
    "fitting": ("FitReport", "correlation_per_neuron", "fit_readout_on_features"),
    "neural": ("NeuralEncoder", "load_encoder"),
    "readout": ("FactorizedReadout", "GaussianReadout", "LinearReadout"),
    "reference": ("InceptionEncoder", "OnOffEncoder", "ResNet50Encoder", "_BaseEncoder", "build_encoder", "build_reference"),
}
_MODULE_OF = {name: module for module, names in _LAZY.items() for name in names}


def __getattr__(name):
    if name in _MODULE_OF:
        from importlib import import_module
        value = getattr(import_module(f".{_MODULE_OF[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
