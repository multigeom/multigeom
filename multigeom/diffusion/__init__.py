"""
Diffusion (core, needs `diffusers`): the conditional diffusion model of an encoder, everything G(x) needs besides the
encoder. `DiffusionConfig` sets the conditioning width from the encoder; the model, the responses of the encoder on
images and the training of the model (resumable) are in the submodules.
"""
from .config import DiffusionConfig, encoder_n_neurons

__all__ = ["DiffusionConfig", "encoder_n_neurons", "NeuralConditionedUNet", "build_diffusion_model", "build_scheduler"]


def __getattr__(name):                       # the model needs diffusers: import it only when it is asked for
    if name in ("NeuralConditionedUNet", "build_diffusion_model", "build_scheduler", "ConditioningMLP"):
        from . import model
        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
