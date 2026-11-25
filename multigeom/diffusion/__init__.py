"""
Diffusion (core, needs `diffusers`): the conditional diffusion model of an encoder,
everything G(x) needs besides the encoder. `DiffusionConfig` sets the conditioning width
from the encoder, `compute_denoised` gives the denoised image x̂(x_t), and
`sample_images` generates images from noise; the model, the responses of the encoder on
images and the training of the model (resumable) are in the submodules.
"""

from .config import DiffusionConfig, encoder_n_neurons
from .denoise import compute_denoised, sample_images

__all__ = [
    "DiffusionConfig",
    "compute_denoised",
    "encoder_n_neurons",
    "sample_images",
    "NeuralConditionedUNet",
    "build_diffusion_model",
    "build_scheduler",
]


def __getattr__(name):  # the model needs diffusers: import it only when it is asked for
    if name in (
        "NeuralConditionedUNet",
        "build_diffusion_model",
        "build_scheduler",
        "ConditioningMLP",
    ):
        from . import model

        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
