"""
Configuration of the conditional diffusion model of an encoder.

The diffusion model that G(x) needs is conditioned on the responses of one encoder, so one model is trained per encoder.
Everything in its configuration is shared (UNet, scheduler, training recipe) except the width of the first layer of the
conditioning network, which is the number of neurons of the encoder: `DiffusionConfig.for_encoder` sets it.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace

import torch

from ..config import load_config


def encoder_n_neurons(encoder, image_shape=(1, 64, 64)) -> int:
    """The number of neurons (output width) of an encoder: its `n_neurons` if it has one, else the width of its output."""
    declared = getattr(encoder, "n_neurons", None)
    if declared is not None:
        return int(declared)
    param = next(iter(encoder.parameters()), None)
    device = param.device if param is not None else torch.device("cpu")
    was_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            out = encoder(torch.zeros(1, *image_shape, device=device))
    finally:
        encoder.train(was_training)
    return int(out.shape[1])


class DiffusionConfig:
    """Namespace of the diffusion-model configuration (`multigeom/configs/diffusion.yaml`) and its per-encoder version."""

    @staticmethod
    def default() -> SimpleNamespace:
        """The packaged configuration (50 neurons)."""
        return load_config("diffusion")

    @staticmethod
    def for_encoder(encoder, base: SimpleNamespace | None = None, image_shape=(1, 64, 64)) -> SimpleNamespace:
        """
        A copy of `base` (default: the packaged configuration) whose conditioning network takes the responses of
        `encoder`: `conditioning.mlp_layers[0]` is set to its number of neurons. The input is not modified.
        """
        cfg = copy.deepcopy(base if base is not None else DiffusionConfig.default())
        n = encoder_n_neurons(encoder, image_shape)
        if n < 1:
            raise ValueError(f"the encoder has no neurons ({n})")
        cfg.conditioning.mlp_layers = [n] + list(cfg.conditioning.mlp_layers[1:])
        return cfg
