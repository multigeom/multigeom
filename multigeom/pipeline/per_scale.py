"""
The estimates at each noise scale for a set of example images (`load_examples`), each
with its own seeded random draws: the integrand M_t(x) of G(x) (`compute_per_scale`)
and the stimulus features M_t^env(x) (`compute_env`).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from ..geometry import compute_env_per_scale, compute_g_per_scale
from ..utils import make_generator


def compute_per_scale(
    examples: dict,
    encoder: torch.nn.Module,
    unet: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
) -> dict:
    """
    The leading eigenpairs of M_t(x) at each noise level, for every example image, with
    one encoder and its diffusion model: {name: PerScaleOutput}.
    """
    # one estimate per image, with its own seeded random draws
    return {
        name: compute_g_per_scale(
            image=image,
            encoder=encoder,
            unet=unet,
            alphas_cumprod=alphas_cumprod,
            cfg=cfg,
            device=device,
            generator=make_generator(seed, device),
        )
        for name, (image, seed) in examples.items()
    }


def compute_env(
    examples: dict,
    unet: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
    params: dict,
) -> dict:
    """
    The leading eigenpairs of the stimulus features M_t^env(x) at each noise level, for
    every example image, from the null token of the diffusion model `unet`, with the
    `n_noisy` noisy images per probe and the step `eta` of `params`: {name:
    PerScaleOutput}. The seed of each image is shifted by `env_seed_offset`, so that
    its random draws are independent of those of M_t(x) (`compute_per_scale`).
    """
    # one estimate per image, with its own seeded random draws
    offset = params["env_seed_offset"]
    return {
        name: compute_env_per_scale(
            image=image,
            unet=unet,
            alphas_cumprod=alphas_cumprod,
            cfg=cfg,
            device=device,
            generator=make_generator(seed + offset, device),
            n_noisy=params["n_noisy"],
            eta=params["eta"],
        )
        for name, (image, seed) in examples.items()
    }
