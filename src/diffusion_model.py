"""
Conditional diffusion model: a UNet2DConditionModel from `diffusers`
augmented with a small response-to-token MLP.

The conditioning vector r ∈ ℝ^{50} is mapped through a 3-layer SiLU MLP to a
single 256-d token, which is supplied to UNet2DConditionModel as the
single-token `encoder_hidden_states` input. Classifier-free guidance dropout
on r is applied at the *training* level (Stage 3 script), not inside the
forward pass — this keeps inference / G(x) estimation independent of any
training-only randomness.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from diffusers import DDPMScheduler, UNet2DConditionModel


# ────────────────────────────────────────────────────────────────────────────
# Conditioning MLP
# ────────────────────────────────────────────────────────────────────────────
class ConditioningMLP(nn.Module):
    """
    Maps a [B, n_neurons] response vector to a [B, 1, embedding_dim] token,
    suitable as `encoder_hidden_states` for a single-token cross-attention
    block.

    Architecture: linear → SiLU → linear → SiLU → linear  (3 layers, default).
    """

    def __init__(self, layer_dims: list[int], activation: str = "silu"):
        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must have at least 2 entries")
        act_cls = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[activation.lower()]
        modules: list[nn.Module] = []
        for i in range(len(layer_dims) - 1):
            modules.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                modules.append(act_cls())
        self.mlp = nn.Sequential(*modules)
        self.in_dim = layer_dims[0]
        self.out_dim = layer_dims[-1]

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # r: [B, n_neurons]
        # → [B, embedding_dim] → unsqueeze to [B, 1, embedding_dim]
        return self.mlp(r).unsqueeze(1)


# ────────────────────────────────────────────────────────────────────────────
# Conditional UNet
# ────────────────────────────────────────────────────────────────────────────
class NeuralConditionedUNet(nn.Module):
    """
    Conditional UNet for the multi-scale-Fisher diffusion estimator.

    The UNet itself is a `UNet2DConditionModel` from `diffusers` with a single
    cross-attention block at the deepest spatial scale (matching the
    reference implementation of Laquitaine et al.). Classifier-free guidance
    is implemented at the data-loading level: with prob `cfg_dropout_rate`
    during training, the conditioning vector is replaced by zeros, so the
    network learns both p(r | x_t, r) and p(r | x_t, ∅).

    Forward signature matches Tweedie / Algorithm 1: ε̂_θ(x_t, t, r).
    """

    def __init__(self, cfg):
        super().__init__()
        u = cfg.unet
        c = cfg.conditioning

        # Block layout: matches the paper's appendix description (single
        # cross-attention block at the deepest spatial scale).
        down_blocks = ("DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")
        up_blocks = ("UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D")

        self.unet = UNet2DConditionModel(
            sample_size=u.image_size,
            in_channels=u.in_channels,
            out_channels=u.out_channels,
            layers_per_block=u.layers_per_block,
            block_out_channels=tuple(u.block_channels),
            down_block_types=down_blocks,
            up_block_types=up_blocks,
            cross_attention_dim=u.cross_attention_dim,
        )
        self.conditioning = ConditioningMLP(
            layer_dims=list(c.mlp_layers),
            activation=c.mlp_activation,
        )
        # Stash these so callers can use the same σ / dropout rate everywhere.
        self.cfg_dropout_rate = float(c.cfg_dropout_rate)
        self.observation_noise_sigma = float(c.observation_noise_sigma)
        self.n_neurons = int(c.mlp_layers[0])

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        x_t : [B, 1, 64, 64]   noisy image at DDPM step t
        t   : [B]              integer DDPM timesteps
        r   : [B, n_neurons]   conditioning vector (or zero for null token)
        returns ε̂_θ : [B, 1, 64, 64]
        """
        encoder_hidden_states = self.conditioning(r)  # [B, 1, 256]
        return self.unet(x_t, t, encoder_hidden_states=encoder_hidden_states).sample


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def build_diffusion_model(cfg) -> NeuralConditionedUNet:
    """Construct a NeuralConditionedUNet from a diffusion config namespace."""
    return NeuralConditionedUNet(cfg)


def build_scheduler(cfg) -> DDPMScheduler:
    """Construct the DDPM scheduler with the paper's linear β schedule."""
    s = cfg.scheduler
    return DDPMScheduler(
        num_train_timesteps=s.num_train_timesteps,
        beta_schedule=s.beta_schedule,
        beta_start=s.beta_start,
        beta_end=s.beta_end,
        prediction_type=s.prediction_type,
    )
