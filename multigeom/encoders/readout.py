"""
The per-neuron Gaussian spatial-attention readout shared by the reference encoders (the part of an encoder that is
fitted to neural data); see `multigeom.encoders.reference` for the encoders built from it.
"""
from __future__ import annotations

import torch
import torch.nn as nn


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


