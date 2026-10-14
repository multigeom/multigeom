"""
Readouts: the part of an encoder that is fitted to neural data. A readout maps the
feature map of a backbone, [B, C, S, S], to one response per neuron, [B, N].

* `GaussianReadout`: each neuron looks at a Gaussian window of the feature map (the
  readout of the paper's encoders);
* `LinearReadout`: average the feature map over space, then one linear map (the simplest
  baseline);
* `FactorizedReadout`: each neuron has its own learned spatial window and its own
  feature weights.

All three take `(n_neurons, n_channels, spatial, dropout)` and are registered by name
("gaussian", "linear", "factorized"), so `NeuralEncoder(backbone, n_neurons,
readout="linear")` builds one and a saved encoder can be reloaded. Register your own
with `register_readout`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_readout


@register_readout("gaussian")
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

    def __init__(
        self, n_neurons: int, n_channels: int, spatial: int, dropout: float = 0.25
    ):
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
        gx = self.grid_x.unsqueeze(0)  # [1, S]
        gy = self.grid_y.unsqueeze(0)  # [1, S]
        mu_x = self.mu[:, 0:1]  # [N, 1]
        mu_y = self.mu[:, 1:2]  # [N, 1]
        sigma = self.sigma.unsqueeze(1).clamp(min=0.05)  # [N, 1]

        dist_x = (gx - mu_x) ** 2  # [N, S]
        dist_y = (gy - mu_y) ** 2  # [N, S]
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
        attn = self._attention()  # [N, S, S]
        feat_exp = features.unsqueeze(1)  # [B, 1, C, S, S]
        attn_exp = attn.unsqueeze(0).unsqueeze(2)  # [1, N, 1, S, S]
        pooled = (feat_exp * attn_exp).sum(dim=(-2, -1))  # [B, N, C]
        weighted = pooled * self.feature_weights.unsqueeze(0)  # [B, N, C]
        return weighted.sum(dim=-1) + self.bias  # [B, N]


@register_readout("linear")
class LinearReadout(nn.Module):
    """
    Global average pooling over space, then a linear map from the channels to the
    neurons:

        r̂_n(x) = Σ_c w_{n,c} · mean_{h,w} φ_c(x)_{h,w} + b_n

    It has no spatial selectivity, so it suits neurons that respond to the image as a
    whole. Parameters: N · (C + 1).
    """

    def __init__(
        self, n_neurons: int, n_channels: int, spatial: int, dropout: float = 0.25
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(n_channels, n_neurons)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, C, S, S] -> [B, N]"""
        return self.linear(self.dropout(features).mean(dim=(-2, -1)))


@register_readout("factorized")
class FactorizedReadout(nn.Module):
    """
    Spatial window times feature weights, both learned (Klindt et al., 2017):

        r̂_n(x) = Σ_c w_{n,c} · Σ_{h,w} a_n(h,w) · φ_c(x)_{h,w} + b_n

    with a_n = softmax over (h, w) of free logits.

    Like `GaussianReadout`, but the window of each neuron can take any shape.
    Parameters: N · (S² + C + 1).
    """

    def __init__(
        self, n_neurons: int, n_channels: int, spatial: int, dropout: float = 0.25
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.spatial_logits = nn.Parameter(
            torch.randn(n_neurons, spatial, spatial) * 0.01
        )
        self.feature_weights = nn.Parameter(torch.randn(n_neurons, n_channels) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_neurons))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, C, S, S] -> [B, N]"""
        features = self.dropout(features)
        attn = torch.softmax(self.spatial_logits.flatten(1), dim=1).view_as(
            self.spatial_logits
        )  # [N, S, S]
        pooled = torch.einsum("bchw,nhw->bnc", features, attn)  # [B, N, C]
        return (pooled * self.feature_weights.unsqueeze(0)).sum(dim=-1) + self.bias
