"""
Tests for multigeom.diffusion_model: conditioning MLP, conditional UNet wrapper and
DDPM scheduler. Expected values: shapes from the documented interfaces, the
linear-β closed form ᾱ_t = Π_{s≤t}(1 − β_s), and the parameter count quoted in
the paper (≈ 68.4 M with the conditioning MLP; App. "Architecture").
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("diffusers")

from conftest import REPO, linear_beta_alphas_cumprod  # noqa: E402
from multigeom.diffusion.model import (  # noqa: E402
    ConditioningMLP, NeuralConditionedUNet, build_diffusion_model, build_scheduler,
)
from multigeom.config import load_config  # noqa: E402
from multigeom.utils import count_parameters  # noqa: E402


def _tiny_cfg():
    return SimpleNamespace(
        unet=SimpleNamespace(image_size=16, in_channels=1, out_channels=1,
                             block_channels=[32, 32, 32, 32], layers_per_block=1,
                             cross_attention_dim=32),
        conditioning=SimpleNamespace(mlp_layers=[5, 32, 32, 32], mlp_activation="silu",
                                     cfg_dropout_rate=0.1, observation_noise_sigma=0.22),
    )


def test_conditioning_mlp_shapes_and_layers():
    mlp = ConditioningMLP([50, 256, 256, 256])
    assert mlp(torch.randn(7, 50)).shape == (7, 1, 256)
    linears = [m for m in mlp.mlp if isinstance(m, nn.Linear)]
    acts = [m for m in mlp.mlp if isinstance(m, nn.SiLU)]
    assert [l.in_features for l in linears] == [50, 256, 256]
    assert len(acts) == 2 and not isinstance(mlp.mlp[-1], nn.SiLU)


def test_conditioning_mlp_rejects_short_spec():
    with pytest.raises(ValueError):
        ConditioningMLP([50])


def test_unet_forward_shape_and_conditioning_is_used():
    torch.manual_seed(0)
    model = NeuralConditionedUNet(_tiny_cfg()).eval()
    x = torch.randn(3, 1, 16, 16)
    t = torch.tensor([1, 500, 999])
    with torch.no_grad():
        out_r = model(x, t, torch.randn(3, 5))
        out_0 = model(x, t, torch.zeros(3, 5))
    assert out_r.shape == (3, 1, 16, 16)
    assert not torch.allclose(out_r, out_0)
    assert model.n_neurons == 5 and model.observation_noise_sigma == pytest.approx(0.22)


def test_scheduler_matches_linear_beta_closed_form():
    sched = build_scheduler(load_config("diffusion"))
    torch.testing.assert_close(sched.alphas_cumprod.double(), linear_beta_alphas_cumprod(),
                               rtol=1e-5, atol=1e-7)


@pytest.mark.slow
def test_paper_config_parameter_count():
    model = build_diffusion_model(load_config("diffusion"))
    trainable, total = count_parameters(model)
    assert trainable == total
    assert total == pytest.approx(68.4e6, rel=0.005)
