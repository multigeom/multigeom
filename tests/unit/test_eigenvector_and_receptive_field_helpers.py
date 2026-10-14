"""match_eigenvectors and readout_receptive_field."""
import pytest
import torch

from multigeom.analysis import match_eigenvectors
from multigeom.encoders.receptive_field import readout_receptive_field


def test_match_eigenvectors_reorders_and_flips_the_signs():
    ref = torch.eye(4)[:3]                                  # e0, e1, e2
    tgt = torch.stack([-torch.eye(4)[1], torch.eye(4)[2], -torch.eye(4)[0], torch.eye(4)[3]])
    out = match_eigenvectors(ref, tgt, n=3, pool=3)
    torch.testing.assert_close(out, ref)


def test_match_eigenvectors_picks_from_the_pool_without_reusing():
    ref = torch.eye(3)[:2]
    tgt = torch.stack([torch.eye(3)[0], torch.eye(3)[2], torch.eye(3)[1]])
    out = match_eigenvectors(ref, tgt, n=2, pool=3)
    torch.testing.assert_close(out, ref)


def test_a_centred_neuron_covers_the_middle_of_the_image():
    state = {"readout.mu": torch.zeros(1, 2), "readout.sigma": torch.full((1,), 0.3), "readout.grid_x": torch.linspace(-1, 1, 12)}
    field, pixels = readout_receptive_field(state, resolution=64)
    assert field.shape == (64, 64) and pixels.shape == (64,)
    assert field[32, 32] > 0.9 and field[0, 0] < 0.01


def test_the_feature_side_is_needed_without_the_readout_grid():
    state = {"readout.mu": torch.zeros(1, 2), "readout.sigma": torch.full((1,), 0.3)}
    with pytest.raises(ValueError, match="feature_side"):
        readout_receptive_field(state)
    field, _ = readout_receptive_field(state, resolution=16, feature_side=12)
    assert field.shape == (16, 16)
