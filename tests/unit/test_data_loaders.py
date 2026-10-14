"""
Tests for multigeom.data_loaders on a synthetic MATLAB-v7.3-like TVSD pair
(write_fake_tvsd in tests/conftest.py). The expected channel selection is computed
here from the oracle vector by sorting; the expected responses are read off
the known construction train_MUA[i, c] = 1000·i + c.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import write_fake_tvsd
from multigeom.datasets import CANONICAL_TRANSFORM
from multigeom.fit.tvsd import AREA_CHANNELS, TVSDDataset, existing_image_mask, load_tvsd

# Old h5py releases call np.product, deprecated in NumPy 1.25.
pytestmark = pytest.mark.filterwarnings("ignore:`product` is deprecated:DeprecationWarning")


@pytest.fixture
def fake(tmp_path):
    return tmp_path, write_fake_tvsd(tmp_path)


@pytest.mark.parametrize("area", ["V1", "V4"])
def test_top_n_channels_by_oracle_descending(fake, area):
    d, truth = fake
    lo, hi = AREA_CHANNELS[area]
    out = load_tvsd(area, n_neurons=10, tvsd_dir=d)
    area_oracle = truth["oracle"][lo:hi + 1]
    expected_local = sorted(range(hi - lo + 1), key=lambda i: -area_oracle[i])[:10]
    np.testing.assert_array_equal(out["neuron_indices"], expected_local)
    np.testing.assert_allclose(out["oracle"], area_oracle[expected_local])
    assert np.all(np.diff(out["oracle"]) < 0)


@pytest.mark.parametrize("area", ["V1", "V4"])
def test_responses_are_selected_channels(fake, area):
    d, truth = fake
    lo, _ = AREA_CHANNELS[area]
    out = load_tvsd(area, n_neurons=10, tvsd_dir=d)
    glob = out["neuron_indices"] + lo
    assert out["train_responses"].shape == (9, 10) and out["test_responses"].shape == (4, 10)
    assert out["train_responses"].dtype == torch.float32
    expected = 1000.0 * np.arange(9)[:, None] + glob[None, :]
    np.testing.assert_array_equal(out["train_responses"].numpy(), expected)
    np.testing.assert_allclose(out["test_responses"].numpy(),
                               truth["test_mean"][:, glob].astype(np.float32))


def test_paths_decoded_with_forward_slashes(fake):
    d, truth = fake
    out = load_tvsd("V1", n_neurons=5, tvsd_dir=d)
    for split in ("train", "test"):
        assert out[f"{split}_paths"] == [p.replace("\\", "/") for p in truth["paths"][split]]
    assert out["train_paths"][3] == "cat03/cat03_03s.jpg"


def test_matlab_orientation_is_rejected(tmp_path):
    write_fake_tvsd(tmp_path, transpose=True)
    with pytest.raises(ValueError):
        load_tvsd("V1", n_neurons=5, tvsd_dir=tmp_path)


def test_unknown_area_and_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tvsd("V1", tvsd_dir=tmp_path)
    write_fake_tvsd(tmp_path)
    with pytest.raises(ValueError):
        load_tvsd("IT", tvsd_dir=tmp_path)


def test_existing_image_mask(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.jpg").write_bytes(b"")
    mask = existing_image_mask(["a/x.jpg", "a/y.jpg", "b/x.jpg"], tmp_path)
    np.testing.assert_array_equal(mask, [True, False, False])
    assert mask.dtype == bool


def test_canonical_transform_range_and_dataset_item(tmp_path):
    img = np.zeros((100, 80, 3), dtype=np.uint8)
    img[:, 40:] = 255
    Image.fromarray(img).save(tmp_path / "half.png")
    ds = TVSDDataset(torch.arange(3.0).view(1, 3), ["half.png"], tmp_path)
    x, r = ds[0]
    assert x.shape == (1, 64, 64)
    assert x.min().item() == pytest.approx(-1.0) and x.max().item() == pytest.approx(1.0)
    torch.testing.assert_close(r, torch.arange(3.0))
    assert CANONICAL_TRANSFORM is not None
