"""
Data loaders for the THINGS Ventral-Stream Spiking Dataset (TVSD) of
Papale et al., and TinyImageNet.

Two TVSD files (MATLAB v7.3 / HDF5) are read:

  monkeyN_THINGS_normMUA.mat
    - `train_MUA`  stored as [n_train_images, n_channels] in HDF5 order
    - `test_MUA`   stored as [n_test_images,  n_channels], trial-averaged
    - `oracle`     [n_channels] split-half reliability
  things_imgs.mat
    - `train_imgs/things_path`, `test_imgs/things_path`: object references to
      the THINGS image paths, in the same order as the response rows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import data_dir


# Channel ranges within the Papale Utah-array recording (monkey N),
# inclusive on both ends (paper App. C.1: V1 0-511, V4 512-767).
AREA_CHANNELS = {
    "V1": (0, 511),       # 512 V1 channels
    "V4": (512, 767),     # 256 V4 channels
}

# Inception/ResNet input convention used in the paper:
#   [-1, 1] grayscale, replicated to 3 channels at the encoder boundary.
DEEP_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),                           # → [1, 64, 64] in [0, 1]
    transforms.Normalize(mean=[0.5], std=[0.5]),     # → [1, 64, 64] in [-1, 1]
])

# Canonical pipeline input, shared by every encoder and by the diffusion
# model: [1, 64, 64] grayscale in [-1, 1]. Encoders whose backbone needs
# another range (ResNet-50 with ImageNet statistics, OnOff in [0, 1]) convert
# internally; see src/encoders.py. Never feed [0, 1] images to an encoder.
CANONICAL_TRANSFORM = DEEP_TRANSFORM


def load_tvsd(
    area: str,
    n_neurons: int = 50,
    tvsd_dir: str | Path | None = None,
) -> dict:
    """
    Load TVSD responses + image paths and select the top-`n_neurons` channels
    in `area` by oracle reliability.

    Neurons are ordered by decreasing oracle reliability, so the first k
    columns are always the k most reliable channels.

    Returns a dict with:
        train_responses   [n_train, n_neurons]   torch.float32
        test_responses    [n_test,  n_neurons]   torch.float32
        train_paths       list[str]              relative to THINGS image dir
        test_paths        list[str]
        neuron_indices    np.ndarray             channel indices within `area`
        oracle            np.ndarray             oracle reliability per selected channel
    """
    tvsd_dir = Path(tvsd_dir) if tvsd_dir else data_dir() / "tvsd"
    mat_path = tvsd_dir / "monkeyN_THINGS_normMUA.mat"
    imgs_path = tvsd_dir / "things_imgs.mat"
    for path in (mat_path, imgs_path):
        if not path.exists():
            raise FileNotFoundError(
                f"TVSD file not found at {path}. See data/README.md for setup."
            )

    if area not in AREA_CHANNELS:
        raise ValueError(f"Unknown area {area!r}; must be one of {list(AREA_CHANNELS)}")
    ch_lo, ch_hi = AREA_CHANNELS[area]

    with h5py.File(mat_path, "r") as f:
        # MATLAB v7.3 files are column-major, so h5py returns the transpose of
        # the MATLAB array: rows are images, columns are channels.
        train_mua = np.asarray(f["train_MUA"])           # [n_train, n_channels]
        test_mua = np.asarray(f["test_MUA"])             # [n_test,  n_channels]
        oracle = np.asarray(f["oracle"]).flatten()       # [n_channels]
    with h5py.File(imgs_path, "r") as f:
        train_paths = _read_path_list(f, f["train_imgs"]["things_path"])
        test_paths = _read_path_list(f, f["test_imgs"]["things_path"])

    n_channels = oracle.shape[0]
    if train_mua.shape[1] != n_channels or test_mua.shape[1] != n_channels:
        raise ValueError(
            f"Unexpected TVSD layout: train {train_mua.shape}, test {test_mua.shape}, "
            f"oracle {oracle.shape}; expected [n_images, {n_channels}]."
        )
    if len(train_paths) != train_mua.shape[0] or len(test_paths) != test_mua.shape[0]:
        raise ValueError(
            f"Image-path count ({len(train_paths)} train / {len(test_paths)} test) "
            f"does not match response rows ({train_mua.shape[0]} / {test_mua.shape[0]})."
        )

    area_oracle = oracle[ch_lo : ch_hi + 1]
    top_local = np.argsort(area_oracle)[::-1][:n_neurons].copy()   # most reliable first
    selected = top_local + ch_lo

    return {
        "train_responses": torch.from_numpy(train_mua[:, selected].astype(np.float32)),
        "test_responses": torch.from_numpy(test_mua[:, selected].astype(np.float32)),
        "train_paths": train_paths,
        "test_paths": test_paths,
        "neuron_indices": top_local,
        "oracle": area_oracle[top_local],
    }


def _read_path_list(f: h5py.File, refs: h5py.Dataset) -> list[str]:
    """Decode an HDF5 array of object references into a list of strings."""
    paths = []
    for ref in np.asarray(refs).flatten():
        chars = np.asarray(f[ref]).flatten().astype(np.uint16)
        paths.append("".join(chr(c) for c in chars).replace("\\", "/"))
    return paths


def existing_image_mask(paths: Sequence[str], images_root: str | Path) -> np.ndarray:
    """Boolean mask of the `paths` that exist under `images_root`."""
    root = Path(images_root)
    return np.array([(root / p).exists() for p in paths], dtype=bool)


class TVSDDataset(Dataset):
    """
    Pairs THINGS images with macaque normMUA responses.

    Args:
        responses: [n_images, n_neurons] tensor.
        paths: list of image-relative paths (THINGS image-database layout).
        images_root: root directory containing THINGS images.
        transform: torchvision transform applied to PIL.Image after RGB load.
    """

    def __init__(
        self,
        responses: torch.Tensor,
        paths: Sequence[str],
        images_root: str | Path,
        transform=None,
    ):
        assert len(responses) == len(paths), \
            f"Got {len(responses)} responses but {len(paths)} paths"
        self.responses = responses
        self.paths = list(paths)
        self.images_root = Path(images_root)
        self.transform = transform if transform is not None else CANONICAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.images_root / self.paths[idx]).convert("RGB")
        return self.transform(img), self.responses[idx]


class TinyImageNetDataset(Dataset):
    """
    Single-channel 64x64 view of TinyImageNet, used as the diffusion-training
    image distribution. Paired with synthetic responses (Stage 2 output) at
    training time.
    """

    def __init__(
        self,
        split: str = "train",
        tinyimagenet_dir: str | Path | None = None,
        transform=None,
    ):
        root = Path(tinyimagenet_dir) if tinyimagenet_dir else data_dir() / "tiny-imagenet-200"
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"TinyImageNet split not found at {split_dir}; see data/README.md."
            )

        self.image_paths: list[Path] = []
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                self.image_paths.extend(sorted(cls_dir.rglob("*.JPEG")))
        if not self.image_paths:
            raise RuntimeError(f"No images found under {split_dir}")
        self.transform = transform if transform is not None else CANONICAL_TRANSFORM

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)
