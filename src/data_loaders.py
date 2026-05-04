"""
Data loaders for the THINGS Ventral-Stream Spiking Dataset (TVSD) of
Papale et al., and TinyImageNet.

The TVSD HDF5 (`.mat`) file contains:
  - `train_MUA`              [n_channels, n_train_images]    z-scored normMUA
  - `test_MUA`               [n_channels, n_test_images]     trial-averaged
  - `train_imgs`, `test_imgs`                                image-path indices
  - `oracle`                 [n_channels]                    split-half reliability
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


# Channel ranges within the Papale Utah-array recording (monkey N).
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

# OnOff input convention: 1-channel luminance in [0, 1] (no centering).
ONOFF_TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),                           # → [1, 64, 64] in [0, 1]
])

# Canonical pipeline input: same as deep encoders ([-1, 1]).  Encoders that
# need [0, 1] (i.e., OnOff) rescale internally; see src/encoders.py.
CANONICAL_TRANSFORM = DEEP_TRANSFORM


def load_tvsd(
    area: str,
    n_neurons: int = 50,
    tvsd_dir: str | Path | None = None,
) -> dict:
    """
    Load TVSD responses + image paths and select the top-`n_neurons` channels
    in `area` by oracle reliability.

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
    if not mat_path.exists():
        raise FileNotFoundError(
            f"TVSD file not found at {mat_path}. See data/README.md for setup."
        )

    if area not in AREA_CHANNELS:
        raise ValueError(f"Unknown area {area!r}; must be one of {list(AREA_CHANNELS)}")
    ch_lo, ch_hi = AREA_CHANNELS[area]

    with h5py.File(mat_path, "r") as f:
        # Datasets are saved with column-major layout in MATLAB v7.3 HDF5.
        train_mua = np.array(f["train_MUA"]).T          # [n_train, n_channels]
        test_mua = np.array(f["test_MUA"]).T            # [n_test,  n_channels]
        oracle = np.array(f["oracle"]).flatten()        # [n_channels]
        # Image paths are stored as references to ASCII strings.
        train_paths = _read_path_list(f, "train_imgs")
        test_paths = _read_path_list(f, "test_imgs")

    area_oracle = oracle[ch_lo : ch_hi + 1]
    # Top-n_neurons indices within the area (relative offsets, 0-based).
    top_local = np.argsort(area_oracle)[::-1][:n_neurons]
    top_local.sort()
    selected = top_local + ch_lo

    train_responses = train_mua[:, selected].astype(np.float32)
    test_responses = test_mua[:, selected].astype(np.float32)

    return {
        "train_responses": torch.from_numpy(train_responses),
        "test_responses": torch.from_numpy(test_responses),
        "train_paths": train_paths,
        "test_paths": test_paths,
        "neuron_indices": top_local,
        "oracle": area_oracle[top_local],
    }


def _read_path_list(f: h5py.File, key: str) -> list[str]:
    """Decode an HDF5 array of object references into a list of strings."""
    paths = []
    refs = f[key]
    for ref in np.array(refs).flatten():
        chars = np.array(f[ref]).flatten().astype(np.uint16)
        paths.append("".join(chr(c) for c in chars))
    return paths


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
        rel = self.paths[idx].replace("\\", "/")
        img = Image.open(self.images_root / rel).convert("RGB")
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
