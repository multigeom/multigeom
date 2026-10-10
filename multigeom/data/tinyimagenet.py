"""
TinyImageNet (the diffusion-training and test images) and the canonical image transform of the pipeline.

Images are 64 x 64, single-channel, in [-1, 1]. Expected layout, under `data_dir()` (see `multigeom.config`):

    tiny-imagenet-200/train/<wnid>/images/<image>.JPEG    (100,000 images: training of the diffusion model)
    tiny-imagenet-200/val/images/<image>.JPEG             (10,000 images; the first 200, in sorted order, are the test images)
"""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..config import data_dir


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
# internally; see multigeom/encoders. Never feed [0, 1] images to an encoder.
CANONICAL_TRANSFORM = DEEP_TRANSFORM


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
