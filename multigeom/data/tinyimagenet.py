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

    Args:
        split: "train" or "val".
        tinyimagenet_dir: the dataset folder (default: `tiny-imagenet-200` under `data_dir()`).
        transform: applied to each PIL image (default: the canonical transform).
        images_per_class: keep only the first this many images of each class (sorted order), to get a small subset.
        return_labels: return `(image, label)` with the class index 0..199 (the order of `wnids.txt`) instead of the image.
    """

    def __init__(
        self,
        split: str = "train",
        tinyimagenet_dir: str | Path | None = None,
        transform=None,
        images_per_class: int | None = None,
        return_labels: bool = False,
    ):
        root = Path(tinyimagenet_dir) if tinyimagenet_dir else data_dir() / "tiny-imagenet-200"
        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"TinyImageNet split not found at {split_dir}; see data/README.md."
            )

        self.image_paths: list[Path] = []
        if split == "val" and images_per_class is not None:
            raise ValueError("images_per_class is only available for the train split")
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                found = sorted(cls_dir.rglob("*.JPEG"))
                self.image_paths.extend(found if images_per_class is None else found[:images_per_class])
        if not self.image_paths:
            raise RuntimeError(f"No images found under {split_dir}")
        self.transform = transform if transform is not None else CANONICAL_TRANSFORM
        self.return_labels = return_labels
        self.labels: list[int] | None = self._read_labels(root, split) if return_labels else None

    def _read_labels(self, root: Path, split: str) -> list[int]:
        wnids_file = root / "wnids.txt"
        if wnids_file.exists():
            wnids = [w.strip() for w in wnids_file.read_text().split() if w.strip()]
        else:
            wnids = sorted(d.name for d in (root / "train").iterdir() if d.is_dir())
        index = {w: i for i, w in enumerate(wnids)}
        if split == "train":
            return [index[p.parent.parent.name] for p in self.image_paths]
        annotations = root / "val" / "val_annotations.txt"
        if not annotations.exists():
            raise FileNotFoundError(f"{annotations} is needed for the labels of the validation images")
        wnid_of = dict(line.split("\t")[:2] for line in annotations.read_text().splitlines() if line.strip())
        return [index[wnid_of[p.name]] for p in self.image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        x = self.transform(img)
        return (x, self.labels[idx]) if self.return_labels else x
