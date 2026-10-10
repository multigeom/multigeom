"""Data: TinyImageNet and the canonical image transform of the pipeline (images 64 x 64, one channel, in [-1, 1])."""
from .tinyimagenet import CANONICAL_TRANSFORM, DEEP_TRANSFORM, TinyImageNetDataset

__all__ = ["CANONICAL_TRANSFORM", "DEEP_TRANSFORM", "TinyImageNetDataset"]
