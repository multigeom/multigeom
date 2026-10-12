"""
Data: TinyImageNet and the canonical image transform of the pipeline (images 64 x 64, one channel, in [-1, 1]), and
synthetic neural data with a known ground truth.
"""
from .synthetic import GaborTeacher, make_synthetic_neural_data, noise_ceiling
from .tinyimagenet import CANONICAL_TRANSFORM, DEEP_TRANSFORM, TinyImageNetDataset

__all__ = ["CANONICAL_TRANSFORM", "DEEP_TRANSFORM", "TinyImageNetDataset",
           "GaborTeacher", "make_synthetic_neural_data", "noise_ceiling"]
