"""
Data: TinyImageNet and the canonical image transform of the pipeline (images 64 x 64, one channel, in [-1, 1]), the functions that
download the datasets (TinyImageNet, TVSD, THINGS images), and
synthetic neural data with a known ground truth.
"""
from .download import download_things_images, download_tinyimagenet, download_tvsd
from .synthetic import GaborTeacher, make_synthetic_neural_data, noise_ceiling
from .tinyimagenet import CANONICAL_TRANSFORM, DEEP_TRANSFORM, TinyImageNetDataset

__all__ = ["CANONICAL_TRANSFORM", "DEEP_TRANSFORM", "TinyImageNetDataset", "download_tinyimagenet", "download_tvsd", "download_things_images",
           "GaborTeacher", "make_synthetic_neural_data", "noise_ceiling"]
