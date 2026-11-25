"""
Data: TinyImageNet and the canonical image transform of the pipeline (images 64 x 64,
one channel, in [-1, 1]), the functions that download the datasets (TinyImageNet, TVSD,
THINGS images), synthetic neural data with a known ground truth, and the band-pass
whitening of images.
"""

from .download import download_things_images, download_tinyimagenet, download_tvsd
from .synthetic import (
    GaborTeacher,
    SyntheticExperiment,
    make_synthetic_experiment,
    make_synthetic_neural_data,
    noise_ceiling,
)
from .tinyimagenet import TinyImageNetDataset, load_examples, load_test_images
from .transforms import CANONICAL_TRANSFORM, DEEP_TRANSFORM
from .whitening import (
    WhiteningFilter,
    estimate_power_spectrum,
    fit_power_law,
)

__all__ = [
    "CANONICAL_TRANSFORM",
    "DEEP_TRANSFORM",
    "TinyImageNetDataset",
    "load_examples",
    "load_test_images",
    "download_tinyimagenet",
    "download_tvsd",
    "download_things_images",
    "GaborTeacher",
    "make_synthetic_neural_data",
    "noise_ceiling",
    "make_synthetic_experiment",
    "SyntheticExperiment",
    "WhiteningFilter",
    "estimate_power_spectrum",
    "fit_power_law",
]
