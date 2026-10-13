"""
Calibrate the observation-noise scale σ of the Gaussian observation model

    p(r | x) = N(f_φ(x), σ² I)                       (paper Eq. (eq:obs-model))

from the trial-to-trial variability of the TVSD recordings (paper App.
"Observation model for responses"):

    "each held-out stimulus is presented 30 times, and for each electrode we
     compute the standard deviation of the z-scored per-trial response around
     the trial-averaged mean. The median of this quantity across electrodes is
     0.22 in V1 and 0.25 in V4."

Computation. `test_MUA_reps` in monkeyN_THINGS_normMUA.mat holds the
normalised MUA of every repetition of every test image; h5py returns it as
[n_trials = 30, n_test_images = 100, n_channels = 1024] (the transpose of the
MATLAB array). For channel c with per-trial responses y_{t,i,c}:

    ȳ_{i,c} = (1/T) Σ_t y_{t,i,c}                         trial-averaged mean
    s_c     = sqrt( (1/(T·I)) Σ_{t,i} (y_{t,i,c} − ȳ_{i,c})² )   pooled over
                                                          trials and images
    σ̂_area  = median_c s_c

The responses are used exactly as provided (normMUA, the per-channel
normalisation of Papale et al.); no further z-scoring is applied. `test_MUA`
in the same file is the trial average of `test_MUA_reps`, which the script
checks.

Channel set. The paper text says "for each electrode ... across electrodes"
without naming the set. σ is the noise scale of the response vector r that
the encoders predict and the diffusion model is conditioned on, i.e. the
top-50 channels of each area by oracle reliability; the original measurement
script (measure_sigma.py, `trial_only_noise`) used exactly this set. The
primary number is therefore computed on the same top-n channels that
`multigeom.fit.tvsd.load_tvsd` selects for the encoders. The median over all
channels of the area is reported alongside it for reference.

Output: JSON with per-area medians (primary: top-n channels; reference: all
area channels), per-channel standard deviations, and a provenance record.

Usage:
    python -m multigeom.fit.calibrate_sigma --out runs/analysis/sigma_calibration.json
    python -m multigeom.fit.calibrate_sigma --tvsd-dir /path/to/tvsd --n-neurons 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np


from ..config import data_dir
from ..utils import provenance
from .tvsd import AREA_CHANNELS, load_tvsd

MAT_NAME = "monkeyN_THINGS_normMUA.mat"


def parse_args(argv: list[str] | None = None):
    """Command-line arguments of `python -m multigeom.fit.calibrate_sigma`."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--tvsd-dir", default=None,
                   help="directory with monkeyN_THINGS_normMUA.mat and things_imgs.mat "
                        "(default: <data_dir>/tvsd)")
    p.add_argument("--n-neurons", type=int, default=50,
                   help="number of top-oracle channels per area (the encoder targets)")
    p.add_argument("--areas", nargs="+", default=list(AREA_CHANNELS))
    p.add_argument("--out", default="runs/analysis/sigma_calibration.json")
    return p.parse_args(argv)


def load_trial_responses(mat_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the per-trial and trial-averaged test responses.

    Returns:
        reps      [n_trials, n_test_images, n_channels]
        test_mean [n_test_images, n_channels]
    """
    with h5py.File(mat_path, "r") as f:
        reps = np.asarray(f["test_MUA_reps"], dtype=np.float64)
        test_mean = np.asarray(f["test_MUA"], dtype=np.float64)
        n_channels = np.asarray(f["oracle"]).size
    if reps.ndim != 3 or reps.shape[2] != n_channels or reps.shape[1] != test_mean.shape[0]:
        raise ValueError(
            f"Unexpected layout: test_MUA_reps {reps.shape}, test_MUA {test_mean.shape}, "
            f"{n_channels} channels; expected [n_trials, n_images, n_channels]."
        )
    return reps, test_mean


def per_channel_trial_std(reps: np.ndarray) -> np.ndarray:
    """
    s_c = RMS over (trial, image) of y_{t,i,c} − mean_t y_{t,i,c}.

    Args:
        reps: [n_trials, n_images, n_channels]
    Returns:
        [n_channels] standard deviations (population normalisation, 1/(T·I)).
    """
    resid = reps - reps.mean(axis=0, keepdims=True)
    return np.sqrt((resid ** 2).mean(axis=(0, 1)))


def calibrate(tvsd_dir: str | Path, areas: list[str], n_neurons: int) -> dict:
    """Per-area σ̂ (median over channels) on the top-n channels and on all channels."""
    tvsd_dir = Path(tvsd_dir)
    reps, test_mean = load_trial_responses(tvsd_dir / MAT_NAME)
    trial_avg_gap = float(np.max(np.abs(reps.mean(axis=0) - test_mean)))

    out = {}
    for area in areas:
        lo, hi = AREA_CHANNELS[area]
        s_area = per_channel_trial_std(reps[:, :, lo:hi + 1])         # [n_area]
        # Same channel selection as the encoders (top-n oracle, descending).
        sel = load_tvsd(area, n_neurons=n_neurons, tvsd_dir=tvsd_dir)
        local = np.asarray(sel["neuron_indices"])
        s_top = s_area[local]
        out[area] = {
            "sigma_median_top_channels": float(np.median(s_top)),
            "sigma_median_all_area_channels": float(np.median(s_area)),
            "n_top_channels": int(local.size),
            "n_area_channels": int(s_area.size),
            "top_channel_indices_global": (local + lo).tolist(),
            "per_channel_std_top": s_top.tolist(),
            "per_channel_std_all": s_area.tolist(),
            "iqr_top": [float(np.percentile(s_top, 25)), float(np.percentile(s_top, 75))],
        }
    return {
        "n_trials": int(reps.shape[0]),
        "n_test_images": int(reps.shape[1]),
        "max_abs_trial_mean_minus_test_MUA": trial_avg_gap,
        "areas": out,
    }


def main(argv: list[str] | None = None) -> dict:
    """Measure the observation noise from the TVSD repetitions and write the JSON (command-line entry point)."""
    args = parse_args(argv)
    tvsd_dir = Path(args.tvsd_dir) if args.tvsd_dir else data_dir() / "tvsd"
    result = calibrate(tvsd_dir, args.areas, args.n_neurons)
    result["definition"] = (
        "per channel: RMS over (trial, image) of the normMUA per-trial response minus "
        "its trial-averaged mean; sigma = median over channels. Primary: top-n oracle "
        "channels (encoder targets); reference: all channels of the area."
    )
    result["provenance"] = provenance(argv=argv)

    print(f"[calibrate_sigma] {result['n_trials']} trials x {result['n_test_images']} "
          f"test images; max|mean(reps) - test_MUA| = "
          f"{result['max_abs_trial_mean_minus_test_MUA']:.2e}")
    for area, r in result["areas"].items():
        print(f"  {area}: sigma (median, top-{r['n_top_channels']}) = "
              f"{r['sigma_median_top_channels']:.4f}   "
              f"(all {r['n_area_channels']} channels: {r['sigma_median_all_area_channels']:.4f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[calibrate_sigma] saved {out_path}")
    return result


if __name__ == "__main__":
    main()
