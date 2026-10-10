"""
Supporting functions of tutorials/4_bures_wasserstein.ipynb: paper Fig. 6 (appendix), the Bures-Wasserstein distance
between the reconstructed geometries of two encoders or two areas.

For two metrics A and B with truncated eigendecompositions A ~ V_A L_A V_A^T and B ~ V_B L_B V_B^T (rank k), the
distance d_BW(A, B)^2 = tr(A) + tr(B) - 2 tr((A^1/2 B A^1/2)^1/2) is computed by Stage 5 (stages/05_compute_metrics.py)
for k = 1..50, for every pair of (encoder, area) configurations that share test images, and averaged over those images.
The inputs are the per-image G(x) and J(x) eigenpairs of notebook 2; this module only inspects them, runs Stage 5
(`src.subspace_alignment.compute_metrics`), checks the ordering the paper reports, and draws the figure.

The paper's statement (Fig. 6): for the multi-scale metric G(x) the cross-model distances (same area, Inception vs
ResNet-50) lie below the cross-area distances (same model, V1 vs V4); for the Fisher information J(x) the ordering
reverses. This is the same ordering as the subspace alignment of Fig. 4D-E.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import subspace_alignment as sa

REPO = sa.REPO
PAPER_N_IMAGES = 200
AREAS = ("V1", "V4")
ENCODERS = ("inception", "resnet50")


def input_summary(runs_dir: str = "runs") -> dict:
    """
    What Stage 5 would work on: per configuration the indices of its per-image files, and per pair of configurations
    the number of shared images (the alignment and distance are averaged over those).
    """
    root = REPO / runs_dir / "per_image"
    images = {}
    for enc, area in sa.configs_with_images(runs_dir):
        images[(enc, area)] = sorted(int(f.stem[3:]) for f in (root / enc / area).glob("img*.pt"))
    configs = list(images)
    shared = {(a, b): len(set(images[a]) & set(images[b])) for i, a in enumerate(configs) for b in configs[i + 1:]}
    return {"images": images, "shared": shared}


def print_inputs(runs_dir: str = "runs", paper_n_images: int = PAPER_N_IMAGES) -> dict:
    """Print the images per configuration and warn when there are fewer than the paper's 200."""
    s = input_summary(runs_dir)
    if not s["images"]:
        print("no per-image files in runs/per_image: estimate G(x) and J(x) first (tutorial 2).")
        return s
    for (enc, area), idx in s["images"].items():
        print(f"{enc:10s} {area}: {len(idx):3d} images")
    fewest = min(s["shared"].values()) if s["shared"] else 0
    print(f"fewest images shared by two configurations: {fewest}" + (
        "" if fewest >= paper_n_images else f"  (the paper averages over {paper_n_images}: with fewer, the bands are wider)"))
    return s


def _mean_curve(metrics: dict, key: str, field: str = "bures_wasserstein_mean") -> list[float]:
    return metrics["pairs"][metrics["named"][key]][field]


def ordering(metrics: dict) -> dict:
    """
    The mean over k of d_BW for the two cross-model comparisons (V1, V4) and the two cross-area ones (Inception,
    ResNet-50), per metric, and whether the ordering is the paper's (G: cross-model below cross-area; J: reversed).
    Comparisons missing from `metrics` are left out; the result for a metric is None if either kind is missing.
    """
    out = {}
    for metric in ("G", "J"):
        model = [f"cross_model_{a}_inception_resnet50__{metric}" for a in AREAS]
        area = [f"cross_area_{e}__{metric}" for e in ENCODERS]
        mean = lambda keys: _avg([_avg(_mean_curve(metrics, k)) for k in keys if k in metrics["named"]])
        m, a = mean(model), mean(area)
        if m is None or a is None:
            out[metric] = None
            continue
        out[metric] = {"cross_model": m, "cross_area": a, "matches_paper": (m < a) if metric == "G" else (m > a)}
    return out


def _avg(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def print_ordering(metrics: dict) -> dict:
    result = ordering(metrics)
    expected = {"G": "cross-model below cross-area", "J": "cross-model above cross-area"}
    for metric, r in result.items():
        if r is None:
            print(f"{metric}: not all four comparisons are available")
            continue
        print(f"{metric}: mean d_BW over k: cross-model {r['cross_model']:.3g}, cross-area {r['cross_area']:.3g}"
              f"  -> paper expects {expected[metric]}: {'yes' if r['matches_paper'] else 'NO'}")
    return result


def load_metrics(path) -> dict:
    return json.loads(Path(path).read_text())


def plot_bures_wasserstein(metrics):
    """Fig. 6 from the Stage-5 JSON (a dict, or the path to it): d_BW against k for G(x) (left) and J(x) (right)."""
    sys.path.insert(0, str(REPO))
    from figures.fig_app_bw import make_figure
    if not isinstance(metrics, dict):
        metrics = load_metrics(metrics)
    return make_figure(metrics)
