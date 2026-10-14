"""
Stage 5: Aggregate cross-(encoder, area) similarity metrics from the
per-image G(x) and J(x) eigendecompositions written by Stage 4.

For every unordered pair of (encoder, area) configurations, for both metrics
(G = multi-scale Fisher, J = Fisher), and on every image the two share, we
compute as a function of k = 1, ..., max_k:

  - the subspace alignment 𝒜_k = (1/k)‖V_Aᵀ V_B‖_F²     (Fig. 4D-E;
    same definition as multigeom.analysis.subspace_alignment)
  - the Bures-Wasserstein distance d_BW between the rank-k
    reconstructions V Λ Vᵀ of the two metrics            (appendix BW figure)

and report the mean and s.e.m. across images.

The paper compares models fitted to the same area ("cross_model") with the
same model fitted to different areas ("cross_area"). Those comparisons are
also listed under named keys in the output, e.g.

    cross_model_V1_inception_resnet50,  cross_area_inception

Output: <out_dir>/cross_encoder_metrics.json

Usage:
    python -m multigeom.pipeline.alignment --runs-dir runs --out-dir runs/metrics
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch


from ..config import project_root
from ..analysis.metrics import bures_wasserstein_from_overlap
from ..utils import provenance, subprocess_env

ENCODER_ORDER = ("inception", "resnet50", "onoff")


def parse_args(argv=None):
    """Command-line arguments of `python -m multigeom.pipeline.alignment`."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs-dir",
        required=True,
        help="root containing per_image/<encoder>/<area>/imgXXXX.pt files",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--max-k",
        type=int,
        default=50,
        help="largest k for the alignment and Bures-Wasserstein curves",
    )
    return p.parse_args(argv)


def discover_configs(per_image_root: Path) -> list[tuple[str, str]]:
    """(encoder, area) pairs with at least one per-image file, in a fixed order."""
    if not per_image_root.exists():
        raise FileNotFoundError(f"{per_image_root} does not exist; run Stage 4 first.")
    configs = [
        (enc_dir.name, area_dir.name)
        for enc_dir in per_image_root.iterdir()
        if enc_dir.is_dir()
        for area_dir in enc_dir.iterdir()
        if area_dir.is_dir() and any(area_dir.glob("img*.pt"))
    ]

    def order(c):
        """Sort key of a configuration: its encoder, then its area."""
        enc, area = c
        rank = ENCODER_ORDER.index(enc) if enc in ENCODER_ORDER else len(ENCODER_ORDER)
        return (rank, enc, area)

    return sorted(configs, key=order)


def load_per_image(per_image_root: Path, encoder: str, area: str) -> dict[int, dict]:
    """Load every img{N}.pt for one (encoder, area), keyed by image index."""
    out = {}
    for f in sorted((per_image_root / encoder / area).glob("img*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        out[int(d["img_idx"])] = d
    return out


def _mean_sem(arr: np.ndarray) -> tuple[list[float], list[float]]:
    """Mean and s.e.m. over axis 0 (images)."""
    n = arr.shape[0]
    sem = arr.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(arr.shape[1])
    return arr.mean(axis=0).tolist(), sem.tolist()


def aggregate_pair(A: dict[int, dict], B: dict[int, dict], metric: str, max_k: int):
    """Alignment and Bures-Wasserstein curves for k = 1..K over shared images."""
    vecs, vals = f"{metric}_eigenvectors", f"{metric}_eigenvalues"
    shared = [
        i
        for i in sorted(set(A) & set(B))
        if A[i][vecs] is not None and B[i][vecs] is not None
    ]
    if not shared:
        return None
    K = min(
        max_k, *(len(A[i][vals]) for i in shared), *(len(B[i][vals]) for i in shared)
    )
    ks = list(range(1, K + 1))

    align = np.empty((len(shared), K))
    bw = np.empty((len(shared), K))
    for row, i in enumerate(shared):
        # Overlap of the top-K eigenvectors, computed once and sliced per k.
        C = A[i][vecs][:K].double() @ B[i][vecs][:K].double().T  # [K, K]
        for k in ks:
            Ck = C[:k, :k]
            align[row, k - 1] = float((Ck**2).sum()) / k
            bw[row, k - 1] = bures_wasserstein_from_overlap(
                A[i][vals][:k], B[i][vals][:k], Ck
            )
    align_mean, align_sem = _mean_sem(align)
    bw_mean, bw_sem = _mean_sem(bw)
    return {
        "n_images": len(shared),
        "k": ks,
        "alignment_mean": align_mean,
        "alignment_sem": align_sem,
        "bures_wasserstein_mean": bw_mean,
        "bures_wasserstein_sem": bw_sem,
    }


def pair_label(a: tuple[str, str], b: tuple[str, str], metric: str) -> str:
    """
    Name of the comparison of two (encoder, area) configurations for one metric (`G` or
    `J`).
    """
    return f"{a[0]}_{a[1]}__VS__{b[0]}_{b[1]}__{metric}"


def named_comparisons(configs: list[tuple[str, str]]) -> dict[str, tuple]:
    """The paper's comparisons: same model across areas, same area across models."""
    named = {}
    encoders = sorted(
        {e for e, _ in configs},
        key=lambda e: (
            ENCODER_ORDER.index(e) if e in ENCODER_ORDER else len(ENCODER_ORDER),
            e,
        ),
    )
    areas = sorted({a for _, a in configs})
    for enc in encoders:
        if (enc, "V1") in configs and (enc, "V4") in configs:
            named[f"cross_area_{enc}"] = ((enc, "V1"), (enc, "V4"))
    for area in areas:
        for ea, eb in combinations([e for e in encoders if (e, area) in configs], 2):
            named[f"cross_model_{area}_{ea}_{eb}"] = ((ea, area), (eb, area))
    return named


def main(argv=None):
    """
    Compare every pair of configurations and write the metrics file (command-line entry
    point).
    """
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    per_image_root = runs_dir / "per_image"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = discover_configs(per_image_root)
    print(f"[Stage 5] Found {len(configs)} (encoder, area) configs:")
    per_image = {}
    for enc, area in configs:
        per_image[(enc, area)] = load_per_image(per_image_root, enc, area)
        print(f"    {enc:>10s} / {area}: {len(per_image[(enc, area)])} images")

    pairs: dict[str, dict] = {}
    for key_A, key_B in combinations(configs, 2):
        for metric in ("G", "J"):
            stats = aggregate_pair(
                per_image[key_A], per_image[key_B], metric, args.max_k
            )
            if stats is None:
                continue
            stats.update(
                metric=metric,
                config_A={"encoder": key_A[0], "area": key_A[1]},
                config_B={"encoder": key_B[0], "area": key_B[1]},
            )
            pairs[pair_label(key_A, key_B, metric)] = stats

    named = {
        f"{name}__{metric}": pair_label(a, b, metric)
        for name, (a, b) in named_comparisons(configs).items()
        for metric in ("G", "J")
        if pair_label(a, b, metric) in pairs
    }

    out_path = out_dir / "cross_encoder_metrics.json"
    with open(out_path, "w") as f:
        json.dump(
            {"pairs": pairs, "named": named, "provenance": provenance()}, f, indent=2
        )
    print(f"[Stage 5] Saved {len(pairs)} pairwise comparisons to {out_path}")

    print("\n[Stage 5] Paper comparisons (mean over images):")
    for name, label in named.items():
        st = pairs[label]
        k_show = min(20, st["k"][-1])
        i = k_show - 1
        print(
            f"  {name:45s} A_{k_show} = {st['alignment_mean'][i]:.3f} ± "
            f"{st['alignment_sem'][i]:.3f}   d_BW(k={k_show}) = "
            f"{st['bures_wasserstein_mean'][i]:.3g}   n = {st['n_images']}"
        )


# ── using the results: the inputs, the order the paper reports, running the stage ──
REPO = project_root()
PAPER_N_IMAGES = 200
AREAS = ("V1", "V4")
ENCODERS = ("inception", "resnet50")
Config = tuple[str, str]


def configs_with_images(runs_dir: str = "runs") -> list[Config]:
    """The (encoder, area) configurations that have per-image files under `runs_dir`."""
    root = REPO / runs_dir / "per_image"
    if not root.exists():
        return []
    return sorted(
        (e.name, a.name)
        for e in root.iterdir()
        if e.is_dir()
        for a in e.iterdir()
        if a.is_dir() and any(a.glob("img*.pt"))
    )


def compute_metrics(
    runs_dir: str = "runs", out_dir: str = "runs/metrics"
) -> Path | None:
    """
    Run the alignment (the former stage 5: alignment and Bures-Wasserstein distance
    between every pair of configurations, over the images they share). Alignment
    compares two models or areas, so with fewer than two configurations there is nothing
    to compute: prints why and returns None.
    """
    found = configs_with_images(runs_dir)
    if len(found) < 2:
        print(
            f"only {len(found)} configuration(s) with per-image files ({found}): the "
            "alignment compares "
            "two configurations, so add another (for example the same encoder fitted "
            "to the other area)."
        )
        return None
    subprocess.run(
        [
            sys.executable,
            "-m",
            "multigeom.pipeline.alignment",
            "--runs-dir",
            runs_dir,
            "--out-dir",
            out_dir,
        ],
        cwd=str(REPO),
        env=subprocess_env(),
        check=True,
    )
    return REPO / out_dir / "cross_encoder_metrics.json"


def input_summary(runs_dir: str = "runs") -> dict:
    """
    What Stage 5 would work on: per configuration the indices of its per-image files,
    and per pair of configurations the number of shared images (the alignment and
    distance are averaged over those).
    """
    root = REPO / runs_dir / "per_image"
    images = {}
    for enc, area in configs_with_images(runs_dir):
        images[(enc, area)] = sorted(
            int(f.stem[3:]) for f in (root / enc / area).glob("img*.pt")
        )
    configs = list(images)
    shared = {
        (a, b): len(set(images[a]) & set(images[b]))
        for i, a in enumerate(configs)
        for b in configs[i + 1 :]
    }
    return {"images": images, "shared": shared}


def print_inputs(runs_dir: str = "runs", paper_n_images: int = PAPER_N_IMAGES) -> dict:
    """
    Print the images per configuration and warn when there are fewer than the paper's
    200.
    """
    s = input_summary(runs_dir)
    if not s["images"]:
        print(
            (
                "no per-image files in runs/per_image: estimate G(x) and J(x) first "
                "(tutorial 2)."
            )
        )
        return s
    for (enc, area), idx in s["images"].items():
        print(f"{enc:10s} {area}: {len(idx):3d} images")
    fewest = min(s["shared"].values()) if s["shared"] else 0
    print(
        f"fewest images shared by two configurations: {fewest}"
        + (
            ""
            if fewest >= paper_n_images
            else (
                f"  (the paper averages over {paper_n_images}: with fewer, the bands "
                "are wider)"
            )
        )
    )
    return s


def _mean_curve(
    metrics: dict, key: str, field: str = "bures_wasserstein_mean"
) -> list[float]:
    return metrics["pairs"][metrics["named"][key]][field]


def ordering(metrics: dict) -> dict:
    """
    The mean over k of d_BW for the two cross-model comparisons (V1, V4) and the two
    cross-area ones (Inception, ResNet-50), per metric, and whether the ordering is the
    paper's (G: cross-model below cross-area; J: reversed). Comparisons missing from
    `metrics` are left out; the result for a metric is None if either kind is missing.
    """
    out = {}
    for metric in ("G", "J"):
        model = [f"cross_model_{a}_inception_resnet50__{metric}" for a in AREAS]
        area = [f"cross_area_{e}__{metric}" for e in ENCODERS]
        mean = lambda keys: _avg(
            [_avg(_mean_curve(metrics, k)) for k in keys if k in metrics["named"]]
        )
        m, a = mean(model), mean(area)
        if m is None or a is None:
            out[metric] = None
            continue
        out[metric] = {
            "cross_model": m,
            "cross_area": a,
            "matches_paper": (m < a) if metric == "G" else (m > a),
        }
    return out


def _avg(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def print_ordering(metrics: dict) -> dict:
    """
    Print, for G and for J, the mean Bures-Wasserstein distance of the cross-model and
    the cross-area comparisons and whether their order is the paper's (G: cross-model
    below cross-area; J: above). Returns the result.
    """
    result = ordering(metrics)
    expected = {
        "G": "cross-model below cross-area",
        "J": "cross-model above cross-area",
    }
    for metric, r in result.items():
        if r is None:
            print(f"{metric}: not all four comparisons are available")
            continue
        print(
            f"{metric}: mean d_BW over k: cross-model {r['cross_model']:.3g}, "
            f"cross-area {r['cross_area']:.3g}"
            f"  -> paper expects {expected[metric]}: "
            f"{'yes' if r['matches_paper'] else 'NO'}"
        )
    return result


def load_metrics(path) -> dict:
    """Read a metrics file written by `compute_metrics`."""
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    main()
