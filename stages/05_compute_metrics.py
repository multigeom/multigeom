"""
Stage 5: Aggregate cross-(encoder, area) metrics from the per-image G(x)
and J(x) eigendecompositions written by Stage 4.

For each pair of (encoder_A, area_A) ↔ (encoder_B, area_B) we compute, on
each shared image:
  - subspace alignment 𝒜_k(G_A, G_B) and 𝒜_k(J_A, J_B) at every k = 1,...,K
  - Bures–Wasserstein distance d_BW(G_A, G_B) and d_BW(J_A, J_B) at k = top_k

Then average across images and dump the results to JSON, ready to feed to
the figure-making script (the paper's Fig. 4D-E and Fig. App-BW).

Usage:
    python stages/05_compute_metrics.py \\
        --runs-dir runs --out-dir runs/metrics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import bures_wasserstein, subspace_alignment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True,
                   help="root containing per_image/<encoder>/<area>/imgXXXX.pt files")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-k", type=int, default=50,
                   help="largest k for the alignment-vs-k curve")
    return p.parse_args()


def discover_configs(per_image_root: Path) -> list[tuple[str, str]]:
    """Discover (encoder, area) pairs available under per_image/."""
    configs = []
    if not per_image_root.exists():
        raise FileNotFoundError(f"{per_image_root} does not exist; run Stage 4 first.")
    for enc_dir in sorted(per_image_root.iterdir()):
        if not enc_dir.is_dir():
            continue
        for area_dir in sorted(enc_dir.iterdir()):
            if not area_dir.is_dir():
                continue
            if any(area_dir.glob("img*.pt")):
                configs.append((enc_dir.name, area_dir.name))
    return configs


def load_per_image(per_image_root: Path, encoder: str, area: str) -> dict[int, dict]:
    """Load every img{N}.pt for a given (encoder, area) keyed by image index."""
    out = {}
    for f in sorted((per_image_root / encoder / area).glob("img*.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        out[int(d["img_idx"])] = d
    return out


def aggregate_pair(
    A: dict[int, dict],
    B: dict[int, dict],
    metric_key: str,                  # 'G' or 'J'
    max_k: int,
):
    """
    For every image present in both A and B, compute alignment 𝒜_k for
    k=1..max_k and Bures–Wasserstein at k=max_k. Returns mean across images
    plus the per-image arrays.
    """
    eigvecs_key = f"{metric_key}_eigenvectors"
    eigvals_key = f"{metric_key}_eigenvalues"
    shared = sorted(set(A.keys()) & set(B.keys()))
    if not shared:
        return None

    align_curves = []
    bw_values = []
    for idx in shared:
        a, b = A[idx], B[idx]
        if a[eigvecs_key] is None or b[eigvecs_key] is None:
            continue
        align_k = [
            subspace_alignment(a[eigvecs_key], b[eigvecs_key], top_k=k)
            for k in range(1, max_k + 1)
        ]
        align_curves.append(align_k)
        bw = bures_wasserstein(
            a[eigvals_key], a[eigvecs_key],
            b[eigvals_key], b[eigvecs_key],
            top_k=max_k,
        )
        bw_values.append(bw)

    if not align_curves:
        return None

    align_arr = np.array(align_curves)              # [n_images, max_k]
    bw_arr = np.array(bw_values)                    # [n_images]
    return {
        "n_images": int(align_arr.shape[0]),
        "alignment_mean": align_arr.mean(axis=0).tolist(),       # [max_k]
        "alignment_sem": (align_arr.std(axis=0, ddof=1) / np.sqrt(align_arr.shape[0])).tolist(),
        "bures_wasserstein_mean": float(bw_arr.mean()),
        "bures_wasserstein_sem": float(bw_arr.std(ddof=1) / np.sqrt(bw_arr.shape[0])),
    }


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    per_image_root = runs_dir / "per_image"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Discover configurations ───────────────────────────────────────
    configs = discover_configs(per_image_root)
    print(f"[Stage 5] Found {len(configs)} (encoder, area) configs:")
    for enc, area in configs:
        print(f"    - {enc} / {area}")

    # ── 2. Load all per-image eigendecompositions ────────────────────────
    print("[Stage 5] Loading per-image eigendecompositions ...")
    per_image: dict[tuple[str, str], dict[int, dict]] = {}
    for enc, area in configs:
        per_image[(enc, area)] = load_per_image(per_image_root, enc, area)
        print(f"    {enc:>10s} / {area}: {len(per_image[(enc, area)])} images loaded")

    # ── 3. Aggregate across all unordered pairs of configurations ────────
    results: dict[str, dict] = {}
    for i, key_A in enumerate(configs):
        for key_B in configs[i + 1:]:
            for metric in ("G", "J"):
                pair_label = f"{key_A[0]}_{key_A[1]}__VS__{key_B[0]}_{key_B[1]}__{metric}"
                stats = aggregate_pair(
                    per_image[key_A], per_image[key_B], metric, args.max_k
                )
                if stats is None:
                    continue
                stats["config_A"] = {"encoder": key_A[0], "area": key_A[1]}
                stats["config_B"] = {"encoder": key_B[0], "area": key_B[1]}
                stats["metric"] = metric
                results[pair_label] = stats

    # ── 4. Save ──────────────────────────────────────────────────────────
    out_path = out_dir / "cross_encoder_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Stage 5] Saved {len(results)} pairwise comparisons to {out_path}")

    # ── 5. Quick textual summary ─────────────────────────────────────────
    print("\n[Stage 5] Summary (alignment at k = 20, mean ± sem):")
    for label, stats in results.items():
        a20_mean = stats["alignment_mean"][min(19, len(stats["alignment_mean"]) - 1)]
        a20_sem = stats["alignment_sem"][min(19, len(stats["alignment_sem"]) - 1)]
        bw = stats["bures_wasserstein_mean"]
        print(f"  {label}")
        print(f"      𝒜_20 = {a20_mean:.3f} ± {a20_sem:.3f}, "
              f"d_BW = {bw:.3f}, n = {stats['n_images']}")


if __name__ == "__main__":
    main()
