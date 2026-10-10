"""
Scan the test images for the reference image whose J(x) local metric field looks like the paper's Fig. 5B.

The paper does not state its reference image, so this scans the first N validation images (the 200 test images of the
G(x) estimation) with several perturbation sizes and reports, for the Inception model fitted to V1 and V4:

  radial   mean |cos| between each ellipse's major axis and the direction from the grid centre (random 0.64, radial 1.0)
  ratio    median Inception / ResNet-50 ellipse area off-centre (above 1: Inception outside, as in the paper)

J(x) is cheap (about 0.04 s per grid point), so 200 images x 6 settings take about 20 minutes on one GPU. To save time the
ResNet-50 grid is computed only when the Inception field is radial enough (--min-radial). One JSON line per
(image, setting) is appended to the output file, which makes the scan resumable.

    python tools/scan_metric_field_images.py --out runs/analysis/metric_field_scan.jsonl
    python tools/scan_metric_field_images.py --summarize runs/analysis/metric_field_scan.jsonl

Result with the checkpoints of this repository (diffusion models at epoch 120): of 1200 combinations 84 are radial in both
areas, and only validation image 64 (fixed steps 0.2 and 0.3) is also clearly Inception-outside-ResNet-50 in both.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SETTINGS = [("fixed", 0.2), ("fixed", 0.3), ("fixed", 0.5), ("fixed", 0.8), ("sqrt_lambda", 0.02), ("sqrt_lambda", 0.04)]


def scan(out_path: Path, n_images: int, min_radial: float, grid_size: int = 5) -> None:
    import torch

    from multigeom.pipeline import metric_field_job as lmf
    from multigeom.data import CANONICAL_TRANSFORM, TinyImageNetDataset
    from multigeom.encoders import build_encoder
    from multigeom.geometry.fisher import fisher_eigendecomposition
    from multigeom.geometry.metric_field import canonical_sign, grid_coefficients, perturbation_grid, project_metric, step_sizes
    from multigeom.config import load_config
    from multigeom.utils import get_device

    dev = get_device(None)
    sigma = float(load_config("estimator").monte_carlo.observation_noise_sigma)
    coeffs = grid_coefficients(grid_size)
    models = {}
    for area in lmf.AREAS:
        for name in ("inception", "resnet50"):
            enc = build_encoder(load_config(name)).to(dev)
            ckpt = torch.load(ROOT / f"runs/encoders/{name}/{area}/best_model.pt", map_location=dev, weights_only=False)
            enc.load_state_dict(ckpt["model_state_dict"])
            models[(name, area)] = enc.eval()
    done = set()
    if out_path.exists():
        done = {(r["img"], r["rule"], r["size"]) for r in map(json.loads, out_path.read_text().splitlines())}
    dataset = TinyImageNetDataset(split="val", transform=CANONICAL_TRANSFORM)

    def field(name, area, images, V):
        import numpy as np
        n = len(coeffs)
        M = np.empty((n, n, 2, 2))
        for i in range(n):
            for j in range(n):
                x = images[i, j].reshape(1, 1, 64, 64).to(dev)
                o = fisher_eigendecomposition(x, models[(name, area)], sigma=sigma, top_k=50, device=dev)
                M[i, j] = project_metric(o.eigenvalues, o.eigenvectors.to(dev), V).cpu().numpy()
        return M

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_path, "a") as out:
        for img in range(n_images):
            x0 = dataset[img].unsqueeze(0).to(dev)
            base = {}
            for area in lmf.AREAS:
                o = fisher_eigendecomposition(x0, models[("inception", area)], sigma=sigma, top_k=50, device=dev)
                base[area] = (canonical_sign(o.eigenvectors[0]), canonical_sign(o.eigenvectors[1]),
                              float(o.eigenvalues[0]), float(o.eigenvalues[1]))
            for rule, size in SETTINGS:
                if (img, rule, size) in done:
                    continue
                rec = {"img": img, "rule": rule, "size": size}
                for area in lmf.AREAS:
                    v1, v2, l1, l2 = base[area]
                    e1, e2 = step_sizes(l1, l2, rule, size)
                    images, _ = perturbation_grid(x0, v1, v2, e1, e2, coeffs)
                    V = torch.stack([v1, v2]).to(dev)
                    blue = field("inception", area, images, V)
                    rec[f"radial_{area}"] = lmf.radial_score(blue, coeffs)
                    if rec[f"radial_{area}"] < min_radial:
                        break
                    rec[f"ratio_{area}"] = lmf.area_ratio(blue, field("resnet50", area, images, V))
                out.write(json.dumps(rec) + "\n")
                out.flush()
            print(f"image {img}: {time.time() - t0:.0f} s", flush=True)


def summarize(path: Path, min_radial: float = 0.7, min_ratio: float = 2.5, top: int = 10) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    full = [r for r in rows if "ratio_V4" in r]
    good = [r for r in full if min(r["ratio_V1"], r["ratio_V4"]) >= min_ratio]
    print(f"{len(rows)} (image, setting) combinations; {len(full)} radial >= {min_radial} in both areas; "
          f"{len(good)} of those with Inception outside ResNet-50 (ratio >= {min_ratio}) in both areas")
    for r in sorted(good, key=lambda r: min(r["radial_V1"], r["radial_V4"]), reverse=True)[:top]:
        print(f"  image {r['img']:3d} {r['rule']:11s} {r['size']:<5}: radial V1 {r['radial_V1']:.2f} V4 {r['radial_V4']:.2f}"
              f" | ratio V1 {r['ratio_V1']:.1f} V4 {r['ratio_V4']:.1f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="runs/analysis/metric_field_scan.jsonl")
    p.add_argument("--n-images", type=int, default=200)
    p.add_argument("--min-radial", type=float, default=0.7)
    p.add_argument("--summarize", metavar="JSONL", help="only summarise an existing scan")
    args = p.parse_args(argv)
    if args.summarize:
        summarize(Path(args.summarize))
    else:
        scan(ROOT / args.out, args.n_images, args.min_radial)
        summarize(ROOT / args.out)


if __name__ == "__main__":
    main()
