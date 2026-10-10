"""
Regenerate the Fig. 5 fixtures of tutorials/3_local_metric_field.ipynb from Stage-6 results in runs/:

    tests/integration/fixtures/fig5/metric_field.npz         what the drawing needs: grid coefficients, reference image x0,
                                                             and the projected 2x2 matrices of every (metric, area, encoder)
    tests/integration/fixtures/fig5/fig5_reference.png       the notebook plot rendered from it
    tests/integration/fixtures/fig5/fixture_info.json        how they were made: image index, grid size, step rule and size,
        encoders, areas, metrics, the steps and eigenvalues at the reference image, the estimator settings (Monte-Carlo samples,
        seeds, sigma), the checkpoints (md5, diffusion epoch), figure settings, git commit and library versions; the test
        checks it against the data

Needs runs/metric_field/img0064/{G,J}_{V1,V4}.pt (the default reference image, 64), for example from the notebook with GRID_SIZE = 5 (about 26 minutes on
one GPU, both areas and both metrics). Usage, from the repository root:

    python tests/integration/fixtures/make_fig5_fixtures.py [--img-idx 64] [--grid-size 5]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))               # tests/integration (nb_helpers)

from nb_helpers import fig5_records  # noqa: E402
from src import local_metric_field as lmf  # noqa: E402
from src.utils import load_config  # noqa: E402

OUT = HERE / "fig5"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--img-idx", type=int, default=lmf.DEFAULT_IMG_IDX)
    p.add_argument("--grid-size", type=int, default=5)
    args = p.parse_args(argv)

    import matplotlib
    import numpy
    import torch
    matplotlib.use("Agg")

    job = lmf.MetricFieldJob(img_idx=args.img_idx, grid_size=args.grid_size)
    records = lmf.load_records(job)
    missing = [(m, a) for m in job.metrics for a in job.areas if (m, a) not in records]
    if missing:
        raise SystemExit(f"missing Stage-6 results for {missing} in {lmf.out_dir(job)}")

    arrays = {"grid_coefficients": records[("G", "V1")]["grid_coefficients"].numpy(),
              "x0": records[("G", "V1")]["x0"].numpy().reshape(64, 64)}
    per_part = {}
    for (metric, area), rec in records.items():
        for enc in rec["encoders"]:
            arrays[f"{metric}_{area}_{enc}"] = rec["matrices_2x2"][enc].numpy()
        per_part[f"{metric}_{area}"] = {
            "eps1": rec["eps1"], "eps2": rec["eps2"], "lambda1": rec["lambda1"], "lambda2": rec["lambda2"],
            "settings": rec["settings"], "encoders": list(rec["encoders"]),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "metric_field.npz", **arrays)

    fig = lmf.plot_metric_field(fig5_records(OUT / "metric_field.npz"))
    figsize = [float(v) for v in fig.get_size_inches()]
    fig.savefig(OUT / "fig5_reference.png", dpi=100, bbox_inches="tight")      # tight: the legend sits outside the axes

    est = load_config(lmf.REPO / "configs" / "estimator.yaml")
    checkpoints = {}
    for area in job.areas:
        for enc in (job.reference_encoder, job.other_encoder):
            d = torch.load(lmf.REPO / f"runs/diffusion/{enc}/{area}/final_model.pt", map_location="cpu", weights_only=False)
            checkpoints[f"{enc}/{area}"] = {
                "encoder_checkpoint_md5": md5(lmf.REPO / f"runs/encoders/{enc}/{area}/best_model.pt"),
                "diffusion_checkpoint_md5": md5(lmf.REPO / f"runs/diffusion/{enc}/{area}/final_model.pt"),
                "diffusion_epoch": d.get("epoch"),
            }
    info = {
        "job": {"img_idx": job.img_idx, "grid_size": job.grid_size,
                "steps": {m: list(v) for m, v in job.steps.items()}, "reference_encoder": job.reference_encoder,
                "other_encoder": job.other_encoder, "areas": list(job.areas), "metrics": list(job.metrics)},
        "parts": per_part,
        "estimator": {"n_samples": int(est.monte_carlo.n_samples), "sigma": float(est.monte_carlo.observation_noise_sigma),
                      "seed": int(est.seed), "top_k": int(est.top_k)},
        "checkpoints": checkpoints,
        "plot": {"function": "figures.fig5.make_figure", "figsize": figsize, "dpi": 100, "bbox_inches": "tight", "fill": 0.9},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=lmf.REPO, capture_output=True, text=True).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=lmf.REPO, capture_output=True, text=True).stdout.strip()),
        "versions": {"torch": torch.__version__, "numpy": numpy.__version__, "matplotlib": matplotlib.__version__},
    }
    (OUT / "fixture_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
