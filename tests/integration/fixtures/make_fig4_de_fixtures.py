"""
Regenerate the Fig. 4D-E fixtures of notebooks/2_subspace_alignment.ipynb from the per-image files in runs/:

    tests/integration/fixtures/fig4_de/cross_encoder_metrics.json   Stage 5 on those files (without provenance)
    tests/integration/fixtures/fig4_de/fig4_de_reference.png        the notebook plot rendered from it
    tests/integration/fixtures/fig4_de/fixture_info.json            how they were made: configurations, images per
        configuration, estimator settings (Monte-Carlo samples, seed, sigma), checkpoints (hash, diffusion epoch),
        k range, plot settings, git commit and library versions; the test checks it against the data

Needs runs/per_image/<encoder>/<area>/img*.pt for at least two configurations (G(x) estimation, see the notebook).
Usage, from the repository root:

    python tests/integration/fixtures/make_fig4_de_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from multigeom import viz  # noqa: E402
from multigeom.pipeline import alignment as sa  # noqa: E402
from multigeom.config import load_config  # noqa: E402

OUT = HERE / "fig4_de"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import matplotlib
    import numpy
    import torch
    matplotlib.use("Agg")

    path = sa.compute_metrics()                          # Stage 5 on the current per-image files
    if path is None:
        raise SystemExit("need per-image files for at least two configurations")
    metrics = json.loads(path.read_text())
    metrics.pop("provenance", None)                      # git state, paths, package versions: kept in fixture_info.json
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cross_encoder_metrics.json").write_text(json.dumps(metrics))
    fig = viz.plot_alignment(OUT / "cross_encoder_metrics.json")
    figsize = [float(v) for v in fig.get_size_inches()]
    fig.savefig(OUT / "fig4_de_reference.png", dpi=100)

    est = load_config("estimator")
    configs, per_config = [], {}
    for enc, area in sa.configs_with_images():
        files = sorted((sa.REPO / "runs" / "per_image" / enc / area).glob("img*.pt"))
        first = torch.load(files[0], map_location="cpu", weights_only=False)
        configs.append([enc, area])
        per_config[f"{enc}/{area}"] = {
            "images": [int(f.stem[3:]) for f in files],
            "diffusion_epoch": first.get("diffusion_epoch"),
            "sigma": first.get("sigma"),
            "encoder_checkpoint_md5": md5(sa.REPO / f"runs/encoders/{enc}/{area}/best_model.pt"),
            "diffusion_checkpoint_md5": md5(sa.REPO / f"runs/diffusion/{enc}/{area}/final_model.pt"),
        }
    info = {
        "configs": configs,
        "per_config": per_config,
        "estimator": {"n_samples": int(est.monte_carlo.n_samples), "batch_size": int(est.monte_carlo.batch_size),
                      "timesteps": [int(est.monte_carlo.timestep_grid.start), int(est.monte_carlo.timestep_grid.stop),
                                    int(est.monte_carlo.timestep_grid.step)],
                      "sigma": float(est.monte_carlo.observation_noise_sigma), "seed": int(est.seed),
                      "top_k": int(est.top_k)},
        "k_max": int(max(max(v["k"]) for v in metrics["pairs"].values())),
        "plot": {"function": "multigeom.viz.plot_alignment", "figsize": figsize, "dpi": 100},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=sa.REPO, capture_output=True, text=True).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=sa.REPO, capture_output=True, text=True).stdout.strip()),
        "versions": {"torch": torch.__version__, "numpy": numpy.__version__, "matplotlib": matplotlib.__version__},
    }
    (OUT / "fixture_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
