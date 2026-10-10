"""
Regenerate the Fig. 6 fixtures of notebooks/4_bures_wasserstein.ipynb:

    tests/integration/fixtures/fig6/fig6_reference.png    the notebook plot (Bures-Wasserstein distance against k)
    tests/integration/fixtures/fig6/fixture_info.json     how it was made: the Stage-5 file it is drawn from (name and md5), the
        images per configuration and the estimator settings of that file (from the Fig. 4D-E fixture, which holds the same
        Stage-5 output), k range, the ordering of the distances the paper reports and its values, the plot settings, git
        commit and library versions

The Stage-5 output is the one of the Fig. 4D-E fixture (it contains the alignment and the Bures-Wasserstein distance),
tests/integration/fixtures/fig4_de/cross_encoder_metrics.json: regenerate that first when the per-image files change
(make_fig4_de_fixtures.py). Usage, from the repository root:

    python tests/integration/fixtures/make_fig6_fixtures.py
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
from multigeom.pipeline import alignment as bw  # noqa: E402

SOURCE = HERE / "fig4_de" / "cross_encoder_metrics.json"
SOURCE_INFO = HERE / "fig4_de" / "fixture_info.json"
OUT = HERE / "fig6"


def main():
    import matplotlib
    import numpy
    import torch
    matplotlib.use("Agg")

    metrics = bw.load_metrics(SOURCE)
    fig = viz.plot_bures_wasserstein(metrics)
    figsize = [float(v) for v in fig.get_size_inches()]
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig6_reference.png", dpi=100, bbox_inches="tight")
    source_info = json.loads(SOURCE_INFO.read_text())
    info = {
        "source": {"file": "fixtures/fig4_de/cross_encoder_metrics.json", "md5": hashlib.md5(SOURCE.read_bytes()).hexdigest()},
        "images_per_config": {k: len(v["images"]) for k, v in source_info["per_config"].items()},
        "estimator": source_info["estimator"],
        "k_max": source_info["k_max"],
        "ordering": bw.ordering(metrics),
        "plot": {"function": "multigeom.viz.fig_app_bw.make_figure", "figsize": figsize, "dpi": 100, "bbox_inches": "tight"},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=bw.REPO, capture_output=True, text=True).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=bw.REPO, capture_output=True, text=True).stdout.strip()),
        "versions": {"torch": torch.__version__, "numpy": numpy.__version__, "matplotlib": matplotlib.__version__},
    }
    (OUT / "fixture_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
