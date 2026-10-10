"""
Integration tests of tutorials/3_local_metric_field.ipynb (paper Fig. 5): they run the notebook's own cells.

  - CPU, run in CI: the parameter cell gives the documented job (reference image 64, 5 x 5 grid, G with 0.5 sqrt(lambda), J
    with a fixed 0.3 step) and prints its cost; the plot cell, run on the stored Stage-6 fields of
    tests/integration/fixtures/fig5/ placed where the notebook looks for results, draws the reference picture.
  - slow, needs the checkpoints and TinyImageNet (skipped without them, so not on GitHub): the parameter, launch, wait and
    plot cells run end to end on a small job (3 x 3 grid, V1, J(x) only) in a scratch copy of the repository, so nothing in
    runs/ is touched: the background process runs, writes its result file, and the figure is drawn.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_helpers import (  # noqa: E402
    BAD_PIXEL_FRACTION, FIG5_FIXTURES, LAUNCH3, MEAN_ABS_TOL, NOTEBOOK3, PLOT3, REPO, SETUP3, WAIT3,
    fig5_records, image_difference, run_cells,
)

REFERENCE_PNG = FIG5_FIXTURES / "fig5_reference.png"


@pytest.fixture
def at_repo_root(monkeypatch):
    """The notebook's first cell moves to the repository root; restore the working directory afterwards."""
    monkeypatch.chdir(REPO)


def test_the_notebook_exists_and_has_the_cells_the_tests_use():
    import json
    nb = json.loads(NOTEBOOK3.read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for marker in (SETUP3, LAUNCH3, WAIT3, PLOT3):
        assert any(marker in c for c in code), marker


def test_parameter_cell_gives_the_documented_job_and_its_cost(at_repo_root, capsys):
    from src import local_metric_field as lmf
    ns = run_cells({}, SETUP3, notebook=NOTEBOOK3)
    job = ns["job"]
    assert job.img_idx == lmf.DEFAULT_IMG_IDX == 64 and job.grid_size == 5
    assert job.steps == {"G": ("sqrt_lambda", 0.5), "J": ("fixed", 0.3)}
    assert (job.reference_encoder, job.other_encoder) == ("inception", "resnet50")
    assert job.areas == ("V1", "V4") and job.metrics == ("G", "J") and job.device == "gpu"
    assert isinstance(ns["RUN"], bool) and isinstance(ns["RESTART"], bool)
    out = capsys.readouterr().out
    assert "102 G(x) estimates and 102 J(x) evaluations" in out and "about 26" in out        # 5x5 grid, 2 areas, 2 encoders


def test_plot_cell_draws_the_reference_figure_from_stored_results(at_repo_root, tmp_path, monkeypatch):
    ns = run_cells({}, SETUP3, notebook=NOTEBOOK3)                       # parameter cell
    lmf, sa, job = ns["lmf"], ns["sa"], ns["job"]
    folder = tmp_path / "runs" / "metric_field" / f"img{job.img_idx:04d}"
    folder.mkdir(parents=True)
    for (metric, area), rec in fig5_records().items():
        torch.save(rec, folder / f"{metric}_{area}.pt")
    monkeypatch.setattr(lmf, "REPO", tmp_path)                           # the notebook looks for results under REPO/runs
    drawn = {}
    monkeypatch.setattr(sa, "show_figure", lambda fig, dpi=110: drawn.setdefault("fig", fig))
    run_cells(ns, PLOT3, notebook=NOTEBOOK3)
    assert "fig" in drawn, "the plot cell did not draw a figure"
    drawn["fig"].savefig(tmp_path / "plot.png", dpi=100, bbox_inches="tight")
    mean_abs, bad = image_difference(mpimg.imread(tmp_path / "plot.png"), mpimg.imread(REFERENCE_PNG))
    assert mean_abs < MEAN_ABS_TOL and bad < BAD_PIXEL_FRACTION, (
        f"the notebook's figure differs from the reference: mean |diff| = {mean_abs:.5f} (tol {MEAN_ABS_TOL}), "
        f"fraction of pixels off by > 0.1 = {bad:.5f} (tol {BAD_PIXEL_FRACTION})")


def test_plot_cell_says_so_when_there_are_no_results(at_repo_root, tmp_path, monkeypatch, capsys):
    ns = run_cells({}, SETUP3, notebook=NOTEBOOK3)
    monkeypatch.setattr(ns["lmf"], "REPO", tmp_path)                     # an empty results folder
    capsys.readouterr()
    run_cells(ns, PLOT3, notebook=NOTEBOOK3)
    assert "Nothing to plot yet" in capsys.readouterr().out


# ── end to end on a small job (needs the checkpoints; not on GitHub) ────────
def _have_inputs() -> bool:
    needed = [REPO / "data/tiny-imagenet-200/val/images"] + [
        REPO / f"runs/encoders/{enc}/V1/best_model.pt" for enc in ("inception", "resnet50")]
    return all(p.exists() for p in needed)


@pytest.mark.slow
@pytest.mark.skipif(not _have_inputs(), reason="needs runs/ encoder checkpoints and TinyImageNet (see README)")
def test_notebook_runs_a_small_job_end_to_end(at_repo_root, tmp_path, monkeypatch):
    """Parameters -> launch -> wait -> plot, with J(x) only on a 3 x 3 grid for V1, in a scratch copy of the repository."""
    scratch = tmp_path / "repo"
    (scratch / "runs").mkdir(parents=True)
    for folder in ("src", "stages", "configs", "figures"):
        (scratch / folder).symlink_to(REPO / folder)                      # code and configs are shared, results are not
    for folder in ("encoders", "diffusion"):
        if (REPO / "runs" / folder).exists():
            (scratch / "runs" / folder).symlink_to(REPO / "runs" / folder)
    (scratch / "pyproject.toml").write_text("")                           # the notebook looks for the repository root by it
    monkeypatch.setenv("NEURIPS_DATA_DIR", str(REPO / "data"))
    monkeypatch.setenv("PYTHONPATH", str(scratch))
    monkeypatch.setenv("MULTIGEOM_REPO", str(scratch))                    # read by the background process

    device = "'gpu'" if torch.cuda.is_available() else "'cpu'"
    overrides = {"IMG_IDX": "7", "GRID_SIZE": "3", "AREAS": "('V1',)", "METRICS": "('J',)", "DEVICE": device,
                 "RESTART": "False"}
    monkeypatch.chdir(scratch)
    ns = run_cells({}, SETUP3, notebook=NOTEBOOK3, overrides=overrides)
    lmf, sa = ns["lmf"], ns["sa"]
    monkeypatch.setattr(lmf, "REPO", scratch)                             # the process started below gets it from MULTIGEOM_REPO
    assert ns["job"].grid_size == 3 and ns["job"].metrics == ("J",)

    run_cells(ns, LAUNCH3, notebook=NOTEBOOK3)                            # starts the background process
    assert ns["proc"] is not None
    try:
        run_cells(ns, WAIT3, notebook=NOTEBOOK3)                          # blocks until it ends
        assert lmf.is_complete(ns["job"]), (scratch / "runs/logs/metric_field.log").read_text()[-2000:]
        rec = torch.load(scratch / "runs/metric_field/img0007/J_V1.pt", map_location="cpu", weights_only=False)
        assert rec["settings"]["step_rule"] == "fixed" and rec["settings"]["step_size"] == 0.3
        M = rec["matrices_2x2"]["inception"].numpy()
        assert M.shape == (3, 3, 2, 2) and np.isfinite(M).all()
        np.testing.assert_allclose(M, np.swapaxes(M, -1, -2), rtol=1e-6, atol=1e-9)
        assert np.linalg.eigvalsh(M).min() > -1e-9 * max(1.0, np.abs(M).max())
        drawn = {}
        monkeypatch.setattr(sa, "show_figure", lambda fig, dpi=110: drawn.setdefault("fig", fig))
        run_cells(ns, PLOT3, notebook=NOTEBOOK3)
        assert "fig" in drawn
    finally:
        lmf.stop()                                                        # nothing may keep running if the test failed
