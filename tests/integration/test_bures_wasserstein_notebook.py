"""
Integration tests of tutorials/4_bures_wasserstein.ipynb (paper Fig. 6, the Bures-Wasserstein distance between geometries):
they run the notebook's own cells.

The reference is stored in tests/integration/fixtures/fig6/ (regenerate with
tests/integration/fixtures/make_fig6_fixtures.py): the notebook plot (fig6_reference.png) and fixture_info.json, which says how
it was made. The data is the Stage-5 output of the Fig. 4D-E fixture (it holds the alignment and the distance for the four
configurations).

  - CPU, run in CI: the fixture agrees with its source file and with the repository's settings; the notebook's ordering cell
    reports the paper's order (G: cross-model below cross-area, J: reversed); the plot cell draws the reference picture.
  - slow, needs the per-image files of notebook 2 (skipped without them, so not on GitHub): the notebook's setup, Stage-5,
    ordering and plot cells run end to end in a scratch copy of the repository, so nothing in runs/ is touched.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_helpers import (  # noqa: E402
    BAD_PIXEL_FRACTION, COMPUTE4, MEAN_ABS_TOL, NOTEBOOK4, ORDER4, PLOT4, REPO, SETUP4, image_difference, run_cells,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SOURCE = FIXTURES / "fig4_de" / "cross_encoder_metrics.json"
INFO = FIXTURES / "fig6" / "fixture_info.json"
REFERENCE_PNG = FIXTURES / "fig6" / "fig6_reference.png"


@pytest.fixture
def at_repo_root(monkeypatch):
    monkeypatch.chdir(REPO)


def info() -> dict:
    return json.loads(INFO.read_text())


def test_the_notebook_has_the_cells_the_tests_use():
    nb = json.loads(NOTEBOOK4.read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for marker in (SETUP4, COMPUTE4, ORDER4, PLOT4):
        assert any(marker in c for c in code), marker


def test_fixture_info_agrees_with_its_source_and_the_repository():
    i = info()
    assert hashlib.md5(SOURCE.read_bytes()).hexdigest() == i["source"]["md5"], \
        "the Stage-5 file of the Fig. 4D-E fixture changed: regenerate the Fig. 6 fixture"
    from src import bures_wasserstein as bw
    metrics = bw.load_metrics(SOURCE)
    assert i["k_max"] == max(max(p["k"]) for p in metrics["pairs"].values()) == 50
    assert i["ordering"] == bw.ordering(metrics)
    assert i["ordering"]["G"]["matches_paper"] and i["ordering"]["J"]["matches_paper"]
    import inspect

    from figures.fig_app_bw import make_figure  # noqa: F401
    assert i["plot"]["function"] == "figures.fig_app_bw.make_figure"
    from src.utils import load_config
    est = load_config(REPO / "configs" / "estimator.yaml")
    assert i["estimator"]["n_samples"] == int(est.monte_carlo.n_samples) and i["estimator"]["seed"] == int(est.seed), \
        "configs/estimator.yaml changed since the fixture was made: regenerate it"
    assert inspect.signature(make_figure).parameters["metrics"]


def test_ordering_cell_reports_the_papers_order(at_repo_root, capsys):
    from src import bures_wasserstein as bw
    ns = run_cells({}, SETUP4, notebook=NOTEBOOK4)
    ns["metrics_path"], ns["metrics"] = SOURCE, bw.load_metrics(SOURCE)
    capsys.readouterr()
    run_cells(ns, ORDER4, notebook=NOTEBOOK4)
    out = capsys.readouterr().out
    assert "cross-model below cross-area: yes" in out and "cross-model above cross-area: yes" in out
    assert ns["order"]["G"]["matches_paper"] and ns["order"]["J"]["matches_paper"]


def test_plot_cell_draws_the_reference_figure(at_repo_root, tmp_path, monkeypatch):
    from src import bures_wasserstein as bw
    ns = run_cells({}, SETUP4, notebook=NOTEBOOK4)
    ns["metrics_path"] = SOURCE
    ns["metrics"] = bw.load_metrics(SOURCE)
    drawn = {}
    monkeypatch.setattr(ns["sa"], "show_figure", lambda fig, dpi=110: drawn.setdefault("fig", fig))
    run_cells(ns, PLOT4, notebook=NOTEBOOK4)
    assert "fig" in drawn, "the plot cell did not draw a figure"
    drawn["fig"].savefig(tmp_path / "plot.png", dpi=100, bbox_inches="tight")
    mean_abs, bad = image_difference(mpimg.imread(tmp_path / "plot.png"), mpimg.imread(REFERENCE_PNG))
    assert mean_abs < MEAN_ABS_TOL and bad < BAD_PIXEL_FRACTION, (
        f"Fig. 6 differs from the reference: mean |diff| = {mean_abs:.5f} (tol {MEAN_ABS_TOL}), "
        f"fraction of pixels off by > 0.1 = {bad:.5f} (tol {BAD_PIXEL_FRACTION})")


def test_plot_and_ordering_cells_say_so_without_results(at_repo_root, capsys):
    ns = run_cells({}, SETUP4, notebook=NOTEBOOK4)
    ns["metrics_path"] = None
    capsys.readouterr()
    run_cells(ns, ORDER4, PLOT4, notebook=NOTEBOOK4)
    out = capsys.readouterr().out
    assert "Nothing to check yet" in out and "Nothing to plot yet" in out


# ── end to end on the real per-image files (not on GitHub) ──────────────────
def _have_per_image() -> bool:
    return all((REPO / "runs/per_image" / enc / area).exists() and any((REPO / "runs/per_image" / enc / area).glob("img*.pt"))
               for enc in ("inception", "resnet50") for area in ("V1", "V4"))


@pytest.mark.slow
@pytest.mark.skipif(not _have_per_image(), reason="needs runs/per_image of the four configurations (tutorial 2)")
def test_notebook_runs_stage5_and_plots_in_a_scratch_copy(at_repo_root, tmp_path, monkeypatch):
    scratch = tmp_path / "repo"
    (scratch / "runs").mkdir(parents=True)
    for folder in ("src", "stages", "configs", "figures"):
        (scratch / folder).symlink_to(REPO / folder)
    (scratch / "runs" / "per_image").symlink_to(REPO / "runs" / "per_image")        # inputs shared, outputs not
    (scratch / "pyproject.toml").write_text("")
    real = REPO / "runs" / "metrics" / "cross_encoder_metrics.json"
    real_mtime = real.stat().st_mtime if real.exists() else None
    monkeypatch.chdir(scratch)
    ns = run_cells({}, SETUP4, notebook=NOTEBOOK4)
    for module in (ns["bw"], ns["sa"]):
        monkeypatch.setattr(module, "REPO", scratch)
    run_cells(ns, "summary = bw.print_inputs(RUNS_DIR)", notebook=NOTEBOOK4)
    run_cells(ns, COMPUTE4, notebook=NOTEBOOK4)
    assert ns["metrics_path"] == scratch / "runs" / "metrics" / "cross_encoder_metrics.json" and ns["metrics_path"].exists()
    assert (real.stat().st_mtime if real.exists() else None) == real_mtime, "the real runs/metrics was touched"
    run_cells(ns, ORDER4, notebook=NOTEBOOK4)
    assert ns["order"]["G"] is not None and ns["order"]["J"] is not None
    drawn = {}
    monkeypatch.setattr(ns["sa"], "show_figure", lambda fig, dpi=110: drawn.setdefault("fig", fig))
    run_cells(ns, PLOT4, notebook=NOTEBOOK4)
    assert "fig" in drawn
