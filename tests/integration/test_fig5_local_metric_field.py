"""
Integration test: do we reproduce the plot of notebooks/3_local_metric_field.ipynb (paper Fig. 5, the local
metric field projected onto the leading eigenvectors of G(x))?

The reference is stored in tests/integration/fixtures/fig5/ (regenerate with
tests/integration/fixtures/make_fig5_fixtures.py, after running the notebook with GRID_SIZE = 5 for both areas):

    metric_field.npz        the projected 2x2 matrices of G(x) and J(x) for the Inception and ResNet-50 encoders, both
                            areas, on the 5 x 5 grid around the reference image, and the reference image itself
    fig5_reference.png      the notebook's plot rendered from it
    fixture_info.json       how they were made: reference image, grid size, step rule and size per metric (G: 0.5 sqrt(lambda),
                            J: fixed 0.3), the steps and eigenvalues at the reference image, estimator settings (200 Monte-Carlo samples, seed 42,
                            sigma 0.22), checkpoint hashes and diffusion epochs, plot settings, git commit, versions

The tests run the notebook's plotting function, multigeom.pipeline.metric_field_job.plot_metric_field, on the stored matrices and compare
the picture with the reference; they check that fixture_info.json agrees with the data and with the repository's settings
(configs/estimator.yaml, the plot defaults), and that the matrices are valid symmetric positive-semidefinite 2 x 2 metrics.
The computation itself (stage 6) is not run here: it needs the checkpoints and takes minutes.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_helpers import BAD_PIXEL_FRACTION, FIG5_FIXTURES, MEAN_ABS_TOL, fig5_records, image_difference  # noqa: E402
from multigeom.pipeline import metric_field_job as lmf  # noqa: E402

FIXTURES = FIG5_FIXTURES
NPZ = FIXTURES / "metric_field.npz"
INFO = FIXTURES / "fixture_info.json"
REFERENCE_PNG = FIXTURES / "fig5_reference.png"
ENCODERS = ("inception", "resnet50")


def render(records, path: Path) -> None:
    fig = lmf.plot_metric_field(records)
    fig.savefig(path, dpi=100, bbox_inches="tight")      # tight: the legend sits outside the axes
    plt.close(fig)


def info() -> dict:
    return json.loads(INFO.read_text())


def test_fixture_matrices_are_valid_metrics():
    data = np.load(NPZ)
    n = len(data["grid_coefficients"])
    for key in data.files:
        if key.startswith(("G_", "J_")):
            m = data[key]
            assert m.shape == (n, n, 2, 2), key
            np.testing.assert_allclose(m, np.swapaxes(m, -1, -2), rtol=1e-6, atol=1e-9, err_msg=key)      # symmetric
            assert np.linalg.eigvalsh(m).min() > -1e-9 * max(1.0, np.abs(m).max()), key                   # PSD
    np.testing.assert_allclose(data["grid_coefficients"], [-2.0, -1.0, 0.0, 1.0, 2.0])
    assert data["x0"].shape == (64, 64)


def test_fixture_info_lists_how_the_fixture_was_made():
    i = info()
    assert {"job", "parts", "estimator", "checkpoints", "plot", "git_commit", "versions"} <= set(i)
    assert i["job"]["reference_encoder"] == "inception" and i["job"]["other_encoder"] == "resnet50"
    for key, c in i["checkpoints"].items():
        assert len(c["encoder_checkpoint_md5"]) == 32 and len(c["diffusion_checkpoint_md5"]) == 32, key
        assert c["diffusion_epoch"] is not None, key
    for part, p in i["parts"].items():
        assert p["settings"]["n_mc"] == i["estimator"]["n_samples"], part
        assert p["settings"]["grid_size"] == i["job"]["grid_size"], part
        rule, size = i["job"]["steps"][part[0]]                                  # the metric is the first letter of "G_V1"
        assert (p["settings"]["step_rule"], p["settings"]["step_size"]) == (rule, size), part


def test_fixture_info_agrees_with_the_stored_matrices():
    i, data = info(), np.load(NPZ)
    job = i["job"]
    assert len(data["grid_coefficients"]) == job["grid_size"]
    expected = {f"{m}_{a}_{e}" for m in job["metrics"] for a in job["areas"] for e in (job["reference_encoder"], job["other_encoder"])}
    assert {k for k in data.files if k[:2] in ("G_", "J_")} == expected
    assert set(i["parts"]) == {f"{m}_{a}" for m in job["metrics"] for a in job["areas"]}


def test_fixture_parameters_match_the_current_settings_of_the_repository():
    """If the estimator config, the job defaults or the plot defaults change, the fixture no longer describes this code."""
    from multigeom.config import load_config
    i = info()
    est = load_config("estimator")
    assert i["estimator"] == {"n_samples": int(est.monte_carlo.n_samples),
                              "sigma": float(est.monte_carlo.observation_noise_sigma),
                              "seed": int(est.seed), "top_k": int(est.top_k)}, \
        "configs/estimator.yaml changed since the fixture was made: regenerate it"
    default = lmf.MetricFieldJob()
    assert i["job"]["img_idx"] == default.img_idx == lmf.DEFAULT_IMG_IDX, "the default reference image changed: regenerate the fixture"
    assert i["job"]["steps"] == {m: list(v) for m, v in default.steps.items()}, \
        "the default step rules of the job changed since the fixture was made: regenerate it"
    from multigeom.viz.fig5 import make_figure
    assert i["plot"]["fill"] == inspect.signature(make_figure).parameters["fill"].default


def test_notebook_plot_reproduces_reference(tmp_path):
    render(fig5_records(), tmp_path / "plot.png")
    mean_abs, bad = image_difference(mpimg.imread(tmp_path / "plot.png"), mpimg.imread(REFERENCE_PNG))
    assert mean_abs < MEAN_ABS_TOL and bad < BAD_PIXEL_FRACTION, (
        f"Fig. 5 plot differs from the reference: mean |diff| = {mean_abs:.5f} (tol {MEAN_ABS_TOL}), "
        f"fraction of pixels off by > 0.1 = {bad:.5f} (tol {BAD_PIXEL_FRACTION})")


def test_plot_comparison_detects_changes(tmp_path):
    """Guards against a vacuous comparison: a different ResNet-50 metric field must fail it."""
    records = fig5_records()
    rec = records[("G", "V1")]
    rec["matrices_2x2"] = {**rec["matrices_2x2"], "resnet50": rec["matrices_2x2"]["resnet50"] * 4.0
                           + torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=rec["matrices_2x2"]["resnet50"].dtype)}
    render(records, tmp_path / "changed.png")
    mean_abs, bad = image_difference(mpimg.imread(tmp_path / "changed.png"), mpimg.imread(REFERENCE_PNG))
    assert mean_abs >= MEAN_ABS_TOL or bad >= BAD_PIXEL_FRACTION
