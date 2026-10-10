"""
Integration test: do we reproduce the plot of notebooks/2_subspace_alignment.ipynb (paper Fig. 4D-E,
subspace alignment of the top-k eigenvectors of G(x) and J(x) across encoders and areas)?

The reference is stored in tests/integration/fixtures/fig4_de/ (regenerate with
tests/integration/fixtures/make_fig4_de_fixtures.py):

    cross_encoder_metrics.json   Stage-5 output for the four configurations
    fig4_de_reference.png        the notebook's plot rendered from it
    fixture_info.json            how they were made: the configurations, the images of each (2 to 6), the estimator
                                 settings (200 Monte-Carlo samples, seed 42, sigma 0.22, ...), the checkpoints (md5,
                                 diffusion epoch), the k range, the plot settings, the git commit and library versions

The tests check that fixture_info.json agrees with the stored data and with the settings in the repository
(configs/estimator.yaml, the plot defaults), so a change of any of them fails here and says to regenerate.
They also run the notebook's plotting function, multigeom.viz.plot_alignment, on the stored
Stage-5 output and compares the picture with the reference. It runs on CPU in CI. It checks the
plotting step only; the computation of the eigenvectors (the G(x) estimation) is covered by the Fig. 3 / Fig. 4B-C
tests and, for the splitting of the work, by tests/unit/test_subspace_alignment.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_helpers import BAD_PIXEL_FRACTION, MEAN_ABS_TOL, image_difference  # noqa: E402
from multigeom import viz  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fig4_de"
METRICS = FIXTURES / "cross_encoder_metrics.json"
INFO = FIXTURES / "fixture_info.json"
REFERENCE_PNG = FIXTURES / "fig4_de_reference.png"


def render(metrics, path: Path) -> None:
    fig = viz.plot_alignment(metrics)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def test_fixture_has_the_four_comparisons_of_both_metrics():
    named = json.loads(METRICS.read_text())["named"]
    for metric in ("G", "J"):
        for key in ("cross_area_inception", "cross_area_resnet50",
                    "cross_model_V1_inception_resnet50", "cross_model_V4_inception_resnet50"):
            assert f"{key}__{metric}" in named


def test_alignment_is_between_zero_and_one():
    pairs = json.loads(METRICS.read_text())["pairs"]
    assert pairs
    for v in pairs.values():
        a = np.asarray(v["alignment_mean"], dtype=float)
        assert a.shape == (50,) and np.all(a >= -1e-9) and np.all(a <= 1 + 1e-9)


def test_notebook_plot_reproduces_reference(tmp_path):
    render(METRICS, tmp_path / "plot.png")
    mean_abs, bad = image_difference(mpimg.imread(tmp_path / "plot.png"), mpimg.imread(REFERENCE_PNG))
    assert mean_abs < MEAN_ABS_TOL and bad < BAD_PIXEL_FRACTION, (
        f"Fig. 4D-E plot differs from the reference: mean |diff| = {mean_abs:.5f} (tol {MEAN_ABS_TOL}), "
        f"fraction of pixels off by > 0.1 = {bad:.5f} (tol {BAD_PIXEL_FRACTION})"
    )


def test_plot_comparison_detects_changes(tmp_path):
    """Guards against a vacuous comparison: changing the curves of one comparison must fail it."""
    metrics = json.loads(METRICS.read_text())
    pair = metrics["named"]["cross_area_inception__G"]
    metrics["pairs"][pair]["alignment_mean"] = [0.9 for _ in metrics["pairs"][pair]["alignment_mean"]]
    render(metrics, tmp_path / "changed.png")
    mean_abs, bad = image_difference(mpimg.imread(tmp_path / "changed.png"), mpimg.imread(REFERENCE_PNG))
    assert mean_abs >= MEAN_ABS_TOL or bad >= BAD_PIXEL_FRACTION


# ── the parameters the fixture was made with are explicit and consistent ────
def info():
    return json.loads(INFO.read_text())


def test_fixture_info_lists_how_the_fixture_was_made():
    i = info()
    assert {"configs", "per_config", "estimator", "k_max", "plot", "git_commit", "versions"} <= set(i)
    assert {"n_samples", "batch_size", "timesteps", "sigma", "seed", "top_k"} <= set(i["estimator"])
    for key, c in i["per_config"].items():
        assert c["images"] and c["diffusion_epoch"] is not None, key
        assert len(c["encoder_checkpoint_md5"]) == 32 and len(c["diffusion_checkpoint_md5"]) == 32, key


def test_fixture_info_agrees_with_the_stored_metrics():
    i, m = info(), json.loads(METRICS.read_text())
    images = {tuple(k.split("/")): set(v["images"]) for k, v in i["per_config"].items()}
    assert {tuple(c) for c in i["configs"]} == set(images)
    seen = set()
    for pair in m["pairs"].values():
        a = (pair["config_A"]["encoder"], pair["config_A"]["area"])
        b = (pair["config_B"]["encoder"], pair["config_B"]["area"])
        seen |= {a, b}
        assert pair["n_images"] == len(images[a] & images[b]), (a, b)          # alignment is averaged over the shared images
        assert max(pair["k"]) == i["k_max"]
    assert seen == set(images)


def test_fixture_parameters_match_the_current_settings_of_the_repository():
    """If the estimator config or the plot defaults change, the fixture no longer describes this code."""
    import inspect

    from multigeom.config import load_config
    i = info()
    est = load_config("estimator")
    mc = est.monte_carlo
    assert i["estimator"] == {
        "n_samples": int(mc.n_samples), "batch_size": int(mc.batch_size),
        "timesteps": [int(mc.timestep_grid.start), int(mc.timestep_grid.stop), int(mc.timestep_grid.step)],
        "sigma": float(mc.observation_noise_sigma), "seed": int(est.seed), "top_k": int(est.top_k),
    }, "configs/estimator.yaml changed since the fixture was made: regenerate it"
    assert all(c["sigma"] == i["estimator"]["sigma"] for c in i["per_config"].values())
    assert i["plot"]["figsize"] == list(inspect.signature(viz.plot_alignment).parameters["figsize"].default)
