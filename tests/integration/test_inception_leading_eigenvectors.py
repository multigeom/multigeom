"""
Integration tests, inception encoder: do we reproduce the notebook's Fig. 3D-F
(notebooks/1_leading_eigenvectors_of_G.ipynb, section 6: the frog image with the V1/V4 receptive fields, and the
leading eigenvectors of G(x) and J(x) for V1 and V4)?

The reference is stored in tests/integration/fixtures/fig3/ (regenerate with
tests/integration/fixtures/make_fig3_fixtures.py):

    fig3_def_data.npz        data behind the plot (image, receptive-field parameters of the inception
                             readouts, leading 4 eigenvectors of G and J, V1 and V4)
    fig3_def_reference.png   the notebook's Fig. 3D-F plot rendered from that data

  - test_*_plot_*       run on CPU in CI: execute the notebook's own plot cell on the stored data and
                        compare the picture with the reference picture.
  - test_recompute_*    run the notebook's setup, model-loading and compute cells on the real
                        checkpoints and compare with the stored data. Skipped unless runs/ and the
                        TinyImageNet data are present (not the case on GitHub).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_helpers import (  # noqa: E402
    AREAS, BAD_PIXEL_FRACTION, CHECKS, FIXTURES, MEAN_ABS_TOL, fixture_inputs, plot_difference,
)

from nb_fixtures import notebook_run  # noqa: E402,F401  (pytest fixture)

ENCODER = "inception"


def test_fixture_is_wellformed():
    ckpts, records = fixture_inputs()
    for area in AREAS:
        assert ckpts[(ENCODER, area)]["model_state_dict"]["readout.mu"].shape[1] == 2
        for metric in ("G", "J"):
            v = records[(ENCODER, area)][f"{metric}_eigenvectors"]
            assert v.shape[1] == 4096 and v.shape[0] >= 4
            np.testing.assert_allclose(v.double().norm(dim=1).numpy(), 1.0, atol=1e-4)
    assert tuple(records[(ENCODER, "V1")]["image"].shape) == (1, 1, 64, 64)


def test_notebook_plot_reproduces_reference(tmp_path):
    ckpts, records = fixture_inputs()
    mean_abs, bad = plot_difference("fig3", ckpts, records, tmp_path)
    assert mean_abs < MEAN_ABS_TOL and bad < BAD_PIXEL_FRACTION, (
        f"Fig. 3D-F plot differs from the reference: mean |diff| = {mean_abs:.5f} (tol {MEAN_ABS_TOL}), "
        f"fraction of pixels off by > 0.1 = {bad:.5f} (tol {BAD_PIXEL_FRACTION})"
    )


def test_plot_comparison_detects_changes(tmp_path):
    """Guards against a vacuous comparison: changing the eigenvectors must fail it."""
    ckpts, records = fixture_inputs()
    rec = records[(ENCODER, "V1")]
    rec["G_eigenvectors"] = rec["G_eigenvectors"].flip(0)       # reverse the stored eigenvectors
    rec["J_eigenvectors"] = rec["J_eigenvectors"].flip(0)
    mean_abs, bad = plot_difference("fig3", ckpts, records, tmp_path)
    assert mean_abs >= MEAN_ABS_TOL or bad >= BAD_PIXEL_FRACTION


@pytest.mark.slow
def test_recompute_matches_fixture(notebook_run):
    """
    Compare what the notebook computes from the checkpoints with the stored data. J(x) is
    deterministic, so it must agree (up to the sign of each eigenvector, and small numerical
    differences between devices). G(x) is a seeded Monte-Carlo estimate: identical on the device
    the fixture was made on, close on another; only its leading eigenvector is stable enough to
    compare across devices.
    """
    import torch

    ref = np.load(FIXTURES / "fig3_def_data.npz")
    assert int(ref["n_mc"]) == int(notebook_run["N_MC"]), "the notebook's N_MC changed: regenerate the fixtures"
    np.testing.assert_allclose(notebook_run["x"].cpu().numpy().reshape(64, 64), ref["image"], atol=1e-6)
    for area in AREAS:
        for metric, n_check, min_cos in CHECKS:
            got = notebook_run["records"][(ENCODER, area)][f"{metric}_eigenvectors"][:n_check].double()
            want = torch.from_numpy(ref[f"{area}_{metric}"][:n_check]).double()
            cos = (got * want).sum(dim=1).abs()
            assert bool((cos > min_cos).all()), f"{area} {metric}: |cos| per eigenvector = {cos.tolist()}"
