"""
Integration tests, resnet50 encoder: do we reproduce the notebook's Fig. 4B-C
(tutorials/demo_inference.ipynb, section 6: the leading eigenvectors of G(x) and J(x) of the
inception and resnet50 encoders, V1 block above the V4 block)?

The plot is the paper's comparison of architectures, so it also shows the inception rows (the
reference the resnet50 eigenvectors are matched to for display); what is specific to this module
is the resnet50 data: its leading 30 eigenvectors per metric and area, from which the displayed
ones are picked.

The reference is stored in tests/integration/fixtures/fig3/ (regenerate with
tests/integration/fixtures/make_fig3_fixtures.py):

    fig4_bc_data.npz         leading eigenvectors of G and J per encoder and area
                             (4 for inception, 30 for resnet50)
    fig4_bc_reference.png    the notebook's Fig. 4B-C plot rendered from that data

  - test_*_plot_*       run on CPU in CI: execute the notebook's own plot cell on the stored data and
                        compare the picture with the reference picture.
  - test_recompute_*    run the notebook's cells on the real checkpoints and compare the resnet50
                        eigenvectors with the stored data. Skipped unless runs/ and the TinyImageNet
                        data are present (not the case on GitHub).
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
    AREAS, BAD_PIXEL_FRACTION, CHECKS, FIXTURES, MEAN_ABS_TOL, N_FIG4_POOL, fixture_inputs, plot_difference,
)

ENCODER = "resnet50"


def test_fixture_is_wellformed():
    _, records = fixture_inputs()
    for area in AREAS:
        for metric in ("G", "J"):
            v = records[(ENCODER, area)][f"{metric}_eigenvectors"]
            assert tuple(v.shape) == (N_FIG4_POOL, 4096)
            np.testing.assert_allclose(v.double().norm(dim=1).numpy(), 1.0, atol=1e-4)


def test_notebook_plot_reproduces_reference(tmp_path):
    ckpts, records = fixture_inputs()
    mean_abs, bad = plot_difference("fig4", ckpts, records, tmp_path)
    assert mean_abs < MEAN_ABS_TOL and bad < BAD_PIXEL_FRACTION, (
        f"Fig. 4B-C plot differs from the reference: mean |diff| = {mean_abs:.5f} (tol {MEAN_ABS_TOL}), "
        f"fraction of pixels off by > 0.1 = {bad:.5f} (tol {BAD_PIXEL_FRACTION})"
    )


def test_plot_comparison_detects_changes(tmp_path):
    """Guards against a vacuous comparison: other resnet50 eigenvectors must fail it."""
    ckpts, records = fixture_inputs()
    records[(ENCODER, "V1")]["G_eigenvectors"] = records[(ENCODER, "V4")]["G_eigenvectors"]
    records[(ENCODER, "V1")]["J_eigenvectors"] = records[(ENCODER, "V4")]["J_eigenvectors"]
    mean_abs, bad = plot_difference("fig4", ckpts, records, tmp_path)
    assert mean_abs >= MEAN_ABS_TOL or bad >= BAD_PIXEL_FRACTION


@pytest.mark.slow
def test_recompute_matches_fixture(notebook_run):
    """
    Compare the resnet50 eigenvectors the notebook computes from the checkpoints with the stored
    ones: J(x) must agree (up to sign and small numerical differences between devices); of the
    seeded Monte-Carlo estimate G(x) only the leading eigenvector is compared across devices.
    """
    import torch

    ref = np.load(FIXTURES / "fig4_bc_data.npz")
    assert int(ref["n_mc"]) == int(notebook_run["N_MC"]), "the notebook's N_MC changed: regenerate the fixtures"
    for area in AREAS:
        for metric, n_check, min_cos in CHECKS:
            got = notebook_run["records"][(ENCODER, area)][f"{metric}_eigenvectors"][:n_check].double()
            want = torch.from_numpy(ref[f"{ENCODER}_{area}_{metric}"][:n_check]).double()
            cos = (got * want).sum(dim=1).abs()
            assert bool((cos > min_cos).all()), f"{area} {metric}: |cos| per eigenvector = {cos.tolist()}"
