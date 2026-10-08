"""
Integration tests: do we reproduce the notebook's Fig. 3D-F (tutorials/demo_inference.ipynb, section 6)?

The reference is stored in tests/integration/fixtures/fig3/ (regenerate with
tests/integration/fixtures/make_fig3_fixtures.py):

    fig3_def_data.npz        the data behind the notebook plot (image, receptive-field parameters,
                             leading 4 eigenvectors of G and J, V1 and V4)
    fig3_def_reference.png   the notebook plot rendered from that data

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

import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_helpers import (  # noqa: E402
    COMPUTE, FIXTURES, LOAD_MODELS, REPO, SETUP, fixture_inputs, render_plot_cell, run_cells,
)

REFERENCE_PNG = FIXTURES / "fig3_def_reference.png"
MEAN_ABS_TOL = 1e-3          # mean |difference| over all pixels, in [0, 1]; text rendering may differ slightly
BAD_PIXEL_FRACTION = 2e-3    # at most this fraction of pixels may differ by more than 0.1

# (metric, number of leading eigenvectors compared, minimum |cosine| with the stored one).
# Measured, CPU against the GPU fixture: J |cos| >= 0.9958 for all four eigenvectors;
# G |cos| = 0.96 (V1) and 0.98 (V4) for the leading one, but down to 0.30 for the 3rd (V4).
CHECKS = (("J", 4, 0.99), ("G", 1, 0.9))


def image_difference(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    assert a.shape == b.shape, f"image shapes differ: {a.shape} vs {b.shape}"
    d = np.abs(a.astype(float) - b.astype(float))
    return float(d.mean()), float((d > 0.1).mean())


# ── fixtures ────────────────────────────────────────────────────────────────
def test_fixture_is_wellformed():
    ckpts, records = fixture_inputs()
    for area in ("V1", "V4"):
        assert ckpts[area]["model_state_dict"]["readout.mu"].shape[1] == 2
        for metric in ("G", "J"):
            v = records[area][f"{metric}_eigenvectors"]
            assert tuple(v.shape) == (4, 4096)
            np.testing.assert_allclose(v.double().norm(dim=1).numpy(), 1.0, atol=1e-4)
    assert tuple(records["V1"]["image"].shape) == (1, 1, 64, 64)
    assert REFERENCE_PNG.exists()


# ── the plot (CPU, runs in CI) ──────────────────────────────────────────────
def test_notebook_plot_reproduces_reference(tmp_path):
    ckpts, records = fixture_inputs()
    out = tmp_path / "plot.png"
    render_plot_cell(ckpts, records, out)
    mean_abs, bad = image_difference(mpimg.imread(out), mpimg.imread(REFERENCE_PNG))
    assert mean_abs < MEAN_ABS_TOL and bad < BAD_PIXEL_FRACTION, (
        f"plot differs from the reference: mean |diff| = {mean_abs:.5f} (tol {MEAN_ABS_TOL}), "
        f"fraction of pixels off by > 0.1 = {bad:.5f} (tol {BAD_PIXEL_FRACTION})"
    )


def test_plot_comparison_detects_changes(tmp_path):
    """Guards against a vacuous comparison: changing one eigenvector must fail it."""
    ckpts, records = fixture_inputs()
    records["V1"]["G_eigenvectors"] = records["V1"]["G_eigenvectors"].flip(0)   # reorder the 4 eigenvectors
    out = tmp_path / "changed.png"
    render_plot_cell(ckpts, records, out)
    mean_abs, bad = image_difference(mpimg.imread(out), mpimg.imread(REFERENCE_PNG))
    assert mean_abs >= MEAN_ABS_TOL or bad >= BAD_PIXEL_FRACTION


# ── recompute from the checkpoints (needs runs/ and data; not on GitHub) ────
def _have_inputs() -> bool:
    needed = [REPO / "data/tiny-imagenet-200/train/n01644900/images/n01644900_0.JPEG"]
    for area in ("V1", "V4"):
        needed += [REPO / f"runs/encoders/inception/{area}/best_model.pt",
                   REPO / f"runs/diffusion/inception/{area}/final_model.pt"]
    return all(p.exists() for p in needed)


@pytest.mark.slow
@pytest.mark.skipif(not _have_inputs(), reason="needs runs/ checkpoints and TinyImageNet (see README)")
def test_recompute_matches_fixture(monkeypatch):
    """
    Run the notebook's setup, model-loading and 6.1 compute cells, then compare with the
    stored data. J(x) is deterministic, so it must agree (up to the sign of each eigenvector,
    and small numerical differences between devices). G(x) is a seeded Monte-Carlo estimate:
    identical on the device the fixture was made on, close on another; only its leading
    eigenvector is stable enough to compare across devices.
    """
    import torch

    monkeypatch.chdir(REPO)
    ns = run_cells({}, SETUP, LOAD_MODELS, COMPUTE)
    ref = np.load(FIXTURES / "fig3_def_data.npz")
    assert int(ref["n_mc"]) == int(ns["N_MC"]), "the notebook's N_MC changed: regenerate the fixtures"

    np.testing.assert_allclose(ns["x"].cpu().numpy().reshape(64, 64), ref["image"], atol=1e-6)
    for area in ("V1", "V4"):
        for metric, n_check, min_cos in CHECKS:
            got = ns["records"][area][f"{metric}_eigenvectors"][:n_check].double()
            want = torch.from_numpy(ref[f"{area}_{metric}"][:n_check]).double()
            cos = (got * want).sum(dim=1).abs()
            assert bool((cos > min_cos).all()), f"{area} {metric}: |cos| per eigenvector = {cos.tolist()}"
