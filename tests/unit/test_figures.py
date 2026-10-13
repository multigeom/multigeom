"""
Smoke test for the figure and table scripts, plus checks of the 2 × 2
projection and ellipse convention of multigeom/geometry/metric_field.py (Fig. 5).

The smoke test writes a small synthetic runs/ tree (Stage-1 checkpoints,
Stage-4 per-image files, Stage-5 JSON, Stage-6 metric fields) into tmp_path,
with the OnOff configurations deliberately missing, and runs every script's
main(). It needs neither diffusers nor pretrained weights.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from multigeom.geometry.metric_field import (  # noqa: E402
    ellipse_grid, ellipse_params, grid_coefficients, perturbation_grid, project_metric,
)

P = 64 * 64
K = 8
N_NEURONS = 50
SIDE = {"inception": 12, "resnet50": 16}


def _orthonormal_rows(k: int, g: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(P, k, generator=g))
    return q.T.contiguous()                                    # [k, P]


def _write_runs(root: Path) -> None:
    g = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    image = torch.rand(1, 1, 64, 64, generator=g) * 2 - 1
    for enc, side in SIDE.items():
        for area in ("V1", "V4"):
            resp = torch.randn(30, N_NEURONS, generator=g)
            pred = resp + 0.5 * torch.randn(30, N_NEURONS, generator=g)
            r = rng.uniform(0.5, 0.9, N_NEURONS)
            oracle = rng.uniform(0.85, 0.95, N_NEURONS)
            d = root / "encoders" / enc / area
            d.mkdir(parents=True)
            torch.save({
                "model_state_dict": {
                    "readout.mu": torch.rand(N_NEURONS, 2, generator=g) * 0.8 - 0.4,
                    "readout.sigma": torch.rand(N_NEURONS, generator=g) * 0.2,
                    "readout.grid_x": torch.linspace(-1, 1, side),
                },
                "test_predictions": pred, "test_responses": resp,
                "per_neuron_test_corr": r, "per_neuron_test_corr_noise_corrected": r / oracle,
                "oracle": oracle, "test_corr": float(r.mean()),
            }, d / "best_model.pt")
            d = root / "per_image" / enc / area
            d.mkdir(parents=True)
            torch.save({
                "img_idx": 0, "image": image,
                "G_eigenvalues": torch.linspace(2, 0.1, K),
                "G_eigenvectors": _orthonormal_rows(K, g),
                "J_eigenvalues": torch.linspace(9, 0.5, K),
                "J_eigenvectors": _orthonormal_rows(K, g),
            }, d / "img0000.pt")

    ks = list(range(1, K + 1))
    pairs, named = {}, {}
    comparisons = {
        "cross_area_inception": ("inception_V1", "inception_V4"),
        "cross_area_resnet50": ("resnet50_V1", "resnet50_V4"),
        "cross_model_V1_inception_resnet50": ("inception_V1", "resnet50_V1"),
        "cross_model_V4_inception_resnet50": ("inception_V4", "resnet50_V4"),
    }
    for name, (a, b) in comparisons.items():
        for metric in ("G", "J"):
            label = f"{a}__VS__{b}__{metric}"
            pairs[label] = {
                "k": ks, "n_images": 3,
                "alignment_mean": rng.uniform(0.2, 0.8, K).tolist(),
                "alignment_sem": [0.02] * K,
                "bures_wasserstein_mean": np.linspace(0.5, 2, K).tolist(),
                "bures_wasserstein_sem": [0.05] * K,
            }
            named[f"{name}__{metric}"] = label
    (root / "metrics").mkdir()
    with open(root / "metrics" / "cross_encoder_metrics.json", "w") as f:
        json.dump({"pairs": pairs, "named": named}, f)

    coeffs = grid_coefficients(3)
    d = root / "metric_field" / "img0000"
    d.mkdir(parents=True)
    for metric in ("G", "J"):
        for area in ("V1", "V4"):
            V = _orthonormal_rows(2, g)
            images, clamp = perturbation_grid(image, V[0], V[1], 0.3, 0.2, coeffs)
            matrices = {}
            for enc in ("inception", "resnet50"):
                M = torch.empty(3, 3, 2, 2, dtype=torch.float64)
                for i in range(3):
                    for j in range(3):
                        M[i, j] = project_metric(torch.rand(K, generator=g) * 3,
                                                 _orthonormal_rows(K, g), V)
                matrices[enc] = M
            torch.save({
                "img_idx": 0, "metric": metric, "area": area,
                "encoders": ["inception", "resnet50"],
                "grid_coefficients": torch.from_numpy(coeffs),
                "x0": image, "perturbed_images": images, "clamp_fraction": clamp,
                "v1": V[0], "v2": V[1], "lambda1": 1.0, "lambda2": 0.5,
                "eps1": 0.3, "eps2": 0.2, "matrices_2x2": matrices,
                "ellipses": {e: torch.from_numpy(ellipse_grid(m.numpy()))
                             for e, m in matrices.items()},
            }, d / f"{metric}_{area}.pt")


@pytest.mark.slow
def test_figure_scripts_write_outputs(tmp_path):
    runs = tmp_path / "runs"
    out = tmp_path / "out"
    _write_runs(runs)
    expected = {
        "table1": ["table1.md", "table1.tex"],
        "fig3": ["fig3_img0000.pdf", "fig3_img0000.png"],
        "fig4": ["fig4_img0000.pdf", "fig4_img0000.png"],
        "fig5": ["fig5_img0000.pdf", "fig5_img0000.png"],
        "fig_app_bw": ["fig_app_bw.pdf", "fig_app_bw.png"],
    }
    for name, files in expected.items():
        module = importlib.import_module(f"multigeom.viz.{name}")
        argv = ["--runs-dir", str(runs), "--out-dir", str(out)]
        if name in ("fig3", "fig4", "fig5"):
            argv += ["--img-idx", "0"]
        module.main(argv)
        for fname in files:
            assert (out / fname).stat().st_size > 0, fname
    table = (out / "table1.tex").read_text()
    assert "Inception" in table and "OnOff" not in table      # missing OnOff is skipped


def test_project_metric_recovers_eigenvalues_in_own_basis():
    g = torch.Generator().manual_seed(1)
    vecs = _orthonormal_rows(5, g)
    lam = torch.tensor([4.0, 1.0, 0.5, 0.2, 0.1])
    M2 = project_metric(lam, vecs, vecs[:2])
    assert torch.allclose(M2, torch.diag(torch.tensor([4.0, 1.0], dtype=torch.float64)),
                          atol=1e-6)
    # A plane orthogonal to every eigenvector sees no metric.
    q, _ = torch.linalg.qr(torch.cat([vecs.T, torch.randn(P, 2, generator=g)], dim=1))
    assert torch.allclose(project_metric(lam, vecs, q[:, 5:].T),
                          torch.zeros(2, 2, dtype=torch.float64), atol=1e-6)


def test_ellipse_axes_are_twice_root_eigenvalues():
    w, h, ang = ellipse_params(np.diag([4.0, 1.0]))
    assert (w, h) == pytest.approx((4.0, 2.0))
    assert abs(np.sin(np.radians(ang))) == pytest.approx(0.0, abs=1e-9)
    w, h, ang = ellipse_params(np.diag([1.0, 9.0]))           # major axis along e2
    assert (w, h) == pytest.approx((6.0, 2.0))
    assert abs(np.cos(np.radians(ang))) == pytest.approx(0.0, abs=1e-9)


def test_style_keeps_the_notebook_backend_in_a_jupyter_kernel(monkeypatch):
    """In a kernel, importing the figure style must not switch to Agg (a figure would show as text, not a picture)."""
    import sys
    import types

    from multigeom.viz import style

    fake = types.ModuleType("IPython")
    fake.get_ipython = lambda: types.SimpleNamespace(kernel=object())
    monkeypatch.setitem(sys.modules, "IPython", fake)
    assert style._in_notebook() is True
    fake.get_ipython = lambda: None
    assert style._in_notebook() is False
    monkeypatch.delitem(sys.modules, "IPython")
    monkeypatch.setitem(sys.modules, "IPython", None)               # not installed
    assert style._in_notebook() is False


@pytest.mark.parametrize("layout", ["older paper checkpoint", "NeuralEncoder.save"])
def test_encoder_checkpoint_gives_the_scores_in_both_layouts(tmp_path, layout):
    from multigeom.viz.style import encoder_checkpoint

    scores = {"per_neuron_test_corr": np.array([0.5, 0.6]), "oracle": np.array([0.9, 0.9]), "test_corr": 0.55}
    ckpt = {"model_state_dict": {"readout.mu": torch.zeros(2, 2)}}
    ckpt.update(scores if layout == "older paper checkpoint" else {"metadata": {"area": "V1", **scores}})
    d = tmp_path / "encoders" / "inception" / "V1"
    d.mkdir(parents=True)
    torch.save(ckpt, d / "best_model.pt")
    got = encoder_checkpoint(tmp_path, "inception", "V1", "test")
    np.testing.assert_array_equal(got["per_neuron_test_corr"], scores["per_neuron_test_corr"])
    assert got["test_corr"] == 0.55 and "model_state_dict" in got
    assert encoder_checkpoint(tmp_path, "inception", "V4", "test") is None
