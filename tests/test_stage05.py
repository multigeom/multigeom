"""
Stage 5 (stages/05_compute_metrics.py) on a synthetic runs/per_image tree.

Expected values are recomputed per image with src.metrics.subspace_alignment
and src.metrics.bures_wasserstein (which are themselves tested against
principal angles and a dense reference in test_metrics.py) and averaged here.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from conftest import load_stage
from src.metrics import bures_wasserstein, subspace_alignment

P, K = 30, 6
CONFIGS = [("inception", "V1"), ("inception", "V4"), ("resnet50", "V1"), ("resnet50", "V4")]


def _eig(seed: int, k: int = K):
    g = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(g.standard_normal((P, k)))
    vals = np.sort(g.uniform(0.1, 10, k))[::-1].copy()
    return torch.from_numpy(vals).float(), torch.from_numpy(Q.T.copy()).float()


def _write_tree(root, missing=(), no_fisher=()):
    per = {}
    for ci, (enc, area) in enumerate(CONFIGS):
        d = root / "per_image" / enc / area
        d.mkdir(parents=True)
        for i in range(4):
            if (enc, area, i) in missing:
                continue
            gv, gV = _eig(100 * ci + i)
            jv, jV = _eig(10_000 + 100 * ci + i)
            rec = {"img_idx": i, "G_eigenvalues": gv, "G_eigenvectors": gV,
                   "J_eigenvalues": None if (enc, area) in no_fisher else jv,
                   "J_eigenvectors": None if (enc, area) in no_fisher else jV}
            torch.save(rec, d / f"img{i:04d}.pt")
            per[(enc, area, i)] = rec
    return per


def _run(tmp_path, monkeypatch, max_k=K):
    stage = load_stage("05_compute_metrics.py")
    out = tmp_path / "metrics"
    monkeypatch.setattr(sys, "argv", ["05", "--runs-dir", str(tmp_path), "--out-dir", str(out),
                                      "--max-k", str(max_k)])
    stage.main()
    return json.loads((out / "cross_encoder_metrics.json").read_text())


def test_structure_and_named_comparisons(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    res = _run(tmp_path, monkeypatch)
    assert len(res["pairs"]) == 6 * 2                         # C(4,2) pairs × {G, J}
    expected_named = {f"{n}__{m}" for m in ("G", "J") for n in (
        "cross_area_inception", "cross_area_resnet50",
        "cross_model_V1_inception_resnet50", "cross_model_V4_inception_resnet50")}
    assert set(res["named"]) == expected_named
    st = res["pairs"][res["named"]["cross_area_inception__G"]]
    assert st["config_A"] == {"encoder": "inception", "area": "V1"}
    assert st["config_B"] == {"encoder": "inception", "area": "V4"}
    assert st["k"] == list(range(1, K + 1)) and st["n_images"] == 4
    assert "git_commit" in res["provenance"]


@pytest.mark.parametrize("metric", ["G", "J"])
def test_curves_equal_mean_of_per_image_metrics(tmp_path, monkeypatch, metric):
    per = _write_tree(tmp_path)
    res = _run(tmp_path, monkeypatch)
    st = res["pairs"][res["named"][f"cross_model_V4_inception_resnet50__{metric}"]]
    for k in range(1, K + 1):
        al, bw = [], []
        for i in range(4):
            a, b = per[("inception", "V4", i)], per[("resnet50", "V4", i)]
            al.append(subspace_alignment(a[f"{metric}_eigenvectors"],
                                         b[f"{metric}_eigenvectors"], top_k=k))
            bw.append(bures_wasserstein(a[f"{metric}_eigenvalues"], a[f"{metric}_eigenvectors"],
                                        b[f"{metric}_eigenvalues"], b[f"{metric}_eigenvectors"],
                                        top_k=k))
        assert st["alignment_mean"][k - 1] == pytest.approx(np.mean(al), rel=1e-5)
        assert st["alignment_sem"][k - 1] == pytest.approx(np.std(al, ddof=1) / 2, rel=1e-4,
                                                           abs=1e-9)
        assert st["bures_wasserstein_mean"][k - 1] == pytest.approx(np.mean(bw), rel=1e-5)


def test_only_shared_images_and_missing_fisher(tmp_path, monkeypatch):
    _write_tree(tmp_path, missing={("resnet50", "V1", 2)}, no_fisher={("resnet50", "V4")})
    res = _run(tmp_path, monkeypatch, max_k=3)
    st = res["pairs"][res["named"]["cross_model_V1_inception_resnet50__G"]]
    assert st["n_images"] == 3 and st["k"] == [1, 2, 3]
    assert "cross_area_resnet50__J" not in res["named"]
    assert "cross_area_resnet50__G" in res["named"]
