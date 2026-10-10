"""
Tests for src.bures_wasserstein: the summary of the Stage-5 inputs, the ordering the paper reports for the
Bures-Wasserstein distances (cross-model below cross-area for G(x), reversed for J(x)), and the messages. Synthetic
data only; no GPU or checkpoints needed.
"""
from __future__ import annotations

import pytest

from src import bures_wasserstein as bw
from src import subspace_alignment as sa

AREAS, ENCODERS = bw.AREAS, bw.ENCODERS


def make_metrics(model_g, area_g, model_j, area_j, k=5):
    """A Stage-5-like dict with constant d_BW curves: cross-model / cross-area for G and J."""
    pairs, named = {}, {}

    def add(key, value, metric):
        pair = f"pair_{key}__{metric}"
        pairs[pair] = {"bures_wasserstein_mean": [value] * k, "k": list(range(1, k + 1))}
        named[f"{key}__{metric}"] = pair

    for metric, model, area in (("G", model_g, area_g), ("J", model_j, area_j)):
        for a in AREAS:
            add(f"cross_model_{a}_inception_resnet50", model, metric)
        for e in ENCODERS:
            add(f"cross_area_{e}", area, metric)
    return {"pairs": pairs, "named": named}


def test_ordering_matches_the_paper_when_g_is_model_below_area_and_j_reversed():
    r = bw.ordering(make_metrics(model_g=0.5, area_g=1.0, model_j=60.0, area_j=45.0))
    assert r["G"]["matches_paper"] and r["J"]["matches_paper"]
    assert r["G"]["cross_model"] == pytest.approx(0.5) and r["J"]["cross_area"] == pytest.approx(45.0)


def test_ordering_detects_a_mismatch():
    r = bw.ordering(make_metrics(model_g=2.0, area_g=1.0, model_j=10.0, area_j=45.0))
    assert not r["G"]["matches_paper"] and not r["J"]["matches_paper"]


def test_ordering_leaves_out_missing_comparisons():
    m = make_metrics(0.5, 1.0, 60.0, 45.0)
    for key in [k for k in m["named"] if k.startswith("cross_area") and k.endswith("__J")]:
        del m["named"][key]
    r = bw.ordering(m)
    assert r["G"] is not None and r["J"] is None


def test_print_ordering_says_what_the_paper_expects(capsys):
    bw.print_ordering(make_metrics(0.5, 1.0, 60.0, 45.0))
    out = capsys.readouterr().out
    assert "G: mean d_BW over k" in out and "cross-model below cross-area: yes" in out
    assert "J: mean d_BW over k" in out and "cross-model above cross-area: yes" in out


def _per_image(tmp_path, spec):
    for (enc, area), indices in spec.items():
        d = tmp_path / "runs" / "per_image" / enc / area
        d.mkdir(parents=True)
        for i in indices:
            (d / f"img{i:04d}.pt").write_bytes(b"x")


def test_input_summary_counts_images_and_shared_images(tmp_path, monkeypatch):
    for module in (bw, sa):
        monkeypatch.setattr(module, "REPO", tmp_path)
    _per_image(tmp_path, {("inception", "V1"): [0, 1, 2], ("inception", "V4"): [1, 2, 3], ("resnet50", "V1"): [5]})
    s = bw.input_summary()
    assert s["images"][("inception", "V1")] == [0, 1, 2] and s["images"][("resnet50", "V1")] == [5]
    assert s["shared"][(("inception", "V1"), ("inception", "V4"))] == 2          # images 1 and 2
    assert s["shared"][(("inception", "V1"), ("resnet50", "V1"))] == 0


def test_print_inputs_warns_when_there_are_fewer_images_than_the_paper(tmp_path, monkeypatch, capsys):
    for module in (bw, sa):
        monkeypatch.setattr(module, "REPO", tmp_path)
    _per_image(tmp_path, {("inception", "V1"): [0, 1], ("inception", "V4"): [0, 1]})
    bw.print_inputs()
    out = capsys.readouterr().out
    assert "inception  V1:   2 images" in out and "the paper averages over 200" in out


def test_print_inputs_without_data_points_to_tutorial_2(tmp_path, monkeypatch, capsys):
    for module in (bw, sa):
        monkeypatch.setattr(module, "REPO", tmp_path)
    assert bw.print_inputs()["images"] == {}
    assert "tutorial 2" in capsys.readouterr().out
