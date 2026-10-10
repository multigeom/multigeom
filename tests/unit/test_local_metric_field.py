"""
Tests for multigeom.pipeline.metric_field_job: the job description and its validation, the cost estimate, the command line of
multigeom.pipeline.metric_field (stage 6), progress counting, clearing only the job's files, and waiting for the process.
No GPU or checkpoints needed.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from multigeom.pipeline import metric_field_job as lmf
from multigeom.pipeline.metric_field_job import MetricFieldJob


def job(**kw):
    return MetricFieldJob(**kw)


def test_defaults_use_sqrt_lambda_for_g_and_a_small_fixed_step_for_j():
    j = job()
    assert (j.img_idx, j.grid_size) == (lmf.DEFAULT_IMG_IDX, 5) and lmf.DEFAULT_IMG_IDX == 64
    assert j.steps == {"G": ("sqrt_lambda", 0.5), "J": ("fixed", 0.3)}
    assert (j.reference_encoder, j.other_encoder, j.areas, j.metrics) == ("inception", "resnet50", ("V1", "V4"), ("G", "J"))


@pytest.mark.parametrize("bad", [dict(grid_size=4), dict(grid_size=0), dict(img_idx=-1),
                                 dict(other_encoder="inception"), dict(reference_encoder="vgg"),
                                 dict(areas=("V2",)), dict(areas=()), dict(metrics=("H",)), dict(metrics=()),
                                 dict(steps={"G": ("random", 0.5)}), dict(steps={"J": ("fixed", 0)}),
                                 dict(steps={"H": ("fixed", 0.3)}), dict(device="tpu")])
def test_job_validation(bad):
    with pytest.raises(ValueError):
        job(**bad)


def test_steps_can_be_overridden_per_metric_and_round_trip_through_json():
    j = job(steps={"J": ("fixed", 0.1)})
    assert j.steps == {"G": ("sqrt_lambda", 0.5), "J": ("fixed", 0.1)}          # G keeps its default
    assert lmf.job_from_json(lmf.job_to_json(j)) == j


def test_single_strings_are_accepted_for_areas_and_metrics():
    j = job(areas="V1", metrics="J")
    assert j.areas == ("V1",) and j.metrics == ("J",)


def test_cost_estimate_follows_the_table_of_the_notebook():
    # grid points x 2 encoders + the reference image, per area
    est = lmf.estimate_cost(job(grid_size=5), seconds_per_g=15.0, seconds_per_j=0.0)
    assert est.g_estimates == (1 + 25 * 2) * 2 == 102 and est.minutes == pytest.approx(102 * 15 / 60)    # ~26 min
    assert lmf.estimate_cost(job(grid_size=5, areas="V1"), seconds_per_j=0).g_estimates == 51
    assert lmf.estimate_cost(job(grid_size=3), seconds_per_j=0).g_estimates == 38
    j_only = lmf.estimate_cost(job(grid_size=3, areas="V1", metrics="J"))
    assert j_only.g_estimates == 0 and j_only.j_evaluations == 19 and j_only.seconds < 10          # seconds


def test_command_uses_only_arguments_of_the_metric_field_module(repo_root):
    import importlib
    stage6 = importlib.import_module("multigeom.pipeline.metric_field")
    import argparse
    parser_args = set()
    real = argparse.ArgumentParser.add_argument

    def record(self, *names, **kw):
        parser_args.update(n for n in names if n.startswith("--"))
        return real(self, *names, **kw)
    argparse.ArgumentParser.add_argument = record
    try:
        stage6.parse_args([])
    finally:
        argparse.ArgumentParser.add_argument = real
    j = job(grid_size=3, areas=("V1",), device="cpu")
    cmds = lmf.metric_field_commands(j, out="x")
    assert len(cmds) == 2                                                   # one run per metric: G, then J
    for cmd, (metric, rule, size) in zip(cmds, (("G", "sqrt_lambda", "0.5"), ("J", "fixed", "0.3"))):
        options = {c for c in cmd if c.startswith("--")}
        assert options <= parser_args and {"--areas", "--metrics", "--grid-size", "--out-dir"} <= options
        assert cmd[cmd.index("--metrics") + 1] == metric
        assert (cmd[cmd.index("--step-rule") + 1], cmd[cmd.index("--step-size") + 1]) == (rule, size)
        assert cmd[cmd.index("--device") + 1] == "cpu" and cmd[cmd.index("--areas") + 1] == "V1"
    assert lmf.metric_field_command(job(), "G", out=None).count("--out-dir") == 0
    assert len(lmf.metric_field_commands(job(metrics=("J",)))) == 1


def test_expected_files_and_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(lmf, "REPO", tmp_path)
    j = job(img_idx=3, areas=("V1", "V4"), metrics=("G", "J"))
    d = tmp_path / "runs" / "metric_field" / "img0003"
    assert [f.name for f in lmf.expected_files(j)] == ["G_V1.pt", "G_V4.pt", "J_V1.pt", "J_V4.pt"]
    assert lmf.progress(j) == (0, 4) and not lmf.is_complete(j)
    d.mkdir(parents=True)
    for name in ("G_V1.pt", "J_V4.pt"):
        (d / name).write_bytes(b"x")
    assert lmf.progress(j) == (2, 4)
    for name in ("G_V4.pt", "J_V1.pt"):
        (d / name).write_bytes(b"x")
    assert lmf.is_complete(j)


def test_clear_results_only_touches_the_jobs_files(tmp_path, monkeypatch):
    monkeypatch.setattr(lmf, "REPO", tmp_path)
    d = tmp_path / "runs" / "metric_field" / "img0000"
    other = tmp_path / "runs" / "metric_field" / "img0001"
    for folder in (d, other):
        folder.mkdir(parents=True)
        for name in ("G_V1.pt", "G_V4.pt"):
            (folder / name).write_bytes(b"x")
    j = job(img_idx=0, areas="V1", metrics="G")
    assert len(lmf.clear_results(j, dry_run=True)) == 1 and (d / "G_V1.pt").exists()
    assert len(lmf.clear_results(j)) == 1
    assert not (d / "G_V1.pt").exists() and (d / "G_V4.pt").exists()          # the other area stays
    assert (other / "G_V1.pt").exists()                                       # the other image stays


def test_clear_results_refuses_while_running(tmp_path, monkeypatch):
    monkeypatch.setattr(lmf, "REPO", tmp_path)
    logs = tmp_path / "runs" / "logs"
    logs.mkdir(parents=True)
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    (logs / "metric_field_worker.json").write_text('[{"pid": %d}]' % p.pid)
    try:
        with pytest.raises(RuntimeError, match="still running"):
            lmf.clear_results(job())
    finally:
        p.kill()
        p.wait()


def test_wait_returns_when_the_process_has_written_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(lmf, "REPO", tmp_path)
    j = job(img_idx=0, areas="V1", metrics="J")
    d = tmp_path / "runs" / "metric_field" / "img0000"
    d.mkdir(parents=True)
    code = f"import time, pathlib; time.sleep(1.0); pathlib.Path(r'{d}/J_V1.pt').write_bytes(b'x')"
    p = subprocess.Popen([sys.executable, "-c", code])
    assert lmf.wait(j, {"pid": p.pid}, poll_s=0.2, verbose=False) is True


def test_wait_reports_an_incomplete_run(tmp_path, monkeypatch):
    monkeypatch.setattr(lmf, "REPO", tmp_path)
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert lmf.wait(job(areas="V1", metrics="J"), {"pid": p.pid}, poll_s=0.1, verbose=False) is False


def test_stop_stops_the_recorded_process(tmp_path, monkeypatch):
    monkeypatch.setattr(lmf, "REPO", tmp_path)
    logs = tmp_path / "runs" / "logs"
    logs.mkdir(parents=True)
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    (logs / "metric_field_worker.json").write_text('[{"pid": %d}]' % p.pid)
    assert lmf.stop() == 1
    p.wait(timeout=5)
    assert lmf.stop() == 0


def test_plot_needs_results():
    with pytest.raises(ValueError, match="no results"):
        lmf.plot_metric_field({})


# ── describing a field: radial orientation and size ratio ───────────────────
def _field(rule, coeffs):
    """[n, n, 2, 2] metrics whose major axis follows rule(a, b) -> unit direction."""
    import numpy as np
    n = len(coeffs)
    M = np.empty((n, n, 2, 2))
    for i, a in enumerate(coeffs):
        for j, b in enumerate(coeffs):
            d = np.asarray(rule(a, b), dtype=float)
            M[i, j] = np.eye(2) + 8.0 * np.outer(d, d)
    return M


def test_radial_score_is_one_for_a_radial_field_and_lower_for_a_uniform_one():
    import numpy as np
    coeffs = np.arange(-2, 3)
    radial = _field(lambda a, b: (a, b) / np.hypot(a, b) if (a or b) else (1.0, 0.0), coeffs)
    horizontal = _field(lambda a, b: (1.0, 0.0), coeffs)
    assert lmf.radial_score(radial, coeffs) == pytest.approx(1.0, abs=1e-9)
    assert lmf.radial_score(horizontal, coeffs) < 0.8


def test_area_ratio_of_a_scaled_field():
    import numpy as np
    coeffs = np.arange(-2, 3)
    a = _field(lambda a, b: (1.0, 0.0), coeffs)
    assert lmf.area_ratio(a, 0.25 * a) == pytest.approx(4.0)         # sqrt(det) scales with the matrix: 0.25 -> 1/4 the area
    assert lmf.area_ratio(a, a) == pytest.approx(1.0)


def test_the_default_reference_image_is_the_best_one_of_the_stored_scan():
    """analysis/results/metric_field_scan.jsonl is the scan of tools/scan_metric_field_images.py (200 images, 6 settings)."""
    import json
    rows = [json.loads(line) for line in (lmf.REPO / "analysis" / "results" / "metric_field_scan.jsonl").read_text().splitlines()]
    assert len(rows) == 200 * 6
    both = [r for r in rows if "ratio_V4" in r and min(r["ratio_V1"], r["ratio_V4"]) >= 2.5]        # radial >= 0.7 and Inception outside
    best = max(both, key=lambda r: min(r["radial_V1"], r["radial_V4"]))
    assert best["img"] == lmf.DEFAULT_IMG_IDX and best["rule"] == "fixed" and best["size"] in (0.2, 0.3)
    assert min(best["radial_V1"], best["radial_V4"]) > 0.8
