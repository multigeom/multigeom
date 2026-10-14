"""
Tests for multigeom.pipeline.distribute (and the progress line of multigeom.utils): sharding of the G(x) estimation work, its command line, scenario
presets, the cost estimate and budget guard, the local-GPU cap, Stage 5 with too few configurations,
and that `modal` is not needed to import the module. No GPU, Modal account or checkpoints needed.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from multigeom.pipeline import distribute as sa
from multigeom.utils import ProgressLine, subprocess_env
from multigeom.pipeline.distribute import Shard, make_shards

CONFIGS = list(sa.ALL_CONFIGS)


def covered(workers):
    """The set of (encoder, area, image) triples covered, with a check that none is covered twice."""
    seen = []
    for shards in workers:
        for s in shards:
            seen += [(s.encoder, s.area, i) for i in range(s.start, s.end)]
    assert len(seen) == len(set(seen)), "an image is in two shards"
    return set(seen)


@pytest.mark.parametrize("n_configs", [1, 2, 4])
@pytest.mark.parametrize("n_images", [1, 7, 200])
@pytest.mark.parametrize("num_workers", [0, 1, 3, 10, 50, 1000])
def test_make_shards_covers_every_image_exactly_once(n_configs, n_images, num_workers):
    configs = CONFIGS[:n_configs]
    workers = make_shards(configs, n_images, num_workers)
    assert covered(workers) == {(e, a, i) for e, a in configs for i in range(n_images)}
    sizes = [sum(s.n_images for s in shards) for shards in workers]
    assert max(sizes) - min(sizes) <= 1                      # balanced
    assert len(workers) == min(max(num_workers, 1), n_configs * n_images)
    assert all(shards for shards in workers)                 # no empty worker


def test_make_shards_one_worker_gets_contiguous_ranges():
    (shards,) = make_shards(CONFIGS[:2], 5, 1)
    assert shards == [Shard("inception", "V1", 0, 5), Shard("inception", "V4", 0, 5)]


def test_make_shards_can_split_one_config_over_workers():
    workers = make_shards(CONFIGS[:1], 10, 4)
    assert [[(s.start, s.end) for s in w] for w in workers] == [[(0, 3)], [(3, 6)], [(6, 8)], [(8, 10)]]


@pytest.mark.parametrize("bad", [dict(configs=[], n_images=3), dict(configs=CONFIGS[:1], n_images=0)])
def test_make_shards_rejects_empty_work(bad):
    with pytest.raises(ValueError):
        make_shards(bad["configs"], bad["n_images"], 2)


def test_per_image_command_matches_the_module_arguments():
    cmd = sa.per_image_command(Shard("resnet50", "V4", 5, 9), device="cuda:2", runs_dir="runs", python="py")
    assert cmd[:3] == ["py", "-m", "multigeom.pipeline.per_image"]
    arg = dict(zip(cmd[3::2], cmd[4::2]))
    assert arg == {
        "--config": "estimator", "--encoder-cfg": "resnet50",
        "--encoder-ckpt": "runs/encoders/resnet50/V4/best_model.pt",
        "--diffusion-ckpt": "runs/diffusion/resnet50/V4/final_model.pt",
        "--area": "V4", "--device": "cuda:2", "--start": "5", "--end": "9",
        "--out-dir": "runs/per_image/resnet50/V4",
    }


def test_per_image_command_only_uses_arguments_of_the_module(repo_root):
    import importlib
    stage4 = importlib.import_module("multigeom.pipeline.per_image")
    options = {a for action in stage4.parse_args.__code__.co_consts if isinstance(action, str) and action.startswith("--")
               for a in [action]}
    cmd = sa.per_image_command(Shard("inception", "V1", 0, 1), device="cpu")
    assert {c for c in cmd if c.startswith("--")} <= options


def test_scenario_normalizes_configs_and_validates_arguments():
    one = sa.Scenario(configs=("inception", "V1"), n_images=3)           # a single (encoder, area) pair
    assert one.configs == (("inception", "V1"),)
    assert (one.backend, one.device, one.num_workers) == ("local", "gpu", 1)
    many = sa.Scenario(configs=[("inception", "V1"), ["resnet50", "V4"]], n_images=2)
    assert many.configs == (("inception", "V1"), ("resnet50", "V4"))
    for bad in (dict(backend="cloud"), dict(device="tpu"), dict(n_images=0), dict(num_workers=0),
                dict(configs=[]), dict(backend="modal", device="cpu")):
        with pytest.raises(ValueError):
            sa.Scenario(**{"configs": ("inception", "V1"), "n_images": 1, **bad})


def test_example_scenarios_of_the_notebook_table():
    """The four examples of the table in tutorials/2_subspace_alignment.ipynb are valid scenarios."""
    sa.Scenario(configs=("inception", "V1"), n_images=3, backend="local", device="gpu", num_workers=1)
    sa.Scenario(configs=("inception", "V1"), n_images=200, backend="local", device="gpu", num_workers=1)
    sa.Scenario(configs=("inception", "V1"), n_images=200, backend="modal", device="gpu", num_workers=10)
    sa.Scenario(configs=sa.ALL_CONFIGS, n_images=200, backend="modal", device="gpu", num_workers=10)
    assert len(sa.ALL_CONFIGS) == 4


def test_seconds_per_image_defaults_to_the_measured_value_of_the_setup():
    s = lambda **kw: sa.Scenario(configs=("inception", "V1"), n_images=1, **kw).seconds
    assert s() == 15.0 and s(device="cpu") == 885.0 and s(backend="modal") == 30.0
    assert s(seconds_per_image=7.0) == 7.0


def test_estimate_cost_arithmetic():
    est = sa.estimate_cost(100, 10, seconds_per_image=30.0, startup_s=15.0, price_per_s=0.0007)
    assert est.gpu_seconds == 100 * 30.0 + 10 * 15.0
    assert est.wall_seconds == est.gpu_seconds / 10
    assert est.cost_usd == pytest.approx(est.gpu_seconds * 0.0007)
    assert sa.estimate_cost(1, 1, 1.0).cost_usd is None


def test_full_modal_run_cost_matches_the_hand_calculation():
    est = sa.scenario_estimate(sa.Scenario(configs=sa.ALL_CONFIGS, n_images=200, backend="modal", num_workers=10))
    # 800 computations x 30 s + 10 x 15 s startup, at the A100 price
    assert est.gpu_seconds == 800 * 30 + 10 * 15
    assert est.cost_usd == pytest.approx((800 * 30 + 150) * sa.GPU_PRICE_USD_PER_S["A100"])


def test_budget_guard_blocks_expensive_and_dry_runs(capsys):
    def modal(**kw):
        return sa.Scenario(configs=sa.ALL_CONFIGS, backend="modal", **{"n_images": 5, "num_workers": 10, "max_cost_usd": 2.0, **kw})
    assert sa.check_budget(modal(), run=True).cost_usd < 2.0
    with pytest.raises(RuntimeError, match="dry run"):
        sa.check_budget(modal(), run=False)
    with pytest.raises(RuntimeError, match="exceeds"):
        sa.check_budget(modal(n_images=200), run=True)
    with pytest.raises(ValueError, match="unknown price"):
        sa.check_budget(modal(gpu_type="B200"), run=True)


def test_local_backend_needs_no_price_or_budget(capsys):
    assert sa.check_budget(sa.Scenario(configs=("inception", "V1"), n_images=3), run=True).cost_usd is None


def test_resolve_workers_caps_local_gpus_but_not_modal(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    local = lambda **kw: sa.resolve_workers(sa.Scenario(configs=("inception", "V1"), n_images=1, **kw))
    assert local(num_workers=8) == ("gpu", 1)
    assert local(device="cpu", num_workers=2) == ("cpu", 2)               # CPU workers: as asked
    assert sa.resolve_workers(sa.Scenario(configs=("inception", "V1"), n_images=1, backend="modal",
                                          num_workers=8)) == ("gpu", 8)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert local(num_workers=2) == ("cpu", 2)                              # no GPU present: falls back to the CPU


def test_progress_counts_existing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "REPO", tmp_path)
    d = tmp_path / "runs" / "per_image" / "inception" / "V1"
    d.mkdir(parents=True)
    for i in (0, 1, 4):
        (d / f"img{i:04d}.pt").write_bytes(b"x")
    assert sa.progress([("inception", "V1"), ("inception", "V4")], 5) == {("inception", "V1"): 3, ("inception", "V4"): 0}
    assert sa.progress([("inception", "V1")], 2) == {("inception", "V1"): 2}


def test_compute_metrics_needs_two_configurations(tmp_path, monkeypatch, capsys):
    from multigeom.pipeline import alignment
    monkeypatch.setattr(alignment, "REPO", tmp_path)
    d = tmp_path / "runs" / "per_image" / "inception" / "V1"
    d.mkdir(parents=True)
    (d / "img0000.pt").write_bytes(b"x")
    assert alignment.compute_metrics("runs") is None
    assert "two configurations" in capsys.readouterr().out


def test_module_imports_without_modal():
    code = ("import sys; sys.modules['modal'] = None; import multigeom.pipeline.distribute as m; "
            "assert 'modal' not in [k for k, v in sys.modules.items() if v is not None]; print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(sa.REPO), env=subprocess_env())
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr


def test_alive_tells_running_from_finished():
    running = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    try:
        assert sa.alive([running.pid, done.pid]) == [running.pid]
    finally:
        running.kill()
        running.wait()


def test_figure_to_png_is_a_png_whatever_the_backend():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    from multigeom.utils import figure_to_png
    png = figure_to_png(fig)
    plt.close(fig)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 1000


def test_is_complete_and_wait_for_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "REPO", tmp_path)
    d = tmp_path / "runs" / "per_image" / "inception" / "V1"
    d.mkdir(parents=True)
    cfg = [("inception", "V1")]
    assert not sa.is_complete(cfg, 2)
    # a worker that finishes after writing the second file
    code = f"import time, pathlib; time.sleep(1.0); pathlib.Path(r'{d}/img0001.pt').write_bytes(b'x')"
    (d / "img0000.pt").write_bytes(b"x")
    p = subprocess.Popen([sys.executable, "-c", code])
    assert sa.wait_for_workers([{"pid": p.pid}], configs=cfg, n_images=2, poll_s=0.2, verbose=False) is True
    assert sa.is_complete(cfg, 2)


def test_wait_for_workers_times_out_but_leaves_workers_running(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "REPO", tmp_path)
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert sa.wait_for_workers([{"pid": p.pid}], poll_s=0.2, timeout_s=0.5, verbose=False) is False
        assert p.poll() is None                                  # still running
    finally:
        p.kill()
        p.wait()


def test_wait_for_workers_reads_the_recorded_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "REPO", tmp_path)
    logs = tmp_path / "runs" / "logs"
    logs.mkdir(parents=True)
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    (logs / "workers.json").write_text('[{"pid": %d}]' % p.pid)
    assert sa.wait_for_workers(poll_s=0.1, verbose=False) is True       # no configs given: True once ended


def _detached_worker(child_too: bool = True):
    """A process in its own session, like launch_local's workers, that starts a child process."""
    code = ("import subprocess, sys, time\n"
            + ("subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n" if child_too else "")
            + "time.sleep(60)")
    return subprocess.Popen([sys.executable, "-c", code], start_new_session=True)


def test_stop_workers_stops_the_worker_and_its_child(tmp_path, monkeypatch):
    import os
    import time
    monkeypatch.setattr(sa, "REPO", tmp_path)
    p = _detached_worker()
    time.sleep(1.0)                                         # let the child start
    assert sa.stop_workers([{"pid": p.pid}], verbose=False) == 1
    p.wait(timeout=5)
    assert sa.alive([p.pid]) == []
    # no member of the worker's process group is left
    with pytest.raises(ProcessLookupError):
        os.killpg(p.pid, 0)


def test_stop_workers_with_nothing_running(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "REPO", tmp_path)
    assert sa.stop_workers(verbose=False) == 0


@pytest.mark.parametrize("stop_on_interrupt", [True, False])
def test_interrupt_while_waiting(tmp_path, monkeypatch, stop_on_interrupt):
    monkeypatch.setattr(sa, "REPO", tmp_path)
    p = _detached_worker(child_too=False)

    import time
    real_sleep, calls = time.sleep, []

    def interrupted(seconds):                       # the first sleep, in the waiting loop, is interrupted
        if not calls:
            calls.append(1)
            raise KeyboardInterrupt
        real_sleep(seconds)
    monkeypatch.setattr(sa.time, "sleep", interrupted)
    try:
        assert sa.wait_for_workers([{"pid": p.pid}], stop_on_interrupt=stop_on_interrupt, verbose=False) is False
        monkeypatch.undo()
        time.sleep(0.5)
        still_running = p.poll() is None
        assert still_running == (not stop_on_interrupt)
    finally:
        if p.poll() is None:
            p.kill()
        p.wait()


def test_progress_line_in_a_terminal_updates_one_line(capsys):
    bar = ProgressLine(6, initial=2, desc="estimate G(x)")
    bar.update(3, "1 worker(s)")
    bar.update(3, "1 worker(s)")                 # nothing changed: no new output
    bar.update(6, "0 worker(s)")
    bar.close()
    out = capsys.readouterr().out
    assert out.count("\r") == 3                  # initial, 3/6, 6/6
    assert "3/6" in out and "6/6" in out and "estimate G(x)" in out and out.endswith("\n")
    assert out.count("\n") == 1                  # a single line: only the final newline


def _fake_ipython(monkeypatch):
    """A stand-in for IPython.display, so that these tests do not need IPython (it is not in the CI environment)."""
    import types

    shown = []

    class Pretty:
        def __init__(self, data):
            self.data = data

    class Image:
        def __init__(self, data=None):
            self.data = data

    class Handle:
        def update(self, obj):
            shown.append(("update", obj.data))

    def display(obj, display_id=None):
        shown.append(("display", obj.data))
        return Handle() if display_id else None

    display_module = types.ModuleType("IPython.display")
    display_module.Pretty, display_module.Image, display_module.display = Pretty, Image, display
    package = types.ModuleType("IPython")
    package.display = display_module
    package.get_ipython = lambda: None
    monkeypatch.setitem(sys.modules, "IPython", package)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    return shown


def test_progress_line_in_a_notebook_updates_in_place(monkeypatch):
    shown = _fake_ipython(monkeypatch)
    bar = ProgressLine(4, notebook=True)
    bar.update(2, "1 worker(s)")
    bar.update(4, "0 worker(s)")
    bar.close()
    assert [k for k, _ in shown] == ["display", "update", "update"]     # one display, then in-place updates
    assert "4/4" in shown[-1][1]


def test_show_figure_displays_a_png_and_closes_the_figure(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shown = _fake_ipython(monkeypatch)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    from multigeom.utils import show_figure
    show_figure(fig)
    assert shown[0][0] == "display" and shown[0][1][:8] == b"\x89PNG\r\n\x1a\n"
    assert not plt.fignum_exists(fig.number)


def test_make_shards_leaves_out_images_already_done(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "REPO", tmp_path)
    d = tmp_path / "runs" / "per_image" / "inception" / "V1"
    d.mkdir(parents=True)
    for i in (0, 1, 3):
        (d / f"img{i:04d}.pt").write_bytes(b"x")
    workers = make_shards(CONFIGS[:1], 5, 2, runs_dir="runs")
    assert covered(workers) == {("inception", "V1", 2), ("inception", "V1", 4)}
    assert make_shards(CONFIGS[:2], 5, 2, runs_dir="runs") != []         # the other config is still to do
    for i in (2, 4):
        (d / f"img{i:04d}.pt").write_bytes(b"x")
    assert make_shards(CONFIGS[:1], 5, 2, runs_dir="runs") == []         # everything done: nothing to launch


def test_clear_results_deletes_only_the_asked_per_image_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "REPO", tmp_path)
    runs = tmp_path / "runs"
    keep = []
    for enc, area, n in [("inception", "V1", 4), ("inception", "V4", 4)]:
        d = runs / "per_image" / enc / area
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"img{i:04d}.pt").write_bytes(b"x")
    (runs / "encoders" / "inception" / "V1").mkdir(parents=True)
    ckpt = runs / "encoders" / "inception" / "V1" / "best_model.pt"
    ckpt.write_bytes(b"x")
    cfg = [("inception", "V1")]
    assert len(sa.clear_results(cfg, 2, dry_run=True)) == 2                           # lists, deletes nothing
    assert len(list((runs / "per_image/inception/V1").glob("img*.pt"))) == 4
    assert len(sa.clear_results(cfg, 2)) == 2
    left = sorted(p.name for p in (runs / "per_image/inception/V1").glob("img*.pt"))
    assert left == ["img0002.pt", "img0003.pt"]                                       # only the first 2 images
    assert len(list((runs / "per_image/inception/V4").glob("img*.pt"))) == 4          # other config untouched
    assert ckpt.exists()                                                              # checkpoints untouched
    assert make_shards(cfg, 4, 1, runs_dir="runs") == [[Shard("inception", "V1", 0, 2)]]   # will be recomputed


def test_clear_results_refuses_while_workers_run(tmp_path, monkeypatch):
    monkeypatch.setattr(sa, "REPO", tmp_path)
    logs = tmp_path / "runs" / "logs"
    logs.mkdir(parents=True)
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    (logs / "workers.json").write_text('[{"pid": %d}]' % p.pid)
    try:
        with pytest.raises(RuntimeError, match="still running"):
            sa.clear_results([("inception", "V1")], 1)
    finally:
        p.kill()
        p.wait()


class _Entry:
    def __init__(self, path):
        self.path = path


class _FakeVolume:
    def __init__(self, paths):
        self.paths = paths

    def listdir(self, path, recursive=False):
        return [_Entry(p) for p in self.paths]


def test_modal_images_done_counts_only_the_expected_images():
    workers = [[Shard("inception", "V1", 0, 3)], [Shard("resnet50", "V4", 2, 4)]]
    expected = sa._expected_images(workers)
    assert len(expected) == 5
    full = ["runs/per_image/inception/V1/img0000.pt", "runs/per_image/inception/V1/img0002.pt",
            "runs/per_image/inception/V1/img0005.pt",          # not expected: index outside the shard
            "runs/per_image/inception/V4/img0000.pt",          # not expected: other config
            "runs/per_image/resnet50/V4/img0003.pt", "runs/per_image/resnet50/V4/notes.txt"]
    assert sa.modal_images_done(_FakeVolume(full), expected) == 3
    relative = [p.replace("runs/per_image/", "") for p in full]  # listdir may also return paths relative to the folder
    assert sa.modal_images_done(_FakeVolume(relative), expected) == 3


class _FakeCall:
    def __init__(self, polls_until_done, fail=False):
        self.left, self.fail, self.cancelled = polls_until_done, fail, False

    def get(self, timeout=None):
        if self.left > 0:
            self.left -= 1
            raise TimeoutError
        if self.fail:
            raise RuntimeError("container failed")
        return 1

    def cancel(self):
        self.cancelled = True


class _Bar:
    def __init__(self):
        self.updates = []

    def update(self, n, postfix=""):
        self.updates.append((n, postfix))


def test_watch_calls_updates_the_bar_until_all_calls_finish():
    calls = [_FakeCall(1), _FakeCall(3)]
    counts = iter(range(0, 100, 10))
    bar = _Bar()
    sa.watch_calls(calls, lambda: next(counts), bar, poll_s=0, sleep=lambda s: None)
    assert [n for n, _ in bar.updates] == [0, 10, 20, 30]                # one update per poll, ends when both are done
    assert bar.updates[0][1] == "2 container(s) running" and bar.updates[-1][1] == "0 container(s) running"


def test_watch_calls_raises_a_container_failure():
    with pytest.raises(RuntimeError, match="container failed"):
        sa.watch_calls([_FakeCall(0, fail=True)], lambda: 0, _Bar(), poll_s=0, sleep=lambda s: None)


def test_watch_calls_cancels_on_interrupt():
    calls = [_FakeCall(5), _FakeCall(5)]

    def interrupted(_):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        sa.watch_calls(calls, lambda: 0, _Bar(), poll_s=0, sleep=interrupted)
    assert all(c.cancelled for c in calls)
