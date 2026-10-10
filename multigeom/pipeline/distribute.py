"""
Supporting functions of 2_subspace_alignment.ipynb of multigeom-tutorials: paper Fig. 4D-E, the subspace alignment of
the top-k eigenvectors of G(x) and J(x) across encoders and areas, averaged over the test images.

The work is the estimation of G(x) and J(x) for every test image of every (encoder, area) configuration, then
Stage 5 (alignment between configurations), then the plot. The G(x) estimation dominates: the images are
independent (no gradients, no communication between GPUs), the stage is resumable and every image is
seeded on its own (seed + image index). So the images can be split across workers (GPUs, local or on
Modal) without changing the results, and no DDP / FSDP is involved.

Everything is parametrised by

    configs      (encoder, area), or a list of them, e.g. ("inception", "V1") or all four
    n_images     first n test images (the paper uses 200)
    backend      "local" (this machine) or "modal" (serverless GPUs, https://modal.com)
    device       "gpu" or "cpu" (Modal is always "gpu")
    num_workers  how many workers run in parallel (one GPU or one CPU process each)

bundled in a `Scenario`. The G(x) estimation itself is not re-implemented here: the workers run
`multigeom.pipeline.per_image`, so this module cannot diverge from it.

`modal` is imported only inside the Modal functions, so the rest works without it installed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

from ..config import package_root, project_root
from ..utils import ProgressLine, subprocess_env

REPO = project_root()

Config = tuple[str, str]
ALL_CONFIGS: tuple[Config, ...] = (
    ("inception", "V1"), ("inception", "V4"), ("resnet50", "V1"), ("resnet50", "V4"),
)
N_TEST_IMAGES = 200                    # n_test_images of the packaged estimator config

# Third-party figures (Modal and others change prices): re-check before relying on them.
GPU_PRICE_USD_PER_S = {"A100": 0.000694, "H100": 0.001097}      # Modal, A100 80GB and H100
MODAL_STARTER_MAX_GPUS = 10
MODAL_PIP = [
    "torch==2.7.0", "torchvision==0.22.0", "diffusers==0.37.1", "huggingface_hub==0.36.0",
    "plenoptic==1.3.1", "numpy", "scipy", "matplotlib", "h5py", "Pillow", "PyYAML", "tqdm",
]


# ── scenario ────────────────────────────────────────────────────────────────
BACKENDS = ("local", "modal")
DEVICES = ("gpu", "cpu")


def normalize_configs(configs) -> tuple[Config, ...]:
    """("inception", "V1") or [("inception", "V1"), ...] -> a tuple of (encoder, area) tuples."""
    if isinstance(configs, (tuple, list)) and len(configs) == 2 and all(isinstance(c, str) for c in configs):
        configs = [configs]
    out = tuple((str(e), str(a)) for e, a in configs)
    if not out:
        raise ValueError("no configuration given")
    return out


@dataclass(frozen=True)
class Scenario:
    """
    What to compute and where. Examples:

        Scenario(configs=("inception", "V1"), n_images=3, backend="local", device="gpu", num_workers=1)
        Scenario(configs=ALL_CONFIGS, n_images=200, backend="modal", device="gpu", num_workers=10)
    """
    configs: tuple[Config, ...]
    n_images: int
    backend: str = "local"             # "local" | "modal"
    device: str = "gpu"                # "gpu" | "cpu" (Modal: "gpu")
    num_workers: int = 1               # parallel workers; local GPU: at most the GPUs present
    gpu_type: str = "A100"             # Modal GPU type
    seconds_per_image: float | None = None   # one image on one worker; None: measured/paper value for the setup
    startup_s: float = 15.0            # per-worker start (imports, model loading); to be measured
    max_cost_usd: float = 2.0          # Modal only

    def __post_init__(self):
        object.__setattr__(self, "configs", normalize_configs(self.configs))
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        if self.backend == "modal" and self.device != "gpu":
            raise ValueError("the Modal backend runs on GPUs: use device='gpu'")
        if self.n_images < 1:
            raise ValueError("n_images must be at least 1")
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if self.backend == "modal" and self.num_workers > MODAL_STARTER_MAX_GPUS:
            print(f"note: {self.num_workers} concurrent GPUs need a Modal plan above Starter "
                  f"(Starter allows {MODAL_STARTER_MAX_GPUS})")

    @property
    def seconds(self) -> float:
        """Seconds per image: the given value, else RTX 5090 15 s (local GPU), A100 ~30 s (paper, Modal), CPU ~885 s."""
        if self.seconds_per_image is not None:
            return self.seconds_per_image
        if self.backend == "modal":
            return 30.0
        return 15.0 if self.device == "gpu" else 885.0


# ── sharding ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Shard:
    """Images [start, end) of one configuration."""
    encoder: str
    area: str
    start: int
    end: int

    @property
    def n_images(self) -> int:
        return self.end - self.start


def make_shards(configs, n_images: int, num_workers: int, runs_dir: str | None = None) -> list[list[Shard]]:
    """
    Split the (config, image) work into `max(num_workers, 1)` balanced workers, each a list of shards
    (one shard = consecutive images of one config). Every (config, image) pair is in exactly one
    shard; workers' sizes differ by at most one image; empty workers are dropped. Because every
    image is seeded on its own, the split does not change any result. With `runs_dir`, images that
    already have their per-image file there are left out (the result is [] if nothing is left).
    """
    configs = list(normalize_configs(configs))
    if n_images < 1:
        raise ValueError("n_images must be at least 1")
    units = [(c, i) for c in configs for i in range(n_images)]
    if runs_dir is not None:
        units = [((enc, area), i) for (enc, area), i in units
                 if not (REPO / runs_dir / "per_image" / enc / area / f"img{i:04d}.pt").exists()]
    if not units:
        return []
    n_workers = max(int(num_workers), 1)
    base, extra = divmod(len(units), n_workers)
    workers, pos = [], 0
    for w in range(n_workers):
        size = base + (1 if w < extra else 0)
        chunk, pos = units[pos:pos + size], pos + size
        shards: list[Shard] = []
        for (enc, area), i in chunk:
            last = shards[-1] if shards else None
            if last is not None and (last.encoder, last.area) == (enc, area) and last.end == i:
                shards[-1] = replace(last, end=i + 1)
            else:
                shards.append(Shard(enc, area, i, i + 1))
        if shards:
            workers.append(shards)
    return workers


def per_image_command(shard: Shard, *, device: str, runs_dir: str = "runs", python: str = sys.executable) -> list[str]:
    """Command line of `python -m multigeom.pipeline.per_image` for one shard (it skips images already done)."""
    enc, area = shard.encoder, shard.area
    return [
        python, "-m", "multigeom.pipeline.per_image",
        "--config", "estimator",
        "--encoder-cfg", enc,
        "--encoder-ckpt", f"{runs_dir}/encoders/{enc}/{area}/best_model.pt",
        "--diffusion-ckpt", f"{runs_dir}/diffusion/{enc}/{area}/final_model.pt",
        "--area", area, "--device", device,
        "--start", str(shard.start), "--end", str(shard.end),
        "--out-dir", f"{runs_dir}/per_image/{enc}/{area}",
    ]


def run_shards(shards, *, device: str, runs_dir: str = "runs", cwd: str | Path = REPO,
               env: dict | None = None) -> None:
    """Run the shards one after the other with `multigeom.pipeline.per_image`, the G(x) estimation (used by the local
    workers and the Modal containers)."""
    full_env = subprocess_env(env)
    for shard in shards:
        subprocess.run(per_image_command(shard, device=device, runs_dir=runs_dir), cwd=str(cwd),
                       env=full_env, check=True)


# ── cost ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CostEstimate:
    n_computations: int
    n_workers: int
    gpu_seconds: float
    wall_seconds: float
    cost_usd: float | None             # None if the GPU price is unknown


def estimate_cost(n_computations: int, n_workers: int, seconds_per_image: float, startup_s: float = 15.0,
                  price_per_s: float | None = None) -> CostEstimate:
    """Total GPU-seconds = images x seconds per image + one startup per worker; wall = that / workers."""
    n_workers = max(int(n_workers), 1)
    gpu_seconds = n_computations * seconds_per_image + n_workers * startup_s
    cost = None if price_per_s is None else gpu_seconds * price_per_s
    return CostEstimate(n_computations, n_workers, gpu_seconds, gpu_seconds / n_workers, cost)


def scenario_estimate(s: Scenario) -> CostEstimate:
    n_workers = len(make_shards(s.configs, s.n_images, s.num_workers))
    price = GPU_PRICE_USD_PER_S.get(s.gpu_type) if s.backend == "modal" else None
    return estimate_cost(len(s.configs) * s.n_images, n_workers, s.seconds, s.startup_s, price)


def print_estimate(s: Scenario) -> CostEstimate:
    """Print what a scenario would take (images, workers, GPU-time, wall-clock and, on Modal, the cost)."""
    est = scenario_estimate(s)
    unit = "GPU" if s.device == "gpu" else "CPU process"
    print(f"{est.n_computations} image computations on {s.backend}, {est.n_workers} {unit} worker(s): "
          f"~{est.gpu_seconds / 60:.1f} {unit}-min in total, ~{est.wall_seconds / 60:.1f} min wall-clock")
    if s.backend == "local" and s.device == "cpu":
        print("note: a CPU is slow for this (about 15 min per image on 32 cores) and every worker uses all cores")
    if s.backend == "modal":
        if est.cost_usd is None:
            print(f"unknown price for GPU type {s.gpu_type!r}; known: {sorted(GPU_PRICE_USD_PER_S)}")
        else:
            print(f"estimated Modal cost ~${est.cost_usd:.2f} (limit ${s.max_cost_usd:.2f}; third-party price list)")
    return est


def check_budget(s: Scenario, run: bool) -> CostEstimate:
    """Print the estimate; raise unless `run` is True and, on Modal, the cost is known and within the budget."""
    est = print_estimate(s)
    if s.backend == "modal":
        if est.cost_usd is None:
            raise ValueError(f"unknown price for GPU type {s.gpu_type!r}; known: {sorted(GPU_PRICE_USD_PER_S)}")
        if est.cost_usd > s.max_cost_usd:
            raise RuntimeError(f"estimated cost ${est.cost_usd:.2f} exceeds max_cost_usd=${s.max_cost_usd:.2f}")
    if not run:
        raise RuntimeError("dry run: set RUN = True to launch")
    return est


# ── local backend ───────────────────────────────────────────────────────────
def resolve_workers(s: Scenario) -> tuple[str, int]:
    """
    The device and number of workers that will really be used: (device, num_workers). Modal: as asked.
    Local GPU: at most the GPUs present, and the CPU if there is none.
    """
    if s.backend != "local" or s.device == "cpu":
        return s.device, s.num_workers
    import torch
    available = torch.cuda.device_count()
    if available == 0:
        print("no local GPU found: using the CPU")
        return "cpu", s.num_workers
    if s.num_workers > available:
        print(f"only {available} local GPU(s): using {available} worker(s) instead of {s.num_workers}")
    return "gpu", min(s.num_workers, available)


def launch_local(workers: list[list[Shard]], *, use_gpu: bool, runs_dir: str = "runs") -> list[dict]:
    """
    Start one detached process per worker (device cuda:i, or the CPU), so the work goes on if the
    notebook kernel dies. Logs are in <runs_dir>/logs/. Relaunching continues where it stopped.
    """
    log_dir = REPO / runs_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, shards in enumerate(workers):
        shard_file = log_dir / f"shards_worker{i}.json"
        shard_file.write_text(json.dumps([s.__dict__ for s in shards]))
        device = f"cuda:{i}" if use_gpu else "cpu"
        log_path = log_dir / f"worker{i}.log"
        with open(log_path, "ab") as log:
            p = subprocess.Popen(
                [sys.executable, "-m", "multigeom.pipeline.distribute", "worker", "--shards", str(shard_file),
                 "--device", device, "--runs-dir", runs_dir],
                cwd=str(REPO), env=subprocess_env(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append({"worker": i, "pid": p.pid, "device": device, "log": str(log_path)})
    (log_dir / "workers.json").write_text(json.dumps(procs))      # so a later session can still wait for them
    return procs


def progress(configs, n_images: int, runs_dir: str = "runs") -> dict[Config, int]:
    """Number of per-image files done, per configuration (out of n_images)."""
    out = {}
    for enc, area in configs:
        d = REPO / runs_dir / "per_image" / enc / area
        out[(enc, area)] = sum(1 for i in range(n_images) if (d / f"img{i:04d}.pt").exists())
    return out


def clear_results(configs, n_images: int, runs_dir: str = "runs", dry_run: bool = False) -> list[Path]:
    """
    Delete the per-image files of the first `n_images` images of the given configurations, so that the G(x) estimation
    recomputes them from scratch (it skips images that exist). Only those files are touched: no other
    image, no checkpoint, no Stage-5 output. Refuses to run while recorded local workers are still
    running. With `dry_run` it only lists what would be deleted. Returns the files.
    """
    f = REPO / runs_dir / "logs" / "workers.json"
    if not dry_run and f.exists() and alive([p["pid"] for p in json.loads(f.read_text())]):
        raise RuntimeError("workers are still running: stop them first with stop_workers()")
    files = [REPO / runs_dir / "per_image" / enc / area / f"img{i:04d}.pt"
             for enc, area in configs for i in range(n_images)]
    files = [p for p in files if p.exists()]
    for p in files:
        if not dry_run:
            p.unlink()
    print(f"{'would delete' if dry_run else 'deleted'} {len(files)} per-image file(s)")
    return files


def is_complete(configs, n_images: int, runs_dir: str = "runs") -> bool:
    """True when every image of every configuration has its per-image file."""
    return all(n == n_images for n in progress(configs, n_images, runs_dir).values())


def stop_workers(procs=None, *, runs_dir: str = "runs", grace_s: float = 5.0, verbose: bool = True) -> int:
    """
    Stop the local workers (POSIX: Linux, macOS). Every worker was started in its own process group, so
    the group is terminated together with the G(x)-estimation process it runs, then killed if it has not exited
    after `grace_s`. Images already written are kept; the one in progress is lost, and launching again
    continues from there. `procs` as for `wait_for_workers` (default: the workers of the last launch).
    Returns the number of workers that were running.
    """
    if procs is None:
        f = REPO / runs_dir / "logs" / "workers.json"
        procs = json.loads(f.read_text()) if f.exists() else []
    running = alive([p["pid"] for p in procs])
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in alive(running):
            try:
                os.killpg(pid, sig)                     # the worker leads its own session / process group
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.time() + (grace_s if sig == signal.SIGTERM else 1.0)
        while alive(running) and time.time() < deadline:
            time.sleep(0.1)
    if verbose:
        print(f"stopped {len(running)} worker(s)")
    return len(running)


def wait_for_workers(procs=None, *, configs=None, n_images: int | None = None, runs_dir: str = "runs",
                     poll_s: float = 15.0, timeout_s: float | None = None, stop_on_interrupt: bool = True,
                     verbose: bool = True) -> bool:
    """
    Block until the local workers have finished (the notebook cell stays busy meanwhile), printing the
    progress. `procs` is what `launch_local` returned; if omitted, the workers recorded by the last launch
    are used, so this also works after a kernel restart. Interrupting the kernel stops the workers too
    (`stop_on_interrupt=True`, the default; finished images are kept, so launching again resumes), or only
    the waiting if `stop_on_interrupt=False`. A `timeout_s` only stops the waiting. Workers are detached,
    so they also keep running if the kernel itself dies; `stop_workers()` stops them from a later session.
    Returns True if the work is complete (needs `configs` and `n_images`; otherwise True once the workers
    have ended).
    """
    if procs is None:
        f = REPO / runs_dir / "logs" / "workers.json"
        procs = json.loads(f.read_text()) if f.exists() else []
    pids = [p["pid"] for p in procs]
    known = configs is not None and n_images is not None
    bar = None
    if verbose and known:
        initial = sum(progress(configs, n_images, runs_dir).values())      # images already done count as such
        bar = ProgressLine(len(configs) * n_images, initial=initial)
    t0, last = time.time(), None
    try:
        while True:
            running = alive(pids)
            status = None
            if known:
                done = progress(configs, n_images, runs_dir)
                status = ", ".join(f"{e}/{a}: {n}/{n_images}" for (e, a), n in done.items())
                if bar is not None:
                    bar.update(sum(done.values()), f"{len(running)} worker(s); {status}")
            if verbose and bar is None:
                line = f"{time.strftime('%H:%M:%S')}  {len(running)} worker(s) running" + (f"  [{status}]" if status else "")
                if line[10:] != (last or "")[10:]:
                    print(line)
                    last = line
            if not running:
                break
            if timeout_s is not None and time.time() - t0 > timeout_s:
                if bar is not None:
                    bar.close()
                print("stopped waiting (timeout); the workers keep running")
                return False
            time.sleep(poll_s)
    except KeyboardInterrupt:
        if bar is not None:
            bar.close()
        if stop_on_interrupt:
            print("interrupted: stopping the workers (finished images are kept; launch again to resume)")
            stop_workers(procs, runs_dir=runs_dir, verbose=verbose)
        else:
            print("stopped waiting; the workers keep running in the background (run this cell again to resume waiting)")
        return False
    if bar is not None:
        bar.close()
    complete = True if not known else is_complete(configs, n_images, runs_dir)
    if verbose:
        print("all images done" if complete else "workers ended but some images are missing: check runs/logs/")
    return complete


def alive(pids) -> list[int]:
    """The pids that are still running (also works for processes started by an earlier kernel)."""
    out = []
    for pid in pids:
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)       # reaps a finished child of this process
            if done == pid:
                continue
        except ChildProcessError:                        # not our child (e.g. started before a kernel restart)
            pass
        try:
            os.kill(pid, 0)
        except OSError:
            continue
        out.append(pid)
    return out


# ── modal backend ───────────────────────────────────────────────────────────
VOLUME_NAME = "multigeom-subspace-alignment"


def _volume_files(configs, n_test: int = N_TEST_IMAGES) -> dict[str, Path]:
    """Files to upload: {path in the volume: local path} (checkpoints of the configs, first test images)."""
    files = {}
    for enc, area in configs:
        files[f"runs/encoders/{enc}/{area}/best_model.pt"] = REPO / f"runs/encoders/{enc}/{area}/best_model.pt"
        files[f"runs/diffusion/{enc}/{area}/final_model.pt"] = REPO / f"runs/diffusion/{enc}/{area}/final_model.pt"
    from ..config import data_dir
    val = sorted((data_dir() / "tiny-imagenet-200" / "val" / "images").glob("*.JPEG"))[:n_test]
    for p in val:        # the first n_test in sorted order are the test images; the dataset sorts the same way
        files[f"data/tiny-imagenet-200/val/images/{p.name}"] = p
    return files


def modal_upload(configs, volume_name: str = VOLUME_NAME) -> None:
    """Copy the checkpoints and test images to a Modal volume (files already there are skipped)."""
    import modal
    vol = modal.Volume.from_name(volume_name, create_if_missing=True)
    present = {e.path.lstrip("/") for e in vol.listdir("/", recursive=True)}
    todo = {remote: local for remote, local in _volume_files(configs).items() if remote not in present}
    print(f"{len(todo)} file(s) to upload to Modal volume {volume_name!r} ({len(present)} already there)")
    if todo:
        with vol.batch_upload() as batch:
            for remote, local in todo.items():
                batch.put_file(str(local), remote)


COMMIT_EVERY_S = 30.0          # how often a Modal container saves the volume, so the notebook can see its progress


def _expected_images(workers) -> set[tuple[str, str, int]]:
    """Every (encoder, area, image index) the workers are to compute."""
    return {(s.encoder, s.area, i) for shards in workers for s in shards for i in range(s.start, s.end)}


def modal_images_done(vol, expected) -> int:
    """How many of the `expected` images have their per-image file on the Modal volume (committed files only)."""
    n = 0
    for e in vol.listdir("runs/per_image", recursive=True):
        m = re.search(r"(?:per_image/)?([^/]+)/([^/]+)/img(\d{4})\.pt$", e.path.lstrip("/"))
        if m and (m.group(1), m.group(2), int(m.group(3))) in expected:
            n += 1
    return n


def watch_calls(calls, count_done, bar, *, poll_s: float = 10.0, sleep=time.sleep) -> None:
    """
    Wait for the Modal function calls to finish while `count_done()` (images on the volume) drives the
    progress bar; this runs in the notebook's own thread, so the bar updates in place. A failure in a
    container is raised here. Interrupting (KeyboardInterrupt) cancels the calls and re-raises.
    """
    pending = list(calls)
    try:
        while True:
            still = []
            for fc in pending:
                try:
                    fc.get(timeout=0)                  # returns the result if done, raises TimeoutError if not
                except TimeoutError:
                    still.append(fc)
            pending = still
            bar.update(count_done(), f"{len(pending)} container(s) running")
            if not pending:
                return
            sleep(poll_s)
    except KeyboardInterrupt:
        for fc in pending:
            try:
                fc.cancel()
            except Exception:                          # best effort: leaving app.run() stops the rest
                pass
        print("interrupted: the Modal calls were cancelled (finished images are kept on the volume)")
        raise


def launch_modal(workers: list[list[Shard]], *, gpu_type: str = "A100", volume_name: str = VOLUME_NAME,
                 timeout_s: int = 3 * 3600, poll_s: float = 10.0) -> None:
    """
    Run each worker's shards in its own Modal container with one GPU, in parallel, and wait. The containers
    run multigeom.pipeline.per_image on the volume and write the per-image files there. While they work, each
    container saves the volume every COMMIT_EVERY_S seconds and the notebook counts the per-image files on
    it every `poll_s` seconds, which drives the progress bar (no extra GPU time). Needs `pip install modal`
    and `modal token new`; not tested against a live account by the repository's tests.
    """
    import modal

    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install(*MODAL_PIP)
             .env({"PYTHONPATH": "/repo", "NEURIPS_DATA_DIR": "/vol/data", "USE_TF": "0", "USE_FLAX": "0"})
             .add_local_dir(str(package_root()), "/repo/multigeom", ignore=["**/__pycache__"]))
    vol = modal.Volume.from_name(volume_name, create_if_missing=True)
    app = modal.App("multigeom-subspace-alignment")
    commit_every = COMMIT_EVERY_S

    @app.function(image=image, gpu=gpu_type, volumes={"/vol": vol}, timeout=timeout_s, serialized=True)
    def worker(shard_dicts: list[dict]) -> int:
        import threading
        from multigeom.pipeline.distribute import Shard, run_shards
        stop = threading.Event()

        def commit_regularly():                        # makes the finished images visible to the notebook
            while not stop.wait(commit_every):
                vol.commit()

        thread = threading.Thread(target=commit_regularly, daemon=True)
        thread.start()
        try:
            run_shards([Shard(**d) for d in shard_dicts], device="cuda:0", runs_dir="/vol/runs", cwd="/repo")
        finally:
            stop.set()
            thread.join()
            vol.commit()
        return sum(d["end"] - d["start"] for d in shard_dicts)

    expected = _expected_images(workers)
    bar = ProgressLine(len(expected), initial=modal_images_done(vol, expected), desc="estimate G(x) (Modal)")
    with modal.enable_output():
        with app.run():
            calls = [worker.spawn([s.__dict__ for s in shards]) for shards in workers]
            watch_calls(calls, lambda: modal_images_done(vol, expected), bar, poll_s=poll_s)
    bar.close()
    print(f"{modal_images_done(vol, expected)}/{len(expected)} images are on the Modal volume")


def modal_clear(configs, n_images: int, volume_name: str = VOLUME_NAME) -> int:
    """Delete the per-image files of the configs on the Modal volume (the containers skip images that exist there)."""
    import modal
    vol = modal.Volume.from_name(volume_name)
    present = set()
    for e in vol.listdir("runs/per_image", recursive=True):
        path = e.path.lstrip("/")
        present.add(path if path.startswith("runs/") else "runs/per_image/" + path)    # full path either way
    n = 0
    for enc, area in configs:
        for i in range(n_images):
            path = f"runs/per_image/{enc}/{area}/img{i:04d}.pt"
            if path in present:
                vol.remove_file(path)
                n += 1
    print(f"deleted {n} per-image file(s) on the Modal volume {volume_name!r}")
    return n


def modal_download(configs, n_images: int, runs_dir: str = "runs", volume_name: str = VOLUME_NAME) -> int:
    """Copy the per-image files of the configs from the Modal volume to <runs_dir>/per_image/ (new ones only)."""
    import modal
    vol = modal.Volume.from_name(volume_name)
    wanted = {f"{enc}/{area}" for enc, area in configs}
    n = 0
    for entry in vol.listdir("runs/per_image", recursive=True):
        rel = entry.path.split("per_image/", 1)[-1]
        parts = rel.split("/")
        if len(parts) != 3 or f"{parts[0]}/{parts[1]}" not in wanted or not parts[2].startswith("img"):
            continue
        if int(parts[2][3:7]) >= n_images:
            continue
        local = REPO / runs_dir / "per_image" / rel
        if local.exists():
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(local, "wb") as f:
            for chunk in vol.read_file(entry.path):
                f.write(chunk)
        n += 1
    print(f"downloaded {n} per-image file(s) to {REPO / runs_dir / 'per_image'}")
    return n


# ── command line (used by the detached local workers) ───────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker", help="run the shards listed in a JSON file with the G(x) estimation script")
    w.add_argument("--shards", required=True)
    w.add_argument("--device", default="cuda:0")
    w.add_argument("--runs-dir", default="runs")
    args = p.parse_args(argv)
    shards = [Shard(**d) for d in json.loads(Path(args.shards).read_text())]
    run_shards(shards, device=args.device, runs_dir=args.runs_dir)


if __name__ == "__main__":
    main()
