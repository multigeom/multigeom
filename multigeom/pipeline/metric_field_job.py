"""
Supporting functions of 3_local_metric_field.ipynb of multigeom-tutorials: paper Fig. 5, the local metric field projected
onto the leading eigenvectors of G(x).

A reference image x0 is perturbed along the two leading eigenvectors of the reference encoder's G(x) (Inception);
at every point of the n x n grid both encoders' metrics G(x) (and the Fisher metric J(x)) are projected onto
that 2-D subspace and drawn as ellipses. The computation is `multigeom.pipeline.metric_field` and the drawing
`multigeom.viz.fig5`; this module only describes the job, estimates its cost, runs it as a detached background
process that survives a notebook crash, shows the progress, and draws the figure.

Step sizes. The perturbation of the reference image is x(a, b) = x0 + a*e1*v1 + b*e2*v2 along the leading two
eigenvectors v1, v2 of the metric being drawn. For G(x) the steps are 0.5*sqrt(lambda) (the stage default), about
0.3. J(x) has far larger eigenvalues, so the same rule gives steps of 5 to 7, the perturbed images are clipped
and the local structure is lost (the Inception ellipses become smaller than the ResNet-50 ones off-centre, the
opposite of the paper's panel B). A small fixed step of 0.3, the size of G's steps, gives smooth fields with the
Inception ellipses outside the ResNet-50 ones, as in the paper. The paper does not state its step sizes, so
these are choices (see `MetricFieldJob.steps`).

Cost: each grid point of each encoder needs one G(x) estimate (about 15 s on an RTX 5090), so the default 5 x 5 grid,
two areas and two encoders take about 26 minutes on one GPU. J(x) is cheap. The grid points are independent
and seeded one by one, so they could be split over GPUs without changing the result, but processes sharing one
GPU are slower than one, so a single process is used.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import project_root
from ..geometry.metric_field import STEP_RULES
from ..utils import ProgressLine, subprocess_env
from .distribute import alive, stop_workers

# The working root (see `multigeom.config.project_root`: MULTIGEOM_REPO overrides it, which the integration tests use to
# run the notebook in a scratch copy of the repository so that nothing in the real runs/ is touched).
REPO = project_root()
ENCODERS = ("inception", "resnet50", "onoff")
AREAS = ("V1", "V4")
METRICS = ("G", "J")
SECONDS_PER_G = 15.0         # one G(x) estimate with 200 Monte-Carlo samples, RTX 5090 (A100: about 30 s)
SECONDS_PER_J = 0.3          # one closed-form J(x)


DEFAULT_STEPS = {"G": ("sqrt_lambda", 0.5), "J": ("fixed", 0.3)}
# The paper does not state its reference image. Validation image 64 is the one that reproduces the pattern of the
# paper's panel B best among the 200 test images (a scan of the 200 test images with six perturbation settings): J(x) ellipses of the Inception
# model radially oriented around the centre (V1 0.81, V4 0.91-0.94, with 1 = perfectly radial) and Inception outside
# ResNet-50 (area ratio 5-6 in V1, 2.5-3 in V4), with the fixed step of 0.2-0.3.
DEFAULT_IMG_IDX = 64


@dataclass(frozen=True)
class MetricFieldJob:
    """
    What to compute for Fig. 5. Example: MetricFieldJob(img_idx=64, grid_size=5, areas=("V1", "V4")).

    `steps` gives, per metric, the step rule and size of the perturbation ("sqrt_lambda": eps_i = size * sqrt(lambda_i);
    "fixed": eps_i = size), as {"G": ("sqrt_lambda", 0.5), "J": ("fixed", 0.3)} (the default). A metric missing from
    the mapping uses its default.
    """
    img_idx: int = DEFAULT_IMG_IDX                       # index into the TinyImageNet validation list
    grid_size: int = 5                                   # odd n: the grid is n x n
    steps: dict = field(default_factory=lambda: dict(DEFAULT_STEPS))
    reference_encoder: str = "inception"
    other_encoder: str = "resnet50"
    areas: tuple[str, ...] = AREAS
    metrics: tuple[str, ...] = METRICS
    device: str = "gpu"                                  # "gpu" | "cpu"

    def __post_init__(self):
        object.__setattr__(self, "areas", _as_tuple(self.areas))
        object.__setattr__(self, "metrics", _as_tuple(self.metrics))
        steps = {**DEFAULT_STEPS, **{m: (str(r), float(sz)) for m, (r, sz) in dict(self.steps).items()}}
        object.__setattr__(self, "steps", steps)
        if self.img_idx < 0:
            raise ValueError("img_idx must be non-negative")
        if self.grid_size < 1 or self.grid_size % 2 == 0:
            raise ValueError(f"grid_size must be odd and at least 1, got {self.grid_size}")
        for name in (self.reference_encoder, self.other_encoder):
            if name not in ENCODERS:
                raise ValueError(f"unknown encoder {name!r}; choose from {ENCODERS}")
        if self.reference_encoder == self.other_encoder:
            raise ValueError("reference_encoder and other_encoder must differ")
        if not self.areas or not set(self.areas) <= set(AREAS):
            raise ValueError(f"areas must be a non-empty subset of {AREAS}, got {self.areas}")
        if not self.metrics or not set(self.metrics) <= set(METRICS):
            raise ValueError(f"metrics must be a non-empty subset of {METRICS}, got {self.metrics}")
        for metric, (rule, size) in steps.items():
            if metric not in METRICS:
                raise ValueError(f"steps: unknown metric {metric!r}; choose from {METRICS}")
            if rule not in STEP_RULES:
                raise ValueError(f"steps[{metric!r}]: rule must be one of {STEP_RULES}, got {rule!r}")
            if size <= 0:
                raise ValueError(f"steps[{metric!r}]: size must be positive")
        if self.device not in ("gpu", "cpu"):
            raise ValueError("device must be 'gpu' or 'cpu'")


def _as_tuple(x) -> tuple[str, ...]:
    return (x,) if isinstance(x, str) else tuple(x)


def out_dir(job: MetricFieldJob, runs_dir: str = "runs") -> Path:
    return REPO / runs_dir / "metric_field" / f"img{job.img_idx:04d}"


def expected_files(job: MetricFieldJob, runs_dir: str = "runs") -> list[Path]:
    """One file per (metric, area), written when that part is finished."""
    d = out_dir(job, runs_dir)
    return [d / f"{m}_{a}.pt" for m in job.metrics for a in job.areas]


# ── cost ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MetricFieldEstimate:
    g_estimates: int
    j_evaluations: int
    seconds: float

    @property
    def minutes(self) -> float:
        return self.seconds / 60


def estimate_cost(job: MetricFieldJob, seconds_per_g: float = SECONDS_PER_G,
                  seconds_per_j: float = SECONDS_PER_J) -> MetricFieldEstimate:
    """Per area and metric: one evaluation at the reference image plus grid_size^2 points for each of the 2 encoders."""
    per_part = 1 + job.grid_size ** 2 * 2
    n_parts = len(job.areas)
    g = per_part * n_parts if "G" in job.metrics else 0
    j = per_part * n_parts if "J" in job.metrics else 0
    return MetricFieldEstimate(g, j, g * seconds_per_g + j * seconds_per_j)


def print_estimate(job: MetricFieldJob) -> MetricFieldEstimate:
    est = estimate_cost(job)
    print(f"{est.g_estimates} G(x) estimates and {est.j_evaluations} J(x) evaluations: about {est.minutes:.1f} min "
          f"on one {'GPU' if job.device == 'gpu' else 'CPU'} (a G(x) estimate takes ~{SECONDS_PER_G:.0f} s on an RTX 5090"
          f"{'' if job.device == 'gpu' else ', and far longer on a CPU'}).")
    return est


# ── running ─────────────────────────────────────────────────────────────────
def metric_field_command(job: MetricFieldJob, metric: str, *, runs_dir: str = "runs", out: str | None = None,
                   python: str = sys.executable) -> list[str]:
    """Command line of `python -m multigeom.pipeline.metric_field` for one metric of the job (the computation is not
    re-implemented here)."""
    rule, size = job.steps[metric]
    cmd = [
        python, "-m", "multigeom.pipeline.metric_field",
        "--img-idx", str(job.img_idx),
        "--reference-encoder", job.reference_encoder, "--other-encoder", job.other_encoder,
        "--areas", *job.areas, "--metrics", metric,
        "--grid-size", str(job.grid_size), "--step-rule", rule, "--step-size", str(size),
        "--runs-dir", runs_dir, "--device", "cuda:0" if job.device == "gpu" else "cpu",
    ]
    if out is not None:
        cmd += ["--out-dir", out]
    return cmd


def metric_field_commands(job: MetricFieldJob, *, runs_dir: str = "runs", out: str | None = None,
                    python: str = sys.executable) -> list[list[str]]:
    """One command per metric: the stage takes one step rule per run, and G and J use different ones."""
    return [metric_field_command(job, m, runs_dir=runs_dir, out=out, python=python) for m in job.metrics]


def job_to_json(job: MetricFieldJob) -> str:
    return json.dumps({"img_idx": job.img_idx, "grid_size": job.grid_size, "steps": job.steps,
                       "reference_encoder": job.reference_encoder, "other_encoder": job.other_encoder,
                       "areas": list(job.areas), "metrics": list(job.metrics), "device": job.device})


def job_from_json(text: str) -> MetricFieldJob:
    return MetricFieldJob(**json.loads(text))


def launch(job: MetricFieldJob, *, runs_dir: str = "runs") -> dict:
    """
    Start the computation as one detached process (its own session) that runs the local metric field computation once per metric, logging to
    <runs_dir>/logs/metric_field.log.
    It keeps running if the notebook kernel dies; the pid is recorded in <runs_dir>/logs/metric_field_worker.json.
    """
    log_dir = REPO / runs_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "metric_field.log"
    job_file = log_dir / "metric_field_job.json"
    job_file.write_text(job_to_json(job))
    with open(log_path, "ab") as log:
        p = subprocess.Popen([sys.executable, "-m", "multigeom.pipeline.metric_field_job", "run", "--job", str(job_file),
                              "--runs-dir", runs_dir],
                             cwd=str(REPO), env=subprocess_env(), stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    proc = {"pid": p.pid, "log": str(log_path), "device": job.device}
    (log_dir / "metric_field_worker.json").write_text(json.dumps([proc]))
    return proc


def progress(job: MetricFieldJob, runs_dir: str = "runs") -> tuple[int, int]:
    """(parts finished, parts in total); a part is one (metric, area) file."""
    files = expected_files(job, runs_dir)
    return sum(f.exists() for f in files), len(files)


def is_complete(job: MetricFieldJob, runs_dir: str = "runs") -> bool:
    done, total = progress(job, runs_dir)
    return done == total


def wait(job: MetricFieldJob, proc=None, *, runs_dir: str = "runs", poll_s: float = 15.0,
         timeout_s: float | None = None, stop_on_interrupt: bool = True, verbose: bool = True) -> bool:
    """
    Block until the process ends, showing a progress line over the (metric, area) parts (the bar moves when a
    part is finished). Interrupting the kernel stops the process too (finished parts are kept; launching again
    recomputes only what is missing), or only the waiting with `stop_on_interrupt=False`. `proc` is what
    `launch` returned; if omitted, the last launch is used (also after a kernel restart). Returns True if complete.
    """
    if proc is None:
        f = REPO / runs_dir / "logs" / "metric_field_worker.json"
        procs = json.loads(f.read_text()) if f.exists() else []
    else:
        procs = [proc]
    pids = [p["pid"] for p in procs]
    done, total = progress(job, runs_dir)
    bar = ProgressLine(total, initial=done, desc="local metric field", unit="part") if verbose else None
    t0 = time.time()
    try:
        while True:
            running = alive(pids)
            done, total = progress(job, runs_dir)
            if bar is not None:
                bar.update(done, f"{len(running)} process running; {done}/{total} (metric, area) parts")
            if not running:
                break
            if timeout_s is not None and time.time() - t0 > timeout_s:
                if bar is not None:
                    bar.close()
                print("stopped waiting (timeout); the process keeps running")
                return False
            time.sleep(poll_s)
    except KeyboardInterrupt:
        if bar is not None:
            bar.close()
        if stop_on_interrupt:
            print("interrupted: stopping the process (finished parts are kept)")
            stop_workers(procs, runs_dir=runs_dir, verbose=verbose)
        else:
            print("stopped waiting; the process keeps running in the background")
        return False
    if bar is not None:
        bar.close()
    complete = is_complete(job, runs_dir)
    if verbose:
        print("all parts done" if complete else "the process ended but some parts are missing: check runs/logs/metric_field.log")
    return complete


def clear_results(job: MetricFieldJob, runs_dir: str = "runs", dry_run: bool = False) -> list[Path]:
    """Delete this job's result files (only those) so that it is recomputed from scratch; refuses while it runs."""
    f = REPO / runs_dir / "logs" / "metric_field_worker.json"
    if not dry_run and f.exists() and alive([p["pid"] for p in json.loads(f.read_text())]):
        raise RuntimeError("the process is still running: stop it first with stop()")
    files = [p for p in expected_files(job, runs_dir) if p.exists()]
    for p in files:
        if not dry_run:
            p.unlink()
    print(f"{'would delete' if dry_run else 'deleted'} {len(files)} result file(s)")
    return files


def stop(runs_dir: str = "runs") -> int:
    """Stop the recorded process (also from a later session)."""
    f = REPO / runs_dir / "logs" / "metric_field_worker.json"
    return stop_workers(json.loads(f.read_text()) if f.exists() else [], runs_dir=runs_dir)


# ── describing a field, for choosing the reference image ────────────────────
def radial_score(matrices, coeffs) -> float:
    """
    How radially the ellipses of an [n, n, 2, 2] field are oriented around the grid centre: the mean over the grid
    points of abs(cos) between the major axis of the ellipse (see `multigeom.geometry.metric_field.ellipse_params`) and the direction
    from the centre to the point (a, b). Random orientations give about 0.64, a perfectly radial field 1.0.
    """
    import numpy as np
    from ..geometry.metric_field import ellipse_grid
    coeffs = np.asarray(coeffs, dtype=float)
    ell = ellipse_grid(matrices)
    cos = []
    for i, a in enumerate(coeffs):
        for j, b in enumerate(coeffs):
            if a == 0 and b == 0:
                continue
            theta = np.deg2rad(ell[i, j, 2])
            cos.append(abs(np.cos(theta) * a + np.sin(theta) * b) / np.hypot(a, b))
    return float(np.mean(cos))


def area_ratio(matrices_a, matrices_b) -> float:
    """Median over the grid points, the centre excluded, of the ratio of the ellipse areas (proportional to
    sqrt(det)) of field a to field b: above 1 means a is outside b."""
    import numpy as np
    area = lambda m: np.sqrt(np.clip(np.linalg.det(np.asarray(m, dtype=float)), 1e-30, None))
    ratio = area(matrices_a) / area(matrices_b)
    n = ratio.shape[0]
    keep = np.ones(ratio.shape, bool)
    keep[n // 2, n // 2] = False
    return float(np.median(ratio[keep]))


# ── the figure ──────────────────────────────────────────────────────────────
def load_records(job: MetricFieldJob, runs_dir: str = "runs") -> dict:
    """Stage-6 records keyed by (metric, area), for the files that exist."""
    import torch
    out = {}
    for m in job.metrics:
        for a in job.areas:
            path = out_dir(job, runs_dir) / f"{m}_{a}.pt"
            if path.exists():
                out[(m, a)] = torch.load(path, map_location="cpu", weights_only=False)
    return out


def plot_metric_field(records, *, fill: float = 0.9, show_x0: bool = False):
    """Fig. 5 from Stage-6 records ({(metric, area): record}), or from a job whose files are in runs/."""
    from ..viz.fig5 import make_figure
    if isinstance(records, MetricFieldJob):
        records = load_records(records)
    if not records:
        raise ValueError("no results to draw: run the job first")
    return make_figure(records, fill=fill, show_x0=show_x0)


# ── command line (used by the detached process) ─────────────────────────────
def run(job: MetricFieldJob, runs_dir: str = "runs") -> None:
    """Run the local metric field computation for each metric of the job, one after the other (G first: it is the long one)."""
    for cmd in metric_field_commands(job, runs_dir=runs_dir):
        subprocess.run(cmd, cwd=str(REPO), env=subprocess_env(), check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="run the stage-6 computation of a MetricFieldJob")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--job", required=True, help="JSON file written by launch()")
    r.add_argument("--runs-dir", default="runs")
    args = p.parse_args(argv)
    run(job_from_json(Path(args.job).read_text()), args.runs_dir)


if __name__ == "__main__":
    main()
