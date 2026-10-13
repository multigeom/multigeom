"""
Shared style and loading helpers for the figure and table scripts.

The figure scripts read only files written by the pipeline stages (torch
checkpoints and JSON), so they run on a CPU machine without `diffusers` or
`plenoptic`. Missing inputs are reported with `skip(...)` and the element is
left out; nothing here raises on a missing file.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib


def _in_notebook() -> bool:
    """True in a Jupyter kernel, where the notebook backend must be kept so that a figure is shown as a picture."""
    try:
        from IPython import get_ipython
        return getattr(get_ipython(), "kernel", None) is not None
    except ImportError:
        return False


if not _in_notebook():
    matplotlib.use("Agg")          # scripts: draw to files, without a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

IMAGE_SIZE = 64
ENCODERS = ("inception", "resnet50", "onoff")
AREAS = ("V1", "V4")

ENCODER_LABELS = {"inception": "inception", "resnet50": "res-net", "onoff": "on-off"}
ENCODER_LABELS_LONG = {"inception": "Inception", "resnet50": "ResNet-50", "onoff": "OnOff"}
ENCODER_COLORS = {"inception": "#1F77B4", "resnet50": "#FF7F0E", "onoff": "#2CA58D"}
AREA_COLORS = {"V1": "#0072B2", "V4": "#E69F00"}          # Fig. 3B-C scatter
RF_COLORS = {"V1": "#56B4E9", "V4": "#E69F00"}            # Fig. 3D outlines (blue, yellow)
METRIC_TITLES = {"G": "multi-scale Fisher", "J": "Fisher information"}
EIGVEC_CMAP = "RdBu_r"

# The four comparisons of Fig. 4D-E and the Bures-Wasserstein appendix figure:
# (key in cross_encoder_metrics.json "named", legend label, colour, line style).
# Dashed: same model, different areas. Solid: same area, different models.
NAMED_COMPARISONS = (
    ("cross_area_inception", "V1 vs V4 (Inc)", "#0072B2", "--"),
    ("cross_area_resnet50", "V1 vs V4 (Res)", "#E69F00", "--"),
    ("cross_model_V1_inception_resnet50", "Inc vs Res (V1)", "#117733", "-"),
    ("cross_model_V4_inception_resnet50", "Inc vs Res (V4)", "#882255", "-"),
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def skip(script: str, what: str) -> None:
    """Report a missing input that leaves one element of a figure out."""
    print(f"[{script}] skipping {what}")


def load_torch(path: Path, script: str, what: str):
    """torch.load `path`, or report and return None if it does not exist."""
    if not path.exists():
        skip(script, f"{what}: {path} not found")
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def encoder_checkpoint(runs_dir: Path, encoder: str, area: str, script: str):
    """
    Encoder checkpoint runs/encoders/<encoder>/<area>/best_model.pt, or None.

    The scores of the fit (`per_neuron_test_corr`, `test_predictions`, `oracle`, ...) are returned as top-level entries
    for both layouts: the paper's older checkpoints have them there, those written by `NeuralEncoder.save` keep them
    in their `metadata`.
    """
    ckpt = load_torch(runs_dir / "encoders" / encoder / area / "best_model.pt",
                      script, f"{encoder}/{area} encoder")
    if isinstance(ckpt, dict) and "metadata" in ckpt:
        ckpt = {**ckpt, **ckpt["metadata"]}
    return ckpt


def per_image(runs_dir: Path, encoder: str, area: str, img_idx: int, script: str):
    """Stage-4 output runs/per_image/<encoder>/<area>/img{idx}.pt, or None."""
    return load_torch(runs_dir / "per_image" / encoder / area / f"img{img_idx:04d}.pt",
                      script, f"{encoder}/{area} eigenvectors of image {img_idx}")


def cross_encoder_metrics(runs_dir: Path, script: str) -> dict | None:
    """Stage-5 output runs/metrics/cross_encoder_metrics.json, or None."""
    path = runs_dir / "metrics" / "cross_encoder_metrics.json"
    if not path.exists():
        skip(script, f"comparison curves: {path} not found (run Stage 5)")
        return None
    with open(path) as f:
        return json.load(f)


def plot_comparisons(ax, metrics: dict | None, metric: str, quantity: str, script: str) -> int:
    """
    Draw the four NAMED_COMPARISONS for `metric` ("G" or "J") as mean ± s.e.m.
    across images versus k. `quantity` is "alignment" or "bures_wasserstein".
    Returns the number of curves drawn.
    """
    if metrics is None:
        return 0
    n_drawn = 0
    for name, label, color, ls in NAMED_COMPARISONS:
        key = f"{name}__{metric}"
        pair = metrics.get("pairs", {}).get(metrics.get("named", {}).get(key, ""))
        if pair is None:
            skip(script, f"curve {key}: not in cross_encoder_metrics.json")
            continue
        k = np.asarray(pair["k"])
        mean = np.asarray(pair[f"{quantity}_mean"])
        sem = np.asarray(pair[f"{quantity}_sem"])
        ax.plot(k, mean, color=color, ls=ls, lw=2.0, label=label)
        ax.fill_between(k, mean - sem, mean + sem, color=color, alpha=0.15, lw=0)
        n_drawn += 1
    return n_drawn


def as_numpy(x) -> np.ndarray:
    """A tensor or array as a NumPy array."""
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def match_to_reference(ref: torch.Tensor, tgt: torch.Tensor, n: int, pool: int) -> torch.Tensor:
    """
    For display only: for each of the first `n` reference eigenvectors (rows of
    `ref`), pick the not-yet-used eigenvector among the first `pool` rows of
    `tgt` with the largest abs(cosine), and flip its sign to make the cosine
    positive. Returns [n, P]. With pool = n this only reorders and sign-flips
    the leading n target eigenvectors.
    """
    ref = ref.double()
    tgt = tgt.double()
    pool = max(min(pool, tgt.shape[0]), min(n, tgt.shape[0]))
    used: set[int] = set()
    out = []
    for i in range(min(n, ref.shape[0], tgt.shape[0])):
        cos = tgt[:pool] @ ref[i]
        score = cos.abs().clone()
        for u in used:
            score[u] = -1.0
        j = int(score.argmax())
        used.add(j)
        out.append(tgt[j] if cos[j] >= 0 else -tgt[j])
    return torch.stack(out).float()


def show_eigenvector(ax, v, cmap: str = EIGVEC_CMAP) -> None:
    """Draw one eigenvector as a 64x64 image with a symmetric diverging colour
    scale, clipped at the 99th percentile of abs(v) so a few extreme pixels do
    not wash out the pattern."""
    img = as_numpy(v).reshape(IMAGE_SIZE, IMAGE_SIZE)
    vmax = float(np.percentile(np.abs(img), 99.0)) + 1e-12
    ax.imshow(img, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def blank(ax, text: str = "n/a") -> None:
    """Placeholder for an element whose input is missing."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color="0.5")


def panel_label(fig, ax, letter: str, dx: float = -0.02, dy: float = 0.01) -> None:
    """Bold panel letter at the top-left corner of `ax` (figure coordinates)."""
    box = ax.get_position()
    fig.text(box.x0 + dx, box.y1 + dy, letter, fontsize=14, fontweight="bold",
             ha="right", va="bottom")


def save_figure(fig, out_dir: Path, name: str, dpi: int = 300) -> list[Path]:
    """Save `fig` as <out_dir>/<name>.pdf and .png and close it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for p in paths:
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    for p in paths:
        print(f"saved {p}")
    return paths
