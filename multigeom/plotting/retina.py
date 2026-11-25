"""
The figures of the retina models, in the layout of open-retina's information-geometry
figures (`run_example._plot_cell_summary`), so that they can be compared with them.

    plot_cell_summary   one row per image: the image, then the leading eigenvectors of
                        a metric (G(x) or J(x)), each with its share of the eigenvalues
    plot_cell_counts    the number of recorded cells of each type, per experiment
    plot_cell_responses one row per recorded cell: a test clip's images, the spike
                        counts of its repeats, their mean rate, and the mean rate over
                        the training clips
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .style import plt


def get_summary_data(images: torch.Tensor, eigenpairs: list, n_eig: int) -> dict:
    """
    The data of `plot_cell_summary`: the images [n, H, W], and per image the leading
    `n_eig` eigenvectors [n_eig, H, W], each over its largest |value|, and their shares
    of the sum of the eigenvalues given [n_eig].
    """
    side = images.shape[-1]
    data = {"images": images[:, 0].cpu().numpy(), "vectors": [], "shares": []}
    for pair in eigenpairs:
        vectors = pair.eigenvectors[:n_eig].reshape(n_eig, side, side).cpu().numpy()
        vectors = vectors / (np.abs(vectors).max(axis=(1, 2), keepdims=True) + 1e-9)
        values = pair.eigenvalues.cpu().double()
        data["vectors"].append(vectors)
        data["shares"].append((values / values.sum()).numpy()[:n_eig])
    return data


def plot_cell_summary(
    images: torch.Tensor,
    eigenpairs: list,
    title: str,
    labels: list | None = None,
    n_eig: int = 5,
) -> tuple[plt.Figure, dict]:
    """
    One row per image of `images` [n, 1, H, W]: the image (grey), then the leading
    `n_eig` eigenvectors of its metric (`eigenpairs`, one output with `eigenvalues` and
    `eigenvectors` per image), each over its largest |value| (red positive, blue
    negative) with its share of the sum of the eigenvalues given; `labels` on the left
    of the rows. Returns the figure and the data it shows.
    """
    # the data drawn, and one panel per image and column
    data = get_summary_data(images, eigenpairs, n_eig)
    n_images = len(data["images"])
    fig, axes = plt.subplots(
        n_images,
        n_eig + 1,
        figsize=(2.0 * (n_eig + 1), 2.0 * n_images),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    # each row: the image, then the eigenvectors with their shares
    for row in range(n_images):
        axes[row, 0].imshow(data["images"][row], cmap="gray")
        axes[row, 0].set_ylabel(labels[row] if labels else f"idx{row}", fontsize=8)
        for k in range(n_eig):
            ax = axes[row, k + 1]
            ax.imshow(data["vectors"][row][k], cmap="RdBu_r", vmin=-1, vmax=1)
            ax.text(
                0.04,
                0.95,
                f"{data['shares'][row][k] * 100:.0f}%",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    # the column titles, no ticks, and the axes frames
    axes[0, 0].set_title("stimulus", fontsize=8)
    for k in range(n_eig):
        axes[0, k + 1].set_title(f"e{k + 1}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)

    # the colour scale, shared, and the title
    bar = fig.colorbar(
        ScalarMappable(norm=Normalize(-1, 1), cmap="RdBu_r"),
        ax=axes,
        fraction=0.02,
        pad=0.01,
        ticks=[-1, 0, 1],
    )
    bar.ax.set_yticklabels(["−", "0", "+"])
    bar.set_label("eigenvector weight (per-panel normalised)", fontsize=8)
    fig.suptitle(title, fontsize=10)
    return fig, data


def plot_cell_counts(rows: list, params: dict) -> tuple[plt.Figure, dict]:
    """
    The number of recorded cells of each type (`rows`, from `list_lorenzi_cells`),
    stacked by experiment, most frequent type first; untyped cells last. Returns the
    figure and the counts ({type: {experiment: n}}).
    """
    # the counts, per type and experiment
    counts = {}
    for row in rows:
        kind = row["cell_type"] or "untyped"
        counts.setdefault(kind, {}).setdefault(row["experiment"], 0)
        counts[kind][row["experiment"]] += 1
    kinds = sorted(counts, key=lambda k: (k == "untyped", -sum(counts[k].values())))
    experiments = sorted({row["experiment"] for row in rows})

    # one horizontal bar per type, stacked by experiment
    fig, ax = plt.subplots(
        figsize=(7.0, 0.22 * len(kinds) + 1.0), dpi=params["screen_dpi"]
    )
    left = np.zeros(len(kinds))
    for experiment in experiments:
        values = np.array([counts[k].get(experiment, 0) for k in kinds])
        ax.barh(kinds, values, left=left, label=experiment)
        left += values
    for i, total in enumerate(left):
        ax.text(total + 0.5, i, str(int(total)), va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("recorded cells")
    ax.legend(title="experiment", fontsize=8, title_fontsize=8, frameon=False)
    ax.set_title(f"{len(rows)} recorded retinal ganglion cells", fontsize=10)
    fig.tight_layout()
    return fig, counts


def plot_cell_responses(
    cells: list, params: dict, clip: int = 0
) -> tuple[plt.Figure, dict]:
    """
    One row per recorded cell of `cells` (from `load_lorenzi_cell`): the adapting and
    the natural image of the test clip `clip`; the spike counts of each of its repeats
    (one row per repeat, 25 ms bins); their mean firing rate ± its standard error over
    the repeats; and the mean rate over all the training clips (one trial each). The
    adapting image is on for the first 16 bins (shaded), the natural image for the
    last 16. Returns the figure and the rates drawn (spikes/s).
    """
    from ..datasets.lorenzi import BIN_SIZE_S, N_ADAPT_BINS

    fig, axes = plt.subplots(
        len(cells),
        5,
        figsize=(13.0, 2.1 * len(cells)),
        dpi=params["screen_dpi"],
        gridspec_kw={"width_ratios": [0.7, 0.7, 1.6, 1.6, 1.6]},
        squeeze=False,
    )
    n_bins = cells[0].test_counts.shape[-1]
    time = (np.arange(n_bins) + 0.5) * BIN_SIZE_S * 1000  # ms from the adaptor's onset
    onset = N_ADAPT_BINS * BIN_SIZE_S * 1000
    drawn = {}
    for row, cell in zip(axes, cells):
        # the two images of the clip, on the cell's gray scale around its grey level
        frames = cell.frames
        spread = np.abs(frames - cell.grey).max()
        for ax, index, name in (
            (row[0], cell.test_adapt[clip], "adapting image"),
            (row[1], cell.test_natural[clip], "natural image"),
        ):
            ax.imshow(
                frames[index],
                cmap="gray",
                vmin=cell.grey - spread,
                vmax=cell.grey + spread,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name, fontsize=8)
        kind = cell.metadata.get("cell_type") or "untyped"
        row[0].set_ylabel(f"{cell.cell}\n{kind}", fontsize=9)

        # the spike counts of each repeat of the clip
        counts = cell.test_counts[clip]
        image = row[2].imshow(
            counts,
            aspect="auto",
            cmap="Greys",
            interpolation="nearest",
            extent=(0, n_bins * BIN_SIZE_S * 1000, counts.shape[0], 0),
        )
        fig.colorbar(image, ax=row[2], fraction=0.05, pad=0.02, label="spikes")
        row[2].set_ylabel("repeat", fontsize=8)

        # their mean rate ± standard error, and the mean rate over the training clips
        rates = counts / BIN_SIZE_S
        mean = rates.mean(axis=0)
        sem = rates.std(axis=0, ddof=1) / np.sqrt(len(rates))
        row[3].plot(time, mean, color="black", lw=1.5)
        row[3].fill_between(time, mean - sem, mean + sem, color="black", alpha=0.2)
        train = cell.train_counts.mean(axis=0) / BIN_SIZE_S
        row[4].plot(time, train, color="black", lw=1.5)
        row[3].set_ylabel("spikes/s", fontsize=8)
        drawn[cell.cell] = {"test_mean": mean, "test_sem": sem, "train_mean": train}

        # the adapting image's period shaded on the time axes
        for ax in row[2:]:
            ax.axvspan(0, onset, color="tab:blue", alpha=0.08, lw=0)
            ax.set_xlim(0, n_bins * BIN_SIZE_S * 1000)

    # the titles and time labels
    axes[0, 2].set_title(f"test clip {clip}: spike counts per repeat", fontsize=9)
    axes[0, 3].set_title("its mean rate ± s.e.m. over repeats", fontsize=9)
    axes[0, 4].set_title(
        f"mean rate over the {len(cells[0].train_counts)} training clips", fontsize=9
    )
    for ax in axes[-1, 2:]:
        ax.set_xlabel("time from the adapting image's onset (ms)", fontsize=8)
    fig.tight_layout()
    return fig, drawn
