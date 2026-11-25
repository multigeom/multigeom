"""
The figures of the estimates at each noise scale: the leading eigenvectors of M_t(x),
the integrand of G(x), and of the stimulus features M_t^env(x) at each noise scale, with
the Fisher information of each neuron (`plot_per_scale`); the envelope spectrum
(`plot_env_spectrum`); and the coding spectrum (`plot_coding_spectrum`). Each returns
the figure and the data it shows.
"""

from __future__ import annotations

import numpy as np
import torch
from matplotlib.gridspec import GridSpec

from ..analysis.per_scale import (
    align_signs,
    compute_band,
    compute_information,
    compute_normalized,
)
from .style import (
    AREA_COLORS,
    RF_COLORS,
    as_numpy,
    get_color,
    panel_label,
    plt,
    show_eigenvector,
)

# the colour of each block of eigenvectors: the areas, and the stimulus features
BLOCK_COLORS = {**AREA_COLORS, "stimulus": "#4D4D4D"}

# the colours of the envelope spectrum: the null token of each model, the Gaussian
# bound, and its limits
SPECTRUM_COLORS = {"V1": "#000000", "V4": "#8C8C8C", "gaussian": "#CC79A7"}
LIMIT_COLOR = "#BBBBBB"

# the layout of the figure of one image: the height of its rows (in units of a tile),
# its width, the height with two blocks and one panel of eigenvalues, and the extra
# height of each further block and of each further panel of eigenvalues
EIGVEC_ROW, BLOCK_GAP, VALUES_GAP, VALUES_ROW = 1.0, 0.45, 0.9, 3.2
FIGURE_WIDTH, TWO_BLOCK_HEIGHT, BLOCK_HEIGHT, VALUES_HEIGHT = 16.5, 8.6, 3.05, 2.8

# the Fisher rows under a block (one row per cell, one for the total), apart by a
# gap, and the height in inches of one unit of row height
FISHER_GAP, INCHES_PER_ROW = 0.45, 3.05 / 4.45

# the labels of the panels of eigenvalues: the areas, normalized, and the stimulus
# features, as they are
AREAS_LABEL = "eigenvalues of $M_t(x)$\nover the largest\n$\\lambda_1$ of each area"
STIMULUS_LABEL = "eigenvalues\nof $M_t^{\\rm env}(x)$"


def get_plot_data(image: torch.Tensor, records: dict) -> dict:
    """
    The data of the figure of one image: the image [H, W], tau = log t [J], and for
    each block its eigenvalues [J, k], the same over its largest leading eigenvalue
    ("normalized"), and its eigenvectors [J, k, P] with the signs of `align_signs`.
    `records` maps each block (an area, or "stimulus") to its PerScaleOutput.
    """
    # the image, and the noise levels (the same for every block)
    first = next(iter(records.values()))
    side = image.shape[-1]
    data = {"image": image.reshape(side, side).numpy(), "log_t": first.log_tau.numpy()}

    # the eigenvalues, raw and normalized, and the signed eigenvectors of each block
    for name, record in records.items():
        values = record.eigenvalues.numpy()
        data[name] = {
            "eigenvalues": values,
            "normalized": compute_normalized(values, values[:, 0]),
            "eigenvectors": align_signs(record.eigenvectors).numpy(),
        }
    return data


def draw_block(
    fig: plt.Figure,
    grid: GridSpec,
    top: int,
    data: dict,
    name: str,
    header: bool,
    outlines: dict,
    label: str | None = None,
) -> plt.Axes:
    """
    The eigenvectors of one block, from row `top` of `grid` (the tiles start at its
    fourth column): one row per eigenvector, one column per noise level, with the
    noise levels above the columns if `header`, and on every tile the outline of the
    region covered by each area of `outlines` (area → `compute_rf_outline`), and
    `label` (the block's name if None) on the left. Returns the top-left tile.
    """
    # the eigenvectors [J, k, P] of the block: J noise levels, k eigenvectors each
    eigvecs = data[name]["eigenvectors"]
    n_levels, n_eigvecs = eigvecs.shape[:2]

    # one tile per eigenvector (row) and noise level (column), each eigenvector drawn
    # with the outlines of the covered regions on top
    tiles = [
        [fig.add_subplot(grid[top + row, 3 + level]) for level in range(n_levels)]
        for row in range(n_eigvecs)
    ]
    for row in range(n_eigvecs):
        for level in range(n_levels):
            show_eigenvector(tiles[row][level], eigvecs[level, row])
            draw_outlines(tiles[row][level], outlines)

    # the names of the rows (v1, v2, ...) in the first column
    for row in range(n_eigvecs):
        tiles[row][0].set_ylabel(
            f"$v_{row + 1}$", fontsize=9, rotation=0, ha="right", va="center"
        )

    # the noise level tau = log t above each column, if asked
    if header:
        for level in range(n_levels):
            scale = f"{data['log_t'][level]:.1f}".replace("-", "−")
            tiles[0][level].set_title(scale, fontsize=7)

    # the label of the block, left of the names of its rows
    draw_row_label(fig, tiles[0][0], tiles[-1][0], name if label is None else label)
    return tiles[0][0]


def draw_row_label(
    fig: plt.Figure, top: plt.Axes, bottom: plt.Axes, label: str, dx: float = 0.028
) -> None:
    """
    `label`, vertical, `dx` (figure units) left of the rows from the tile `top` to the
    tile `bottom`.
    """
    top_box, bottom_box = top.get_position(), bottom.get_position()
    middle = (top_box.y1 + bottom_box.y0) / 2
    fig.text(
        top_box.x0 - dx,
        middle,
        label,
        fontsize=12,
        rotation=90,
        ha="center",
        va="center",
    )


def crop_around_peak(pattern: np.ndarray, size: int) -> np.ndarray:
    """The `size` x `size` window of the square image `pattern` (flattened) centred
    on its largest |value|, moved inside the image if it would cross an edge,
    flattened."""
    side = int(round(pattern.size**0.5))
    image = pattern.reshape(side, side)
    row, column = np.unravel_index(np.abs(image).argmax(), image.shape)
    top = int(np.clip(row - size // 2, 0, side - size))
    left = int(np.clip(column - size // 2, 0, side - size))
    return image[top : top + size, left : left + size].ravel()


def draw_fisher_rows(
    fig: plt.Figure,
    grid: GridSpec,
    top: int,
    fisher: dict,
    outlines: dict,
    label: str,
    crop: int | None = None,
    title: str = "J(x) per cells",
) -> None:
    """
    From row `top` of `grid` (the tiles start at its fourth column): a row with the
    patterns of the neurons with the largest Fisher information, one per tile, largest
    first ("per cell"), and below it, in one tile, their total summed over the cells
    pixel by pixel: the diagonal of J(x) = Σ_n g_n g_nᵀ, Σ_n g_n², the Fisher
    information at each pixel ("diag J(x)"); the outlines of `outlines` on every tile,
    `label` on the left, and a title above. `fisher` is the output of
    `compute_fisher_per_cell`, possibly shortened to its strongest cells with the
    total stored as "diag" (then used instead of the sum over the stored cells). With
    `crop`, each cell's tile shows only the `crop` x `crop` pixels around its peak
    (without outlines), for cells much smaller than the image. `title` goes above the
    rows.
    """
    # the neurons in order of their Fisher information, one tile each
    order = np.argsort(fisher["cell_values"])[::-1]
    n_tiles = min(grid.ncols - 3, len(order))
    for column in range(n_tiles):
        ax = fig.add_subplot(grid[top, 3 + column])
        pattern = fisher["cells"][order[column]]
        if crop:
            show_eigenvector(ax, crop_around_peak(np.asarray(pattern), crop))
        else:
            show_eigenvector(ax, pattern)
            draw_outlines(ax, outlines)
        if column == 0:
            name = f"per cell\n({crop} x {crop} px)" if crop else "per cell"
            ax.set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
            first = ax

    # the total over the cells, pixel by pixel: the diagonal of J(x), Σ_n g_n², as a
    # 64 x 64 image like the other tiles
    total = fisher["diag"] if "diag" in fisher else (fisher["cells"] ** 2).sum(axis=0)
    ax_total = fig.add_subplot(grid[top + 1, 3])
    show_eigenvector(ax_total, total)
    draw_outlines(ax_total, outlines)
    ax_total.set_ylabel("diag J(x)", fontsize=8, rotation=0, ha="right", va="center")

    # the label of the two rows, left of their names, and their title above
    draw_row_label(fig, first, ax_total, label, dx=0.045)
    draw_rows_title(fig, grid, top, title)


def draw_rows_title(
    fig: plt.Figure, grid: GridSpec, row: int, title: str, pad: float = 0.0
) -> None:
    """`title` above row `row` of `grid`, centred over its tiles (4th column on),
    `pad` inches higher (to clear the noise scales above the first row)."""
    left = grid[row, 3].get_position(fig)
    right = grid[row, grid.ncols - 1].get_position(fig)
    fig.text(
        (left.x0 + right.x1) / 2,
        left.y1 + 0.003 + pad / fig.get_figheight(),
        title,
        fontsize=10,
        ha="center",
        va="bottom",
    )


def draw_outlines(ax: plt.Axes, outlines: dict) -> None:
    """
    The outline of the region covered by each area of `outlines` on a tile drawn by
    `show_eigenvector` (pixel k centred at k), in the colours of Fig. 3D.
    """
    for area, (field, coords) in outlines.items():
        ax.contour(
            coords - 0.5,
            coords - 0.5,
            field,
            levels=[0.5],
            colors=[RF_COLORS[area]],
            linewidths=0.8,
        )


def draw_eigenvalues(
    ax: plt.Axes, data: dict, names: list, key: str, ylabel: str
) -> None:
    """
    The eigenvalues of each block of `names` against tau = log t, raw ("eigenvalues")
    or over the block's largest leading eigenvalue ("normalized") as `key` says, the
    leading one thick and the others thin, in the columns of the tiles (one per noise
    level), on a linear axis from 0.
    """
    # one line per eigenvalue and block
    n_levels, n_eigvecs = data[names[0]][key].shape
    for index, name in enumerate(names):
        for k in range(n_eigvecs):
            ax.plot(
                np.arange(n_levels),
                data[name][key][:, k],
                color=get_color(BLOCK_COLORS, name, index),
                lw=2.0 if k == 0 else 0.8,
                alpha=1.0 if k == 0 else 0.6,
                label=name if k == 0 else None,
            )

    # the columns of the tiles, labelled with tau = log t; a linear axis for the values
    labels = [f"{value:.1f}".replace("-", "−") for value in data["log_t"]]
    ax.set_xlim(-0.5, n_levels - 0.5)
    ax.set_xticks(np.arange(n_levels), labels, fontsize=7)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\tau = \log t$   (t: variance of the noise, in image units)")
    ax.set_ylabel(ylabel)

    # which line is which, where the lines are not (the values are near 0 at the coarse
    # noise scales, on the right)
    ax.legend(
        title=rf"thick: $\lambda_1$, thin: $\lambda_2$ to $\lambda_{n_eigvecs}$",
        fontsize=8,
        title_fontsize=8,
        frameon=False,
        loc="upper right",
    )


def plot_per_scale(
    image: torch.Tensor,
    records: dict,
    params: dict,
    title: str,
    suptitle: str,
    outlines: dict,
    fisher: dict | None = None,
    eigenvalues: bool = True,
    stimulus: tuple = ("stimulus",),
    images: dict | None = None,
) -> tuple[plt.Figure, dict]:
    """
    The figure of one image: the image (A); for each block of `records` (an area, or
    "stimulus", mapped to its PerScaleOutput, in that order) its leading eigenvectors at
    each noise level, one row per eigenvector and one column per level (B, C, ...); then
    the eigenvalues against tau = log t, on linear axes: those of the stimulus features
    in their own panel, if there are any, and those of the areas, each over its largest
    leading eigenvalue, in the last panel. On the tiles of an area, the outline of the
    region its neurons cover (`outlines`: area → `compute_rf_outline`); on those of the
    stimulus features, the outlines of every area. With `fisher` (area →
    `compute_fisher_per_cell`), two more rows under each area: the Fisher information
    of its neurons, one per tile, and the eigenvectors of their total. Without the
    panels of eigenvalues if not `eigenvalues`. `stimulus` names the blocks that are
    stimulus features (their eigenvalues share their own panel, unnormalized), and
    `images` (name → [1, 1, H, W]) more images drawn under A, each from −m to m, m the
    99.5th percentile of its |values| (for images not in [-1, 1], such as whitened
    ones). Returns the figure and the data it shows.
    """
    # the data drawn, its blocks, and the numbers of noise levels and eigenvectors
    data = get_plot_data(image, records)
    names = list(records)
    n_levels, n_eigvecs = data[names[0]]["eigenvalues"].shape
    fisher = fisher or {}
    for area, record in fisher.items():
        data[area]["fisher"] = record

    # the panels of eigenvalues: the stimulus features as they are, then the areas,
    # normalized to compare where along tau each one sits
    stimuli = [name for name in names if name in stimulus]
    areas = [name for name in names if name not in stimulus]
    panels = [(areas, "normalized", AREAS_LABEL)] if areas else []
    if stimuli:
        panels.insert(0, (stimuli, "eigenvalues", STIMULUS_LABEL))
    panels = panels if eigenvalues else []

    # the grid: the image on the left; for each block one row per eigenvector and one
    # column per noise level (and the Fisher rows of an area, if any), blocks apart by
    # a gap; the eigenvalues at the bottom
    heights, tops = [], []
    for block, name in enumerate(names):
        heights += [BLOCK_GAP] if block else []
        tops.append(len(heights))
        heights += [EIGVEC_ROW] * n_eigvecs
        heights += [FISHER_GAP, EIGVEC_ROW, EIGVEC_ROW] if name in fisher else []
    first_value_row = len(heights) + 1
    heights = heights + [VALUES_GAP, VALUES_ROW] * len(panels)
    height = TWO_BLOCK_HEIGHT + BLOCK_HEIGHT * (len(names) - 2)
    height = height + VALUES_HEIGHT * (len(panels) - 1)
    fisher_rows = FISHER_GAP + 2 * EIGVEC_ROW
    height = height + len(fisher) * fisher_rows * INCHES_PER_ROW
    fig = plt.figure(figsize=(FIGURE_WIDTH, height), dpi=params["screen_dpi"])
    grid = fig.add_gridspec(
        len(heights),
        3 + n_levels,
        height_ratios=heights,
        width_ratios=[1, 1, 1.6] + [1] * n_levels,
        hspace=0.08,
        wspace=0.06,
    )

    # A: the image, with the regions the neurons of every area cover
    ax_image = fig.add_subplot(grid[0:2, 0:2])
    ax_image.imshow(data["image"], cmap="gray", vmin=-1, vmax=1)
    ax_image.set_title(title, fontsize=10)
    ax_image.axis("off")
    draw_outlines(ax_image, outlines)

    # more images under it, each on its own symmetric scale
    for index, (name, extra) in enumerate((images or {}).items()):
        values = as_numpy(extra.reshape(extra.shape[-2:]))
        limit = np.percentile(np.abs(values), 99.5)
        ax_extra = fig.add_subplot(grid[2 * index + 2 : 2 * index + 4, 0:2])
        ax_extra.imshow(values, cmap="gray", vmin=-limit, vmax=limit)
        ax_extra.set_title(name, fontsize=10)
        ax_extra.axis("off")

    # B, C, ...: the eigenvectors of each block, the first with the noise levels above,
    # with the outline of the area of the block, or of every area for the stimulus;
    # under an area, its Fisher rows if any
    first_tiles = []
    for block, name in enumerate(names):
        shown = outlines if name in stimulus else {}
        shown = {name: outlines[name]} if name in outlines else shown
        label = f"{name}\nG(x)" if name in fisher else None
        tile = draw_block(fig, grid, tops[block], data, name, block == 0, shown, label)
        first_tiles.append(tile)
        if name in fisher:
            pad = 0.18 if block == 0 else 0.0
            draw_rows_title(fig, grid, tops[block], "G(x) per noise scale", pad)
        if name in fisher:
            fisher_top = tops[block] + n_eigvecs + 1
            # (without the name of the block if `params` says so: it labels the rows
            # above, and a long label would run into the next block)
            short = params.get("fisher_label_without_name", False)
            fisher_label = "J(x)" if short else f"{name}\nJ(x)"
            crop = params.get("fisher_crop")
            title = params.get("fisher_title", "J(x) per cells")
            draw_fisher_rows(
                fig, grid, fisher_top, fisher[name], shown, fisher_label, crop, title
            )

    # the panels of eigenvalues, in the columns of the tiles
    value_axes = []
    for index, (blocks, key, ylabel) in enumerate(panels):
        ax = fig.add_subplot(grid[first_value_row + 2 * index, 3:])
        draw_eigenvalues(ax, data, blocks, key, ylabel)
        value_axes.append(ax)

    # the panel letters, left of the names of the rows and blocks
    axes = [ax_image, *first_tiles, *value_axes]
    for letter, ax in zip("ABCDEFG", axes):
        panel_label(fig, ax, letter, dx=-0.01 if letter == "A" else -0.04)

    # the title, on two lines, just above the noise scales of the first block (the
    # figure's height is only estimated, so a title at the top of the figure can leave
    # a gap)
    top = max(ax.get_position().y1 for ax in fig.axes)
    fig.suptitle(suptitle, fontsize=11, y=top + 0.55 / fig.get_figheight(), va="bottom")
    return fig, data


def plot_env_spectrum(
    spectra: dict, gaussian: torch.Tensor, params: dict
) -> tuple[plt.Figure, dict]:
    """
    The envelope spectrum against tau = log t, on a log scale, for the null token of the
    first model of `spectra` (name → curves of the images, with `log_tau` and
    `values`): the mean over the images, the 10–90% band of the images, and the
    Gaussian bound `gaussian`. Returns the figure and the data it shows.
    """
    # the data drawn: the noise scales, the Gaussian bound, the mean and the band
    name, first = next(iter(spectra.items()))
    values = first.values.double()
    data = {
        "log_t": first.log_tau.double().numpy(),
        "gaussian": gaussian.numpy(),
        "mean": values.mean(dim=0).numpy(),
        "band": values.quantile(torch.tensor([0.1, 0.9]).double(), dim=0).numpy(),
    }

    # one panel: the band of the images, their mean, and the Gaussian bound
    fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=params["screen_dpi"])
    color = get_color(SPECTRUM_COLORS, name, 0)
    low, high = data["band"]
    ax.fill_between(
        data["log_t"],
        low,
        high,
        color=color,
        alpha=0.15,
        lw=0,
        label="10–90% of the images",
    )
    ax.plot(data["log_t"], data["mean"], color=color, lw=2, label="mean of the images")
    ax.plot(
        data["log_t"],
        data["gaussian"],
        color=SPECTRUM_COLORS["gaussian"],
        lw=1.5,
        label="Gaussian bound: images assumed Gaussian,\nwith the pixel covariance"
        " of the training images",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau = \log t$")
    ax.set_ylabel(
        r"$\rho^{\rm env}(\tau) = \mathbb{E}\,\frac{1}{t}\,\|\hat x(x_t) - x\|^2$"
    )
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    # the title
    fig.suptitle(
        f"Envelope spectrum of the {len(values)} test images ({params['encoder']}"
        f" {name} model,\n{params['n_noise']} noisy versions per image and noise"
        " scale)",
        fontsize=11,
        y=1.04,
    )
    return fig, data


def plot_coding_spectrum(
    spectra: dict, bootstrap: dict, params: dict
) -> tuple[plt.Figure, dict]:
    """
    The coding spectrum ρ(τ) of each area of `spectra` (area → CodingSpectrumOutput)
    against tau = log t, on linear axes, each with the 95% band of its paired
    bootstrap over the images (`bootstrap`, from `compute_bootstrap`). A: on the same
    axes as the envelope spectrum ρ^env(τ) of the first area's model, fitted to ρ
    (ρ^env, hundreds of times larger at fine noise scales, is above the axes there);
    the legend gives ½ ∫ ρ dτ of each area. B: each ρ over its own maximum, to compare
    where along tau each area sits. Returns the figure and the data it shows.
    """
    # the data drawn: the noise scales, the envelope, and per area ρ, ρ over its
    # maximum, their 95% bands, and ½ ∫ ρ dτ
    areas = list(spectra)
    colors = {area: get_color(AREA_COLORS, area, i) for i, area in enumerate(areas)}
    log_t = spectra[areas[0]].log_tau.double()
    data = {"log_t": log_t.numpy()}
    data["env"] = spectra[areas[0]].env.double().mean(dim=0).numpy()
    for area in areas:
        record = spectra[area]
        rho = record.rho.double().numpy()
        data[area] = {
            "rho": rho,
            "rho_band": compute_band(bootstrap[area]["mean"]),
            "normalized": compute_normalized(rho, rho),
            "normalized_band": compute_band(bootstrap[area]["normalized"]),
            "information": compute_information(record.env - record.coded, log_t)[0],
        }

    # two panels side by side
    fig, (ax_rho, ax_shape) = plt.subplots(
        1, 2, figsize=(12, 4.4), dpi=params["screen_dpi"]
    )

    # A: the envelope, then ρ of each area with its band, and the line ρ = 0
    ax_rho.plot(
        data["log_t"],
        data["env"],
        color=SPECTRUM_COLORS["V1"],
        lw=2,
        label=r"$\rho^{\rm env}(\tau)$ (above the axes at fine scales)",
    )
    for area in areas:
        draw_band(ax_rho, data["log_t"], data[area]["rho_band"], colors[area])
        ax_rho.plot(
            data["log_t"],
            data[area]["rho"],
            "o-",
            ms=3,
            color=colors[area],
            lw=2,
            label=rf"$\rho(\tau)$ {area}: $\frac{{1}}{{2}}\int \rho\, d\tau$ = "
            f"{data[area]['information']:.1f} nats",
        )
    ax_rho.axhline(0.0, color=LIMIT_COLOR, lw=0.8)

    # A: a linear axis fitted to ρ, with room for the legend above the curves
    lowest = min(data[area]["rho_band"].min() for area in areas)
    highest = max(data[area]["rho_band"].max() for area in areas)
    ax_rho.set_ylim(min(lowest, 0.0) - 0.05 * highest, 1.35 * highest)
    ax_rho.set_xlabel(r"$\tau = \log t$")
    ax_rho.set_ylabel(r"$\rho(\tau)$ and $\rho^{\rm env}(\tau)$")
    ax_rho.legend(fontsize=8, frameon=False, loc="upper left")

    # B: each ρ over its own maximum, with its band, and the line ρ = 0
    for area in areas:
        band = data[area]["normalized_band"]
        draw_band(ax_shape, data["log_t"], band, colors[area])
        ax_shape.plot(
            data["log_t"],
            data[area]["normalized"],
            "o-",
            ms=3,
            color=colors[area],
            lw=2,
            label=area,
        )
    ax_shape.axhline(0.0, color=LIMIT_COLOR, lw=0.8)
    ax_shape.set_xlabel(r"$\tau = \log t$")
    ax_shape.set_ylabel(r"$\rho(\tau)$ over its maximum")
    ax_shape.legend(fontsize=8, frameon=False, loc="upper left")

    # the panel letters, and the title
    for letter, ax in zip("AB", (ax_rho, ax_shape)):
        panel_label(fig, ax, letter)
    fig.suptitle(
        f"Coding spectrum of the {len(spectra[areas[0]].env)} test images"
        f" ({params['encoder']} models),\n{params['n_noise']} noisy versions and"
        " responses per image and noise scale; bands: 95% paired bootstrap",
        fontsize=11,
        y=1.03,
    )
    return fig, data


def draw_band(ax: plt.Axes, x: np.ndarray, band: np.ndarray, color: str) -> None:
    """A shaded band between the two rows of `band` [2, len(x)], behind the curves."""
    ax.fill_between(x, band[0], band[1], color=color, alpha=0.25, lw=0)
