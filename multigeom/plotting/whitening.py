"""
The figures of the whitening of the images (`multigeom.datasets.whitening`): the power
spectra and the filter, its spatial kernel, and example images raw and whitened.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..datasets.whitening import WhiteningFilter, compute_filter_gain
from .style import OTHER_COLORS, as_numpy, panel_label

RAW_COLOR = "#4D4D4D"
FILTER_COLOR = OTHER_COLORS[1]


def get_filter_label(whitening: WhiteningFilter) -> str:
    """The name of the filter in the figures."""
    return f"Olshausen & Field's whitening ({whitening.cutoff:g} cycles/pixel)"


def plot_whitening_spectra(
    spectrum: tuple[torch.Tensor, torch.Tensor],
    fit: tuple[float, float],
    band: tuple[float, float],
    whitening: WhiteningFilter,
    whitened: tuple[torch.Tensor, torch.Tensor],
    pixels: dict,
    params: dict,
) -> tuple[plt.Figure, dict]:
    """
    Four panels. A: the radially averaged power spectrum of the training images
    (`spectrum`: frequencies and power) with its power-law fit A / f^α (`fit`: A, α)
    within `band`, log–log. B: the gain of the filter `whitening`. C: the power spectrum
    of the whitened images (`whitened`: frequencies and power), log–log, flat below the
    cut-off if the images are white there. D: the histograms of the pixel values
    (`pixels`: "images" and "whitened" → values), with the fraction outside [-1, 1].
    Returns the figure and the data it shows.
    """
    freqs, power = (as_numpy(v) for v in spectrum)
    amplitude, exponent = fit
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9), dpi=params["screen_dpi"])

    # A: the spectrum and its fit
    keep = freqs > 0
    axes[0].axvspan(band[0], band[1], color="0.92", zorder=0, label="fitted band")
    axes[0].loglog(freqs[keep], power[keep], "o", ms=3, color=RAW_COLOR, label="images")
    axes[0].loglog(
        freqs[keep],
        amplitude / freqs[keep] ** exponent,
        color=OTHER_COLORS[2],
        label=rf"fit $A/f^{{\alpha}}$, $\alpha$ = {exponent:.2f}",
    )
    axes[0].set(xlabel="spatial frequency f (cycles/pixel)", ylabel="power S(f)")
    axes[0].set_title("power spectrum of the training images")
    axes[0].legend(frameon=False, fontsize=8)

    # B: the gain of the filter
    grid = np.linspace(0.0, 0.5, 501)
    gain = as_numpy(compute_filter_gain(torch.tensor(grid), whitening.cutoff))
    axes[1].plot(grid, gain, color=FILTER_COLOR)
    axes[1].axvline(whitening.cutoff, color=FILTER_COLOR, ls=":", lw=1)
    axes[1].set(xlabel="spatial frequency f (cycles/pixel)", ylabel="gain K(f)")
    axes[1].set_title(
        r"$K(f) = f\,e^{-(f/f_0)^4}$, $f_0$ = " + f"{whitening.cutoff:g} (dotted)"
    )

    # C: the spectrum of the whitened images
    w_freqs, w_power = (as_numpy(v) for v in whitened)
    axes[2].loglog(w_freqs[w_freqs > 0], w_power[w_freqs > 0], color=FILTER_COLOR)
    axes[2].axvline(whitening.cutoff, color=FILTER_COLOR, ls=":", lw=1)
    axes[2].set(xlabel="spatial frequency f (cycles/pixel)", ylabel="power")
    axes[2].set_title("power spectrum of the whitened images")

    # D: the pixel values
    bins = np.linspace(-3, 3, 121)
    colors = {"images": RAW_COLOR, "whitened": FILTER_COLOR}
    for name, values in pixels.items():
        values = as_numpy(values).ravel()
        outside = float(np.mean(np.abs(values) > 1))
        axes[3].hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            color=colors[name],
            label=f"{name}: {100 * outside:.1f}% outside [-1, 1]",
        )
    axes[3].set_yscale("log")
    axes[3].set(xlabel="pixel value", ylabel="density")
    axes[3].set_title("pixel values")
    axes[3].legend(frameon=False, fontsize=8)

    # log axes: labels on the decades only, so they do not overlap
    for ax in (axes[0], axes[2]):
        ax.xaxis.set_minor_formatter(plt.NullFormatter())

    fig.suptitle(get_filter_label(whitening), y=1.02)
    fig.tight_layout()
    for ax, letter in zip(axes, "ABCD"):
        panel_label(fig, ax, letter)
    data = {"freqs": freqs, "power": power, "gain": gain}
    return fig, data


def plot_whitening_kernel(
    whitening: WhiteningFilter, params: dict, half: int = 8
) -> tuple[plt.Figure, np.ndarray]:
    """
    The spatial kernel of the filter `whitening` within ±`half` pixels of its centre
    (left), and its profile across the centre, with the surround magnified (right).
    Returns the figure and the kernel.
    """
    kernel = as_numpy(whitening.kernel(64))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), dpi=params["screen_dpi"])

    # the kernel
    crop = kernel[32 - half : 33 + half, 32 - half : 33 + half]
    limit = np.abs(crop).max()
    axes[0].imshow(crop, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title(f"kernel, ±{half} pixels")

    # its profile, the centre off scale so the surround shows
    offsets = np.arange(-half, half + 1)
    profile = kernel[32, 32 - half : 33 + half]
    surround = profile[np.abs(offsets) > 1]
    axes[1].axhline(0, color="0.7", lw=0.8)
    axes[1].plot(offsets, profile, "o-", ms=3, color=FILTER_COLOR)
    axes[1].set_ylim(1.5 * surround.min(), 0.5 * profile[half])
    axes[1].set(xlabel="pixels from the centre", ylabel="kernel, across the centre")
    axes[1].set_title(
        f"centre {profile[half]:.2f} (off scale), surround min {surround.min():.2f}",
        fontsize=9,
    )
    fig.suptitle(get_filter_label(whitening))
    fig.tight_layout()
    return fig, kernel


def plot_whitening_examples(
    images: dict, whitening: WhiteningFilter, params: dict
) -> tuple[plt.Figure, dict]:
    """
    One column per image of `images` (name → [1, 1, H, W]): the image (top) and the
    image whitened by `whitening` (bottom), drawn in gray from −m to m, m the 99.5th
    percentile of its |values|. Returns the figure and the images drawn.
    """
    rows = ["image", get_filter_label(whitening).replace(" (", "\n(")]
    fig, axes = plt.subplots(
        2,
        len(images),
        figsize=(1.9 * len(images) + 1.2, 4.2),
        dpi=params["screen_dpi"],
        squeeze=False,
    )
    drawn = {}
    for j, (name, image) in enumerate(images.items()):
        raw = as_numpy(image[0, 0].double())
        white = as_numpy(whitening(image.double())[0, 0])
        drawn[name] = (raw, white)
        limit = np.percentile(np.abs(white), 99.5)
        axes[0, j].imshow(raw, cmap="gray", vmin=-1, vmax=1)
        axes[1, j].imshow(white, cmap="gray", vmin=-limit, vmax=limit)
        axes[0, j].set_title(name, fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for i, row in enumerate(rows):
        axes[i, 0].set_ylabel(row, fontsize=9)
    fig.tight_layout()
    return fig, drawn


def plot_training_loss(histories: dict, params: dict) -> tuple[plt.Figure, dict]:
    """
    The training loss per epoch of each model of `histories` (name → the
    `train_log.json` of its training, {"epoch": [...], "loss": [...]}), on a linear
    axis.
    Returns the figure and the final losses.
    """
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=params["screen_dpi"])
    final = {}
    for index, (name, history) in enumerate(histories.items()):
        color = [RAW_COLOR, FILTER_COLOR][index % 2]
        final[name] = history["loss"][-1]
        ax.plot(
            history["epoch"],
            history["loss"],
            color=color,
            label=f"{name}: final {final[name]:.4f}",
        )
    ax.set(xlabel="epoch", ylabel="training loss (MSE of the noise)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, final


def plot_denoised(
    image: torch.Tensor,
    noisy: dict,
    denoised: dict,
    title: str,
    params: dict,
) -> tuple[plt.Figure, dict]:
    """
    The image `image` [1, 1, H, W] (left), and at each noise scale of `noisy` and
    `denoised` (log t → [1, 1, H, W]) the noisy image x_t (top) and the denoised image
    x̂(x_t) (bottom). The image and the denoised images share one gray scale, from −m
    to m, m the 99.5th percentile of the |values| of the image; each noisy image has
    its own. Returns the figure and the images drawn.
    """
    scales = list(noisy)
    clean = as_numpy(image[0, 0].double())
    limit = np.percentile(np.abs(clean), 99.5)
    fig, axes = plt.subplots(
        2,
        len(scales) + 1,
        figsize=(1.8 * (len(scales) + 1), 4.0),
        dpi=params["screen_dpi"],
        squeeze=False,
    )
    axes[0, 0].imshow(clean, cmap="gray", vmin=-limit, vmax=limit)
    axes[0, 0].set_title(title, fontsize=9)
    drawn = {"image": clean}
    for j, scale in enumerate(scales, start=1):
        x_t = as_numpy(noisy[scale][0, 0].double())
        x_hat = as_numpy(denoised[scale][0, 0].double())
        drawn[scale] = (x_t, x_hat)
        noisy_limit = np.percentile(np.abs(x_t), 99.5)
        axes[0, j].imshow(x_t, cmap="gray", vmin=-noisy_limit, vmax=noisy_limit)
        axes[1, j].imshow(x_hat, cmap="gray", vmin=-limit, vmax=limit)
        axes[0, j].set_title(rf"$\tau$ = {round(scale, 1) + 0.0:.1f}", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[1, 0].axis("off")
    axes[0, 1].set_ylabel(r"noisy $x_t$", fontsize=9)
    axes[1, 1].set_ylabel(r"denoised $\hat x(x_t)$", fontsize=9)
    fig.tight_layout()
    return fig, drawn


def plot_recovered(
    image: torch.Tensor | None,
    whitened: torch.Tensor,
    unwhitened: torch.Tensor,
    rows: dict,
    title: str,
    params: dict,
    labels: tuple[str, str] = ("whitened", "whitened, un-whitened"),
) -> tuple[plt.Figure, dict]:
    """
    What the denoiser of whitened images recovers of an image at each noise scale. Left
    column: the image `image`, its whitened version `whitened`, and that un-whitened,
    `unwhitened` (the best the inverse filter can do), each [1, 1, H, W]. Then one
    column per noise scale of `rows` (log t → (x_t, x̂, x̂ un-whitened), each
    [1, 1, H, W]): the noisy whitened image, the denoised one, and the denoised one
    un-whitened. Whitened images
    share one gray scale, un-whitened ones another (the 99.5th percentile of |values| of
    the whitened image and of the image minus its mean); each noisy image has its own.
    `image` None (a generated image, which has no original) leaves its place empty,
    titles the column `title`, and sets the un-whitened gray scale from `unwhitened`;
    `labels` titles the whitened and un-whitened images of the left column. Returns
    the figure and the images drawn.
    """
    # the gray scales, and the left column (the image minus its mean, which the
    # whitening removes)
    scales = list(rows)
    restored = as_numpy(unwhitened[0, 0].double())
    clean = None if image is None else as_numpy(image[0, 0].double())
    clean = None if clean is None else clean - clean.mean()
    white = as_numpy(whitened[0, 0].double())
    white_limit = np.percentile(np.abs(white), 99.5)
    image_limit = np.percentile(np.abs(restored if clean is None else clean), 99.5)
    drawn = {"image": clean, "whitened": white, "unwhitened": restored}

    # three rows: noisy, denoised, denoised un-whitened
    fig, axes = plt.subplots(
        3,
        len(scales) + 1,
        figsize=(1.5 * (len(scales) + 1), 4.9),
        dpi=params["screen_dpi"],
        squeeze=False,
    )
    if clean is not None:
        axes[0, 0].imshow(clean, cmap="gray", vmin=-image_limit, vmax=image_limit)
        axes[0, 0].set_title(f"{title}\n(mean removed)", fontsize=8)
    axes[1, 0].imshow(white, cmap="gray", vmin=-white_limit, vmax=white_limit)
    axes[1, 0].set_title(
        labels[0] if clean is not None else f"{title}\n{labels[0]}", fontsize=8
    )
    axes[2, 0].imshow(restored, cmap="gray", vmin=-image_limit, vmax=image_limit)
    axes[2, 0].set_title(labels[1], fontsize=8)

    # each noise scale
    for j, scale in enumerate(scales, start=1):
        x_t, x_hat, recovered = (as_numpy(r[0, 0].double()) for r in rows[scale])
        drawn[scale] = (x_t, x_hat, recovered)
        noisy_limit = np.percentile(np.abs(x_t), 99.5)
        axes[0, j].imshow(x_t, cmap="gray", vmin=-noisy_limit, vmax=noisy_limit)
        axes[1, j].imshow(x_hat, cmap="gray", vmin=-white_limit, vmax=white_limit)
        axes[2, j].imshow(recovered, cmap="gray", vmin=-image_limit, vmax=image_limit)
        axes[0, j].set_title(rf"$\tau$ = {round(scale, 1) + 0.0:.1f}", fontsize=9)

    # no axes; the row labels
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0, 1].set_ylabel(r"noisy $x_t$", fontsize=9)
    axes[1, 1].set_ylabel(r"denoised $\hat x(x_t)$", fontsize=9)
    axes[2, 1].set_ylabel(r"$\hat x$ un-whitened", fontsize=9)
    fig.tight_layout()
    return fig, drawn


def plot_samples(
    samples: torch.Tensor, title: str, params: dict
) -> tuple[plt.Figure, np.ndarray]:
    """
    Generated images `samples` [n, 1, H, W] on a square grid (n a square number), on one
    gray scale from −m to m, m the 99.5th percentile of their |values|. Returns the
    figure and the images drawn [n, H, W].
    """
    images = as_numpy(samples[:, 0].double())
    side = int(round(len(images) ** 0.5))
    limit = np.percentile(np.abs(images), 99.5)
    fig, axes = plt.subplots(
        side, side, figsize=(1.6 * side, 1.6 * side + 0.4), dpi=params["screen_dpi"]
    )
    for ax, image in zip(axes.flat, images):
        ax.imshow(image, cmap="gray", vmin=-limit, vmax=limit)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig, images


def plot_trajectories(
    blocks: dict, params: dict, title: str = ""
) -> tuple[plt.Figure, dict]:
    """
    How several diffusion models build images, step by step: one block of rows per
    model of `blocks` (name → (finals [n, 1, H, W], {log t: x̂ [n, 1, H, W]})), one row
    per image: the model's estimate x̂(x_t) of it at each noise scale, from the
    coarsest (left) to the finest (right, titled as the final image). Each row is on
    the gray scale of its final image (its 0.5th to 99.5th percentiles). Returns the
    figure and the images drawn ({name: [n, J, H, W]}).
    """
    # the rows: one per image of each model, with an empty row between two models so
    # that the gap between them is nine times that between two rows (an empty row of
    # height s adds two gaps of hspace: 2 hspace + s = 9 hspace, in mean axes heights)
    names = list(blocks)
    scales = list(next(iter(blocks.values()))[1])
    n_rows = sum(len(blocks[name][0]) for name in names)
    hspace = 0.12
    gap = 7 * hspace * n_rows / (n_rows + 1 - 7 * hspace)
    heights, places = [], []
    for k, name in enumerate(names):
        if k > 0:
            heights.append(gap)
        for _ in range(len(blocks[name][0])):
            places.append(len(heights))
            heights.append(1.0)
    fig = plt.figure(
        figsize=(1.3 * len(scales) + 0.6, 1.3 * sum(heights) + 0.6),
        dpi=params["screen_dpi"],
    )
    grid = fig.add_gridspec(
        len(heights),
        len(scales),
        height_ratios=heights,
        hspace=hspace,
        wspace=0.08,
        left=0.04,
        right=0.995,
        bottom=0.01,
        top=0.95,
    )
    axes = np.array(
        [[fig.add_subplot(grid[i, j]) for j in range(len(scales))] for i in places]
    )

    # each image: its estimates, on the gray scale of the final image
    drawn, row = {}, 0
    for name in names:
        finals, estimates = blocks[name]
        stack = [estimates[scale] for scale in scales]
        drawn[name] = as_numpy(torch.cat(stack, dim=1).double())
        for i, images in enumerate(drawn[name]):
            final = as_numpy(finals[i, 0].double())
            low, high = np.percentile(final, [0.5, 99.5])
            for j, image in enumerate(images):
                axes[row, j].imshow(image, cmap="gray", vmin=low, vmax=high)
            if i == len(finals) // 2:  # the label at the middle of the block
                axes[row, 0].set_ylabel(name, fontsize=9)
            row += 1

    # no axes; the titles of the columns
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for j, scale in enumerate(scales):
        label = rf"$\tau$ = {round(scale, 1) + 0.0:.1f}"
        last = j == len(scales) - 1
        axes[0, j].set_title(label + ("\n(final image)" if last else ""), fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)

    # a red line across the figure in the middle of each gap between two models
    first = 0
    for name in names[:-1]:
        first += len(blocks[name][0])
        above = axes[first - 1, 0].get_position().y0
        below = axes[first, 0].get_position().y1
        middle = (above + below) / 2
        fig.add_artist(
            plt.Line2D([0.01, 0.995], [middle, middle], color="red", linewidth=1.5)
        )
    return fig, drawn


def plot_whitened_vs_standard(
    image: torch.Tensor,
    whitened: torch.Tensor,
    unwhitened: torch.Tensor,
    rows: dict,
    title: str,
    params: dict,
) -> tuple[plt.Figure, dict]:
    """
    The denoised images of the model of whitened images next to those of the standard
    model (trained on the raw images), at each noise scale of `rows` (log t →
    {"whitened": (x_t, x̂, x̂ un-whitened), "standard": (x_t, x̂)}, each [1, 1, H, W]).
    Top three rows, whitened images: the noisy whitened image, the denoised one, and
    that un-whitened. Bottom two rows, raw images: the noisy image and the denoised one.
    A red line separates the two. Left column, the reference of each row: the whitened
    image `whitened` (twice), the whitened image un-whitened `unwhitened`, and the image
    `image` (twice, mean removed). Raw images share the gray scale of the image (mean
    removed), whitened ones that of the whitened image; each noisy image has its own.
    Returns the figure and the images drawn.
    """
    # the gray scales: the image minus its mean (which the whitening removes), and the
    # whitened image
    scales = list(rows)
    clean = as_numpy(image[0, 0].double())
    clean = clean - clean.mean()
    white = as_numpy(whitened[0, 0].double())
    restored = as_numpy(unwhitened[0, 0].double())
    white_limit = np.percentile(np.abs(white), 99.5)
    image_limit = np.percentile(np.abs(clean), 99.5)
    drawn = {"image": clean, "whitened": white, "unwhitened": restored}
    on_image = {"cmap": "gray", "vmin": -image_limit, "vmax": image_limit}
    on_white = {"cmap": "gray", "vmin": -white_limit, "vmax": white_limit}

    # five rows of panels and, between the whitened and the raw images, a spacer row
    # that makes that gap three times the others; little space around the panels
    gap, n_columns = 0.05, len(scales) + 1
    fig = plt.figure(figsize=(1.3 * n_columns, 1.3 * 5.2), dpi=params["screen_dpi"])
    grid = fig.add_gridspec(
        6,
        n_columns,
        height_ratios=[1, 1, 1, gap, 1, 1],
        hspace=gap,
        wspace=gap,
        left=0.07,
        right=0.995,
        top=0.95,
        bottom=0.01,
    )
    axes = np.array(
        [
            [fig.add_subplot(grid[r, c]) for c in range(n_columns)]
            for r in (0, 1, 2, 4, 5)
        ]
    )

    # the left column: the reference of each row
    for row, (picture, scale) in enumerate(
        ((white, on_white), (white, on_white), (restored, on_image))
        + ((clean, on_image), (clean, on_image))
    ):
        axes[row, 0].imshow(picture, **scale)
    axes[0, 0].set_title(f"{title}\n(reference)", fontsize=9)

    # each noise scale: the whitened images, then the raw images
    for j, scale in enumerate(scales, start=1):
        x_t, x_hat, back = (as_numpy(r[0, 0].double()) for r in rows[scale]["whitened"])
        s_t, s_hat = (as_numpy(r[0, 0].double()) for r in rows[scale]["standard"])
        drawn[scale] = {"whitened": (x_t, x_hat, back), "standard": (s_t, s_hat)}
        for row, noisy in ((0, x_t), (3, s_t - s_t.mean())):
            limit = np.percentile(np.abs(noisy), 99.5)
            axes[row, j].imshow(noisy, cmap="gray", vmin=-limit, vmax=limit)
        axes[1, j].imshow(x_hat, **on_white)
        axes[2, j].imshow(back, **on_image)
        axes[4, j].imshow(s_hat - s_hat.mean(), **on_image)
        axes[0, j].set_title(rf"$\tau$ = {scale:.1f}", fontsize=10)

    # no axes; the label of each row, left of its reference
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    labels = (
        r"noisy $x_t$",
        r"denoised $\hat x(x_t)$",
        r"$\hat x$ un-whitened",
        r"noisy $x_t$",
        r"denoised $\hat x(x_t)$",
    )
    for row, label in enumerate(labels):
        axes[row, 0].set_ylabel(label, fontsize=9)

    # the label of each condition, vertically centred on its rows, at the far left
    for name, group in (("whitened images", (0, 2)), ("raw images", (3, 4))):
        top = axes[group[0], 0].get_position().y1
        bottom = axes[group[1], 0].get_position().y0
        fig.text(
            0.008,
            (top + bottom) / 2,
            name,
            rotation=90,
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )

    # a red line between the two conditions, in the middle of the gap
    middle = (axes[2, 0].get_position().y0 + axes[3, 0].get_position().y1) / 2
    left = axes[0, 0].get_position().x0
    fig.add_artist(plt.Line2D([left, 0.995], [middle, middle], color="red", lw=1.5))
    return fig, drawn
