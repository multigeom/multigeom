"""
Figure 4: robustness of the geometry across encoding-model architectures.

  A    predictive performance of the Inception, ResNet-50 and OnOff encoders in
       V1 and V4: box plot of the per-neuron test Pearson correlations (box =
       quartiles, line = median), with the mean ± s.e.m. across neurons
       overlaid as a black point. Source: Stage-1 checkpoints.
  B    leading eigenvectors of G(x) for the example image, one row per model,
       V1 block above the V4 block. Source: Stage-4 per-image files.
  C    same as B for J(x).
  D-E  subspace alignment 𝒜_k = (1/k)‖V_Aᵀ V_B‖_F² of the top-k eigenvectors
       of G (D) and J (E), mean ± s.e.m. across images. Solid: models fitted
       to the same area (Inception vs ResNet-50); dashed: the same model fitted
       to V1 and V4. Source: runs/metrics/cross_encoder_metrics.json (Stage 5).

Display matching in B-C: the Inception row shows its own leading eigenvectors
in order. For ResNet-50 and OnOff, each displayed eigenvector is the one,
among that model's first --match-pool eigenvectors, with the largest |cosine|
to the Inception eigenvector of the same column (each used once), sign-flipped
to make the cosine positive. This only changes what is displayed;
--match-pool equal to --n-eigvecs restricts it to reordering the leading
eigenvectors. Eigenvector colour scales as in fig3.py.

Usage:
    python -m multigeom.viz.fig4 --runs-dir runs --out-dir figures/out --img-idx 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


from .style import (
    AREAS, ENCODER_COLORS, ENCODER_LABELS, ENCODERS, METRIC_TITLES, as_numpy, blank,
    cross_encoder_metrics, encoder_checkpoint, match_to_reference, panel_label, per_image,
    plot_comparisons, plt, save_figure, show_eigenvector, skip,
)

SCRIPT = "fig4"
REFERENCE = "inception"


def parse_args(argv: list[str] | None = None):
    """Command-line arguments of `python -m multigeom.viz.fig4`."""
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="figures/out")
    p.add_argument("--img-idx", type=int, default=0, help="example image (Stage-4 index)")
    p.add_argument("--n-eigvecs", type=int, default=3)
    p.add_argument("--match-pool", type=int, default=30,
                   help="eigenvectors of ResNet-50/OnOff searched for the best match "
                        "to each displayed Inception eigenvector")
    return p.parse_args(argv)


def panel_performance(ax, ckpts: dict) -> None:
    """Box plot of the per-neuron test correlations of each encoder and area (Fig. 4A)."""
    width, gap = 0.24, 0.04
    handles = []
    for ei, enc in enumerate(ENCODERS):
        for gi, area in enumerate(AREAS):
            ckpt = ckpts.get((enc, area))
            if ckpt is None or "per_neuron_test_corr" not in ckpt:
                if ckpt is not None:
                    skip(SCRIPT, f"{enc}/{area} performance: checkpoint has no per_neuron_test_corr "
                                 "(run Stage 1 with --evaluate-only)")
                continue
            r = as_numpy(ckpt["per_neuron_test_corr"]).astype(float)
            pos = gi + (ei - 1) * (width + gap)
            bp = ax.boxplot([r], positions=[pos], widths=width, patch_artist=True,
                            showfliers=True, flierprops=dict(marker="o", markersize=2.5,
                                                             markerfacecolor="0.6",
                                                             markeredgecolor="none"))
            bp["boxes"][0].set_facecolor(ENCODER_COLORS[enc])
            bp["boxes"][0].set_edgecolor("black")
            bp["boxes"][0].set_linewidth(0.7)
            bp["medians"][0].set_color("black")
            sem = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else 0.0
            ax.errorbar(pos, r.mean(), yerr=sem, fmt="o", color="black", markersize=3,
                        capsize=0, lw=1.0, zorder=5)
        if any((enc, a) in ckpts for a in AREAS):
            handles.append(enc)
    ax.axhline(0, color="0.6", ls=":", lw=0.7)
    ax.set_xticks(range(len(AREAS)))
    ax.set_xticklabels(AREAS)
    ax.set_xlim(-0.6, len(AREAS) - 0.4)
    ax.set_ylim(-0.15, 1.05)
    ax.set_ylabel("Test Pearson $r$")
    for i, enc in enumerate(handles):
        ax.text(0.98, 0.99 - 0.07 * i, ENCODER_LABELS[enc], transform=ax.transAxes,
                ha="right", va="top", color=ENCODER_COLORS[enc], fontweight="bold",
                fontsize=10)


def panel_eigvecs(fig, spec, records: dict, metric: str, n_show: int, pool: int):
    """V1 block above V4 block; one row per encoder; returns the title axes."""
    n_enc = len(ENCODERS)
    heights = [0.25] + ([0.22] + [1.0] * n_enc) * len(AREAS)
    inner = spec.subgridspec(len(heights), n_show + 2, height_ratios=heights,
                             width_ratios=[0.35] + [1.0] * n_show + [0.3],
                             hspace=0.08, wspace=0.04)
    title_ax = fig.add_subplot(inner[0, 1:-1])
    title_ax.axis("off")
    title_ax.text(0.5, 0.0, METRIC_TITLES[metric], ha="center", va="bottom", fontsize=13,
                  transform=title_ax.transAxes)
    row = 1
    for area in AREAS:
        head = fig.add_subplot(inner[row, 1:-1])
        head.axis("off")
        head.text(0.5, 0.0, area, ha="center", va="bottom", fontsize=11,
                  transform=head.transAxes)
        row += 1
        ref = records.get((REFERENCE, area))
        ref_vecs = None if ref is None else ref.get(f"{metric}_eigenvectors")
        for enc in ENCODERS:
            rec = records.get((enc, area))
            vecs = None if rec is None else rec.get(f"{metric}_eigenvectors")
            if vecs is not None and enc != REFERENCE and ref_vecs is not None:
                vecs = match_to_reference(ref_vecs, vecs, n_show, pool)
            lab = fig.add_subplot(inner[row, 0])
            lab.axis("off")
            lab.text(1.0, 0.5, ENCODER_LABELS[enc], rotation=90, ha="right", va="center",
                     fontsize=10, transform=lab.transAxes)
            for k in range(n_show):
                ax = fig.add_subplot(inner[row, 1 + k])
                if vecs is None or k >= len(vecs):
                    blank(ax)
                else:
                    show_eigenvector(ax, vecs[k])
            dots = fig.add_subplot(inner[row, -1])
            dots.axis("off")
            dots.text(0.5, 0.5, "...", ha="center", va="center", fontsize=12,
                      transform=dots.transAxes)
            row += 1
    return title_ax


def panel_alignment(ax, metrics, metric: str) -> None:
    """Subspace alignment against the number of eigenvectors, for one metric (Fig. 4D-E)."""
    n = plot_comparisons(ax, metrics, metric, "alignment", SCRIPT)
    if n == 0:
        blank(ax)
        return
    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.set_xlabel("Number of eigenvectors")
    ax.set_ylabel("Subspace alignment")
    ax.set_title(METRIC_TITLES[metric], fontsize=12)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def main(argv: list[str] | None = None) -> list[Path]:
    """Draw Figure 4 from the pipeline outputs and write it (command-line entry point)."""
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    ckpts, records = {}, {}
    for enc in ENCODERS:
        for area in AREAS:
            ckpt = encoder_checkpoint(runs_dir, enc, area, SCRIPT)
            if ckpt is not None:
                ckpts[(enc, area)] = ckpt
            rec = per_image(runs_dir, enc, area, args.img_idx, SCRIPT)
            if rec is not None:
                records[(enc, area)] = rec
    metrics = cross_encoder_metrics(runs_dir, SCRIPT)

    fig = plt.figure(figsize=(17, 8.5))
    outer = fig.add_gridspec(1, 4, width_ratios=[0.8, 1.6, 1.6, 1.25], wspace=0.22)

    a_spec = outer[0].subgridspec(3, 1, height_ratios=[0.6, 1.0, 0.6])
    ax_a = fig.add_subplot(a_spec[1])
    panel_performance(ax_a, ckpts)

    ax_b = panel_eigvecs(fig, outer[1], records, "G", args.n_eigvecs, args.match_pool)
    ax_c = panel_eigvecs(fig, outer[2], records, "J", args.n_eigvecs, args.match_pool)

    de_spec = outer[3].subgridspec(5, 1, height_ratios=[0.25, 1.0, 0.35, 1.0, 0.25])
    ax_d = fig.add_subplot(de_spec[1])
    ax_e = fig.add_subplot(de_spec[3])
    panel_alignment(ax_d, metrics, "G")
    panel_alignment(ax_e, metrics, "J")

    for letter, ax in zip("ABCDE", (ax_a, ax_b, ax_c, ax_d, ax_e)):
        panel_label(fig, ax, letter)
    return save_figure(fig, Path(args.out_dir), f"fig4_img{args.img_idx:04d}")


if __name__ == "__main__":
    main()
