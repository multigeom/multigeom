"""
Table 1 (tab:encoder_perf): predictive performance of every encoder.

For each encoder and area, from the Stage-1 checkpoint
runs/encoders/<encoder>/<area>/best_model.pt:

    r̄     = mean over the 50 neurons of the Pearson test correlation r_n
            between predicted and trial-averaged responses (Papale test set);
    r̄_nc  = mean over neurons of r_n / ρ_n, with ρ_n the split-half oracle
            reliability of neuron n (noise-corrected correlation).

The best value of each column is set in bold, as in the paper. Writes
table1.md (Markdown) and table1.tex (booktabs tabular) to --out-dir.

Usage:
    python figures/table1.py --runs-dir runs --out-dir figures/out
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from figures._style import AREAS, ENCODER_LABELS_LONG, ENCODERS, as_numpy, encoder_checkpoint

SCRIPT = "table1"


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="figures/out")
    return p.parse_args(argv)


def encoder_scores(ckpt: dict) -> tuple[float, float]:
    """(mean r, mean r_nc) from a Stage-1 checkpoint."""
    r = as_numpy(ckpt["per_neuron_test_corr"]).astype(float)
    if "per_neuron_test_corr_noise_corrected" in ckpt:
        r_nc = as_numpy(ckpt["per_neuron_test_corr_noise_corrected"]).astype(float)
    else:
        r_nc = r / as_numpy(ckpt["oracle"]).astype(float)
    return float(r.mean()), float(r_nc.mean())


def collect(runs_dir: Path) -> dict[tuple[str, str], tuple[float, float]]:
    scores = {}
    for enc in ENCODERS:
        for area in AREAS:
            ckpt = encoder_checkpoint(runs_dir, enc, area, SCRIPT)
            if ckpt is not None:
                scores[(enc, area)] = encoder_scores(ckpt)
    return scores


def _cells(scores, bold) -> dict[str, list[tuple[str, bool]]]:
    """Per encoder: four (value, is_best) cells in the order V1 r, V1 r_nc, V4 r, V4 r_nc."""
    columns = [(area, m) for area in AREAS for m in (0, 1)]
    best = {}
    for area, m in columns:
        vals = [scores[(e, area)][m] for e in ENCODERS if (e, area) in scores]
        best[(area, m)] = max(vals) if vals else None
    rows = {}
    for enc in ENCODERS:
        if not any((enc, a) in scores for a in AREAS):
            continue
        cells = []
        for area, m in columns:
            if (enc, area) in scores:
                v = scores[(enc, area)][m]
                cells.append((f"{v:.2f}", bold and np.isclose(v, best[(area, m)])))
            else:
                cells.append(("--", False))
        rows[enc] = cells
    return rows


def to_markdown(scores) -> str:
    lines = ["| Encoder | V1 r | V1 r_nc | V4 r | V4 r_nc |",
             "|---|---|---|---|---|"]
    for enc, cells in _cells(scores, bold=True).items():
        txt = [f"**{v}**" if b else v for v, b in cells]
        lines.append(f"| {ENCODER_LABELS_LONG[enc]} | " + " | ".join(txt) + " |")
    return "\n".join(lines) + "\n"


def to_latex(scores) -> str:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{V1} & \multicolumn{2}{c}{V4} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Encoder & $\bar r$ & $\bar r_{\text{nc}}$ & $\bar r$ & $\bar r_{\text{nc}}$ \\",
        r"\midrule",
    ]
    for enc, cells in _cells(scores, bold=True).items():
        txt = [(rf"$\mathbf{{{v}}}$" if b else f"${v}$") if v != "--" else "--"
               for v, b in cells]
        lines.append(f"{ENCODER_LABELS_LONG[enc]:<10} & " + " & ".join(txt) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    scores = collect(Path(args.runs_dir))
    if not scores:
        print(f"[{SCRIPT}] no Stage-1 checkpoints found under {args.runs_dir}/encoders; "
              "nothing to write.")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    md, tex = to_markdown(scores), to_latex(scores)
    paths = [out_dir / "table1.md", out_dir / "table1.tex"]
    paths[0].write_text(md)
    paths[1].write_text(tex)
    print(md)
    for p in paths:
        print(f"saved {p}")
    return paths


if __name__ == "__main__":
    main()
