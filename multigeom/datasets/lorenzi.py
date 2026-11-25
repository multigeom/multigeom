"""
The recordings the retina encoders were fitted to: the flashed-natural-image adaptation
dataset of Lorenzi et al., one file `<cell>.npz` per retinal ganglion cell (e.g.
"176-exp7"), as open-retina exports it.

Each clip shows an adapting image (the first 16 bins) then a natural image (the last
16), and the responses are spike counts in 32 bins of 25 ms from the adaptor's onset:
one trial per training clip, and every repeat of each test clip. The cell types and
other metadata are open-retina's table (`openretina.data_io.lorenzi_adaptation`), read
if `openretina` is installed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

BIN_SIZE_S = 0.025
N_ADAPT_BINS = 16


def load_lorenzi_cell(data_dir: Path, cell: str) -> SimpleNamespace:
    """
    The recording of the cell `cell` (e.g. "176-exp7") in `data_dir`: its stimulus
    frames `frames` [n_frames, H, W] and grey level `grey`; the spike counts of the
    training clips `train_counts` [n_clips, 32] and of the test clips `test_counts`
    [n_clips, n_repeats, 32]; the frame of each clip's adapting and natural image
    (`train_adapt`, `train_natural`, `test_adapt`, `test_natural`); and its `metadata`
    (cell type, experiment, reliability...; {} without openretina).
    """
    data = np.load(Path(data_dir) / f"{cell}.npz")
    return SimpleNamespace(
        cell=cell,
        frames=data["frames"],
        grey=float(data["grey_value"]),
        train_counts=data["train_psth"],
        test_counts=data["test_psth_by_trial"],
        train_adapt=data["train_adapt_idx"],
        train_natural=data["train_nat_idx"],
        test_adapt=data["test_adapt_idx"],
        test_natural=data["test_nat_idx"],
        metadata=get_lorenzi_metadata().get(cell, {}),
    )


def get_lorenzi_metadata() -> dict:
    """open-retina's metadata of every cell, {cell: row}, or {} without openretina."""
    try:
        from openretina.data_io.lorenzi_adaptation.metadata import load_cell_metadata
    except ImportError:
        return {}
    return {row["session"]: row for row in load_cell_metadata()}


def list_lorenzi_cells(data_dir: Path) -> list[dict]:
    """
    The cells recorded in `data_dir`, one row each: "cell", "experiment" and
    "cell_type" ("" if untyped or without openretina's metadata).
    """
    metadata = get_lorenzi_metadata()
    rows = []
    for path in sorted(Path(data_dir).glob("*.npz")):
        cell = path.stem
        row = metadata.get(cell, {})
        rows.append(
            {
                "cell": cell,
                "experiment": row.get("experiment", cell.split("-")[-1]),
                "cell_type": row.get("cell_type", ""),
            }
        )
    return rows
