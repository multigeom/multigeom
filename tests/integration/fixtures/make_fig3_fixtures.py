"""
Regenerate the Fig. 3D-F fixtures from the trained checkpoints, by running the
notebook's own cells (setup, model loading, 6.1 compute, 6.2 plot):

    tests/integration/fixtures/fig3/fig3_def_data.npz         data of the notebook plot (image, receptive-field
                                                  parameters, leading 4 eigenvectors of G and J, V1 and V4)
    tests/integration/fixtures/fig3/fig3_def_reference.png    the notebook plot rendered from that data

Needs runs/ (checkpoints, see README), the TinyImageNet data and a GPU (or a lot of patience).
Usage, from the repository root:

    python tests/integration/fixtures/make_fig3_fixtures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))               # tests/integration (nb_helpers)
sys.path.insert(0, str(HERE.parents[2]))           # repository root

from nb_helpers import COMPUTE, FIXTURES, LOAD_MODELS, SETUP, fixture_inputs, render_plot_cell, run_cells  # noqa: E402

N_EIGVECS = 4


def main():
    ns = run_cells({}, SETUP, LOAD_MODELS, COMPUTE)
    out = {"image": ns["x"].cpu().numpy().reshape(64, 64), "n_mc": np.array(ns["N_MC"])}
    for area in ("V1", "V4"):
        state = ns["ckpts"][area]["model_state_dict"]
        for k in ("mu", "sigma", "grid_x"):
            out[f"{area}_{k}"] = state[f"readout.{k}"].numpy()
        out[f"{area}_G"] = ns["records"][area]["G_eigenvectors"][:N_EIGVECS].numpy()
        out[f"{area}_J"] = ns["records"][area]["J_eigenvectors"][:N_EIGVECS].numpy()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURES / "fig3_def_data.npz", **out)
    ckpts, records = fixture_inputs()
    render_plot_cell(ckpts, records, FIXTURES / "fig3_def_reference.png")
    print("wrote", FIXTURES)


if __name__ == "__main__":
    main()
