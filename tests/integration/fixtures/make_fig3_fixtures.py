"""
Regenerate the fixtures of the notebook plots (Fig. 3D-F and Fig. 4B-C) from the trained
checkpoints, by running the notebook's own cells (setup, model loading, 6.1 compute, plots):

    tests/integration/fixtures/fig3/fig3_def_data.npz         Fig. 3D-F data: image, receptive-field parameters of the
                                                              inception readouts, leading 4 eigenvectors of G and J
    tests/integration/fixtures/fig3/fig3_def_reference.png    the notebook's Fig. 3D-F plot rendered from that data
    tests/integration/fixtures/fig3/fig4_bc_data.npz          Fig. 4B-C data: leading eigenvectors of G and J per
                                                              (encoder, area): 4 for inception, 30 for resnet50
    tests/integration/fixtures/fig3/fig4_bc_reference.png     the notebook's Fig. 4B-C plot rendered from that data

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

from nb_helpers import (  # noqa: E402
    AREAS, COMPUTE, ENCODERS, FIXTURES, LOAD_MODELS, N_FIG3, N_FIG4_POOL, N_FIG4_REF, SETUP,
    fixture_inputs, render_plot_cell, run_cells,
)


def main():
    ns = run_cells({}, SETUP, LOAD_MODELS, COMPUTE)
    fig3 = {"image": ns["x"].cpu().numpy().reshape(64, 64), "n_mc": np.array(ns["N_MC"])}
    fig4 = {"n_mc": np.array(ns["N_MC"])}
    for area in AREAS:
        state = ns["ckpts"][("inception", area)]["model_state_dict"]
        for k in ("mu", "sigma", "grid_x"):
            fig3[f"{area}_{k}"] = state[f"readout.{k}"].numpy()
        for metric in ("G", "J"):
            fig3[f"{area}_{metric}"] = ns["records"][("inception", area)][f"{metric}_eigenvectors"][:N_FIG3].numpy()
            for enc in ENCODERS:
                n = N_FIG4_REF if enc == "inception" else N_FIG4_POOL
                fig4[f"{enc}_{area}_{metric}"] = ns["records"][(enc, area)][f"{metric}_eigenvectors"][:n].numpy()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURES / "fig3_def_data.npz", **fig3)
    np.savez_compressed(FIXTURES / "fig4_bc_data.npz", **fig4)
    ckpts, records = fixture_inputs()
    render_plot_cell(ckpts, records, FIXTURES / "fig3_def_reference.png", "fig3")
    render_plot_cell(ckpts, records, FIXTURES / "fig4_bc_reference.png", "fig4")
    print("wrote", FIXTURES)


if __name__ == "__main__":
    main()
