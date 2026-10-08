"""Helpers to run cells of tutorials/demo_inference.ipynb inside the tests."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "tutorials" / "demo_inference.ipynb"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fig3"

# A unique line of each notebook code cell used by the tests. The plot cells rely on the
# notebook's setup cell for their imports (plt, panel_rf, ...), supplied by `render_plot_cell`.
SETUP = "# import built-in packages"
LOAD_MODELS = "alphas_cumprod = scheduler.alphas_cumprod"
COMPUTE = "ckpts, records = {}, {}"
PLOT_FIG3 = "panel_rf(ax_d"
PLOT_FIG4 = "panel_eigvecs_fig4(fig, spec[0]"

AREAS = ("V1", "V4")
ENCODERS = ("inception", "resnet50")
N_FIG3 = 4          # eigenvectors of the Fig. 3 encoder stored per metric and area
N_FIG4_REF = 4      # ... of the reference encoder (inception) for Fig. 4B-C
N_FIG4_POOL = 30    # ... of the other encoders: the notebook searches their first 30 for the best match


def notebook_cell(marker: str) -> str:
    """Source of the code cell containing `marker`, without IPython magics (%%time)."""
    nb = json.loads(NOTEBOOK.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if marker in src:
            return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("%%"))
    raise KeyError(f"no code cell containing {marker!r} in {NOTEBOOK}")


def run_cells(namespace: dict, *markers: str) -> dict:
    """Execute the notebook cells containing `markers`, in order, in `namespace`."""
    for marker in markers:
        exec(compile(notebook_cell(marker), f"<notebook cell: {marker}>", "exec"), namespace)
    return namespace


def fixture_inputs() -> tuple[dict, dict]:
    """
    (ckpts, records), keyed by (encoder, area) like the notebook's compute cell, from the
    stored fixtures. ckpts hold the readout parameters of the Fig. 3 encoder (inception);
    records hold the image and the leading eigenvectors of G and J.
    """
    import numpy as np
    import torch

    d3 = np.load(FIXTURES / "fig3_def_data.npz")
    d4 = np.load(FIXTURES / "fig4_bc_data.npz")
    image = torch.from_numpy(d3["image"]).reshape(1, 1, 64, 64)
    ckpts, records = {}, {}
    for area in AREAS:
        ckpts[("inception", area)] = {"model_state_dict": {
            "readout.mu": torch.from_numpy(d3[f"{area}_mu"]),
            "readout.sigma": torch.from_numpy(d3[f"{area}_sigma"]),
            "readout.grid_x": torch.from_numpy(d3[f"{area}_grid_x"]),
        }}
        for enc in ENCODERS:
            records[(enc, area)] = {
                "image": image,
                "G_eigenvectors": torch.from_numpy(d4[f"{enc}_{area}_G"]),
                "J_eigenvectors": torch.from_numpy(d4[f"{enc}_{area}_J"]),
            }
    # Fig. 3 uses the first N_FIG3 eigenvectors; they are the same numbers as in fig3_def_data.npz
    for area in AREAS:
        for metric in ("G", "J"):
            np.testing.assert_array_equal(records[("inception", area)][f"{metric}_eigenvectors"][:N_FIG3].numpy(),
                                          d3[f"{area}_{metric}"])
    return ckpts, records


def render_plot_cell(ckpts: dict, records: dict, out_png: Path, which: str = "fig3") -> None:
    """Run the notebook's plot cell for Fig. 3D-F ("fig3") or Fig. 4B-C ("fig4") and save the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figures._style import panel_label
    from figures.fig3 import panel_eigvecs, panel_rf
    from figures.fig4 import panel_eigvecs as panel_eigvecs_fig4

    plt.close("all")
    ns = run_cells({"plt": plt, "ckpts": ckpts, "records": records, "ENC": "inception", "AREAS": list(AREAS),
                    "panel_rf": panel_rf, "panel_eigvecs": panel_eigvecs,
                    "panel_eigvecs_fig4": panel_eigvecs_fig4, "panel_label": panel_label},
                   {"fig3": PLOT_FIG3, "fig4": PLOT_FIG4}[which])
    ns["fig"].savefig(out_png, dpi=100)
    plt.close("all")


# ── comparison of rendered plots ────────────────────────────────────────────
REFERENCE_PNG = {"fig3": FIXTURES / "fig3_def_reference.png", "fig4": FIXTURES / "fig4_bc_reference.png"}
MEAN_ABS_TOL = 1e-3          # mean |difference| over all pixels, in [0, 1]; text rendering may differ slightly
BAD_PIXEL_FRACTION = 2e-3    # at most this fraction of pixels may differ by more than 0.1

# Recompute checks: (metric, number of leading eigenvectors compared, minimum |cosine| with the stored one).
# Measured, CPU against the GPU fixture (inception): J |cos| >= 0.9958 for all four eigenvectors;
# G |cos| = 0.96 (V1) and 0.98 (V4) for the leading one, but down to 0.30 for the 3rd (V4).
CHECKS = (("J", 4, 0.99), ("G", 1, 0.9))


def image_difference(a, b) -> tuple[float, float]:
    """(mean |a - b|, fraction of pixels with |a - b| > 0.1) of two images with values in [0, 1]."""
    import numpy as np

    assert a.shape == b.shape, f"image shapes differ: {a.shape} vs {b.shape}"
    d = np.abs(a.astype(float) - b.astype(float))
    return float(d.mean()), float((d > 0.1).mean())


def plot_difference(which: str, ckpts: dict, records: dict, tmp_path: Path) -> tuple[float, float]:
    """Render the notebook plot `which` from (ckpts, records) and compare it with its reference picture."""
    import matplotlib.image as mpimg

    out = tmp_path / f"{which}.png"
    render_plot_cell(ckpts, records, out, which)
    return image_difference(mpimg.imread(out), mpimg.imread(REFERENCE_PNG[which]))
