"""Helpers to run cells of tutorials/demo_inference.ipynb inside the tests."""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "tutorials" / "demo_inference.ipynb"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fig3"

# A unique line of each notebook code cell used by the tests. The plot cell relies on the
# notebook's setup cell for its imports (plt, panel_rf, panel_eigvecs, panel_label), supplied below.
SETUP = "# import built-in packages"
LOAD_MODELS = "alphas_cumprod = scheduler.alphas_cumprod"
COMPUTE = "ckpts, records = {}, {}"
PLOT = "panel_rf(ax_d"


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


def fixture_inputs(path: Path | None = None) -> tuple[dict, dict]:
    """(ckpts, records) as the notebook's plot cell expects them, from the stored fixture."""
    import numpy as np
    import torch

    data = np.load(path or FIXTURES / "fig3_def_data.npz")
    image = torch.from_numpy(data["image"]).reshape(1, 1, 64, 64)
    ckpts, records = {}, {}
    for area in ("V1", "V4"):
        ckpts[area] = {"model_state_dict": {
            "readout.mu": torch.from_numpy(data[f"{area}_mu"]),
            "readout.sigma": torch.from_numpy(data[f"{area}_sigma"]),
            "readout.grid_x": torch.from_numpy(data[f"{area}_grid_x"]),
        }}
        records[area] = {
            "image": image,
            "G_eigenvectors": torch.from_numpy(data[f"{area}_G"]),
            "J_eigenvectors": torch.from_numpy(data[f"{area}_J"]),
        }
    return ckpts, records


def render_plot_cell(ckpts: dict, records: dict, out_png: Path) -> None:
    """Run the notebook's plot cell (Fig. 3D-F) on (ckpts, records) and save the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figures._style import panel_label
    from figures.fig3 import panel_eigvecs, panel_rf

    plt.close("all")
    ns = run_cells({"plt": plt, "ckpts": ckpts, "records": records, "panel_rf": panel_rf,
                    "panel_eigvecs": panel_eigvecs, "panel_label": panel_label}, PLOT)
    ns["fig"].savefig(out_png, dpi=100)
    plt.close("all")
