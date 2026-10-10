"""Integration test of notebooks/0_bring_your_own_encoder.ipynb: its cells run on a CPU and give a valid J(x)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb_helpers import REPO, run_cells  # noqa: E402

NOTEBOOK0 = REPO / "notebooks" / "0_bring_your_own_encoder.ipynb"


@pytest.fixture
def at_repo_root(monkeypatch):
    monkeypatch.chdir(REPO)


def test_notebook_runs_and_returns_a_valid_fisher_metric(at_repo_root):
    ns = run_cells({}, "class MyEncoder", "report = multigeom.validate_encoder", "fisher = multigeom.compute_fisher",
                   "axes[0].imshow", notebook=NOTEBOOK0)
    report, fisher = ns["report"], ns["fisher"]
    assert report.n_neurons == 20 and not report.warnings
    assert fisher.eigenvectors.shape == (10, 4096)
    ev = fisher.eigenvalues
    assert torch.isfinite(ev).all() and (ev > 0).all() and (ev[:-1] >= ev[1:]).all()
    norms = fisher.eigenvectors.norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-4, rtol=0)


def test_a_broken_encoder_is_rejected_with_the_rule(at_repo_root):
    ns = run_cells({}, "class MyEncoder", notebook=NOTEBOOK0)
    import multigeom

    class Detached(ns["MyEncoder"]):
        def forward(self, x):
            return super().forward(x).detach()

    with pytest.raises(multigeom.EncoderContractError, match="R5"):
        multigeom.validate_encoder(Detached())
