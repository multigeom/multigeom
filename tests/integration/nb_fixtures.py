"""Shared pytest fixture of the integration tests (imported by the test modules, not a conftest.py:
a second conftest.py would shadow tests/conftest.py, whose helpers the unit tests import by name)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nb_helpers import AREAS, COMPUTE, ENCODERS, LOAD_MODELS, REPO, SETUP, run_cells  # noqa: E402


def have_inputs() -> bool:
    """Checkpoints and data the notebook needs (not available on GitHub)."""
    needed = [REPO / "data/tiny-imagenet-200/train/n01644900/images/n01644900_0.JPEG"]
    for enc in ENCODERS:
        for area in AREAS:
            needed += [REPO / f"runs/encoders/{enc}/{area}/best_model.pt",
                       REPO / f"runs/diffusion/{enc}/{area}/final_model.pt"]
    return all(p.exists() for p in needed)


@pytest.fixture(scope="session")
def notebook_run():
    """
    Namespace after running the notebook's setup, model-loading and 6.1 compute cells once
    (all encoders and areas). Skips the test when the checkpoints or data are missing.
    """
    if not have_inputs():
        pytest.skip("needs runs/ checkpoints and TinyImageNet (see README)")
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        yield run_cells({}, SETUP, LOAD_MODELS, COMPUTE)
    finally:
        os.chdir(cwd)
