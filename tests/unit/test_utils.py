"""Tests for src.utils: provenance, config loading, seeding helpers."""
from __future__ import annotations

import subprocess

import pytest
import torch

from conftest import REPO
from src.utils import load_config, make_generator, project_root, provenance


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def test_provenance_records_commit_config_and_seed():
    cfg = REPO / "configs" / "estimator.yaml"
    prov = provenance({"estimator": cfg, "absent": None}, seed=7, argv=["x", "--y"])
    assert prov["git_commit"] == _git_head()
    if prov["git_commit"] is not None:
        assert len(prov["git_commit"]) == 40
        assert isinstance(prov["git_dirty"], bool)
    assert prov["configs"] == {"estimator": {"path": str(cfg), "text": cfg.read_text()}}
    assert prov["seed"] == 7 and prov["argv"] == ["x", "--y"]
    assert prov["packages"]["torch"].split("+")[0] == torch.__version__.split("+")[0]


def test_project_root_contains_src():
    assert (project_root() / "src" / "utils.py").exists()


def test_load_config_nested_namespace(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a:\n  b: 3\n  c: [1, {d: 2}]\n")
    cfg = load_config(p)
    assert cfg.a.b == 3 and cfg.a.c[0] == 1 and cfg.a.c[1].d == 2


def test_make_generator_reproducible():
    a = torch.randn(5, generator=make_generator(3, "cpu"))
    b = torch.randn(5, generator=make_generator(3, "cpu"))
    assert torch.equal(a, b)
