"""Tests for multigeom.utils: provenance, config loading, seeding helpers."""
from __future__ import annotations

import subprocess

import pytest
import torch

from conftest import REPO
from multigeom.config import CONFIG_DIR, load_config, package_root, project_root
from multigeom.utils import make_generator, provenance


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def test_provenance_records_commit_config_and_seed():
    cfg = CONFIG_DIR / "estimator.yaml"
    prov = provenance({"estimator": cfg, "absent": None}, seed=7, argv=["x", "--y"])
    assert prov["git_commit"] == _git_head()
    if prov["git_commit"] is not None:
        assert len(prov["git_commit"]) == 40
        assert isinstance(prov["git_dirty"], bool)
    assert prov["configs"] == {"estimator": {"path": str(cfg), "text": cfg.read_text()}}
    assert prov["seed"] == 7 and prov["argv"] == ["x", "--y"]
    assert prov["packages"]["torch"].split("+")[0] == torch.__version__.split("+")[0]


def test_package_and_project_roots(tmp_path, monkeypatch):
    assert (package_root() / "utils.py").exists() and (package_root() / "configs" / "estimator.yaml").exists()
    monkeypatch.delenv("MULTIGEOM_REPO", raising=False)
    monkeypatch.chdir(tmp_path)
    assert project_root() == tmp_path.resolve()                                # the working folder
    monkeypatch.setenv("MULTIGEOM_REPO", str(tmp_path / "x"))
    assert project_root() == tmp_path / "x"


def test_load_config_nested_namespace(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a:\n  b: 3\n  c: [1, {d: 2}]\n")
    cfg = load_config(p)
    assert cfg.a.b == 3 and cfg.a.c[0] == 1 and cfg.a.c[1].d == 2


def test_make_generator_reproducible():
    a = torch.randn(5, generator=make_generator(3, "cpu"))
    b = torch.randn(5, generator=make_generator(3, "cpu"))
    assert torch.equal(a, b)
