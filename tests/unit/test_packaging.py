"""Packaging checks: one version, the configuration files travel with the package, the metadata names the extras."""
from __future__ import annotations

import re
import tomllib
from importlib import metadata
from pathlib import Path

import pytest

import multigeom
from multigeom.config import CONFIG_DIR, load_config

ROOT = Path(__file__).resolve().parents[2]


def test_the_version_is_defined_once_and_pyproject_reads_it():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "version" in pyproject["project"]["dynamic"] and "version" not in pyproject["project"]
    assert pyproject["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "multigeom.__version__"}
    assert re.fullmatch(r"\d+\.\d+\.\d+([abrc]+\d+|\.dev\d+)?", multigeom.__version__)


def test_the_installed_version_equals_the_package_version():
    try:
        installed = metadata.version("multigeom")
    except metadata.PackageNotFoundError:
        pytest.skip("multigeom is not installed (run from a source checkout)")
    assert installed == multigeom.__version__


def test_the_extras_are_the_documented_ones():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert set(extras) == {"paper", "all", "dev", "docs"}
    assert set(extras["paper"]) <= set(extras["all"])


def test_every_packaged_configuration_loads_and_is_listed_for_the_wheel():
    names = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
    assert names == ["diffusion", "estimator", "inception", "onoff", "resnet50"]
    for name in names:
        load_config(name)
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert pyproject["tool"]["setuptools"]["package-data"]["multigeom"] == ["configs/*.yaml"]


def test_the_readme_has_no_latex_that_pypi_would_show_raw():
    text = (ROOT / "README.md").read_text()
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    assert "$$" not in text and not re.search(r"\$[A-Za-z\\][^$\n]*\$", text)
