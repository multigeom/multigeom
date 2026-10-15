"""
Tests for multigeom.cli: the commands are dispatched to the modules that do the work.
"""

from __future__ import annotations

import pytest

from multigeom import cli


def test_help_lists_every_command(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in ("jacobian", *cli.COMMANDS):
        assert name in out


def test_unknown_command_fails(capsys):
    assert cli.main(["nope"]) == 2
    assert "unknown command" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command", [c for c in cli.COMMANDS if c not in ("train-encoder", "responses")]
)
def test_command_calls_the_main_of_its_module(command, monkeypatch):
    import importlib

    module = importlib.import_module(cli.COMMANDS[command][0])
    seen = {}
    monkeypatch.setattr(module, "main", lambda argv=None: seen.setdefault("argv", argv))
    assert cli.main([command, "--x", "1"]) == 0
    assert seen["argv"] == ["--x", "1"]


def test_every_command_module_has_a_main():
    import ast
    import importlib.util

    for module, _ in cli.COMMANDS.values():
        path = importlib.util.find_spec(module).origin
        names = {
            n.name
            for n in ast.parse(open(path).read()).body
            if isinstance(n, ast.FunctionDef)
        }
        assert "main" in names, module


def test_jacobian_requires_an_encoder():
    with pytest.raises(SystemExit):
        cli.main(["jacobian"])
