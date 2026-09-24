"""Tests for the CLI entry point."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

import ssgrep.cli as cli


def test_main_rejects_windows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.sys, "platform", "win32")

    with pytest.raises(SystemExit) as caught:
        cli.main()

    assert caught.value.code == 1
    assert "not supported on Windows" in capsys.readouterr().err


def test_main_delegates_to_usecli(monkeypatch: pytest.MonkeyPatch) -> None:
    usecli_main = Mock()
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr("usecli.main", usecli_main)

    assert cli.main() is None

    usecli_main.assert_called_once_with()


def test_module_main_guard_runs_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    usecli_main = Mock()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("usecli.main", usecli_main)
    module_path = Path(cli.__file__)

    namespace = runpy.run_path(str(module_path), run_name="__main__")

    assert namespace["__name__"] == "__main__"
    usecli_main.assert_called_once_with()
