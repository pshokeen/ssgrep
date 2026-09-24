"""Tests for the package-level console entry point."""

from __future__ import annotations

import runpy
from pathlib import Path
from unittest.mock import Mock

import pytest

import ssgrep


def test_main_delegates_to_usecli_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # ssgrep/__init__.py binds `from usecli import run` at import time, so the
    # module-level reference must be patched rather than usecli.run itself.
    usecli_run = Mock()
    monkeypatch.setattr(ssgrep, "run", usecli_run)

    assert ssgrep.main() is None

    usecli_run.assert_called_once_with()


def test_module_main_guard_runs_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    usecli_run = Mock()
    monkeypatch.setattr("usecli.run", usecli_run)
    module_path = Path(ssgrep.__file__)

    namespace = runpy.run_path(str(module_path), run_name="__main__")

    assert namespace["__name__"] == "__main__"
    usecli_run.assert_called_once_with()
