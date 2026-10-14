"""Copied terminal blocks preserve arguments after Rich applies actual display widths."""

from __future__ import annotations

import io
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.console import Console

from ssgrep.cli.savings_render import emit
from ssgrep.cli.savings_text import Doc


@pytest.mark.parametrize("width", [40, 80, 120])
def test_whole_displayed_command_preserves_wide_unicode_and_shell_literals(
    tmp_path: Path, width: int
) -> None:
    value = "/workspace/" + "航空預訂路径 " * 20 + "O'Brien;$(touch injected)"
    doc = Doc(width)
    doc.command("Next:", "printf %s " + shlex.quote(value))
    output = io.StringIO()
    Console(
        file=output,
        width=width,
        force_terminal=False,
        color_system=None,
        highlight=False,
        markup=False,
    ).print(doc.text(), soft_wrap=False)
    rendered = output.getvalue()
    assert all(cell_len(line) <= width for line in rendered.splitlines())
    pasted = subprocess.run(
        ["/bin/sh", "-c", rendered],
        cwd=tmp_path,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert pasted.returncode == 0, pasted.stderr
    assert pasted.stdout == value.encode("utf-8")
    assert not pasted.stderr
    assert not (tmp_path / "injected").exists()


def test_optimized_python_keeps_command_indentation_progress_check(tmp_path: Path) -> None:
    code = """
import shlex
import subprocess
from ssgrep.cli.savings_text import Doc

try:
    Doc(30).command("", "printf %s value", indent=30)
except ValueError:
    pass
else:
    raise RuntimeError("unrenderable indentation accepted")

doc = Doc(30)
doc.command("", "printf %s " + shlex.quote("O'Brien;$(touch injected)"), indent=2)
subprocess.run(["/bin/sh", "-c", doc.text().plain], check=True)
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"O'Brien;$(touch injected)"
    assert not (tmp_path / "injected").exists()
    with pytest.raises(ValueError):
        Doc(30).command("", "printf %s value", indent=30)


@pytest.mark.parametrize("terminal_width", [12, 24, 29, 30, 160])
def test_final_emission_preserves_commands_when_terminal_and_layout_widths_differ(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    terminal_width: int,
) -> None:
    monkeypatch.setenv("COLUMNS", str(terminal_width))
    value = "/workspace/" + "航空預訂路径 " * 20 + "O'Brien;$(touch injected)"
    doc = Doc(terminal_width)
    doc.command("Next:", "printf %s " + shlex.quote(value))
    emit(doc.text())
    rendered = capsys.readouterr().out
    pasted = subprocess.run(
        ["/bin/sh", "-c", rendered],
        cwd=tmp_path,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert pasted.returncode == 0, pasted.stderr
    assert pasted.stdout == value.encode("utf-8")
    assert not pasted.stderr
    assert not (tmp_path / "injected").exists()
