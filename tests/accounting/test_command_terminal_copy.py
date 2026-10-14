"""Copied terminal blocks preserve arguments after Rich applies actual display widths."""

from __future__ import annotations

import io
import shlex
import subprocess
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.console import Console

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
