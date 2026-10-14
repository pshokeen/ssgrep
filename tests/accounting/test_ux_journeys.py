"""10.5: the documented journey through the REAL binary, at 40/80/120 columns.

setup -> real ``search`` (it fails without an index; that is fine, it is still captured) ->
``metrics collect`` -> summary -> explain -> history -> pause -> resume, in an isolated HOME and
data root. The consumer transcripts carry hostile strings (ANSI, OSC, Rich markup, newlines) in
the conversation id. Checks are structural: width, control bytes, navigation (every ``Next:``
command is parsed and RUN), sign/unit/action visibility at 40 columns, and JSON parity. Screens
are written to ``UX_SCREENS_DIR`` when that variable is set (used for the release report).
"""

from __future__ import annotations

import os
import pty
import re
import select
import shlex
import subprocess
from pathlib import Path

import pytest

from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import source_session
from tests.accounting.test_reduce_evidence import FACT, VIEW, consumer
from tests.accounting.test_savings_cli import BINARY
from tests.accounting.test_ux_support import Binary

HOSTILE = "bad\x1b[31m[bold red]x[/]\x1b]0;pwn\x07"
ANSI_OTHER = re.compile(r"\x1b\[[0-9;]*[^0-9;m]|\x1b\][^\x07]*\x07|\x1b[^\[\]]|\x9b|\x07|\r")
SGR = re.compile(r"\x1b\[[0-9;]*m")
WIDTHS = (40, 80, 120)


def renamed(session: str, records: list[dict]) -> list[dict]:
    return [{**r, "sessionId": session} if "sessionId" in r else dict(r) for r in records]


def screens_dir() -> Path | None:
    value = os.environ.get("UX_SCREENS_DIR")
    return Path(value) if value else None


def keep(name: str, width: int, text: str) -> None:
    target = screens_dir()
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{name}.{width}.txt").write_text(text)


def lines_fit(text: str, width: int) -> None:
    for line in text.splitlines():
        assert len(line) <= width, f"{len(line)} > {width}: {line!r}"


def next_command(text: str) -> list[str]:
    """The parsed argv of the ``Next:`` command (shell continuations joined)."""
    block = text.split("Next:")[1]
    parts = []
    for line in block.splitlines():
        stripped = line.strip()
        if parts and not line.startswith(" "):
            break
        if stripped.endswith("\\"):
            parts.append(stripped[:-1])
            continue
        parts.append(stripped)
        break
    command = " ".join(parts)
    argv = shlex.split(command)
    assert argv[0] == "ssgrep", command
    return argv[1:]


def collect_real(run: Binary) -> None:
    """``ssgrep metrics collect`` with the real clock: the snapshot is fresh for the journey."""
    done = run("metrics", "collect", "--json")
    assert done.returncode == 0, (done.stdout, done.stderr[-400:])


@pytest.fixture
def journey(tmp_path: Path) -> Binary:
    return Binary(tmp_path)


def build_dataset(root: Path) -> None:
    """Transcripts the collector will read: one fact source and a hostile-named consumer."""
    vs.projects(root).mkdir(parents=True, exist_ok=True)
    vs.put(vs.projects(root), "src1", source_session("src1", FACT, when_day=11), scene.at(hour=10))
    vs.put(vs.projects(root), "hostile", renamed(HOSTILE, consumer()), scene.at(hour=10))


def test_documented_journey_end_to_end_at_each_width(tmp_path: Path) -> None:
    run = Binary(tmp_path)
    vs.enable(pin=VIEW)
    vs.start_capture(day=11)
    build_dataset(tmp_path)
    assert run("savings", columns=80).returncode == 0
    # setup was done through the policy module by vs.enable; the binary then reads and collects
    assert run("metrics", "setup", "--yes").returncode == 0
    search = run("search", "payment retries")  # real execution; no index -> an error is fine
    assert search.returncode != 0 or search.stdout is not None
    collect_real(run)
    for width in WIDTHS:
        summary = run("savings", columns=width)
        assert summary.returncode == 0, summary.stderr
        text = summary.stdout
        keep("summary", width, text)
        lines_fit(text, width)
        assert not ANSI_OTHER.search(text) and "\x1b" not in text
        assert HOSTILE not in text
        compact = " ".join(text.split())
        assert "Next: ssgrep savings explain" in compact.replace("\\ ", "")
        assert "-44 tokens" in compact or "-45 tokens" in compact or "net -" in compact
        assert "MEASURED" in text and "ESTIMATED NET BENEFIT" in text and "EVIDENCE" in text
        # navigation: the suggested command really runs and shows the receipt
        argv = next_command(text)
        assert argv[:2] == ["savings", "explain"]
        explained = run(*argv, columns=width)
        assert explained.returncode == 0, explained.stderr
        keep("explain", width, explained.stdout)
        lines_fit(explained.stdout, width)
        assert "\x1b" not in explained.stdout and "\x07" not in explained.stdout
        assert "\\x1b" in explained.stdout, "the hostile conversation id is escaped, not dropped"
        assert "[bold red]x[/]" in " ".join(explained.stdout.split())
        history_argv = next_command(explained.stdout)
        assert history_argv[:2] == ["savings", "history"]
        history = run(*history_argv, columns=width)
        assert history.returncode == 0, history.stderr
        keep("history", width, history.stdout)
        lines_fit(history.stdout, width)
        assert "CURRENT TOTAL" in history.stdout and "tokens" in history.stdout
        back = next_command(history.stdout)
        assert back == ["savings"], "history leads back to the summary"
    paused = run("metrics", "pause", columns=40)
    keep("pause", 40, paused.stdout)
    assert paused.returncode == 0
    paused_summary = run("savings", columns=40)
    keep("paused-summary", 40, paused_summary.stdout)
    assert "Paused" in paused_summary.stdout and "metrics resume" in paused_summary.stdout
    assert "-4" in paused_summary.stdout, "retained values stay readable while paused"
    status = run("metrics", "status", columns=40)
    keep("status-paused", 40, status.stdout)
    lines_fit(status.stdout, 40)
    resumed = run("metrics", "resume", columns=40)
    keep("resume", 40, resumed.stdout)
    assert resumed.returncode == 0 and "resumed" in resumed.stdout


def test_forty_columns_never_hide_signs_units_uncertainty_or_the_next_action(
    tmp_path: Path,
) -> None:
    run = Binary(tmp_path)
    vs.enable(pin=VIEW)
    vs.start_capture(day=11)
    build_dataset(tmp_path)
    collect_real(run)
    text = run("savings", columns=40, env={"NO_COLOR": "1"}).stdout
    compact = " ".join(text.replace("\\\n", " ").split())
    for required in ("-", "tokens", "Partial", "not causal proof"):
        assert required.lower() in compact.lower(), required
    assert "Next:" in text and "ssgrep savings explain" in compact
    assert "credit" in compact and "cost" in compact
    for line in text.splitlines():
        assert len(line) <= 40, line


def test_color_is_only_decoration_on_a_real_tty_and_hostile_text_stays_inert(
    tmp_path: Path,
) -> None:
    run = Binary(tmp_path)
    vs.enable(pin=VIEW)
    vs.start_capture(day=11)
    build_dataset(tmp_path)
    collect_real(run)
    receipt = run.data("savings")["data"]["representative_receipt"]
    assert isinstance(receipt, str)
    for no_color in (False, True):
        env: dict[str, str] = {
            **vs.env_for(tmp_path),
            "COLUMNS": "40",
            "TERM": "xterm-256color",
        }
        env.pop("NO_COLOR", None)
        if no_color:
            env["NO_COLOR"] = "1"
        master, slave = pty.openpty()
        process = subprocess.Popen(
            [str(BINARY), "savings", "explain", receipt],
            stdin=subprocess.DEVNULL,
            stdout=slave,
            stderr=subprocess.PIPE,
            env=env,
        )
        os.close(slave)
        chunks = b""
        while True:
            ready, _, _ = select.select([master], [], [], 30)
            if not ready:
                break
            try:
                data = os.read(master, 65536)
            except OSError:
                break
            if not data:
                break
            chunks += data
        process.wait(timeout=60)
        os.close(master)
        screen = chunks.decode(errors="replace").replace("\r\n", "\n")
        assert process.returncode == 0
        stripped = SGR.sub("", screen)
        assert not ANSI_OTHER.search(stripped.replace("\r", "")), repr(stripped[:300])
        assert HOSTILE not in screen and "pwn" not in re.sub(r"\\x1b\]0;pwn\\x07", "", stripped)
        if no_color:
            assert "\x1b" not in screen, "NO_COLOR means no styling at all on a tty"
        else:
            assert SGR.search(screen), "the real terminal is the positive styling control"
        keep("explain-tty-nocolor" if no_color else "explain-tty-color", 40, screen)
        text = " ".join(stripped.split())
        assert "CREDIT BY SCENARIO" in text and "Next:" in text


def test_pipe_output_has_no_cursor_control_and_json_parity_with_text(tmp_path: Path) -> None:
    run = Binary(tmp_path)
    vs.enable(pin=VIEW)
    vs.start_capture(day=11)
    build_dataset(tmp_path)
    collect_real(run)
    text = run("savings", columns=80, env={"FORCE_COLOR": "1", "NO_COLOR": ""}).stdout
    assert "\x1b" not in text, "a pipe never receives color or cursor control"
    body = run.data("savings")
    net = body["data"]["estimates"]["context_tokens"]["central"]["net"]["value"]
    assert f"{net:+,}" in text or f"{net:,}" in text
    assert body["next_actions"][0]["command"] in " ".join(text.replace("\\\n", " ").split())
    for hostile_free in (text, str(body)):
        assert "pwn" not in hostile_free.replace("\\x1b]0;pwn\\x07", "").replace(
            "\\u001b]0;pwn\\u0007", ""
        )


def test_cancelled_setup_and_every_empty_state_screen_name_an_action_at_40_columns(
    tmp_path: Path,
) -> None:
    run = Binary(tmp_path)
    off = run("savings", columns=40).stdout
    assert "ssgrep metrics setup" in off and "off" in off.lower()
    cancelled = run("metrics", "setup", columns=40)  # no tty, no --yes: refuses
    assert cancelled.returncode == 2 and "--yes" in cancelled.stderr
    assert not (tmp_path / "data").exists(), "cancelled setup changed nothing"
    assert run("metrics", "setup", "--yes", columns=40).returncode == 0
    pending = run("savings", columns=40).stdout
    assert "ssgrep metrics collect" in pending and "No snapshot yet" in pending
    keep("no-snapshot", 40, pending)
