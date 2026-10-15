"""Real subprocess runs of the installed ``ssgrep`` binary in an isolated HOME/data dir."""

from __future__ import annotations

import json
import os
import pty
import select
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from ssgrep.accounting import policy
from ssgrep.accounting.ledger_contracts import DestinationPolicy, ProfilePolicy
from ssgrep.accounting.service_types import Scope
from ssgrep.cli import metrics_render, savings_state_render
from ssgrep.cli.accounting_cli import stdin_is_tty
from tests.accounting.snapshot_builders import RECEIPT_ID, write_receipt, write_snapshot
from tests.accounting.test_savings_render import flat
from tests.accounting.verify_scenes import env_for

BINARY = Path(sys.executable).parent / "ssgrep"


class Run:
    def __init__(self, home: Path, data: Path) -> None:
        self.home, self.data = home, data

    def __call__(
        self, *args: str, columns: int = 80, stdin: str | None = "", env: dict | None = None
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            **os.environ,
            "HOME": str(self.home),
            "SSGREP_DATA_DIR": str(self.data),
            "COLUMNS": str(columns),
            "NO_COLOR": "1",
            **(env or {}),
        }
        return subprocess.run(
            [str(BINARY), *args],
            input=stdin,
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
            check=False,
        )

    def files(self) -> list[str]:
        if not self.data.exists():
            return []
        return sorted(str(p.relative_to(self.data)) for p in self.data.rglob("*"))


@pytest.fixture
def run(tmp_path: Path) -> Run:
    home = tmp_path / "home"
    home.mkdir()
    return Run(home, tmp_path / "data")


def document(result: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_summary_json_envelope_for_hand_built_snapshot(run: Run) -> None:
    write_snapshot(run.data)
    body = document(run("savings", "--json"))["data"]
    # the fixed 2026 timestamp is far in the past, so this is a stale last-good snapshot
    assert body["state"] == "stale"
    inner = body["data"]
    assert inner["freshness"]["stale"] is True and inner["freshness"]["stale_after_seconds"] == 180
    net = inner["estimates"]["context_tokens"]["conservative"]["net"]
    assert net == {
        "value": -12000,
        "unit": "tokens",
        "basis": "estimated",
        "method": "chars_div4_v1",
        "reason": None,
    }
    assert inner["estimates_are_causal"] is False
    assert inner["measured"]["searches"]["value"] == 100
    assert inner["coverage"]["attributed_lookups"] == 91
    assert body["provenance"]["snapshot_id"] == "a" * 16
    assert body["next_actions"][0]["command"] == "ssgrep metrics collect"
    assert "\x1b" not in run("savings", "--json").stdout
    assert "." not in json.dumps(inner["estimates"]).replace("chars_div4_v1", "")


def test_setup_without_tty_and_without_yes_exits_2_and_changes_nothing(run: Run) -> None:
    result = run("metrics", "setup", stdin="")
    assert result.returncode == 2 and "--yes" in result.stderr
    assert run.files() == []
    as_json = run("metrics", "setup", "--json")
    assert as_json.returncode == 2 and document(as_json)["condition"] == "confirmation_required"
    assert run.files() == []


def test_setup_rejects_invalid_policy_without_side_effects(run: Run, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"profile_id": "acme", "roots": ["relative/path"]}')
    result = run("metrics", "setup", "--yes", "--policy", str(bad), "--profile", "acme", "--json")
    assert result.returncode == 2 and document(result)["condition"] == "invalid_policy"
    garbled = tmp_path / "garbled.json"
    garbled.write_text("{nope")
    assert run("metrics", "setup", "--yes", "--policy", str(garbled)).returncode == 2
    assert (
        run("metrics", "setup", "--yes", "--policy", str(tmp_path / "absent.json")).returncode == 2
    )
    assert run.files() == []


def test_cancelled_interactive_setup_leaves_no_files(run: Run) -> None:
    master, slave = pty.openpty()
    environment = {
        **os.environ,
        "HOME": str(run.home),
        "SSGREP_DATA_DIR": str(run.data),
        "NO_COLOR": "1",
    }
    process = subprocess.Popen(
        [str(BINARY), "metrics", "setup"],
        stdin=slave,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
    )
    os.write(master, b"n\n")
    stdout, stderr = process.communicate(timeout=120)
    os.close(master)
    os.close(slave)
    assert process.returncode == 0
    assert "approved roots".upper() in stdout.upper() and "No export is enabled" in stdout
    assert "Cancelled" in stderr
    assert run.files() == []


def test_first_use_journey_end_to_end(run: Run) -> None:
    assert "metrics setup" in run("savings").stdout
    assert run("metrics", "setup", stdin="").returncode == 2
    assert run.files() == []
    confirmed = run("metrics", "setup", "--yes")
    assert confirmed.returncode == 0 and "Export remains off" in confirmed.stdout
    policy_file = run.data / "accounting" / "policy.json"
    assert policy_file.stat().st_mode & 0o777 == 0o600
    stored = json.loads(policy_file.read_text())["profiles"]["personal-local"]
    assert stored["collection_enabled"] is True and stored["enabled_exports"] == []
    pending = run("savings")
    assert "ssgrep metrics collect" in pending.stdout
    assert "tokens" not in pending.stdout
    paused = run("metrics", "pause")
    assert paused.returncode == 0 and "resume" in paused.stdout
    assert "PAUSED" in run("savings").stdout and "ssgrep metrics resume" in run("savings").stdout
    status = document(run("metrics", "status", "--json"))["data"]
    assert status["state"] == "paused" and status["data"]["paused_intervals"][0]["until"] is None
    resumed = run("metrics", "resume", "--json")
    assert document(resumed)["data"]["state"] == "resumed"
    after = json.loads(policy_file.read_text())["profiles"]["personal-local"]
    assert after["paused"] is False and after["enabled_exports"] == []
    assert document(run("metrics", "status", "--json"))["data"]["state"] == "no_snapshot"
    assert not (run.data / "lancedb").exists(), "no index was created by any accounting command"


def test_profile_is_preserved_in_real_next_actions(run: Run, tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    policy_path = tmp_path / "acme.json"
    policy_path.write_text(json.dumps({"profile_id": "acme", "roots": [str(root)]}))
    ok = run(
        "metrics", "setup", "--yes", "--policy", str(policy_path), "--profile", "acme", "--json"
    )
    assert ok.returncode == 0, ok.stderr
    assert (
        document(ok)["data"]["next_actions"][0]["command"]
        == "ssgrep metrics collect --profile acme"
    )
    summary = document(run("savings", "--profile", "acme", "--json"))["data"]
    assert summary["state"] == "no_snapshot"
    assert summary["next_actions"][0]["command"] == "ssgrep metrics collect --profile acme"
    personal = document(run("savings", "--json"))["data"]
    assert personal["state"] == "disabled", "an organization profile is never inferred or shared"


def test_hostile_receipt_text_through_the_real_binary_has_no_control_bytes(run: Run) -> None:
    write_receipt(run.data, root="/w/\x1b[2J[bold red]x[/]\x07", candidate_type="\x1b]0;t\x07")
    result = run(
        "savings", "explain", RECEIPT_ID, columns=40, env={"FORCE_COLOR": "1", "NO_COLOR": ""}
    )
    assert result.returncode == 0
    assert "\x1b" not in result.stdout and "\x07" not in result.stdout
    assert "\\x1b[2J" in " ".join(result.stdout.split())


# ------------------------------------------------------------ consent prompts and policy


def environment(run: Run) -> dict[str, str]:
    return {**os.environ, "HOME": str(run.home), "SSGREP_DATA_DIR": str(run.data), "NO_COLOR": "1"}


@contextmanager
def prompt(run: Run, *arguments: str):
    """Wait for an actual consent prompt before sending input or changing its policy."""
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [str(BINARY), *arguments],
        stdin=slave,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment(run),
    )
    os.close(slave)
    preview = bytearray()
    try:
        assert process.stdout is not None
        deadline = time.monotonic() + 30
        while b"[y/N] " not in preview:
            remaining = deadline - time.monotonic()
            assert remaining > 0, bytes(preview)
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            assert ready, bytes(preview)
            chunk = os.read(process.stdout.fileno(), 65536)
            assert chunk, (process.poll(), bytes(preview))
            preview.extend(chunk)
        yield process, master
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
        os.close(master)


def export_arguments() -> tuple[str, ...]:
    from tests.accounting.test_export_consent import DEST

    return ("metrics", "export", "--enable", "--destination", DEST, "--signal", "metrics")


@pytest.mark.parametrize("command", ["setup", "export"])
@pytest.mark.parametrize("answer", [b"yes\n", b"no\n", b"\x04"], ids=["approve", "refuse", "eof"])
def test_interactive_consent_changes_only_the_approved_state(run: Run, command: str, answer: bytes):
    from tests.accounting.test_export_consent import DEST, configure, enabled_pairs

    if command == "export":
        configure()
    before = policy.policy_path().read_bytes() if policy.policy_path().exists() else None
    args = ("metrics", "setup") if command == "setup" else export_arguments()
    with prompt(run, *args) as (process, terminal):
        os.write(terminal, answer)
        stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stderr
    assert b"Traceback" not in stdout + stderr
    approved = answer == b"yes\n"
    if command == "setup":
        assert policy.load_profile_policy().collection_enabled is approved
    else:
        assert enabled_pairs() == (((DEST, "metrics"),) if approved else ())
    if not approved:
        after = policy.policy_path().read_bytes() if policy.policy_path().exists() else None
        assert after == before


@pytest.mark.parametrize("command", ["setup", "export"])
def test_consent_revalidates_policy_changed_while_prompt_is_open(run: Run, command: str):
    from tests.accounting.test_export_consent import configure

    configure()
    args = ("metrics", "setup") if command == "setup" else export_arguments()
    with prompt(run, *args) as (process, terminal):
        policy.policy_path().write_text("{")
        os.write(terminal, b"yes\n")
        stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 1
    assert policy.policy_path().read_bytes() == b"{"
    assert b"ssgrep metrics setup" in stderr
    assert b"Traceback" not in stdout + stderr


def test_oversized_setup_policy_is_rejected_before_creating_accounting_state(
    run: Run, tmp_path: Path
):
    source = tmp_path / "oversized.json"
    source.write_bytes(b" " * (256 * 1024 + 1))
    result = run("metrics", "setup", "--yes", "--policy", str(source), "--json")
    assert result.returncode == 2
    assert document(result)["condition"] == "invalid_policy"
    assert not run.data.exists()


def test_closed_or_absent_stdin_cannot_grant_interactive_consent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    with (tmp_path / "stdin").open("w+") as stream:
        monkeypatch.setattr(sys, "stdin", stream)
        assert stdin_is_tty() is False
    assert stdin_is_tty() is False
    monkeypatch.setattr(sys, "stdin", None)
    assert stdin_is_tty() is False
    master, slave = pty.openpty()
    try:
        with os.fdopen(slave) as terminal:
            monkeypatch.setattr(sys, "stdin", terminal)
            assert stdin_is_tty() is True
    finally:
        os.close(master)


@pytest.mark.parametrize("roots", [(), ("/w/team\x1b[2J",)])
def test_setup_preview_preserves_roots_retention_and_disabled_destinations(roots: tuple[str, ...]):
    from tests.accounting.test_export_consent import DEST, ENDPOINT

    p = ProfilePolicy(
        profile_id="acme",
        roots=roots,
        evidence_retention_days=17,
        receipt_retention_days=43,
        destinations=(DestinationPolicy(DEST, ENDPOINT),),
    )
    text = flat(
        metrics_render.render_setup_preview(
            p, width=80, already_enabled=False, source="managed"
        ).plain
    )
    assert "17 days" in text and "43 days" in text and DEST in text
    assert "disabled" in text
    assert "\x1b" not in text
    if roots:
        assert "/w/team\\x1b[2J" in text
    else:
        assert "none approved" in text and "all working directories" not in text


def test_raw_scope_command_preserves_hash_prefixed_selection_in_real_cli(tmp_path: Path) -> None:
    scope = Scope(
        since=None,
        until=None,
        project="#notes",
        runtime="omp",
        session="#session",
        task="#task",
        by=None,
        include_children=True,
    )
    command = savings_state_render.scope_command("personal-local", scope, 20, None)
    script = f'ssgrep() {{ command {shlex.quote(str(BINARY))} "$@"; }};\n{command} --json'
    result = subprocess.run(
        ["/bin/sh", "-c", script],
        env=env_for(tmp_path),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    selected = json.loads(result.stdout)["data"]["data"]["scope"]
    assert {key: selected[key] for key in ("project", "runtime", "session", "task")} == {
        "project": "#notes",
        "runtime": "omp",
        "session": "#session",
        "task": "#task",
    }
    assert selected["include_children"] is True
