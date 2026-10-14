"""Real subprocess runs of the installed ``ssgrep`` binary in an isolated HOME/data dir."""

from __future__ import annotations

import json
import os
import pty
import subprocess
import sys
from pathlib import Path

import pytest

from tests.accounting.snapshot_builders import (
    RECEIPT_ID,
    window_7d,
    write_history,
    write_receipt,
    write_snapshot,
)

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


def test_commands_are_discoverable_in_the_real_cli(run: Run) -> None:
    help_text = run("--help").stdout
    assert "savings" in help_text and "metrics" in help_text
    savings = run("savings", "--help").stdout
    metrics = run("metrics", "--help").stdout
    for name in ("explain", "history"):
        assert name in savings
    for name in ("setup", "status", "pause", "resume"):
        assert name in metrics
    assert run.files() == []


def test_disabled_summary_exits_0_in_text_and_json_and_creates_nothing(run: Run) -> None:
    text = run("savings")
    assert text.returncode == 0 and "metrics setup" in text.stdout and "off" in text.stdout
    body = document(run("savings", "--json"))
    assert body["ok"] is True
    data = body["data"]
    assert set(data) == {
        "schema_version",
        "state",
        "data",
        "warnings",
        "next_actions",
        "provenance",
    }
    assert data["state"] == "disabled"
    assert data["next_actions"][0]["command"] == "ssgrep metrics setup"
    assert data["data"]["measured"] == {} and data["data"]["estimates"] == {}
    assert run.files() == []


@pytest.mark.parametrize(
    "args",
    [
        ("savings", "--limit", "101"),
        ("savings", "--limit", "0"),
        ("savings", "--since", "yesterday"),
        ("savings", "--since", "2026-10-02", "--until", "2026-10-01"),
        ("savings", "--runtime", "vim"),
        ("savings", "--by", "colour"),
        ("savings", "--include-children"),
        ("savings", "--profile", "../etc"),
        ("savings", "--cursor", "garbage"),
        ("savings", "history", "--limit", "500"),
        ("savings", "explain", "x", "--profile", "A/B"),
        ("metrics", "status", "--profile", ".."),
        ("metrics", "pause", "--profile", "../x"),
    ],
)
def test_invalid_arguments_exit_2_with_error_document_before_side_effects(
    run: Run, args: tuple[str, ...]
) -> None:
    result = run(*args, "--json")
    assert result.returncode == 2, result.stdout
    error = document(result)
    assert error["ok"] is False
    assert {"condition", "message", "command"} <= set(error)
    assert run.files() == []


def test_unknown_receipt_exits_3_with_error_document_and_text_message(run: Run) -> None:
    result = run("savings", "explain", "rcpt-missing", "--json")
    assert result.returncode == 3
    assert document(result) == {
        "ok": False,
        "condition": "receipt_not_found",
        "message": "No receipt with that id exists for this profile.",
        "command": "ssgrep savings",
    }
    plain = run("savings", "explain", "rcpt-missing")
    assert plain.returncode == 3 and "No receipt" in plain.stderr and plain.stdout == ""
    traversal = run("savings", "explain", "../../x", "--json")
    assert traversal.returncode == 3 and document(traversal)["condition"] == "receipt_not_found"
    assert run.files() == []


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


def test_unknown_quantity_is_null_with_reason_in_json(run: Run) -> None:
    write_snapshot(run.data)
    inner = document(run("savings", "--json"))["data"]["data"]
    credit = inner["estimates"]["context_tokens"]["optimistic"]["net"]
    assert credit["value"] == 4000
    window = window_7d()
    window["estimates"]["context_tokens"]["central"]["net"] = {
        "value": None,
        "unit": "tokens",
        "basis": "unavailable",
        "reason": "source_costs_missing",
    }
    write_snapshot(run.data, windows={"7d": window})
    unknown = document(run("savings", "--json"))["data"]["data"]["estimates"]["context_tokens"][
        "central"
    ]["net"]
    assert unknown["value"] is None and unknown["reason"] == "source_costs_missing"


def test_explain_and_history_over_hand_built_files(run: Run) -> None:
    write_receipt(run.data)
    write_history(
        run.data,
        [
            {
                "period": "2026-10-01",
                "revision": 1,
                "status": "sealed",
                "net": {"value": 100, "unit": "tokens", "basis": "estimated"},
            },
            {
                "period": "2026-10-01",
                "revision": 2,
                "status": "revised",
                "supersedes": 1,
                "net": {"value": 70, "unit": "tokens", "basis": "estimated"},
            },
        ],
    )
    explained = run("savings", "explain", RECEIPT_ID)
    assert explained.returncode == 0 and "atom-1" in explained.stdout
    assert "SECRET-CONTENT-CANARY" not in explained.stdout
    with_content = run("savings", "explain", RECEIPT_ID, "--include-content")
    assert "SECRET-CONTENT-CANARY" in with_content.stdout
    as_json = document(run("savings", "explain", RECEIPT_ID, "--json"))["data"]
    assert as_json["data"]["content"] is None and as_json["state"] == "available"
    history = document(run("savings", "history", "--json"))["data"]
    assert [(e["revision"], e["current"]) for e in history["data"]["entries"]] == [
        (1, False),
        (2, True),
    ]
    assert history["data"]["current_total"]["value"] == 70
    text = run("savings", "history", columns=40)
    assert text.returncode == 0 and "superseded" in text.stdout


def test_status_works_without_any_index(run: Run) -> None:
    result = run("metrics", "status", "--json")
    assert result.returncode == 0
    data = document(result)["data"]
    assert data["state"] == "disabled" and data["data"]["capture"] == "off"
    assert data["data"]["destinations"] == []
    assert run.files() == []


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
    assert "Next: ssgrep metrics collect" in pending.stdout
    assert "tokens" not in pending.stdout
    paused = run("metrics", "pause")
    assert paused.returncode == 0 and "resume" in paused.stdout
    assert (
        "PAUSED" in run("savings").stdout and "Next: ssgrep metrics resume" in run("savings").stdout
    )
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
