"""10.3: one matrix over EVERY new command through the real binary.

Each row runs a command with ``--json`` in an isolated HOME/data and checks the contract:
stdout is exactly one JSON document, no ANSI, no floats/NaN, the exit code is the specified
one, successes carry the envelope keys, application errors carry ``ok/condition/message/command``
(framework usage errors keep usecli's own ``{ok:false,error:{type,message,code}}`` document),
and rows that must not change anything leave the data directory byte-identical.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.accounting.test_ux_support import Binary, collected_scene, fingerprint

ENVELOPE = {"schema_version", "state", "data", "warnings", "next_actions", "provenance"}
ERROR = {"ok", "condition", "message", "command"}
ANSI = re.compile(r"\x1b|\x9b")
FLOAT = re.compile(r"(?<![\w.\"])-?\d+\.\d+(?:[eE][+-]?\d+)?(?![\w.\"])")


def no_float(text: str) -> Any:
    raise AssertionError(f"float in JSON output: {text}")


def parse(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 1, f"exactly one JSON document expected, got {len(lines)}: {stdout!r}"
    assert not ANSI.search(stdout), "ANSI escape in JSON"
    assert "NaN" not in stdout and "Infinity" not in stdout
    return json.loads(lines[0], parse_float=no_float)


def shape_ok(body: dict[str, Any], expected: int) -> None:
    if expected == 0:
        assert body["ok"] is True and ENVELOPE <= set(body["data"]), body
    elif "error" in body:  # usecli's own document for argument parsing errors
        assert body["ok"] is False and {"type", "message", "code"} <= set(body["error"])
        assert body["error"]["code"] == expected
    else:
        assert ERROR <= set(body) and body["ok"] is False, body
        assert isinstance(body["condition"], str) and body["condition"]
        assert isinstance(body["message"], str) and isinstance(body["command"], str)


#: (arguments, expected exit, expected condition-or-state, must leave files unchanged)
EMPTY_ROWS = [
    (("savings",), 0, "disabled", True),
    (("savings", "history"), 0, "empty", True),
    (("savings", "explain", "nope"), 3, "receipt_not_found", True),
    (("savings", "explain", "../../etc/passwd"), 3, "receipt_not_found", True),
    (("savings", "--limit", "101"), 2, "invalid_limit", True),
    (("savings", "--limit", "0"), 2, "invalid_limit", True),
    (("savings", "--since", "yesterday"), 2, "invalid_interval", True),
    (("savings", "--since", "2026-10-02", "--until", "2026-10-01"), 2, "invalid_interval", True),
    (("savings", "--runtime", "vim"), 2, "invalid_filter", True),
    (("savings", "--include-children"), 2, "invalid_filter", True),
    (("savings", "--cursor", "g1.1.deadbeefdeadbeef", "--by", "task"), 2, "invalid_cursor", True),
    (("savings", "history", "--cursor", "h1.1.deadbeefdeadbeef"), 2, "invalid_cursor", True),
    (("savings", "--profile", "../x"), 2, "invalid_profile", True),
    (("savings", "--bogus"), 2, None, True),
    (("savings", "explain"), 2, None, True),
    (("metrics", "status"), 0, "disabled", True),
    (("metrics", "status", "--profile", ".."), 2, "invalid_profile", True),
    (("metrics", "collect"), 1, "collection_disabled", True),
    (
        ("metrics", "collect", "--backfill", "--since", "2026-01-01"),
        2,
        "backfill_requires_interval",
        True,
    ),
    (("metrics", "collect", "--since", "2026-01-01"), 2, "interval_requires_mode", True),
    (
        (
            "metrics",
            "collect",
            "--backfill",
            "--recompute",
            "--since",
            "2026-01-01",
            "--until",
            "2026-02-01",
        ),
        2,
        "conflicting_modes",
        True,
    ),
    (("metrics", "collect", "--profile", "A/B"), 2, "invalid_profile", True),
    (("metrics", "setup"), 2, "confirmation_required", True),
    (("metrics", "export"), 2, "destination_required", True),
    (("metrics", "export", "--destination", "x", "--preview", "--check"), 2, "invalid_mode", True),
    (("metrics", "export", "--destination", "x", "--preview"), 2, "unknown_destination", True),
    (("metrics", "purge", "--before", "2026-01-01"), 2, "confirmation_required", True),
    (("metrics", "purge"), 2, "invalid_interval", True),
    (("metrics", "serve", "--profile", "../x"), 2, "invalid_profile", True),
    (("metrics", "census", "--profile", "../x"), 2, "invalid_profile", True),
    (("metrics", "pause", "--profile", "../x"), 2, "invalid_profile", True),
    (("metrics", "resume", "--profile", "../x"), 2, "invalid_profile", True),
]


@pytest.fixture
def run(tmp_path: Path) -> Binary:
    return Binary(tmp_path)


@pytest.mark.parametrize(
    ("args", "code", "label", "readonly"), EMPTY_ROWS, ids=lambda v: str(v)[:60]
)
def test_matrix_row_on_an_empty_profile(
    run: Binary, args: tuple[str, ...], code: int, label: str | None, readonly: bool
) -> None:
    before = fingerprint(run.root / "data") if (run.root / "data").exists() else {}
    done = run(*args, "--json")
    assert done.returncode == code, (args, done.stdout, done.stderr)
    body = parse(done.stdout)
    shape_ok(body, code)
    if label is not None:
        got = body["data"]["state"] if code == 0 else body["condition"]
        assert got == label, body
    if readonly:
        after = fingerprint(run.root / "data") if (run.root / "data").exists() else {}
        assert after == before, "a rejected or read-only command changed files"


def test_paused_mutations_exit_1_with_stable_conditions_and_readonly_commands_still_work(
    run: Binary,
) -> None:
    assert run("metrics", "setup", "--yes", "--json").returncode == 0
    assert run("metrics", "pause", "--json").returncode == 0
    before = fingerprint(run.root / "data")
    code, body = run.json("metrics", "collect")
    assert code == 1 and body["condition"] == "collection_paused"
    assert body["command"] == "ssgrep metrics resume --profile personal-local"
    for readonly in (("savings",), ("savings", "history"), ("metrics", "status")):
        done = run(*readonly, "--json")
        assert done.returncode == 0, readonly
        assert parse(done.stdout)["data"]["state"] in {"paused", "empty"}
    assert fingerprint(run.root / "data") == before, "paused reads and refusals wrote nothing"
    code, resumed = run.json("metrics", "resume")
    assert code == 0 and resumed["data"]["state"] == "resumed"


def test_paused_export_send_and_check_exit_1_export_paused_and_preview_stays_usable(
    run: Binary, tmp_path: Path
) -> None:
    from dataclasses import replace

    from ssgrep.accounting import policy
    from ssgrep.accounting.ledger_contracts import DestinationPolicy, Signal

    policy.enable_collection("personal-local")
    current = policy.load_profile_policy("personal-local")
    dest = DestinationPolicy(
        "org-primary", "https://collector.example.test:4318", (Signal.METRICS,)
    )
    policy.save_profile(replace(current, destinations=(dest,)))
    assert run("metrics", "pause", "--json").returncode == 0
    for mode in ((), ("--check",)):
        done = run("metrics", "export", "--destination", "org-primary", *mode, "--json")
        assert done.returncode == 1, (mode, done.stdout, done.stderr[-300:])
        body = parse(done.stdout)
        shape_ok(body, 1)
        assert body["condition"] == "export_paused"
    preview = run("metrics", "export", "--destination", "org-primary", "--preview", "--json")
    assert preview.returncode == 0, preview.stderr[-300:]
    parse(preview.stdout)


def test_successes_over_a_real_collection_are_float_free_ansi_free_one_document(
    tmp_path: Path,
) -> None:
    collected_scene(tmp_path)
    run = Binary(tmp_path)
    receipt = run.data("savings")["provenance"]["snapshot_id"]
    assert receipt
    rows = [
        ("savings",),
        ("savings", "--by", "task"),
        ("savings", "--by", "day", "--limit", "1"),
        ("savings", "--runtime", "claude", "--since", "2026-10-12", "--until", "2026-10-13"),
        ("savings", "history"),
        ("metrics", "status"),
        ("metrics", "census", "--root", str(tmp_path / "home")),
    ]
    for args in rows:
        done = run(*args, "--json", env={"FORCE_COLOR": "1", "NO_COLOR": ""})
        assert done.returncode == 0, (args, done.stderr[-300:])
        body = parse(done.stdout)
        shape_ok(body, 0)
        assert not FLOAT.search(done.stdout), (args, FLOAT.search(done.stdout))
    explained = run.data("savings")["data"]["representative_receipt"]
    done = run("savings", "explain", explained, "--json", env={"FORCE_COLOR": "1"})
    assert done.returncode == 0 and ENVELOPE <= set(parse(done.stdout)["data"])
    unknown = run("savings", "explain", "ffffffffffffffffffffffff", "--json")
    assert unknown.returncode == 3
    assert parse(unknown.stdout) == {
        "ok": False,
        "condition": "receipt_not_found",
        "message": "No receipt with that id exists for this profile.",
        "command": "ssgrep savings",
    }


def test_unknown_values_are_null_with_a_reason_never_zero_in_json(tmp_path: Path) -> None:
    collected_scene(tmp_path)
    run = Binary(tmp_path)
    data = run.data("savings", "--runtime", "codex")["data"]
    net = data["estimates"]["context_tokens"]["central"]["net"]
    assert net["value"] is None and net["reason"] and net["basis"] == "unavailable"
    by_task = run.data("savings", "--by", "task")["data"]["groups"]["rows"]
    unknown = next(r for r in by_task if r["key"] == "task_unknown")
    cell = unknown["estimates"]["context_tokens"]["central"]["net"]
    assert cell["value"] is None and cell["reason"]
    status = run.data("metrics", "status")["data"]
    assert status["lag"]["lag_bytes"] is not None or status["lag"]["reason"]
    assert all(
        v is None or isinstance(v, int)
        for v in (
            status["storage_pressure"]["free_bytes"],
            status["storage_pressure"]["total_bytes"],
        )
    )


def test_a_newer_schema_receipt_exits_3_unavailable_and_is_not_rewritten(tmp_path: Path) -> None:
    collected_scene(tmp_path)
    run = Binary(tmp_path)
    receipt_id = run.data("savings")["data"]["representative_receipt"]
    target = tmp_path / "data" / "accounting" / "personal-local" / "receipts" / f"{receipt_id}.json"
    document = json.loads(target.read_text())
    document["schema"] = 99
    target.write_text(json.dumps(document))
    target.chmod(0o600)
    before = target.read_bytes()
    done = run("savings", "explain", receipt_id, "--json")
    assert done.returncode == 3
    body = parse(done.stdout)
    assert body["condition"] == "receipt_unsupported_schema" and "newer schema" in body["message"]
    assert target.read_bytes() == before, "an unsupported receipt is never rewritten"
