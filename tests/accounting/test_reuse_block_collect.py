"""The reuse block of stored receipts: how reuse was established, from persisted facts only.

Real end to end: native Claude transcripts, the real reducer through ``collect_once``, then the
installed ``savings explain`` binary reading the files the collector published.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import collector_publish as publish, service
from ssgrep.accounting.contracts import canonical_bytes
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import (
    MCP,
    PROFILE,
    card,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    results_json,
    source_session,
    ts,
)
from tests.accounting.test_savings_cli import BINARY

NOW = scene.at(hour=20)
CLOCK = scene.at(hour=9)
SECRET_QUERY = "why did the alpha deploy fail"
REUSE_KEYS = {"class", "relay", "report", "action_ok", "novel", "user_supplied"}


def consumer(n: int, kind: str) -> list[dict[str, Any]]:
    """Search, then act on the fact; ``contradicted`` errors, ``validated`` then runs pytest."""
    session = f"cons{n}"
    h, m = 9 + n, 10 * n
    records = [
        claude_prompt(session, "fix the deploy", uuid=f"u{n}", ts=ts(12, h, m)),
        claude_tool_use(
            session, f"t{n}a", MCP, {"query": vs.query(n)}, ts=ts(12, h, m, 10), message_id=f"a{n}"
        ),
        claude_tool_result(
            session,
            f"t{n}a",
            results_json(card(f"claude:src{n}:ep:0", vs.fact(n))),
            ts=ts(12, h, m, 12),
        ),
        claude_tool_use(
            session,
            f"t{n}b",
            "Bash",
            {"command": f"vim {vs.path(n)}"},
            ts=ts(12, h, m, 20),
            message_id=f"b{n}",
        ),
        claude_tool_result(
            session, f"t{n}b", "ok", is_error=kind == "contradicted", ts=ts(12, h, m, 22)
        ),
    ]
    if kind == "validated":
        records += [
            claude_tool_use(
                session,
                f"t{n}c",
                "Bash",
                {"command": "uv run pytest -q"},
                ts=ts(12, h, m, 30),
                message_id=f"c{n}",
            ),
            claude_tool_result(
                session, f"t{n}c", "12 passed in 0.4s", is_error=False, ts=ts(12, h, m, 35)
            ),
        ]
    return records


def build(root: Path) -> dict[int, str]:
    """Three real consumers (action_reference, validated, contradicted); class by session."""
    vs.enable(pin=True)
    vs.start_capture()
    base = vs.projects(root)
    kinds: dict[int, str] = {0: "action_reference", 1: "validated", 2: "contradicted"}
    for n, kind in kinds.items():
        vs.put(base, f"src{n}", source_session(f"src{n}", vs.fact(n), when_day=11), CLOCK)
        vs.put(base, f"cons{n}", consumer(n, kind), CLOCK)
        scene.journal_events(
            scene.execution(f"n-{n}", ts(12, 9 + n, 10 * n, 11), text=vs.query(n), transport="mcp")
        )
    result = scene.collect(NOW)
    assert result.state == "published" and scene.report(result).complete
    return kinds


def stored_blocks() -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted((vs.accounting_dir() / "receipts").glob("*.json")):
        document = json.loads(path.read_bytes())
        out[path.stem] = document.get("reuse")
    return out


def test_collected_receipts_record_how_reuse_was_established(tmp_path: Path) -> None:
    kinds = build(tmp_path)
    blocks = stored_blocks()
    assert sorted(b["class"] for b in blocks.values()) == sorted(kinds.values())
    for block in blocks.values():
        assert set(block) == REUSE_KEYS
        assert all(isinstance(block[k], bool) for k in REUSE_KEYS - {"class"})
        assert block["novel"] is True
    by_class = {b["class"]: b for b in blocks.values()}
    assert by_class["contradicted"]["action_ok"] is False
    assert by_class["validated"]["action_ok"] is True
    assert by_class["action_reference"]["action_ok"] is True


def test_block_holds_no_text_paths_or_facts(tmp_path: Path) -> None:
    build(tmp_path)
    for path in (vs.accounting_dir() / "receipts").glob("*.json"):
        raw = path.read_text()
        block = json.dumps(json.loads(raw)["reuse"])
        for forbidden in (vs.fact(1), vs.path(1), vs.query(1), "pytest", "/"):
            assert forbidden not in block
        assert vs.query(0) not in raw and vs.fact(0) not in raw


def test_savings_explain_shows_how_reuse_was_established(tmp_path: Path) -> None:
    build(tmp_path)
    by_class = {b["class"]: rid for rid, b in stored_blocks().items()}
    for kind, succeeded in (("validated", "yes"), ("contradicted", "no")):
        done = subprocess.run(
            [str(BINARY), "savings", "explain", by_class[kind]],
            capture_output=True,
            text=True,
            env={**os.environ, "NO_COLOR": "1", "COLUMNS": "100"},
            timeout=120,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert "not recorded" not in done.stdout
        klass = next(x for x in done.stdout.splitlines() if "Reuse class" in x)
        assert klass.split()[-1] == kind
        line = next(x for x in done.stdout.splitlines() if "Acted-on step succeeded" in x)
        assert line.split()[-1] == succeeded


def test_ledger_digest_and_snapshot_id_ignore_the_block(tmp_path: Path) -> None:
    build(tmp_path)
    facts = vs.facts()
    receipts = vs.accounting_dir() / "receipts"
    for path in receipts.glob("*.json"):  # strip every block: an older collector's files
        document = json.loads(path.read_bytes())
        document.pop("reuse")
        path.write_bytes(canonical_bytes(document) + b"\n")
    assert vs.facts() == facts
    assert all(b is None for b in stored_blocks().values())


def test_older_files_without_a_block_still_read(tmp_path: Path) -> None:
    build(tmp_path)
    rid = next(iter(stored_blocks()))
    path = vs.accounting_dir() / "receipts" / f"{rid}.json"
    document = json.loads(path.read_bytes())
    document.pop("reuse")
    path.write_bytes(canonical_bytes(document) + b"\n")
    view = service.read_receipt(PROFILE, rid)
    assert view is not None and view.reuse is None


@pytest.mark.parametrize("stale", [b"", b"{"])
def test_rewrite_without_fresh_facts_keeps_the_stored_block(tmp_path: Path, stale: bytes) -> None:
    build(tmp_path)
    before = stored_blocks()
    state = vs.checkpoint()
    assert state is not None
    from ssgrep.accounting.collector_ledger import VersionedLedger
    from ssgrep.accounting.collector_state import load_state

    loaded = load_state(PROFILE, state, state.methodology)
    assert loaded.ledger is not None
    ledger = VersionedLedger.restore_versioned(loaded.ledger, loaded.superseded)
    files = vs.accounting_dir() / "receipts"
    for path in files.glob("*.json"):
        path.unlink()
        path.write_bytes(stale)  # unreadable stale file must not raise nor invent a block
        path.chmod(0o600)
    publish.write_receipts(PROFILE, ledger.rows())
    assert all(b is None for b in stored_blocks().values())
    publish.write_receipts(
        PROFILE,
        ledger.rows(),
        publish.reuse_blocks(loaded.ledger.candidates, ledger.methodology, loaded.superseded),
    )
    assert stored_blocks() == before
    publish.write_receipts(PROFILE, ledger.rows())  # no fresh blocks: stored ones survive
    assert stored_blocks() == before


def test_block_is_deterministic_across_collections(tmp_path: Path) -> None:
    build(tmp_path)
    first = {p.name: p.read_bytes() for p in (vs.accounting_dir() / "receipts").glob("*.json")}
    scene.collect(NOW, scene.LIVE)
    second = {p.name: p.read_bytes() for p in (vs.accounting_dir() / "receipts").glob("*.json")}
    assert first == second
