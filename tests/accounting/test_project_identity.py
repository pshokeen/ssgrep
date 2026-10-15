"""Matched CLI journal paths anchor project counters and exposure debits together."""

from __future__ import annotations

from pathlib import Path

from ssgrep.accounting import policy, service
from ssgrep.accounting.collector import collect_once
from ssgrep.accounting.collector_types import CollectOptions
from ssgrep.accounting.scope import build_scope
from ssgrep.accounting.service_ledger import load_ledger, load_reduction
from tests.accounting import collector_scene as scene
from tests.accounting.reduce_builders import (
    SHOW,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    ts,
)

PROFILE = "org-project-identity"
BODY = "x" * 896  # exactly 224 context tokens; no modeled reuse credit
REF = "claude:earlier:ep:0"
PROJECTS = ("target", "target-neighbor")


def _collect(tmp_path: Path, *, transport: str = "cli") -> tuple[Path, Path]:
    base = tmp_path.resolve() / "workspaces"
    for name in PROJECTS:
        (base / name).mkdir(parents=True)
    alias = tmp_path / "workspace-alias"
    alias.symlink_to(base, target_is_directory=True)
    policy.enable_collection(PROFILE, {"roots": [str(base)]})
    scene.install_key(PROFILE)
    for minute, name in enumerate(PROJECTS):
        cwd = str(alias / name)
        command = {"command": f"ssgrep show {REF} --json"}
        records = [
            claude_prompt(name, "Inspect the earlier work", ts=ts(12, 9, minute)),
            claude_tool_use(
                name,
                "lookup",
                "Bash" if transport == "cli" else SHOW,
                command if transport == "cli" else {"ref": REF},
                cwd=cwd,
                ts=ts(12, 9, minute, 1),
            ),
            claude_tool_result(name, "lookup", BODY, ts=ts(12, 9, minute, 3)),
        ]
        for record in records:
            record["cwd"] = cwd
        scene.write_transcript(tmp_path, name, records, scene.at(hour=10))
        execution_cwd = base / name if transport == "cli" else base / "mcp-server"
        scene.journal_events(
            scene.execution(
                name,
                ts(12, 9, minute, 2),
                operation="show",
                text=REF,
                transport=transport,
                cwd=str(execution_cwd),
                profile=PROFILE,
                refs=(REF,),
                rendered=len(BODY),
            ),
            PROFILE,
        )
    result = collect_once(PROFILE, CollectOptions(), now=scene.at(hour=11))
    assert result.state == "published", result
    return base, alias


def _read(**scope):
    return service.read_summary(
        PROFILE,
        scope=build_scope(since="2026-10-01", until="2026-10-22", by="project", **scope),
        now=scene.at(day=21, hour=12),
    )


def _tokens(view, field: str) -> list[int | None]:
    row = next(row for row in view.scenarios if row.unit_key == "context_tokens")
    return [quantity.value for _, quantity in getattr(row, field)]


def test_matched_cli_aliases_share_project_counters_debits_and_filters(tmp_path: Path) -> None:
    base, alias = _collect(tmp_path)
    view = _read()
    rows = {row.key: row for row in view.groups}
    assert set(rows) == {str(base / name) for name in PROJECTS}
    assert view.group_total == 2
    assert dict(view.measured)["shows"].value == 2
    assert view.coverage["attributed_lookups"] == 2
    assert _tokens(view, "debit") == [448] * 3
    assert _tokens(view, "net") == [-448] * 3
    for name in PROJECTS:
        row = rows[str(base / name)]
        assert dict(row.measured)["shows"].value == 1
        assert dict(row.measured)["visible_bytes"].value == len(BODY)
        assert _tokens(row, "debit") == [224] * 3
        assert _tokens(row, "net") == [-224] * 3
        for project in (base / name, alias / name):
            filtered = _read(project=str(project))
            assert [group.key for group in filtered.groups] == [str(base / name)]
            assert dict(filtered.measured)["shows"].value == 1
            assert filtered.coverage["attributed_lookups"] == 1
            assert _tokens(filtered, "debit") == [224] * 3
            assert _tokens(filtered, "net") == [-224] * 3
    doc, reason = load_reduction(PROFILE)
    assert reason is None and doc is not None
    assert set(doc.session_cwd) == {("claude", name, str(base / name)) for name in PROJECTS}


def test_retargeting_alias_cannot_move_committed_or_sealed_project_history(tmp_path: Path) -> None:
    base, alias = _collect(tmp_path)
    # Seal the original day without reading any new source evidence.
    result = collect_once(PROFILE, CollectOptions(), now=scene.at(day=21, hour=11))
    assert result.state == "published", result
    ledger, reason = load_ledger(PROFILE)
    assert reason is None and ledger is not None
    assert "2026-10-12" in ledger.sealed_days
    before = _read()
    files = scene.file_digests(tmp_path / "data")
    outside = tmp_path / "unapproved"
    (outside / "target").mkdir(parents=True)
    alias.unlink()
    alias.symlink_to(outside, target_is_directory=True)
    after = _read()
    assert after.groups == before.groups
    assert after.measured == before.measured and after.scenarios == before.scenarios
    assert _tokens(_read(project=str(base / "target")), "debit") == [224] * 3
    assert _read(project=str(alias / "target")).groups == ()
    assert _read(project=str(outside)).groups == ()
    assert scene.file_digests(tmp_path / "data") == files


def test_mcp_server_cwd_never_replaces_caller_project(tmp_path: Path) -> None:
    base, alias = _collect(tmp_path, transport="mcp")
    doc, reason = load_reduction(PROFILE)
    assert reason is None and doc is not None
    expected = {("claude", name, str(base / name)) for name in PROJECTS}
    assert set(doc.session_cwd) == expected
    assert {execution.cwd for execution in doc.executions} == {
        str(base / name) for name in PROJECTS
    }
    assert all(execution.scope_basis == "native_call_metadata" for execution in doc.executions)
    assert doc.coverage["attributed_lookups"] == 2
    before = _read()
    assert {row.key for row in before.groups} == {str(base / name) for name in PROJECTS}
    assert dict(before.measured)["shows"].value == 2
    assert _tokens(before, "debit") == [448] * 3
    outside = tmp_path / "unapproved"
    (outside / "target").mkdir(parents=True)
    alias.unlink()
    alias.symlink_to(outside, target_is_directory=True)
    after = _read()
    assert after.groups == before.groups
    assert after.measured == before.measured and after.scenarios == before.scenarios
    assert _tokens(_read(project=str(base / "target")), "debit") == [224] * 3
    assert _read(project=str(alias / "target")).groups == ()
    assert _read(project=str(base / "mcp-server")).groups == ()
