"""V09 every accepted lookup has independent evidence and bounded read-only pages."""

import json
from dataclasses import replace

import pytest

from ssgrep.accounting import collector_state, collector_store, service_ledger
from ssgrep.accounting.contracts import Attribution, Exposure, InvocationClass, Outcome, Reuse
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.scope import build_scope
from ssgrep.accounting.service_lookups import read_lookups
from tests.accounting.test_ux_support import Binary, collected_scene, fingerprint


@pytest.fixture
def scene(tmp_path):
    collected_scene(tmp_path, pairs=(0,))
    data, reason = service_ledger.load_ledger("personal-local")
    assert data is not None and data.doc is not None
    assert reason is None and data.doc.invocations
    invocation = next(
        r
        for r in data.doc.invocations
        if set(r.operations) & {"search", "show"} and r.execution_ids == ("n-0",)
    )
    additions = tuple(
        replace(
            invocation,
            invocation_id=name,
            call_id=name,
            execution_ids=(),
            ambiguous_execution_ids=(),
            outcome=outcome,
            exposure=Exposure.OBSERVED,
            reuse=reuse,
            cls=InvocationClass.TRANSCRIPT_ONLY,
            attribution=Attribution.UNATTRIBUTED,
        )
        for name, outcome, reuse in (
            ("empty-lookup", Outcome.EMPTY, Reuse.NONE_OBSERVED),
            ("error-lookup", Outcome.ERROR, Reuse.NONE_OBSERVED),
            ("inspected-lookup", Outcome.OK, Reuse.INSPECTED),
        )
    )
    doc = replace(data.doc, invocations=(*data.doc.invocations, *additions))
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    checkpoint.blobs[collector_state.REDUCTION] = collector_store.write_blob(
        "personal-local", collector_state.REDUCTION, doc.to_dict()
    )
    collector_store.commit_checkpoint("personal-local", checkpoint)
    return tmp_path, invocation


def test_pages_enumerate_all_lookup_ids_once_including_no_receipt_and_journal_only(scene):
    root, invocation = scene
    before = fingerprint(root / "data")
    page = read_lookups(limit=1)
    rows = list(page.rows)
    while page.next_cursor:
        page = read_lookups(limit=1, cursor=page.next_cursor)
        rows.extend(page.rows)
    assert {r.lookup_id for r in rows} == {
        f"invocation:{invocation.invocation_id}",
        "invocation:empty-lookup",
        "invocation:error-lookup",
        "invocation:inspected-lookup",
        "execution:n-start",
    }
    assert len(rows) == len({r.lookup_id for r in rows}) == 5
    by_id = {r.lookup_id: r for r in rows}
    assert by_id["invocation:empty-lookup"].receipt_ids == ()
    assert by_id["invocation:empty-lookup"].reuse == "none_observed"
    assert by_id["invocation:error-lookup"].outcome == "error"
    assert by_id["invocation:inspected-lookup"].reuse == "inspected"
    assert by_id["execution:n-start"].eligibility == "unavailable"
    assert fingerprint(root / "data") == before


def test_lookup_filters_intersect_and_cursor_is_bound_to_selection_and_rows(scene):
    _, invocation = scene
    page = read_lookups(limit=1)
    with pytest.raises(PolicyError, match="Cursor"):
        read_lookups(scope=build_scope(runtime="codex"), limit=1, cursor=page.next_cursor)
    scoped = read_lookups(
        scope=build_scope(
            runtime=invocation.runtime,
            session=invocation.session_id,
            task=invocation.task.root,
            project=invocation.cwd,
        ),
        limit=100,
    )
    assert len(scoped.rows) == 4 and all(r.runtime == invocation.runtime for r in scoped.rows)
    assert read_lookups(scope=build_scope(runtime="codex"), limit=100).rows == ()


def test_lookup_old_generation_is_honestly_expired_not_current_data(scene):
    root, _ = scene
    page = read_lookups(limit=1)
    path = root / "data" / "accounting" / "personal-local" / "snapshots" / "current.json"
    raw = json.loads(path.read_text())
    raw["generation"] += 1
    raw["snapshot_id"] = "new-snapshot"
    path.write_text(json.dumps(raw))
    with pytest.raises(PolicyError) as caught:
        read_lookups(
            snapshot_id=page.snapshot_id,
            generation=page.generation,
            limit=1,
            cursor=page.next_cursor,
        )
    assert caught.value.condition == "lookup_snapshot_expired"


def test_actual_cli_lookup_command_is_read_only_bounded_and_json_complete(scene):
    root, _ = scene
    before = fingerprint(root / "data")
    binary = Binary(root)
    status, body = binary.json("savings", "lookups", "--limit", "2")
    assert status == 0, body
    data = body["data"]["data"]
    assert len(data["rows"]) == 2 and data["total"] == 5 and data["next_cursor"]
    for row in data["rows"]:
        assert {
            "execution",
            "exposure",
            "attribution",
            "reuse",
            "eligibility",
            "outcome",
            "scenarios",
        } <= set(row)
        assert row["causal_usefulness"] is None
    assert fingerprint(root / "data") == before


def test_default_lookup_page_keeps_unplaced_calls_without_inventing_time_membership(scene):
    _, invocation = scene
    data, _ = service_ledger.load_ledger("personal-local")
    assert data is not None and data.doc is not None
    doc = replace(
        data.doc,
        invocations=tuple(
            replace(r, ts_utc=None, timing_known=False)
            if r.invocation_id == invocation.invocation_id
            else r
            for r in data.doc.invocations
        ),
    )
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    checkpoint.blobs[collector_state.REDUCTION] = collector_store.write_blob(
        "personal-local", collector_state.REDUCTION, doc.to_dict()
    )
    collector_store.commit_checkpoint("personal-local", checkpoint)
    view = read_lookups(limit=100)
    row = next(r for r in view.rows if r.invocation_id == invocation.invocation_id)
    assert row.ts_utc is None and not row.timing_known
    assert any(w.startswith("untimed_lookup_rows:") for w in view.warnings)
    exact = read_lookups(scope=build_scope(since="2026-10-12", until="2026-10-13"), limit=100)
    assert all(r.invocation_id != invocation.invocation_id for r in exact.rows)


def test_skipped_reuse_evaluation_is_not_observed_no_reuse_or_known_zero_eligibility(scene):
    _, invocation = scene
    data, _ = service_ledger.load_ledger("personal-local")
    assert data is not None and data.doc is not None
    skipped = replace(
        invocation,
        invocation_id="not-evaluated",
        execution_ids=(),
        ambiguous_execution_ids=(),
        reuse=Reuse.NONE_OBSERVED,
        reuse_evaluated=False,
        outcome=Outcome.OK,
        exposure=Exposure.PARTIAL,
        cls=InvocationClass.TRANSCRIPT_ONLY,
        attribution=Attribution.UNATTRIBUTED,
    )
    doc = replace(data.doc, invocations=(*data.doc.invocations, skipped))
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    checkpoint.blobs[collector_state.REDUCTION] = collector_store.write_blob(
        "personal-local", collector_state.REDUCTION, doc.to_dict()
    )
    collector_store.commit_checkpoint("personal-local", checkpoint)
    row = next(r for r in read_lookups(limit=100).rows if r.invocation_id == "not-evaluated")
    assert row.reuse == "unavailable" and row.useful_outcome == "not_evaluated"
    assert row.eligibility == "unavailable" and "reuse_not_evaluated" in row.missing_evidence


@pytest.mark.parametrize("width", [40, 80, 120])
def test_lookup_binary_text_and_json_preserve_evidence_and_receipt_actions(scene, width):
    from rich.cells import cell_len

    root, _ = scene
    before = fingerprint(root / "data")
    binary = Binary(root)
    status, envelope = binary.json("savings", "lookups", "--limit", "100")
    assert status == 0, envelope
    rows = envelope["data"]["data"]["rows"]
    assert len(rows) == 5
    result = binary("savings", "lookups", "--limit", "100", columns=width)
    assert result.returncode == 0, result.stderr
    assert "\x1b" not in result.stdout
    assert all(cell_len(line) <= width for line in result.stdout.splitlines())
    compact = "".join(result.stdout.split())
    links = []
    known_costs = []
    unknown = []
    for row in rows:
        assert row["lookup_id"] in compact
        for field in ("execution", "outcome", "exposure", "attribution", "reuse", "eligibility"):
            assert row[field] in compact
        for name, quantity in row["known_costs"].items():
            known_costs.append(quantity)
            assert name in compact
            if quantity["value"] is not None:
                unit = {"context_tokens": "tokens", "work_seconds": "s"}.get(
                    quantity["unit"], quantity["unit"]
                )
                assert f"{quantity['value']:,}{unit}" in compact
        unknown.extend(row["missing_evidence"])
        for reason in row["missing_evidence"]:
            assert reason in compact
        for receipt_id in row["receipt_ids"]:
            links.append(receipt_id)
            assert f"ssgrepsavingsexplain{receipt_id}" in compact
            code, receipt = binary.json("savings", "explain", receipt_id)
            assert code == 0 and receipt["data"]["data"]["receipt_id"] == receipt_id
        assert row["causal_usefulness"] is None
    assert links and known_costs and unknown
    assert fingerprint(root / "data") == before


@pytest.mark.parametrize("width", [40, 80, 120])
def test_lookup_next_command_roundtrips_quoted_scope_and_children_without_writes(scene, width):
    import subprocess

    from rich.cells import cell_len

    root, invocation = scene
    data, _ = service_ledger.load_ledger("personal-local")
    assert data is not None and data.doc is not None
    project = str(root / "project 'quoted' ; $not_a_command")
    doc = replace(
        data.doc,
        invocations=tuple(replace(row, cwd=project) for row in data.doc.invocations),
    )
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    checkpoint.blobs[collector_state.REDUCTION] = collector_store.write_blob(
        "personal-local", collector_state.REDUCTION, doc.to_dict()
    )
    collector_store.commit_checkpoint("personal-local", checkpoint)
    binary = Binary(root)
    selection = (
        "--project",
        project,
        "--session",
        invocation.session_id,
        "--runtime",
        invocation.runtime,
        "--include-children",
        "--limit",
        "1",
    )
    before = fingerprint(root / "data")
    code, first = binary.json("savings", "lookups", *selection)
    assert code == 0, first
    first_data = first["data"]["data"]
    assert first_data["next_cursor"] and first_data["total"] >= 2
    text = binary("savings", "lookups", *selection, columns=width)
    assert text.returncode == 0
    lines = text.stdout.splitlines()
    assert all(cell_len(line) <= width for line in lines)
    start = next(i for i, line in enumerate(lines) if line.startswith("ssgrep savings lookups"))
    command = "\n".join(lines[start:])
    parsed = subprocess.run(
        ["/bin/sh", "-c", "set -e; ssgrep() { printf '%s\\0' ssgrep \"$@\"; };\n" + command],
        cwd=root,
        capture_output=True,
        check=True,
    )
    tokens = [part.decode() for part in parsed.stdout.split(b"\0")[:-1]]
    assert tokens[:3] == ["ssgrep", "savings", "lookups"]
    assert tokens[tokens.index("--project") + 1] == project
    assert tokens[tokens.index("--session") + 1] == invocation.session_id
    assert "--include-children" in tokens
    code, second = binary.json(*tokens[1:])
    assert code == 0, second
    second_data = second["data"]["data"]
    expected = read_lookups(
        scope=build_scope(
            project=project,
            session=invocation.session_id,
            runtime=invocation.runtime,
            include_children=True,
        ),
        limit=1,
        cursor=first_data["next_cursor"],
    )
    assert [row["lookup_id"] for row in second_data["rows"]] == [
        row.lookup_id for row in expected.rows
    ]
    assert second_data["rows"][0]["lookup_id"] != first_data["rows"][0]["lookup_id"]
    assert second["data"]["provenance"] == first["data"]["provenance"]
    assert fingerprint(root / "data") == before


@pytest.mark.parametrize("corruption", ["prefix", "bool_generation", "page", "snapshot"])
def test_lookup_cursor_rejects_forged_fields_without_rebinding_or_writes(scene, corruption):
    import base64

    root, _ = scene
    first = read_lookups(limit=1)
    assert first.next_cursor is not None
    token = first.next_cursor
    if corruption == "prefix":
        token = "history1." + token.partition(".")[2]
    else:
        raw = json.loads(base64.urlsafe_b64decode(token[9:] + "=" * (-len(token[9:]) % 4)))
        raw[
            {"bool_generation": "generation", "page": "page", "snapshot": "snapshot"}[corruption]
        ] = {
            "bool_generation": True,
            "page": "diag1.9999999.0000000000000000",
            "snapshot": "not-the-selected-snapshot",
        }[corruption]
        token = "lookups1." + base64.urlsafe_b64encode(json.dumps(raw).encode()).decode()
    before = fingerprint(root / "data")
    with pytest.raises(PolicyError) as caught:
        read_lookups(
            cursor=token, limit=1, snapshot_id=first.snapshot_id, generation=first.generation
        )
    assert caught.value.condition == "invalid_cursor"
    assert fingerprint(root / "data") == before


def test_lookup_authenticated_past_end_cursor_and_grouping_are_typed_errors(scene):
    from ssgrep.accounting.contracts import digest_of
    from ssgrep.accounting.lookup_cursor import next_cursor
    from ssgrep.accounting.service_types import Scope

    root, _ = scene
    page = read_lookups(limit=1)
    assert page.snapshot_id is not None and page.generation is not None
    doc, _, ledger, reason = service_ledger.load_lookup_data("personal-local")
    assert doc is not None and ledger is not None
    binding = digest_of(
        [
            "personal-local",
            Scope(),
            1,
            page.snapshot_id,
            page.generation,
            doc.to_dict(),
            ledger.rows.digest if ledger else reason,
        ]
    )
    cursor = next_cursor(page.total, binding, page.snapshot_id, page.generation)
    before = fingerprint(root / "data")
    with pytest.raises(PolicyError) as past_end:
        read_lookups(limit=1, cursor=cursor)
    assert past_end.value.condition == "invalid_cursor"
    with pytest.raises(PolicyError) as grouped:
        read_lookups(scope=Scope(by="runtime"))
    assert grouped.value.condition == "invalid_filter"
    assert fingerprint(root / "data") == before


def test_deleted_receipt_is_not_offered_as_an_explainable_lookup(scene):
    root, _ = scene
    original = read_lookups(limit=100)
    linked = next(row for row in original.rows if row.receipt_ids)
    receipt_id = linked.receipt_ids[0]
    path = root / "data/accounting/personal-local/receipts" / f"{receipt_id}.json"
    assert path.is_file()
    path.unlink()
    before = fingerprint(root / "data")
    reread = read_lookups(limit=100)
    row = next(row for row in reread.rows if row.lookup_id == linked.lookup_id)
    assert receipt_id not in row.receipt_ids
    assert "receipt_explanation_unavailable" in row.missing_evidence
    assert row.execution == linked.execution and row.exposure == linked.exposure
    assert row.known_costs == linked.known_costs
    assert reread.total == original.total
    assert fingerprint(root / "data") == before


@pytest.mark.parametrize("enabled", [False, True])
def test_lookup_unavailable_states_have_reachable_status_and_no_invented_rows(tmp_path, enabled):
    from ssgrep.accounting import policy

    if enabled:
        policy.enable_collection("personal-local")
    root = tmp_path
    before = fingerprint(root / "data")
    binary = Binary(root)
    code, envelope = binary.json("savings", "lookups")
    assert code == 0, envelope
    payload = envelope["data"]
    assert payload["state"] == "unavailable"
    assert payload["data"]["rows"] == [] and payload["data"]["total"] == 0
    assert payload["data"]["next_cursor"] is None
    assert payload["data"]["reason"] == "no_snapshot"
    result = binary("savings", "lookups")
    assert result.returncode == 0 and "ssgrep metrics status" in result.stdout
    status, recovered = binary.json("metrics", "status")
    assert status == 0, recovered
    assert fingerprint(root / "data") == before


def test_new_snapshot_generation_without_matching_ledger_withholds_old_lookup_rows(scene):
    root, _ = scene
    first = read_lookups(limit=1)
    assert first.rows and first.next_cursor
    assert first.generation is not None
    path = root / "data/accounting/personal-local/snapshots/current.json"
    raw = json.loads(path.read_text())
    raw["generation"] += 1
    path.write_text(json.dumps(raw))
    before = fingerprint(root / "data")
    pending = read_lookups(limit=1)
    assert pending.rows == () and pending.next_cursor is None
    assert pending.generation == first.generation + 1
    assert pending.snapshot_id == first.snapshot_id
    assert pending.reason == "lookup_generation_pending"
    with pytest.raises(PolicyError) as caught:
        read_lookups(limit=1, cursor=first.next_cursor)
    assert caught.value.condition == "lookup_snapshot_expired"
    assert fingerprint(root / "data") == before
