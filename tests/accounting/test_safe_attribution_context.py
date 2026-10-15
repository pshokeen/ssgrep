"""Safe attribution uses local invocation authority, not client arguments."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting import capture, capture_policy, fingerprint, journal, journal_reader, policy
from ssgrep.accounting.contracts import Attribution, Operation, Outcome, Phase, Transport
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.reconcile import Request, fold_events, reconcile
from ssgrep.services import api, mcp_server
from ssgrep.utilities.types import SearchResponse


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    journal.close_all()
    fingerprint.forget_keys()
    capture._unsafe.clear()
    capture_policy._cache = ("", None, capture_policy._DISABLED)
    for name in ("SSGREP_MCP_CALLER_WORKSPACE", "SSGREP_CAPTURE_PROFILE", capture.CORRELATION_ENV):
        monkeypatch.delenv(name, raising=False)
    yield
    journal.close_all()


def request(fact, index=0, **overrides):
    base = Request(
        invocation_id=f"native-{index}",
        index=0,
        operation=fact.operation,
        transport=fact.transport,
        fingerprint=fact.fingerprint,
        cwd=fact.cwd,
        correlation=None,
        window=(fact.ts_ms - 1000, fact.ts_ms + 1000),
        capacity=1,
        result_refs=None,
        result_outcome=None,
        fingerprint_key_id=fact.fingerprint_key_id,
        cwd_trusted=True,
    )
    return replace(base, **overrides)


def captured(query="query", transport=Transport.CLI):
    capture.arm(transport)
    execution = capture.begin(Operation.SEARCH, text=query)
    assert execution is not None
    capture.complete(execution, Outcome.EMPTY)
    return execution


def facts(profile="personal-local"):
    return fold_events(journal_reader.read_journal(profile, quarantine=False).events)


def test_seven_unique_cli_mcp_controls_join_once_including_copied_evidence(monkeypatch):
    policy.enable_collection("personal-local")
    monkeypatch.setenv("SSGREP_MCP_CALLER_WORKSPACE", str(Path.cwd()))
    monkeypatch.setattr(api.search_module, "search", lambda *a, **k: SearchResponse([], 0))
    for index in range(7):
        if index % 2:
            assert mcp_server.search_sessions(f"unique query {index}")["total_matches"] == 0
        else:
            captured(f"unique query {index}")
    events = journal_reader.read_journal("personal-local", quarantine=False).events
    folded = fold_events(events + events)
    requests = [request(fact, index) for index, fact in enumerate(folded)]
    links = reconcile(folded, requests + requests)
    assert len(links) == 7
    assert {link.attribution for link in links.values()} == {Attribution.INFERRED_UNIQUE}
    assert len({link.requests[0] for link in links.values()}) == 7


def test_parallel_equal_queries_never_choose_first():
    policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()

    link = reconcile([fact], [request(fact), request(fact, 1)])[fact.execution_id]
    assert link.attribution is Attribution.AMBIGUOUS
    assert "equal_query_concurrency" in link.reasons


def test_rotated_key_is_observed_without_process_restart(tmp_path):
    key = fingerprint.load_key("personal-local", create=True)
    assert fingerprint.load_key("personal-local", create=False) == key
    path = tmp_path / "data" / "accounting" / "personal-local" / fingerprint.KEY_FILE
    replacement = b"r" * fingerprint.KEY_BYTES
    path.write_bytes(replacement)
    assert fingerprint.load_key("personal-local", create=False) == replacement
    assert fingerprint.key_id(key) != fingerprint.key_id(replacement)


def test_capacity_collision_has_explicit_reason():
    policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    other = replace(fact, execution_id="second")
    links = reconcile([fact, other], [request(fact)])
    assert all(link.attribution is Attribution.AMBIGUOUS for link in links.values())
    assert all("capacity_ambiguity" in link.reasons for link in links.values())


@pytest.mark.parametrize(
    "override,reason",
    [
        ({"fingerprint_key_id": "rotated-key"}, "key_mismatch"),
        ({"fingerprint": "0" * 64}, "query_mismatch"),
        ({"window": None}, "clock_uncertainty"),
        ({"window": (2000, 1000)}, "clock_uncertainty"),
        ({"cwd_trusted": False}, "untrusted_cwd"),
    ],
)
def test_unsafe_join_reports_rejection(override, reason):
    policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    link = reconcile([fact], [request(fact, **override)])[fact.execution_id]
    assert link.attribution is Attribution.UNATTRIBUTED
    assert reason in link.reasons


def test_unknown_execution_clock_cannot_infer():
    policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    link = reconcile([replace(fact, timing_known=False)], [request(fact)])[fact.execution_id]
    assert link.attribution is Attribution.UNATTRIBUTED
    assert "clock_uncertainty" in link.reasons


def test_mcp_server_cwd_is_not_caller_scope_and_client_cwd_cannot_authorize(monkeypatch):
    policy.enable_collection("personal-local")
    capture.arm(Transport.MCP)
    execution = capture.begin(Operation.SEARCH, text="q", cwd="/client-supplied")
    assert execution is not None
    (event,) = journal_reader.read_journal("personal-local", quarantine=False).events
    assert event.cwd is None and event.scope_basis == "unknown_scope"
    (fact,) = fold_events([event])
    link = reconcile([fact], [request(fact, cwd=None, cwd_trusted=False)])[fact.execution_id]
    assert link.attribution is Attribution.UNATTRIBUTED
    assert "unknown_scope" in link.reasons


def test_trusted_mcp_launch_context_captures_only_workspace_owner(tmp_path, monkeypatch):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    for profile, root in (("org-a", root_a), ("org-b", root_b)):
        policy.save_profile(
            ProfilePolicy(profile_id=profile, roots=(str(root),), collection_enabled=True)
        )
    monkeypatch.chdir(root_b)
    monkeypatch.setenv("SSGREP_MCP_CALLER_WORKSPACE", str(root_a))
    monkeypatch.setattr(api.search_module, "search", lambda *a, **k: SearchResponse([], 0))
    assert mcp_server.search_sessions("positive control")["total_matches"] == 0
    events = journal_reader.read_journal("org-a", quarantine=False).events
    assert len(events) == 3
    assert all(
        event.cwd == str(root_a) and event.scope_basis == "trusted_invocation" for event in events
    )
    assert journal_reader.read_journal("org-b", quarantine=False).events == []


def test_unknown_scope_selected_profile_never_broadcasts_or_fabricates_scope(tmp_path, monkeypatch):
    for profile in ("org-a", "org-b"):
        root = tmp_path / profile
        root.mkdir()
        policy.save_profile(
            ProfilePolicy(profile_id=profile, roots=(str(root),), collection_enabled=True)
        )
    monkeypatch.setenv("SSGREP_CAPTURE_PROFILE", "org-a")
    monkeypatch.setattr(api.search_module, "search", lambda *a, **k: SearchResponse([], 0))
    assert mcp_server.search_sessions("unknown caller")["total_matches"] == 0
    events = journal_reader.read_journal("org-a", quarantine=False).events
    assert len(events) == 3 and all(event.cwd is None for event in events)
    assert all(event.scope_basis == "unknown_scope" for event in events)
    assert journal_reader.read_journal("org-b", quarantine=False).events == []


def test_profile_selector_cannot_override_known_workspace_owner(tmp_path, monkeypatch):
    for profile in ("org-a", "org-b"):
        root = tmp_path / profile
        root.mkdir()
        policy.save_profile(
            ProfilePolicy(profile_id=profile, roots=(str(root),), collection_enabled=True)
        )
    monkeypatch.setenv("SSGREP_MCP_CALLER_WORKSPACE", str(tmp_path / "org-a"))
    monkeypatch.setenv("SSGREP_CAPTURE_PROFILE", "org-b")
    monkeypatch.setattr(api.search_module, "search", lambda *a, **k: SearchResponse([], 0))
    assert mcp_server.search_sessions("wrong tenant")["total_matches"] == 0
    assert journal_reader.read_journal("org-a", quarantine=False).events == []
    assert journal_reader.read_journal("org-b", quarantine=False).events == []


def test_native_seven_lookup_controls_have_independent_expected_counts(tmp_path):
    from tests.accounting.reduce_builders import (
        CWD,
        KEY,
        MCP,
        claude_prompt,
        claude_tool_result,
        claude_tool_use,
        event,
        results_json,
        ts,
    )
    from tests.accounting.test_reduce_reconcile import run

    records = [claude_prompt("s1", "seven controls", ts=ts(12, 9))]
    events = []
    expected_ids = {f"control-{index}" for index in range(7)}
    for index in range(7):
        query = f"independent query {index}"
        transport = "mcp" if index % 2 else "cli"
        name = MCP if transport == "mcp" else "Bash"
        arguments = (
            {"query": query} if transport == "mcp" else {"command": f"ssgrep search '{query}'"}
        )
        records.extend(
            [
                claude_tool_use("s1", f"tool-{index}", name, arguments, ts=ts(12, 10), cwd=CWD),
                claude_tool_result("s1", f"tool-{index}", results_json(), ts=ts(12, 10, 0, 3)),
            ]
        )
        events.extend(
            replace(
                row,
                fingerprint_key_id=fingerprint.key_id(KEY),
                scope_basis="unknown_scope" if transport == "mcp" else "process_cwd",
                cwd=None if transport == "mcp" else CWD,
            )
            for row in event(f"control-{index}", ts(12, 10, 0, 1), text=query, transport=transport)
        )
    reduction = run(tmp_path, records, events + events)
    assert {row.execution_id for row in reduction.executions} == expected_ids
    assert len(reduction.invocations) == 7
    assert all(row.attribution is Attribution.INFERRED_UNIQUE for row in reduction.executions)
    assert {
        identifier for row in reduction.invocations for identifier in row.execution_ids
    } == expected_ids
    assert all(len(row.execution_ids) == 1 for row in reduction.invocations)


def test_conflicting_request_copies_never_pick_supported_variant():
    policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    original = request(fact)
    conflicting = replace(original, fingerprint="0" * 64)
    link = reconcile([fact], [original, conflicting])[fact.execution_id]
    assert link.attribution is Attribution.AMBIGUOUS
    assert link.reasons == ("request_conflict",)


def test_conflicting_journal_phases_never_gain_unique_attribution():
    policy.enable_collection("personal-local")
    captured()
    events = journal_reader.read_journal("personal-local", quarantine=False).events
    started = next(row for row in events if row.phase is Phase.STARTED)
    folded = fold_events(events + [replace(started, fingerprint="0" * 64)])
    (fact,) = folded
    link = reconcile(folded, [request(fact)])[fact.execution_id]
    assert link.attribution is Attribution.UNATTRIBUTED
    assert "journal_conflict" in link.reasons


@pytest.mark.parametrize("client_argument", [False, True])
def test_client_cwd_argument_does_not_replace_native_workspace(tmp_path, client_argument):
    from tests.accounting.reduce_builders import (
        CWD,
        MCP,
        claude_prompt,
        claude_tool_result,
        claude_tool_use,
        event,
        results_json,
        ts,
    )
    from tests.accounting.test_reduce_reconcile import run

    arguments = {"query": "deploy failure"}
    if client_argument:
        arguments["cwd"] = "/forged-org"
    records = [
        claude_prompt("s1", "investigate", ts=ts(12, 9)),
        claude_tool_use("s1", "tool", MCP, arguments, ts=ts(12, 10), cwd=CWD),
        claude_tool_result("s1", "tool", results_json(), ts=ts(12, 10, 0, 3)),
    ]
    events = [
        replace(row, cwd=None, scope_basis="unknown_scope")
        for row in event("one", ts(12, 10, 0, 1))
    ]
    reduction = run(tmp_path, records, events)
    (row,) = reduction.executions
    if client_argument:
        # cwd is not part of the actual MCP tool schema. Such a transcript
        # cannot manufacture a valid lookup, let alone organization authority.
        assert reduction.invocations == ()
        assert reduction.coverage["inv_unclassified"] == 1
        assert row.attribution is Attribution.UNATTRIBUTED
        assert row.cwd is None
        assert "unknown_scope" in row.join_rejection_reasons
    else:
        assert row.attribution is Attribution.INFERRED_UNIQUE
        assert row.cwd == CWD


@pytest.mark.parametrize("authorized", [True, False])
def test_unknown_org_journal_requires_same_profile_native_workspace(tmp_path, authorized):
    from ssgrep.accounting.frequency import FrequencyView
    from ssgrep.accounting.policy_rules import PolicyFile
    from ssgrep.accounting.reduce import reduce_profile
    from tests.accounting.reduce_builders import (
        KEY,
        MCP,
        claude_evidence,
        claude_prompt,
        claude_tool_result,
        claude_tool_use,
        event,
        results_json,
        ts,
    )

    root_a, root_b = str(tmp_path / "org-a"), str(tmp_path / "org-b")
    profiles = {
        "org-a": ProfilePolicy(profile_id="org-a", roots=(root_a,), collection_enabled=True),
        "org-b": ProfilePolicy(profile_id="org-b", roots=(root_b,), collection_enabled=True),
    }
    native_cwd = root_a if authorized else root_b
    records = [
        claude_prompt("s1", "org controls", ts=ts(12, 9)),
        claude_tool_use(
            "s1", "tool", MCP, {"query": "deploy failure"}, ts=ts(12, 10), cwd=native_cwd
        ),
        claude_tool_result("s1", "tool", results_json(), ts=ts(12, 10, 0, 3)),
    ]
    evidence = claude_evidence(tmp_path, {"-work/s1.jsonl": records})
    events = [
        replace(
            row, cwd=None, scope_basis="unknown_scope", fingerprint_key_id=fingerprint.key_id(KEY)
        )
        for row in event("one", ts(12, 10, 0, 1), profile="org-a")
    ]
    reduction = reduce_profile(
        "org-a",
        evidence,
        events,
        policy=profiles["org-a"],
        frequency=FrequencyView.unknown(),
        fingerprint_key=KEY,
        ownership=PolicyFile(profiles=profiles),
    )
    if authorized:
        (row,) = reduction.executions
        assert row.in_scope and row.cwd == root_a
        assert row.attribution is Attribution.INFERRED_UNIQUE
        assert row.scope_basis == "native_call_metadata"
        from ssgrep.accounting.join_scope import effective_scoped_journal
        from ssgrep.accounting.scope_events import Scoper

        scoped = effective_scoped_journal(
            events, reduction.executions, Scoper(profiles["org-a"], PolicyFile(profiles=profiles))
        )
        assert len(scoped) == len(events)
        assert all(
            item.cwd == root_a and item.scope_basis == "native_call_metadata" for item in scoped
        )
        assert all(item.cwd is None and item.scope_basis == "unknown_scope" for item in events)
    else:
        assert not any(row.in_scope for row in reduction.executions)
        assert reduction.candidates == ()


def test_trusted_explicit_correlation_can_replace_clock_but_not_key_or_scope():
    policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    fact = replace(fact, correlation_id="one-local-call", timing_known=False)
    native = request(fact, correlation="one-local-call", window=None)
    link = reconcile([fact], [native, native])[fact.execution_id]
    assert link.attribution is Attribution.EXPLICIT
    assert link.requests == (native.key,)
    bad = replace(native, fingerprint_key_id="rotated-key")
    rejected = reconcile([fact], [bad])[fact.execution_id]
    assert rejected.attribution is Attribution.UNATTRIBUTED
    assert "key_mismatch" in rejected.reasons


@pytest.mark.parametrize(
    ("visible_refs", "expected"),
    [
        (frozenset({"source-a:ep:0"}), Attribution.EXPLICIT),
        (frozenset({"source-b:ep:0"}), Attribution.UNATTRIBUTED),
        (None, Attribution.EXPLICIT),
    ],
)
def test_explicit_correlation_cannot_override_complete_result_contradiction(visible_refs, expected):
    policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    fact = replace(
        fact,
        correlation_id="one-local-call",
        timing_known=False,
        refs=("source-a:ep:0",),
        refs_complete=True,
        outcome=Outcome.OK,
    )
    native = request(fact, correlation="one-local-call", window=None, result_refs=visible_refs)
    link = reconcile([fact], [native])[fact.execution_id]
    assert link.attribution is expected
    if expected is Attribution.UNATTRIBUTED:
        assert link.requests == ()
        assert "result_mismatch" in link.reasons
    else:
        assert link.requests == (native.key,)


def test_cwd_and_timing_without_query_or_result_are_not_join_evidence():
    policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    native = request(fact, fingerprint=None)
    link = reconcile([fact], [native])[fact.execution_id]
    assert link.attribution is Attribution.UNATTRIBUTED
    assert "insufficient_evidence" in link.reasons


@pytest.mark.parametrize("profile", ["personal-local", "org-a"])
def test_unjoined_unknown_journal_scope_is_never_company_counter_authority(profile, tmp_path):
    from ssgrep.accounting.join_scope import effective_scoped_journal
    from ssgrep.accounting.scope_events import Scoper
    from tests.accounting.reduce_builders import event, ts

    selected = ProfilePolicy(
        profile_id=profile,
        roots=() if profile == "personal-local" else (str(tmp_path / "org"),),
        collection_enabled=True,
    )
    events = [
        replace(item, cwd=None, scope_basis="unknown_scope")
        for item in event("unjoined", ts(12, 10), profile=profile)
    ]
    projected = effective_scoped_journal(events, (), Scoper(selected))
    assert len(projected) == (len(events) if profile == "personal-local" else 0)
    assert all(item.cwd is None and item.scope_basis == "unknown_scope" for item in projected)


@pytest.mark.parametrize(
    ("correlation", "expected"),
    [
        ("recorded-call", Attribution.EXPLICIT),
        ("different-call", Attribution.UNATTRIBUTED),
        (None, Attribution.INFERRED_UNIQUE),
    ],
)
def test_known_correlation_conflict_is_not_an_inference_candidate(correlation, expected):
    policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    fact = replace(fact, correlation_id="recorded-call")
    native = request(fact, correlation=correlation)
    link = reconcile([fact], [native])[fact.execution_id]
    assert link.attribution is expected
    if expected is Attribution.UNATTRIBUTED:
        assert not link.requests
        assert "correlation_mismatch" in link.reasons
    else:
        assert link.requests == (native.key,)


def test_workspace_alias_retargeting_is_checked_fresh_between_reconciliations(tmp_path):
    policy.enable_collection("personal-local")
    captured()
    (fact,) = facts()
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    alias = tmp_path / "workspace-alias"
    alias.symlink_to(first, target_is_directory=True)
    fact = replace(fact, cwd=str(first.resolve()), scope_basis="process_cwd")
    native = request(fact, cwd=str(alias))
    assert reconcile([fact], [native])[fact.execution_id].attribution is Attribution.INFERRED_UNIQUE
    alias.unlink()
    alias.symlink_to(second, target_is_directory=True)
    link = reconcile([fact], [native])[fact.execution_id]
    assert link.attribution is Attribution.UNATTRIBUTED and not link.requests
    assert "cwd_mismatch" in link.reasons
    alias.unlink()
    alias.symlink_to(first, target_is_directory=True)
    assert reconcile([fact], [native])[fact.execution_id].attribution is Attribution.INFERRED_UNIQUE
