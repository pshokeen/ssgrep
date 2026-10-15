"""Permanent affected-closure regressions, with real native input positive controls."""

from __future__ import annotations

import os
from pathlib import Path

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_types import CycleBudget
from tests.accounting import collector_scene as scene
from tests.accounting.reduce_builders import PROFILE


def seed(root: Path) -> dict[str, Path]:
    scene.install_key()
    from ssgrep.accounting.policy import enable_collection

    enable_collection(PROFILE)
    scene.journal_events(scene.start_marker())
    scene.collect(
        scene.at(hour=8, minute=5)
    )  # Establish live watermarks before native input exists.
    paths = {
        name: scene.write_transcript(
            root,
            name,
            scene.search_records(name, name + "-first", "query-" + name, (10, n, 0)),
            scene.at(),
        )
        for n, name in enumerate(("left", "right"))
    }
    first = scene.collect(scene.at())
    assert scene.report(first).new_evidence > 0
    assert len(current_ledger()[1].reduction.invocations) == 2
    return paths


def test_unchanged_cycle_never_hydrates_retained_native_content(tmp_path, monkeypatch):
    from ssgrep.accounting.evidence_replay import EvidenceReplay

    seed(tmp_path)
    calls = []
    original = EvidenceReplay.hydrate_many

    def observed(self, items):
        items = list(items)
        calls.extend((item.runtime, item.session_id) for item in items)
        return original(self, items)

    monkeypatch.setattr(EvidenceReplay, "hydrate_many", observed)
    result = scene.collect(scene.at(minute=1))
    assert scene.report(result).complete
    assert calls == []
    assert len(current_ledger()[1].reduction.invocations) == 2


def test_one_native_source_changes_only_its_affected_root(tmp_path, monkeypatch):
    from ssgrep.accounting.evidence_replay import EvidenceReplay

    paths = seed(tmp_path)
    calls = []
    original = EvidenceReplay.hydrate_many

    def observed(self, items):
        items = list(items)
        calls.extend(item.session_id for item in items)
        return original(self, items)

    monkeypatch.setattr(EvidenceReplay, "hydrate_many", observed)
    scene.append_transcript(
        paths["left"],
        scene.search_records("left", "left-next", "different-query", (11, 0, 0)),
        scene.at(hour=11),
    )
    result = scene.collect(scene.at(hour=11))
    assert scene.report(result).new_evidence > 0
    assert calls and set(calls) == {"left"}
    assert len(current_ledger()[1].reduction.invocations) == 3


def test_budgeted_reduction_backlog_is_durable_and_never_complete(tmp_path):
    from ssgrep.accounting.collector_partitions import load_partitions

    paths = seed(tmp_path)
    scene.append_transcript(
        paths["left"],
        scene.search_records("left", "left-next", "different-query", (11, 0, 0)),
        scene.at(hour=11),
    )
    result = scene.collect(scene.at(hour=11), budget=CycleBudget(seconds=0))
    assert not scene.report(result).complete
    assert result.state == "partial"
    # Reader backlog or a durable partition queue must explain every omitted update.
    checkpoint = scene.committed()
    partitions = load_partitions(PROFILE, checkpoint)
    assert partitions.pending or any(source.more for source in checkpoint.sources.values())
    resumed = scene.collect(scene.at(hour=11, minute=1))
    assert scene.report(resumed).complete
    assert len(current_ledger()[1].reduction.invocations) == 3


def test_targeted_compact_reads_do_not_decode_unrelated_sessions(tmp_path):
    from tests.accounting.conftest import build_evidence

    left = build_evidence(session_id="left", origin_domain="left", record_id="l")
    right = build_evidence(session_id="right", origin_domain="right", record_id="r")
    refs = store.write_segments(PROFILE, [left, right])
    selected = list(store.iter_evidence(PROFILE, refs, sessions={("claude", "left")}))
    assert [item.session_id for item in selected] == ["left"]
    selected = list(store.iter_evidence(PROFILE, refs, native_keys={right.native_key}))
    assert [item.native_key for item in selected] == [right.native_key]


def test_partition_blob_references_survive_crash_recovery(tmp_path):
    from ssgrep.accounting.collector_partitions import load_partitions

    seed(tmp_path)
    checkpoint = scene.committed()
    before = load_partitions(PROFILE, checkpoint)
    store.sweep_orphans(PROFILE, checkpoint)
    after = load_partitions(PROFILE, checkpoint)
    assert before == after
    assert after.parts


def test_unavailable_frequency_probe_uses_typed_state_not_integer_checkpoint():
    from ssgrep.accounting.collector_state import State, load_state, save_state
    from ssgrep.accounting.frequency import FrequencyView

    probe = FrequencyView.unknown("index_absent")
    checkpoint = store.Checkpoint(PROFILE, frequency={"documents": 0})
    save_state(PROFILE, State(frequency_probe=probe), "test-namespace", checkpoint.blobs)
    restored = store.Checkpoint.from_dict(checkpoint.to_dict())
    assert restored.frequency == {"documents": 0}
    assert load_state(PROFILE, restored, "test-namespace").frequency_probe == probe


def current_ledger():
    from ssgrep.accounting.collector_ledger import VersionedLedger
    from ssgrep.accounting.collector_state import load_state

    checkpoint = scene.committed()
    state = load_state(PROFILE, checkpoint, checkpoint.methodology)
    assert state.ledger is not None
    return VersionedLedger.restore_versioned(state.ledger, state.superseded, state.archives), state


def assert_full_replay_equal():
    from ssgrep.accounting.collector_frequency import load_pinned, load_view
    from ssgrep.accounting.collector_inputs import Retained, load_fingerprint_key
    from ssgrep.accounting.collector_ledger import VersionedLedger
    from ssgrep.accounting.fingerprint import key_id
    from ssgrep.accounting.frequency import FrequencyView
    from ssgrep.accounting.frequency_assignment import FrequencyAssignments
    from ssgrep.accounting.journal_reader import read_journal
    from ssgrep.accounting.policy import read_policy_file
    from ssgrep.accounting.reduce import reduce_profile

    ledger, state = current_ledger()
    checkpoint = scene.committed()
    view = state.frequency_probe or load_pinned(PROFILE)
    assert isinstance(view, FrequencyView)
    key = load_fingerprint_key(PROFILE)
    policy = read_policy_file()
    helper = FrequencyAssignments(
        view,
        assignments=state.frequency_assignments,
        loader=lambda view_id: load_view(PROFILE, view_id),
        fingerprint_key_id=key_id(key),
    )
    reduced = reduce_profile(
        PROFILE,
        Retained(PROFILE, checkpoint.segments).items(),
        read_journal(PROFILE).events,
        policy=policy.profiles[PROFILE],
        ownership=policy,
        frequency=view,
        fingerprint_key=key,
        frequency_assignments=helper,
    )
    reference = VersionedLedger(ledger.methodology)
    reference.reconcile(
        dict(zip(reduced.bundle_roots, reduced.bundles, strict=True)),
        today=scene.at().date().isoformat(),
        exhaustive=True,
    )
    assert reference.rows().canonical_bytes() == ledger.rows().canonical_bytes()


def test_structured_result_refs_and_real_join_survive_partition_restart(tmp_path):
    from dataclasses import replace

    from ssgrep.accounting.contracts import Attribution
    from ssgrep.accounting.fingerprint import key_id
    from ssgrep.accounting.policy import enable_collection
    from tests.accounting.reduce_builders import (
        CWD,
        KEY,
        card,
        claude_tool_result,
        claude_tool_use,
        results_json,
        source_session,
        ts,
    )

    enable_collection(PROFILE)
    scene.install_key()
    scene.journal_events(scene.start_marker())
    scene.collect(scene.at(hour=8, minute=5))
    ref = "claude:witness:ep:0"
    scene.write_transcript(
        tmp_path, "witness", source_session("witness", "E_TRACE_UNIQUE", when_day=12), scene.at()
    )
    scene.write_transcript(
        tmp_path,
        "consumer",
        [
            claude_tool_use(
                "consumer",
                "lookup",
                "Bash",
                {"command": "SSGREP_CORRELATION_ID=cor-refs ssgrep search deployment"},
                ts=ts(12, 11),
                cwd=CWD,
            ),
            {
                **claude_tool_result(
                    "consumer",
                    "lookup",
                    results_json(card(ref, "E_TRACE_UNIQUE")),
                    ts=ts(12, 11, 0, 2),
                ),
                "cwd": CWD,
            },
        ],
        scene.at(hour=11),
    )
    scene.journal_events(
        replace(event, fingerprint_key_id=key_id(KEY), scope_basis="process_cwd")
        for event in scene.execution(
            "e-refs",
            ts(12, 11),
            text="deployment",
            transport="cli",
            refs=(ref,),
            correlation="cor-refs",
        )
    )
    first = scene.collect(scene.at(hour=11, minute=1))
    assert scene.report(first).complete
    before, state = current_ledger()
    assert any(
        row.execution_id == "e-refs" and row.attribution is Attribution.EXPLICIT
        for row in state.reduction.executions
    )
    scene.collect(scene.at(hour=11, minute=2))  # Reload actual durable state, not a codec proxy.
    after, restored = current_ledger()
    assert after.rows().canonical_bytes() == before.rows().canonical_bytes()
    assert restored.reduction.invocations == state.reduction.invocations
    assert_full_replay_equal()


def test_source_witness_change_invalidates_its_dependent_partition(tmp_path, monkeypatch):
    from ssgrep.accounting.evidence_replay import EvidenceReplay
    from ssgrep.accounting.policy import enable_collection
    from tests.accounting.reduce_builders import (
        card,
        claude_tool_result,
        claude_tool_use,
        results_json,
        source_session,
        ts,
    )

    enable_collection(PROFILE)
    scene.install_key()
    scene.journal_events(scene.start_marker())
    scene.collect(scene.at(hour=8, minute=5))
    source = scene.write_transcript(
        tmp_path, "witness", source_session("witness", "E_SOURCE_VALUE", when_day=12), scene.at()
    )
    scene.write_transcript(
        tmp_path,
        "dependent",
        [
            claude_tool_use(
                "dependent",
                "read",
                "mcp__acme_recall__search_sessions",
                {"query": "deployment"},
                ts=ts(12, 11),
            ),
            claude_tool_result(
                "dependent",
                "read",
                results_json(card("claude:witness:ep:0", "E_SOURCE_VALUE")),
                ts=ts(12, 11, 0, 2),
            ),
        ],
        scene.at(hour=11),
    )
    scene.collect(scene.at(hour=11))
    read = []
    original = EvidenceReplay.hydrate_many

    def observed(self, items):
        items = list(items)
        read.extend(row.session_id for row in items)
        return original(self, items)

    monkeypatch.setattr(EvidenceReplay, "hydrate_many", observed)
    scene.append_transcript(
        source,
        [
            claude_tool_use(
                "witness", "tail", "Bash", {"command": "echo continue"}, ts=ts(12, 10, 1)
            )
        ],
        scene.at(hour=11, minute=1),
    )
    scene.collect(scene.at(hour=11, minute=1))
    assert {"witness", "dependent"} <= set(read)
    assert_full_replay_equal()


def test_copied_native_history_does_not_duplicate_and_restart_matches_full_replay(tmp_path):
    paths = seed(tmp_path)
    original, _ = current_ledger()
    copied = paths["left"].with_name("left-copy.jsonl")
    copied.write_bytes(paths["left"].read_bytes())
    scene.collect(scene.at(minute=1))
    actual, _ = current_ledger()
    assert actual.rows().canonical_bytes() == original.rows().canonical_bytes()
    assert_full_replay_equal()


def test_late_parent_merges_the_old_root_and_survives_restart(tmp_path):
    from ssgrep.accounting.policy import enable_collection
    from tests.accounting.readers_fixtures import write_records
    from tests.accounting.reduce_builders import (
        omp_header,
        omp_launch,
        omp_search,
        omp_user,
        results_json,
        ts,
    )

    enable_collection(PROFILE)
    scene.install_key()
    scene.journal_events(scene.start_marker())
    scene.collect(scene.at(hour=8, minute=5))
    parent = Path(os.environ["OMP_SESSIONS_DIR"]) / "parent-root.jsonl"
    child = parent.with_name("child-root.jsonl")
    write_records(
        child,
        [
            omp_header("child-root", ts(12, 10), parent=parent),
            *omp_search("lookup", "lookup", "deployment", results_json(), (12, 10, 0, 1)),
        ],
    )
    scene.collect(scene.at())
    write_records(
        parent,
        [
            omp_header("parent-root", ts(12, 9)),
            omp_user("turn", "delegate investigation", ts(12, 9)),
            *omp_launch("launch", "task", "child-root", ts(12, 9, 1), ts(12, 9, 1, 1)),
        ],
    )
    scene.collect(scene.at(minute=1))
    _, state = current_ledger()
    assert ("omp", "child-root", "omp:parent-root") in state.reduction.session_roots
    scene.collect(scene.at(minute=2))
    assert_full_replay_equal()


def test_clock_only_expiry_never_rehydrates_native_evidence(tmp_path, monkeypatch):
    from datetime import timedelta

    from ssgrep.accounting.evidence_replay import EvidenceReplay

    seed(tmp_path)
    read = []
    original = EvidenceReplay.hydrate_many

    def observed(self, items):
        items = list(items)
        read.extend(items)
        return original(self, items)

    monkeypatch.setattr(EvidenceReplay, "hydrate_many", observed)
    scene.collect(scene.at() + timedelta(days=31))
    scene.collect(scene.at() + timedelta(days=31, minutes=1))
    assert read == []
    retention = scene.current().coverage["retention"]
    assert isinstance(retention, dict)
    assert retention["evidence_expired_before"]


def test_legacy_candidate_projection_rebuilds_from_accepted_partitions_without_native_io(
    tmp_path, monkeypatch
):
    from ssgrep.accounting import collector_ledger
    from ssgrep.accounting.collector_partitions import load_partitions, save_partitions
    from ssgrep.accounting.contracts import Scenario
    from ssgrep.accounting.evidence_replay import EvidenceReplay
    from ssgrep.accounting.reduce_cands import CHARS
    from ssgrep.accounting.revisions import unit_key
    from ssgrep.accounting.valuation_inputs import Eligibility
    from tests.accounting import verify_scenes as vs
    from tests.accounting.reduce_builders import (
        MCP,
        card,
        claude_tool_result,
        claude_tool_use,
        results_json,
        ts,
    )
    from tests.accounting.test_ux_support import current_rarity_index

    current_rarity_index(tmp_path, monkeypatch)
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
    path = vs.projects(tmp_path) / "-work" / "cons0.jsonl"
    scene.append_transcript(
        path,
        [
            claude_tool_use("cons0", "repeat", MCP, {"query": vs.query(0)}, ts=ts(12, 10)),
            claude_tool_result(
                "cons0",
                "repeat",
                results_json(card("claude:src0:ep:0", vs.fact(0))),
                ts=ts(12, 10, 0, 2),
            ),
            claude_tool_use(
                "cons0",
                "repeat-action",
                "Bash",
                {"command": f"vim {vs.path(0)}"},
                ts=ts(12, 10, 0, 5),
            ),
            claude_tool_result("cons0", "repeat-action", "ok", ts=ts(12, 10, 0, 8)),
        ],
        scene.at(hour=10),
    )
    # Produce the historical collapse using its actual reconciliation key.
    with monkeypatch.context() as old_binary:
        old_binary.setattr(
            collector_ledger,
            "candidate_identity",
            lambda row: (row.task.root, row.task.turn or "", row.task.branch, row.candidate_id),
        )
        scene.collect(scene.at(hour=11))
    before, accepted = current_ledger()
    assert accepted.reduction is not None
    lookups = {row.call_id: row.invocation_id for row in accepted.reduction.invocations}
    first_lookup, repeat_lookup = lookups["t0a"], lookups["repeat"]
    held = before.state()
    assert held.candidates
    assert {row.exposure_ids[0] for row in held.candidates} <= {first_lookup, repeat_lookup}
    counters = scene.observations(scene.current())["counters"]
    immutable = {
        name: body for name, body in vs.snapshot_files().items() if name.startswith("gen-")
    }
    checkpoint = scene.committed()
    partitions = load_partitions(PROFILE, checkpoint)
    for part in partitions.parts.values():
        part.pop("candidate_identity_method", None)  # Real pre-cutover manifest shape.
    save_partitions(PROFILE, partitions, checkpoint.blobs)
    store.commit_checkpoint(PROFILE, checkpoint)

    def forbidden(*args, **kwargs):
        raise AssertionError("accepted partition reprojection must not hydrate native text")

    with monkeypatch.context() as blocked_native:
        blocked_native.setattr(EvidenceReplay, "hydrate_many", forbidden)
        stopped = scene.collect(scene.at(hour=11, minute=1), budget=CycleBudget(seconds=0))
        assert not scene.report(stopped).complete
        assert load_partitions(PROFILE, scene.committed()).pending_projections
        assert current_ledger()[0].state() == held
        resumed = scene.collect(scene.at(hour=11, minute=2))
        assert scene.report(resumed).complete
        assert scene.report(resumed).bytes_read == scene.report(resumed).new_evidence == 0
        after, _ = current_ledger()
        observations = {
            exposure: row for row in after.state().candidates for exposure in row.exposure_ids
        }
        assert set(observations) == {first_lookup, repeat_lookup}
        assert observations[first_lookup].eligibility is Eligibility.SUPPORTED
        assert observations[repeat_lookup].eligibility is Eligibility.KNOWN_REJECTED
        assert observations[repeat_lookup].eligibility_reasons == ("already_visible",)
        assert observations[first_lookup].candidate_id == observations[repeat_lookup].candidate_id
        assert (
            len({row.debit_id for row in after.state().debits if row.unit.unit == "context_tokens"})
            == 2
        )
        assert any(
            estimate.credit.value is not None and estimate.credit.value > 0
            for period in after.rows().periods
            for unit, estimates in period.estimates
            if unit == unit_key(CHARS)
            for estimate in estimates
            if estimate.scenario is Scenario.CENTRAL
        )
        assert scene.observations(scene.current())["counters"] == counters
        assert not load_partitions(PROFILE, scene.committed()).pending_projections
        frozen = after.rows().canonical_bytes()
        scene.collect(scene.at(hour=11, minute=3))
        assert current_ledger()[0].rows().canonical_bytes() == frozen
    assert_full_replay_equal()
    published = vs.snapshot_files()
    assert all(published[name] == body for name, body in immutable.items())
