"""Collection authority, not first-send timing, authorizes both exported signals."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.accounting import capture_policy, collector_store, policy
from ssgrep.accounting.collector_types import CollectOptions
from ssgrep.accounting.delivery_candidate import preview_candidate
from ssgrep.accounting.frequency import FrequencyView
from ssgrep.accounting.reduce import reduce_profile
from ssgrep.accounting.scope_events import Scoper
from ssgrep.accounting.valuation_inputs import AtomStatus
from tests.accounting import collector_scene as scene, delivery_fixtures as fx
from tests.accounting.delivery_fixtures import DEST, PROFILE, L, M
from tests.accounting.lab_receiver import LabReceiver
from tests.accounting.reduce_builders import MCP, claude_evidence, claude_tool_use, ts


@pytest.mark.parametrize("signal", [M, L])
def test_first_send_after_narrowing_without_version_bump_is_refused(tmp_path, signal):
    with LabReceiver.start() as lab:
        roots = (str(tmp_path / "a"), str(tmp_path / "b"))
        fx.configure(lab.url, roots=roots, signals=(M, L), logs_permitted=True, enabled=())
        fx.commit_ledger(1)
        fx.publish(1)
        fx.amend(roots=roots[:1])
        fx.set_enabled(signal, True)
        assert preview_candidate(PROFILE, DEST, signal, now=fx.at()).state == "ineligible_scope"
        result = fx.cycle(signal)
        assert result.sent == 0
        assert lab.requests == []


@pytest.mark.parametrize("signal", [M, L])
@pytest.mark.parametrize(
    "bad",
    [
        None,
        {},
        {"schema": 1, "profile": PROFILE, "digest": "bad"},
        {"schema": 1, "profile": "other", "digest": "0" * 64},
    ],
)
def test_missing_or_malformed_snapshot_provenance_refuses_export(signal, bad):
    from ssgrep.accounting import snapshots
    from ssgrep.accounting.delivery_candidate import current_envelope

    with LabReceiver.start() as lab:
        current = fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=(M, L))
        fx.commit_ledger(1)
        fx.publish(1)
        envelope = current_envelope(PROFILE)
        assert envelope is not None
        coverage = dict(envelope.coverage)
        coverage.pop("collection_scope", None)
        if bad is not None:
            coverage["collection_scope"] = bad
        fields = {
            k: v
            for k, v in vars(envelope).items()
            if k not in {"schema_version", "snapshot_id", "generation", "coverage"}
        }
        snapshots.publish(
            PROFILE, snapshots.build_envelope(generation=2, coverage=coverage, **fields)
        )
        assert current.policy_version == 1
        assert fx.cycle(signal).sent == 0
        assert lab.requests == []


@pytest.mark.parametrize("signal", [M, L])
def test_unchanged_scope_and_consent_only_enable_keep_snapshot_eligible(signal):
    with LabReceiver.start() as lab:
        fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=())
        fx.commit_ledger(1)
        fx.publish(1)
        fx.amend_destination(credential_ref="env:ROTATED_TEST_INGEST")
        fx.set_enabled(signal, True)
        result = fx.cycle(signal, deps=fx.deps(environ={"ROTATED_TEST_INGEST": fx.SECRET}))
        assert result.sent > 0
        assert len(lab.requests) > 0


@pytest.mark.parametrize("signal", [M, L])
@pytest.mark.parametrize("bad_state", [None, [], {"schema": 1, "digest": 42}])
def test_corrupt_previous_scope_requires_recollection(signal, bad_state):
    from ssgrep.accounting import export_files

    with LabReceiver.start() as lab:
        fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=(M, L))
        fx.commit_ledger(1)
        fx.publish(1)
        with export_files.locked(PROFILE, ("export", DEST), f"scope-{signal.value}.json") as fd:
            assert fd is not None
            export_files.write_json(fd, f"scope-{signal.value}.json", bad_state)
        assert preview_candidate(PROFILE, DEST, signal, now=fx.at()).state == "ineligible_scope"
        assert fx.cycle(signal).sent == 0
        assert lab.requests == []
        fx.publish(2, collected_at=fx.stamp(fx.at(1)))
        assert fx.cycle(signal, seconds=2).sent > 0


def test_explicit_assignment_excludes_other_organization_capture_raw_and_source(tmp_path: Path):
    root = str(tmp_path / "shared")
    a = policy.enable_collection("org-a", {"roots": [root], "assignments": {root: "org-a"}})
    b = policy.enable_collection("org-b", {"roots": [root], "assignments": {root: "org-a"}})
    assert capture_policy.capture_profile(root) == "org-a"
    assert policy.path_in_scope(a, root)
    assert not policy.path_in_scope(b, root)
    assert Scoper(a).source_status(root)[0] is AtomStatus.OK
    assert Scoper(b).source_status(root)[0] is AtomStatus.OUT_OF_SCOPE
    evidence = claude_evidence(
        tmp_path,
        {
            "shared/s.jsonl": [
                claude_tool_use("s", "lookup", MCP, {"query": "fixture"}, cwd=root, ts=ts())
            ]
        },
    )
    allowed = reduce_profile("org-a", evidence, [], policy=a, frequency=FrequencyView.unknown())
    denied = reduce_profile("org-b", evidence, [], policy=b, frequency=FrequencyView.unknown())
    assert len(allowed.invocations) == 1
    assert denied.invocations == ()
    assert denied.coverage["invocations_out_of_scope"] == 1


def test_recollection_does_not_relabel_retained_old_counters_or_frozen_ledger(tmp_path):
    from ssgrep.accounting import collector_state, export_records
    from ssgrep.accounting.collector import collect_once
    from ssgrep.accounting.collector_ledger import VersionedLedger

    roots = (str(tmp_path / "a"), str(tmp_path / "b"))
    fx.configure("http://127.0.0.1:1", roots=roots, signals=(M, L), logs_permitted=True)
    scene.journal_events(scene.execution("from-b", ts(12, 9), cwd=roots[1]))
    collect_once(PROFILE, CollectOptions(), now=scene.at(12, 10))
    old = scene.current()
    assert scene.counter_total(old) == 1
    # Add frozen, no-longer-recomputable history to the existing broad scope namespace.
    checkpoint = scene.committed()
    fx.commit_ledger(1, generation=checkpoint.generation)
    synthetic = scene.committed()
    frozen_state = collector_state.load_state(PROFILE, synthetic, synthetic.methodology)
    assert frozen_state.ledger is not None
    ledger = VersionedLedger.restore_versioned(frozen_state.ledger, frozen_state.superseded)
    assert ledger.expire_roots(["root-1"]) == ["root-1"]
    frozen_state.ledger = ledger.state()
    collector_state.save_state(PROFILE, frozen_state, synthetic.methodology, synthetic.blobs)
    checkpoint.blobs.update({k: v for k, v in synthetic.blobs.items() if k.startswith("ledger-")})
    checkpoint.methodology = synthetic.methodology
    collector_store.commit_checkpoint(PROFILE, checkpoint)
    old_rows = export_records.ledger_rows(PROFILE)
    assert old_rows is not None and old_rows.periods
    fx.amend(roots=roots[:1])
    collect_once(PROFILE, CollectOptions(), now=scene.at(12, 11))
    fresh = scene.current()
    committed = scene.committed()
    assert fresh.counter_epoch > old.counter_epoch
    assert scene.counter_total(fresh) == 0
    assert fresh.coverage["collection_scope"] != old.coverage["collection_scope"]
    assert fresh.coverage["scope_history_unavailable"] is True
    assert any("-scope-" in name for name in committed.blobs if name.startswith("ledger-"))
    state = collector_state.load_state(PROFILE, committed, committed.methodology)
    assert state.ledger is not None and not state.ledger.frozen
    fresh_rows = export_records.ledger_rows(PROFILE)
    assert fresh_rows is not None and not fresh_rows.periods
    # Clock-only republishing cannot resurrect the private old epoch.
    collect_once(PROFILE, CollectOptions(), now=scene.at(12, 12))
    assert scene.counter_total(scene.current()) == 0
    assert scene.current().counter_epoch == fresh.counter_epoch


@pytest.mark.parametrize("signal", [M, L])
def test_corrupt_scope_history_also_blocks_already_queued_bytes(signal):
    from ssgrep.accounting import export_files
    from tests.accounting.lab_receiver import Respond

    with LabReceiver.start() as lab:
        fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=(M, L))
        fx.commit_ledger(1)
        fx.publish(1)
        lab.queue(Respond(503))
        assert fx.cycle(signal).sent == 1
        with export_files.locked(PROFILE, ("export", DEST), f"scope-{signal.value}.json") as fd:
            assert fd is not None
            export_files.write_json(fd, f"scope-{signal.value}.json", {"invalid": True})
        assert fx.cycle(signal, seconds=1000).sent == 0
        assert len(lab.requests) == 1


@pytest.mark.parametrize("signal", [M, L])
def test_assignment_change_invalidates_collection_digest_without_root_or_version_change(
    tmp_path, signal
):
    from ssgrep.accounting import export_policy
    from ssgrep.accounting.ledger_contracts import DestinationPolicy
    from tests.accounting.http_builders import make_snapshot

    root = str(tmp_path / "shared")
    a = policy.enable_collection("org-a", {"roots": [root], "assignments": {root: "org-a"}})
    policy.enable_collection("org-b", {"roots": [root], "assignments": {root: "org-a"}})
    old_scope = policy.collection_scope(a)
    base = make_snapshot(1)
    from dataclasses import replace

    envelope = replace(base, coverage={**base.coverage, "collection_scope": old_scope})
    policy.enable_collection("org-a", {"roots": [root], "assignments": {root: "org-b"}})
    destination = DestinationPolicy(DEST, "http://127.0.0.1:1", (M, L), logs_permitted=True)
    new_scope = export_policy.scope_document(a, destination, signal, "test-install")[
        "collection_scope"
    ]
    assert old_scope != new_scope
    assert not export_policy.snapshot_eligible(
        envelope, export_policy.ScopeState("", None, new_scope), a.policy_version
    )
    assert capture_policy.capture_profile(root) == "org-b"
    assert not Scoper(a).in_scope(root)


def test_logs_refuse_missing_committed_ledger_provenance():
    from ssgrep.accounting import collector_state

    with LabReceiver.start() as lab:
        fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=(L,))
        fx.commit_ledger(1)
        fx.publish(1)
        checkpoint = collector_store.load_checkpoint(PROFILE)
        assert checkpoint is not None
        name = collector_state.ledger_blob_name(checkpoint.methodology)
        raw = collector_store.read_blob(PROFILE, checkpoint.blobs[name])
        assert raw is not None
        raw.pop("collection_scope")
        checkpoint.blobs[name] = collector_store.write_blob(PROFILE, name, raw)
        collector_store.commit_checkpoint(PROFILE, checkpoint)
        assert fx.cycle(L).sent == 0
        assert lab.requests == []


def test_unchanged_real_collection_keeps_scope_and_counter_epoch():
    from ssgrep.accounting.collector import collect_once

    fx.configure("http://127.0.0.1:1")
    scene.journal_events(scene.execution("allowed", ts(12, 9)))
    collect_once(PROFILE, CollectOptions(), now=scene.at(12, 10))
    before = scene.current()
    collect_once(PROFILE, CollectOptions(), now=scene.at(12, 11))
    after = scene.current()
    assert after.coverage["collection_scope"] == before.coverage["collection_scope"]
    assert after.counter_epoch == before.counter_epoch
    assert scene.counter_total(after) == scene.counter_total(before) == 1
