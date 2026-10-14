"""Pre-dispatch revalidation (tasks 12.3 / 13.6): queued bytes are never sent on old authority.

Batches are queued by the REAL delivery code, the policy is then changed through the REAL policy
writer, and the delivery cycle runs again against a loopback lab receiver. The assertion that
matters is always the receiver's request count (zero for a stale batch) together with the state
files: unsent bytes deleted or reprojected, registry ``queued`` entries reset (logs), and an
explicit ``policy_contraction`` loss when data was discarded. Gate/scope helpers are also covered
directly where their contract is the thing being specified.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting import acks, export_policy, policy
from ssgrep.accounting.ledger_contracts import DestinationPolicy, ProfilePolicy
from tests.accounting import delivery_fixtures as fx
from tests.accounting.delivery_fixtures import DEST, PROFILE, L, M, cycle, deps
from tests.accounting.lab_receiver import LabReceiver, Respond


@pytest.fixture
def lab():
    with LabReceiver.start() as receiver:
        yield receiver


def queue_metrics(lab: LabReceiver, **config) -> None:
    """Leave one metrics batch queued (the receiver is unavailable), nothing acknowledged."""
    fx.configure(lab.url, **config)
    fx.publish(1)
    lab.queue(Respond(503))
    assert cycle().state == "waiting"
    assert [m.state for m in fx.metas()] == ["queued"] and len(lab.requests) == 1


def queue_logs(lab: LabReceiver, records: int = 3) -> None:
    fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=(L,))
    fx.commit_ledger(records)
    fx.publish(1)
    lab.queue(Respond(503))
    assert cycle(L).state == "waiting"
    assert [m.state for m in fx.metas(L)] == ["queued"] and len(lab.requests) == 1


def only_loss(signal=M) -> acks.Loss:
    state = fx.delivery(signal)
    assert state is not None
    assert [loss.reason for loss in state.loss_summary] == ["policy_contraction"]
    return state.loss_summary[0]


# --- the gate and scope documents, directly --------------------------------------------------


def test_gate_distinguishes_pause_disabled_revoked_and_policy_errors(lab: LabReceiver) -> None:
    fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=(M,))
    ok = export_policy.gate(PROFILE, DEST, M)
    assert ok.ok and ok.destination is not None and ok.destination.endpoint == lab.url
    disabled = export_policy.gate(PROFILE, DEST, L)
    assert (disabled.reason, disabled.revoked) == ("not_enabled", False)
    assert export_policy.gate(PROFILE, DEST, L, need_enabled=False).ok
    unknown = export_policy.gate(PROFILE, "elsewhere", M)
    assert (unknown.reason, unknown.revoked) == ("unknown_destination", True)
    policy.pause(PROFILE)
    paused = export_policy.gate(PROFILE, DEST, M)
    assert (paused.reason, paused.revoked) == ("export_paused", False)
    assert export_policy.gate(PROFILE, DEST, M, need_unpaused=False).ok
    policy.resume(PROFILE)
    assert export_policy.gate(PROFILE, DEST, M).ok


def test_resume_never_reenables_a_disabled_signal(lab: LabReceiver) -> None:
    fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=(M, L))
    fx.set_enabled(L, False)
    policy.pause(PROFILE)
    policy.resume(PROFILE)
    assert export_policy.gate(PROFILE, DEST, L).reason == "not_enabled"
    assert export_policy.gate(PROFILE, DEST, M).ok


def test_scope_documents_change_exactly_with_what_they_authorize(lab: LabReceiver) -> None:
    fx.configure(lab.url, signals=(M, L), logs_permitted=True, enabled=(M, L), roots=())
    base = export_policy.gate(PROFILE, DEST, M)
    assert base.destination is not None
    one = export_policy.scope_document(base.policy, base.destination, M, "inst-1")
    version = replace(base.policy, policy_version=2)
    two = export_policy.scope_document(version, base.destination, M, "inst-1")
    assert export_policy.scope_changes(one, two) == ("policy_version",)
    moved = replace(base.destination, endpoint="https://other.example.test")
    three = export_policy.scope_document(base.policy, moved, M, "inst-1")
    assert export_policy.scope_changes(one, three) == ("endpoint",)
    other = export_policy.scope_document(base.policy, base.destination, M, "inst-2")
    assert export_policy.scope_changes(one, other) == ("identity",)
    logs = export_policy.scope_document(base.policy, base.destination, L, "inst-1")
    assert logs["fields"] != one["fields"] and logs["logs_permitted"] is True
    assert one["logs_permitted"] is False  # metrics never inherit logs permission


def test_batch_current_judges_data_by_every_key_and_tombstones_by_the_deletion_keys() -> None:
    scope = {
        "roots": ["/a"],
        "endpoint": "https://e",
        "proxy": None,
        "fields": "logs-v1",
        "policy_version": 1,
        "logs_permitted": True,
        "identity": "i",
    }
    assert export_policy.batch_current(scope, scope)
    for key, value in (
        ("roots", ["/b"]),
        ("endpoint", "https://x"),
        ("proxy", "http://p"),
        ("fields", "logs-v2"),
        ("policy_version", 2),
        ("logs_permitted", False),
        ("identity", "j"),
    ):
        assert not export_policy.batch_current(scope, {**scope, key: value}), key
    # a narrowed root set or a new policy version must not block a request to DELETE exported data
    assert export_policy.batch_current(scope, {**scope, "roots": ["/b"]}, "tombstone")
    assert export_policy.batch_current(scope, {**scope, "policy_version": 9}, "tombstone")
    assert not export_policy.batch_current(scope, {**scope, "endpoint": "https://x"}, "tombstone")
    assert not export_policy.batch_current(scope, {**scope, "logs_permitted": False}, "tombstone")
    assert not export_policy.batch_current("not-a-scope", scope)


# --- contraction: a stale batch is NEVER sent -----------------------------------------------


def test_policy_version_change_discards_the_stale_batch_and_reprojects_nothing_stale(
    lab: LabReceiver,
) -> None:
    queue_metrics(lab)
    (old,) = fx.metas()
    fx.amend(policy_version=2)  # administrators bumped the policy: the old bytes are void
    result = cycle(seconds=1000)
    assert len(lab.requests) == 1  # THE assertion: nothing stale was dispatched
    assert result.invalidated_batches == 1 and result.sent == 0
    # the snapshot was collected under policy 1: it can never leave under policy 2
    assert fx.metas() == [] and not any(
        old.batch_id in str(p) for p in fx.profile_root().rglob("*.json")
    )
    loss = only_loss()
    assert (loss.batches, loss.records) == (1, old.items)
    state = fx.delivery()
    assert state is not None and state.last_acked_generation is None
    assert state.pending_batches == 0 and state.lost_records == old.items
    assert [p.discarded for p in state.progress] == [(0,)]  # the generation can never be acked


def test_a_fresh_snapshot_under_the_new_policy_is_delivered_instead(lab: LabReceiver) -> None:
    queue_metrics(lab)
    fx.amend(policy_version=2)
    cycle(seconds=1000)
    assert len(lab.requests) == 1
    fx.publish(2, collected_at=fx.stamp(fx.at(1500)))  # collected AFTER the change, under policy 2
    result = cycle(seconds=2000)
    assert result.state == "sent" and result.acked == 1 and len(lab.requests) == 2
    state = fx.delivery()
    assert state is not None and state.last_acked_generation == 2


def test_root_removal_blocks_old_data_until_a_newer_snapshot_exists(
    lab: LabReceiver, tmp_path: Path
) -> None:
    work = tmp_path / "team"
    other = tmp_path / "team-two"
    work.mkdir()
    other.mkdir()
    queue_metrics(lab, roots=(str(work), str(other)))
    (old,) = fx.metas()
    fx.amend(roots=(str(work),))  # a workspace was removed while the destination was offline
    result = cycle(seconds=1000)
    assert len(lab.requests) == 1 and result.invalidated_batches == 1
    assert fx.metas() == [] and only_loss().batches == 1
    assert old.scope["roots"] == sorted(os.path.realpath(r) for r in (work, other))
    fx.publish(2, collected_at=fx.stamp(fx.at(1500)))  # collected after the roots were narrowed
    after = cycle(seconds=2000)
    assert after.state == "sent" and len(lab.requests) == 2
    (fresh,) = [m for m in fx.metas() if m.generation == 2]
    assert fresh.scope["roots"] == [os.path.realpath(work)]


def test_a_snapshot_collected_before_the_scope_change_is_never_reprojected(
    lab: LabReceiver, tmp_path: Path
) -> None:
    work, other = tmp_path / "a", tmp_path / "b"
    work.mkdir()
    other.mkdir()
    queue_metrics(lab, roots=(str(work), str(other)))
    fx.amend(roots=(str(work),))
    cycle(seconds=1000)
    cycle(seconds=1400)  # even many cycles later, the same (older) snapshot stays blocked
    assert len(lab.requests) == 1 and fx.metas() == []
    assert cycle(seconds=1800).state == "idle"


def test_logs_permission_withdrawn_resets_queued_registry_entries_and_records_the_loss(
    lab: LabReceiver,
) -> None:
    queue_logs(lab, records=3)
    (old,) = fx.metas(L)
    assert all(e.queued for e in fx.registry().entries.values())
    current = policy.load_profile_policy(PROFILE)
    revoked = replace(current.destinations[0], logs_permitted=False, signals=(M,))
    fx.amend(destinations=(revoked,), enabled_exports=())
    result = cycle(L, seconds=1000)
    assert (result.state, result.reason) == ("revoked", "signal_not_permitted")
    assert len(lab.requests) == 1  # nothing was sent
    assert fx.metas(L) == [] and fx.registry().entries == {}  # unsent bytes + queued entries gone
    loss = only_loss(L)
    assert (loss.batches, loss.records) == (1, old.items)
    state = fx.delivery(L)
    assert state is not None and state.last_acked_generation is None and state.pending_batches == 0


def test_dropping_the_destination_from_policy_discards_its_queue_with_an_explicit_loss(
    lab: LabReceiver,
) -> None:
    queue_metrics(lab)
    other = DestinationPolicy("other-org", "http://127.0.0.1:1")
    fx.amend(destinations=(other,), enabled_exports=())
    result = cycle(seconds=1000)
    assert (result.state, result.reason) == ("revoked", "unknown_destination")
    assert len(lab.requests) == 1 and fx.metas() == []
    assert only_loss().batches == 1
    assert fx.state().last_acked_generation is None
    # resume / re-adding the destination cannot resurrect the discarded bytes
    fx.amend(
        destinations=(DestinationPolicy(DEST, lab.url), other), enabled_exports=((DEST, "metrics"),)
    )
    assert fx.metas() == []


def test_a_disabled_signal_keeps_its_bounded_queue_and_sends_nothing(lab: LabReceiver) -> None:
    queue_metrics(lab)
    fx.set_enabled(M, False)
    for step in (1, 2):
        result = cycle(seconds=step * 1000)
        assert (result.state, result.reason) == ("not_enabled", "not_enabled")
    assert len(lab.requests) == 1
    assert [m.state for m in fx.metas()] == ["queued"]  # preserved, not discarded
    state = fx.delivery()
    assert state is not None and state.loss_summary == () and state.pending_batches == 1
    # only an explicit re-enable (never resume) lets the SAME bytes go
    policy.pause(PROFILE)
    policy.resume(PROFILE)
    assert cycle(seconds=3000).state == "not_enabled" and len(lab.requests) == 1
    fx.set_enabled(M, True)
    assert cycle(seconds=4000).acked == 1 and len(lab.requests) == 2


def test_pause_holds_the_queue_without_any_write_or_request(lab: LabReceiver) -> None:
    queue_metrics(lab)
    policy.pause(PROFILE)
    before = fx.tree()
    result = cycle(seconds=1000)
    assert (result.state, result.reason) == ("paused", "export_paused")
    assert len(lab.requests) == 1 and fx.tree() == before  # spool/state untouched
    policy.resume(PROFILE)
    done = cycle(seconds=2000)
    assert done.acked == 1 and len(lab.requests) == 2  # the same queued bytes, after resume


def test_a_changed_endpoint_invalidates_bytes_queued_for_the_old_receiver(
    lab: LabReceiver,
) -> None:
    queue_metrics(lab)
    (old,) = fx.metas()
    assert old.scope["endpoint"] == lab.url
    with LabReceiver.start() as moved:
        fx.amend_destination(endpoint=moved.url)
        result = cycle(seconds=1000)
        assert len(lab.requests) == 1  # the old receiver never sees the old bytes again
        assert result.invalidated_batches == 1
        # the snapshot itself stays exportable (an endpoint change does not widen WHAT leaves):
        # it is rebuilt for the new target with the same batch identity, then delivered there
        assert result.enqueued_batches == 1 and result.acked == 1 and len(moved.requests) == 1
        (fresh,) = fx.metas()
        assert fresh.batch_id == old.batch_id and fresh.scope["endpoint"] == moved.url
        assert (fresh.state, fresh.attempts) == ("acked", 1)  # a NEW batch, not a retry
        assert moved.requests[0].sha256 == fresh.sha256
        state = fx.delivery()
        assert state is not None and state.last_acked_generation == 1
        assert state.loss_summary == ()  # reprojected, not lost


def test_policy_that_becomes_unreadable_holds_without_discarding(lab: LabReceiver) -> None:
    queue_metrics(lab)
    policy.policy_path().write_text("{ not json")
    result = cycle(seconds=1000)
    assert (result.state, result.reason) == ("paused", "policy_unreadable")
    assert len(lab.requests) == 1 and [m.state for m in fx.metas()] == ["queued"]
    assert fx.state().loss_summary == ()


def test_an_invalid_managed_update_pauses_sending_without_dropping_the_queue(
    lab: LabReceiver,
) -> None:
    queue_metrics(lab)
    ok, error = policy.apply_managed_update(PROFILE, {"roots": 42})
    assert not ok and error is not None
    result = cycle(seconds=1000)
    assert (result.state, result.reason) == ("paused", "export_paused")
    assert len(lab.requests) == 1 and [m.state for m in fx.metas()] == ["queued"]


# --- revocation racing a request that is already out ----------------------------------------


def test_revocation_with_a_request_in_flight_reports_it_as_nonrecallable(
    lab: LabReceiver,
) -> None:
    import threading

    fx.configure(lab.url)
    fx.publish(1)
    lab.queue(Respond(200, delay=1.0))
    out: list = []
    runner = threading.Thread(target=lambda: out.append(cycle()))
    runner.start()
    try:
        assert lab.wait_for_requests(1)  # the POST is in flight right now
        fx.set_enabled(M, False)  # consent revoked while it cannot be recalled
    finally:
        runner.join(timeout=10)
    (result,) = out
    assert result.nonrecallable_batch == fx.metas()[0].batch_id
    state = fx.delivery()
    assert state is not None and state.in_flight_nonrecallable == fx.metas()[0].batch_id
    assert len(lab.requests) == 1
    fx.publish(2)
    assert cycle(seconds=5000).state == "not_enabled" and len(lab.requests) == 1


def test_revocation_between_batches_stops_before_the_next_request(
    lab: LabReceiver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.split_small(monkeypatch)
    fx.configure(lab.url)
    fx.publish(1)
    total_batches = 0

    real_post = deps().post

    def revoke_after_first(*args, **kwargs):
        outcome = real_post(*args, **kwargs)
        nonlocal total_batches
        total_batches += 1
        if total_batches == 1:
            fx.set_enabled(M, False)  # consent withdrawn right after the first request returned
        return outcome

    result = cycle(deps=deps(post=revoke_after_first))
    assert len(lab.requests) == 1 and result.sent == 1
    assert (result.state, result.reason) == ("sent", None) or result.reason == "not_enabled"
    states = [m.state for m in fx.metas()]
    assert states[0] == "acked" and set(states[1:]) == {"queued"} and len(states) >= 3
    assert fx.state().last_acked_generation is None


def test_a_batch_whose_scope_changed_mid_cycle_is_not_sent(
    lab: LabReceiver, monkeypatch: pytest.MonkeyPatch
) -> None:
    fx.split_small(monkeypatch)
    fx.configure(lab.url)
    fx.publish(1)
    real_post = deps().post
    calls = 0

    def narrow_after_first(*args, **kwargs):
        nonlocal calls
        outcome = real_post(*args, **kwargs)
        calls += 1
        if calls == 1:
            fx.amend(policy_version=2)  # policy moved on between request 1 and request 2
        return outcome

    result = cycle(deps=deps(post=narrow_after_first))
    assert len(lab.requests) == 1  # the remaining (now stale) batches were never dispatched
    assert result.invalidated_batches >= 2
    assert only_loss().batches >= 2
    assert [m.state for m in fx.metas()] == ["acked"]  # only the already-delivered one remains
    assert fx.state().last_acked_generation is None


def test_policy_contraction_never_regresses_a_previous_acknowledgment(lab: LabReceiver) -> None:
    fx.configure(lab.url)
    fx.publish(1)
    assert cycle().acked == 1
    fx.publish(2)
    lab.queue(Respond(503))
    cycle(seconds=400)
    fx.amend(policy_version=2)
    cycle(seconds=2000)
    state = fx.delivery()
    assert state is not None and state.last_acked_generation == 1
    assert state.last_acked_snapshot_id is not None and only_loss().batches == 1
    resolved = acks.resolve_acknowledged(PROFILE, DEST, M)
    assert resolved.state == "acknowledged" and resolved.envelope is not None
    assert resolved.envelope.generation == 1


def test_an_unauthorized_batch_is_not_sent_even_by_a_direct_second_runner(
    lab: LabReceiver,
) -> None:
    queue_metrics(lab)
    fx.amend(policy_version=2)
    fx.run_driver("run", "metrics", "1000")  # a REAL separate process, current policy applies
    assert len(lab.requests) == 1 and fx.metas() == []


def test_profile_pause_flag_is_not_needed_for_preview_but_blocks_sends(lab: LabReceiver) -> None:
    from ssgrep.accounting.delivery import preview_candidate

    fx.configure(lab.url)
    fx.publish(1)
    policy.pause(PROFILE)
    previewed = preview_candidate(PROFILE, DEST, M)
    assert previewed.state == "ok" and previewed.candidate is not None
    assert cycle().state == "paused" and lab.requests == []
    assert policy.load_profile_policy(PROFILE).paused is True
    assert isinstance(policy.load_profile_policy(PROFILE), ProfilePolicy)
