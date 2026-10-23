"""Purge erases DATA payloads, not delivery truth or authorized deletion requests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting.export import (
    acks,
    ids as export_ids,
    otlp as export_http,
    projection as export_build,
    records as export_records,
    registry as export_registry,
    spool as export_spool,
)
from ssgrep.accounting.export.delivery import run_destination
from ssgrep.accounting.ledger.ledger_contracts import Signal
from ssgrep.accounting.privacy import purge, purge_spool, storage
from ssgrep.accounting.privacy.purge import PurgeError
from tests.accounting.export import delivery_fixtures as fx
from tests.accounting.export.lab_receiver import DropAfterRead, LabReceiver, Partial
from tests.accounting.ledger.reduce_builders import PROFILE
from tests.accounting.privacy.test_purge import (
    ALPHA,
    PURGE_AT,
    accounting_dir,
    build_scene,
    current_envelope,
    evidence_by_session,
    periods,
    select,
)

DEST = fx.DEST
STAMP = "2026-10-21T12:00:00Z"
LOG_SECRET = b"purge-settled-private-log-73c49e"
METRIC_SECRET = b"purge-settled-private-total-081cfa"


def _canary(body: bytes, value: bytes, signal: Signal) -> bytes:
    document = json.loads(body)
    resource = document["resourceLogs" if signal is Signal.LOGS else "resourceMetrics"][0]
    resource["resource"]["attributes"].append(
        {"key": "test.privacy_canary", "value": {"stringValue": value.decode()}}
    )
    return json.dumps(document, separators=(",", ":")).encode()


def _seed(tmp_path: Path, endpoint: str):
    build_scene(tmp_path)
    full = fx.configure(endpoint, signals=(fx.M, fx.L), logs_permitted=True, enabled=(fx.M, fx.L))
    envelope = current_envelope()
    key = export_ids.load_key(PROFILE, create=True)
    assert key is not None
    logs = export_build.build_logs(PROFILE, full.destinations[0], full, envelope, key)
    assert len(logs.batches) == 1
    original = logs.batches[0]
    members = logs.planned[0]
    roots = [export_records.root_key_of(f"claude:{session}") for session in ("a1", "b1", "a2")]
    batches = []
    for index, root in enumerate(roots):
        group = tuple(m for m in members if m.root_key == root)
        assert group
        document = json.loads(original.body)
        records = document["resourceLogs"][0]["scopeLogs"][0]["logRecords"]
        ids = {m.record_id for m in group}
        records[:] = [
            r for r in records if json.loads(r["body"]["stringValue"])["record_id"] in ids
        ]
        body = json.dumps(document, separators=(",", ":")).encode()
        if index == 0:
            body = _canary(body, LOG_SECRET, fx.L)
        batch = replace(
            original,
            batch_id=original.batch_id[:-4] + f"{index:04d}",
            index=index,
            total=len(roots),
            body=body,
            items=len(group),
            record_ids=tuple(sorted(ids)),
        )
        export_spool.write_batch(PROFILE, DEST, fx.L, batch, created_utc=STAMP, scope=logs.scope)
        export_registry.mark_queued(PROFILE, DEST, group, batch.batch_id)
        batches.append(batch)
    metrics = export_build.build_metrics(PROFILE, full.destinations[0], full, envelope, key)
    assert len(metrics.batches) == 1
    metric = replace(metrics.batches[0], body=_canary(metrics.batches[0].body, METRIC_SECRET, fx.M))
    export_spool.write_batch(PROFILE, DEST, fx.M, metric, created_utc=STAMP, scope=metrics.scope)
    return batches, metric


def _send(signal: Signal, **kwargs):
    return run_destination(PROFILE, DEST, signal, now=PURGE_AT, enqueue=False, **kwargs)


def _body_path(batch) -> Path:
    return accounting_dir() / "spool" / DEST / batch.signal.value / f"{batch.batch_id}.json"


def _local_bytes() -> bytes:
    return b"\n".join(p.read_bytes() for p in accounting_dir().rglob("*") if p.is_file())


def _history(signal: Signal):
    state = acks.read_delivery(PROFILE, DEST, signal)
    assert state is not None
    return (
        state.last_acked_generation,
        state.last_acked_at,
        state.partial_batches,
        state.rejected_points,
        state.identity_unavailable,
        state.lost_records,
        state.loss_summary,
        state.progress,
    )


def _assert_deleted_and_scoped(batches, metric, *, settled: str) -> None:
    selected, *unrelated = batches
    assert not _body_path(selected).exists()
    assert not _body_path(metric).exists()
    local = _local_bytes()
    assert LOG_SECRET not in local and METRIC_SECRET not in local
    for batch in unrelated:
        assert _body_path(batch).read_bytes() == batch.body
    assert periods() == {"2026-10-06", "2026-10-15"}
    evidence = evidence_by_session()
    assert "a1" not in evidence and "src-a1" not in evidence
    assert "b1" in evidence and "a2" in evidence
    registry = export_registry.load_registry(PROFILE, DEST)
    for rid in selected.record_ids:
        entry = registry.entries[rid]
        assert getattr(entry, settled) is not None
        assert entry.latest is not None and entry.latest.type == "deletion"
    tombstones = [
        m for m in export_spool.list_batches(PROFILE, DEST, fx.L) if m.kind == "tombstone"
    ]
    assert len(tombstones) == 1 and tombstones[0].state == "queued"
    assert set(tombstones[0].record_ids) == set(selected.record_ids)
    assert b'"scenarios":{}' in export_spool.read_body(
        PROFILE, DEST, fx.L, tombstones[0].batch_id
    ).replace(b'\\"', b'"')


@pytest.mark.parametrize("partial", [False, True], ids=["full-ack", "partial-ack"])
def test_real_receiver_settled_bodies_are_erased_without_rewriting_history(
    tmp_path: Path, partial: bool
) -> None:
    with LabReceiver.start() as lab:
        batches, metric = _seed(tmp_path, lab.url)
        if partial:
            lab.queue(Partial(1, kind="logs"), path="/v1/logs")
            lab.queue(Partial(1, kind="metrics"), path="/v1/metrics")
        logs, metrics = _send(fx.L, deps=fx.deps()), _send(fx.M, deps=fx.deps())
        assert logs.sent == 3 and metrics.sent == 1
        assert logs.partial == metrics.partial == int(partial)
        assert LOG_SECRET in lab.requests[0].body and METRIC_SECRET in lab.requests[-1].body
        history = {signal: _history(signal) for signal in (fx.L, fx.M)}
        other = accounting_dir().parent / "other-profile" / "proof.json"
        other.parent.mkdir(mode=0o700)
        other.write_bytes(b"unrelated-profile-proof")
        before_other = other.read_bytes()
        preview = purge.dry_run(select(workspace=ALPHA))
        assert preview.local_deleted.spool_batches == 2
        report = purge.execute(select(workspace=ALPHA), now=PURGE_AT)
        assert report.local_deleted.spool_batches == 2
        _assert_deleted_and_scoped(batches, metric, settled="unknown" if partial else "acked")
        assert {signal: _history(signal) for signal in (fx.L, fx.M)} == history
        assert other.read_bytes() == before_other
        again = purge.execute(select(workspace=ALPHA), now=PURGE_AT)
        assert again.local_deleted.spool_batches == 0
        _assert_deleted_and_scoped(batches, metric, settled="unknown" if partial else "acked")
        assert len(lab.requests) == 4  # purge only queues remote deletion; no erasure claim


@pytest.mark.parametrize("boundary", ["body", "meta"])
@pytest.mark.parametrize("delivered", [False, True], ids=["queued", "acked"])
def test_restart_finishes_marked_purge_before_body_or_meta_unlink_without_double_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str, delivered: bool
) -> None:
    with LabReceiver.start() as lab:
        batches, metric = _seed(tmp_path, lab.url)
        if delivered:
            assert _send(fx.L, deps=fx.deps()).acked == 3
            assert _send(fx.M, deps=fx.deps()).acked == 1
        target = batches[0]
        original_delete = storage.delete_file
        suffix = ".json" if boundary == "body" else ".meta.json"

        def crash(fd: int, name: str):
            if name == target.batch_id + suffix:
                raise OSError("injected purge interruption")
            return original_delete(fd, name)

        with monkeypatch.context() as scoped:
            scoped.setattr(storage, "delete_file", crash)
            with pytest.raises(PurgeError):
                purge.execute(select(workspace=ALPHA), now=PURGE_AT)
        marked = export_spool.read_meta(PROFILE, DEST, fx.L, target.batch_id)
        assert marked is not None and marked.purge_requested
        loss_before = acks.read_delivery(PROFILE, DEST, fx.L)
        assert loss_before is not None
        script = (
            "from tests.accounting.privacy.test_purge import select, ALPHA, PURGE_AT; "
            "from ssgrep.accounting.privacy import purge; "
            "purge.execute(select(workspace=ALPHA), now=PURGE_AT)"
        )
        restarted = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            env=os.environ.copy(),
            timeout=60,
            check=False,
        )
        assert restarted.returncode == 0, restarted.stderr.decode()
        assert LOG_SECRET not in _local_bytes() and METRIC_SECRET not in _local_bytes()
        assert not _body_path(target).exists() and not _body_path(metric).exists()
        loss_after = acks.read_delivery(PROFILE, DEST, fx.L)
        assert loss_after is not None and loss_after.lost_records == loss_before.lost_records
        assert loss_after.lost_records == (0 if delivered else target.items)
        assert purge.execute(select(workspace=ALPHA), now=PURGE_AT).local_deleted.spool_batches == 0
        for batch in batches[1:]:
            assert _body_path(batch).read_bytes() == batch.body


@pytest.mark.parametrize("response", ["full", "partial", "disconnect"])
@pytest.mark.parametrize("start_after_plan", [False, True], ids=["already-sending", "racing-plan"])
def test_purge_during_real_send_keeps_tombstone_and_erases_body_at_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: str, start_after_plan: bool
) -> None:
    with LabReceiver.start() as lab:
        batches, _ = _seed(tmp_path, lab.url)
        if response == "partial":
            lab.queue(Partial(1, kind="logs"))
        elif response == "disconnect":
            lab.queue(DropAfterRead())
        received, release = threading.Event(), threading.Event()
        results = []

        def held_post(*args, **kwargs):
            outcome = export_http.post(*args, **kwargs)
            received.set()
            assert release.wait(20)
            return outcome

        thread = threading.Thread(
            target=lambda: results.append(_send(fx.L, deps=fx.deps(post=held_post, max_batches=1)))
        )
        if start_after_plan:
            real_apply = purge_spool.apply_spool

            def start_then_apply(profile, plan, stamp):
                thread.start()
                assert received.wait(20)
                return real_apply(profile, plan, stamp)

            monkeypatch.setattr(purge_spool, "apply_spool", start_then_apply)
        else:
            thread.start()
            assert received.wait(20)
        try:
            report = purge.execute(select(workspace=ALPHA), now=PURGE_AT)
            assert report.retained.in_flight_batches_not_recallable == 1
            assert LOG_SECRET in lab.requests[0].body
            marked = export_spool.read_meta(PROFILE, DEST, fx.L, batches[0].batch_id)
            assert marked is not None and marked.purge_requested
            assert any(
                m.kind == "tombstone" for m in export_spool.list_batches(PROFILE, DEST, fx.L)
            )
        finally:
            release.set()
            thread.join(20)
        assert not thread.is_alive() and results[0].state != "failed"
        assert not _body_path(batches[0]).exists() and LOG_SECRET not in _local_bytes()
        state = acks.read_delivery(PROFILE, DEST, fx.L)
        assert state is not None and state.lost_records == 0
        assert state.in_flight_nonrecallable == batches[0].batch_id
        _send(fx.L, deps=fx.deps())
        assert sum(LOG_SECRET in request.body for request in lab.requests) == 1


def test_crashed_in_flight_request_is_purged_without_resending_it(tmp_path: Path) -> None:
    with LabReceiver.start() as lab:
        batches, _ = _seed(tmp_path, lab.url)
        script = (
            "import os; from ssgrep.accounting.export import otlp as export_http; "
            "from tests.accounting.privacy.test_purge_settled_privacy import _send, fx; "
            "real = export_http.post; "
            "post = lambda *a, **k: (real(*a, **k), os._exit(19))[0]; "
            "_send(fx.L, deps=fx.deps(post=post))"
        )
        crashed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            env=os.environ.copy(),
            timeout=60,
            check=False,
        )
        assert crashed.returncode == 19
        assert LOG_SECRET in lab.requests[0].body
        purge.execute(select(workspace=ALPHA), now=PURGE_AT)
        assert not _body_path(batches[0]).exists() and LOG_SECRET not in _local_bytes()
        entry = export_registry.load_registry(PROFILE, DEST).entries[batches[0].record_ids[0]]
        assert entry.unknown is not None and entry.latest is not None
        assert entry.latest.type == "deletion"
        _send(fx.L, deps=fx.deps())
        assert sum(LOG_SECRET in request.body for request in lab.requests) == 1


@pytest.mark.parametrize("state", ["quarantined", "lost", "discarded"])
def test_all_nonlive_data_states_are_privacy_selected(tmp_path: Path, state: str) -> None:
    with LabReceiver.start() as lab:
        batches, metric = _seed(tmp_path, lab.url)
        for batch in (batches[0], metric):
            export_spool.update_meta(
                PROFILE, DEST, batch.signal, batch.batch_id, lambda m: replace(m, state=state)
            )
        purge.execute(select(workspace=ALPHA), now=PURGE_AT)
        assert LOG_SECRET not in _local_bytes() and METRIC_SECRET not in _local_bytes()
        for batch in batches[1:]:
            assert _body_path(batch).read_bytes() == batch.body
        losses = acks.read_delivery(PROFILE, DEST, fx.L)
        assert losses is not None
        assert losses.lost_records == (batches[0].items if state == "quarantined" else 0)


def test_marked_batch_is_cleaned_on_delivery_restart_without_a_resend(tmp_path: Path) -> None:
    with LabReceiver.start() as lab:
        batches, _ = _seed(tmp_path, lab.url)
        target = batches[0]
        export_spool.update_meta(
            PROFILE, DEST, fx.L, target.batch_id, lambda m: replace(m, purge_requested=True)
        )
        with pytest.raises(ValueError):
            export_spool.update_meta(
                PROFILE,
                DEST,
                fx.L,
                target.batch_id,
                lambda m: replace(m, purge_requested=False),
            )
        _send(fx.L, deps=fx.deps())
        assert not _body_path(target).exists()
        assert all(LOG_SECRET not in request.body for request in lab.requests)
        assert len(lab.requests) == 2  # positive proof: the unrelated tasks still send
