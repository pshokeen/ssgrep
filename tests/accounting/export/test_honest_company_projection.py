"""22.3/27.1: honest units/uncertainty through actual HTTP and OTLP bytes."""

import copy
import json
from dataclasses import replace

import pytest
from prometheus_client.openmetrics.parser import text_string_to_metric_families

from ssgrep.accounting.collect.frequency import FrequencyView
from ssgrep.accounting.collect.replay import Ledger
from ssgrep.accounting.export import (
    otlp,
    projection as export_build,
    records as export_records,
    registry as export_registry,
)
from ssgrep.accounting.export.otlp import Point, Resource
from ssgrep.accounting.export.receiver import Key, Outcome, ReceiverState, apply
from ssgrep.accounting.export.registry import Entry, Registry, State
from ssgrep.accounting.ledger.ledger_contracts import Signal
from ssgrep.accounting.ledger.reduce import methodology_for
from ssgrep.accounting.serve import metrics
from ssgrep.accounting.serve.metrics_render import render_openmetrics
from ssgrep.accounting.views import snapshots
from ssgrep.accounting.views.history import window_estimates
from tests.accounting.export import delivery_fixtures as fx, http_builders as http
from tests.accounting.export.lab_receiver import LabReceiver, Respond
from tests.accounting.views.test_lookup_time import time_bundle

KEY = bytes(range(32))
DEST = "org-primary"
METHOD = methodology_for(FrequencyView.unknown())


def rows(missing=True, milliseconds=2000):
    ledger = Ledger(METHOD)
    ledger.ingest(time_bundle(missing=missing, milliseconds=milliseconds))
    return ledger.rows()


def envelope(generation=1, missing=True, milliseconds=2000):
    window = {
        "estimates": window_estimates(rows(missing, milliseconds), start_day=None, end_day=None),
        "observations": {
            "measured_service_ms": {
                "value": milliseconds,
                "unit": "work_ms",
                "basis": "observed",
                "method": "monotonic_v1",
                "reason": "",
            },
            "measured_service_ms_known": {
                "value": milliseconds,
                "unit": "work_ms",
                "basis": "observed",
                "method": "monotonic_v1",
                "reason": "",
            },
            "measured_service_missing": {
                "value": 0,
                "unit": "count",
                "basis": "observed",
                "method": "timing_coverage",
                "reason": "",
            },
        },
    }
    return http.make_snapshot(generation, window=window, methodology_id=METHOD.hash)


def matching(snapshot, family, **wanted):
    return [
        s
        for s in metrics.snapshot_samples(snapshot)
        if s.family == family and all(dict(s.labels).get(k) == v for k, v in wanted.items())
    ]


def test_unknown_net_has_known_subtotal_and_no_finite_lower_bound_in_openmetrics():
    snapshot = envelope()
    assert matching(snapshot, "ssgrep_estimated_net_work_seconds") == []
    known = matching(snapshot, "ssgrep_known_net_work_seconds", scenario="conservative")
    assert len(known) == 1 and known[0].value == 10
    assert dict(known[0].labels)["method"] == "capped_gap_v1"
    assert (
        matching(snapshot, "ssgrep_estimate_bound_work_seconds", quantity="net", bound="lower")
        == []
    )
    upper = matching(snapshot, "ssgrep_estimate_bound_work_seconds", quantity="net", bound="upper")
    assert sorted(s.value for s in upper) == [10, 20, 30]
    families = list(
        text_string_to_metric_families(
            render_openmetrics(metrics.snapshot_samples(snapshot)).decode()
        )
    )
    assert families and not any("human" in f.name for f in families)


def test_method_labels_use_compatible_unit_not_changing_global_estimate_hash():
    first = envelope(missing=False)
    second = replace(first, methodology_id="a" * 32)
    for scenario in second.windows["7d"]["estimates"]["work_seconds"].values():
        for part in ("credit", "debit", "net"):
            scenario[part]["method"] = "estimate:" + "b" * 32
    labels = [
        dict(s.labels)["method"]
        for s in metrics.snapshot_samples(second)
        if s.family == "ssgrep_estimated_net_work_seconds"
    ]
    assert labels == ["capped_gap_v1"] * 3
    assert not any(
        "b" * 32 in value for s in metrics.snapshot_samples(second) for _, value in s.labels
    )


def test_incompatible_quantity_unit_never_becomes_a_token_net():
    snapshot = envelope(missing=False)
    entry = snapshot.windows["7d"]["estimates"]["context_tokens"]["conservative"]
    entry["net"]["unit"] = "work_seconds"
    assert matching(snapshot, "ssgrep_estimated_net_tokens", scenario="conservative") == []


def test_measured_latency_is_a_separate_observation_not_a_time_net():
    snapshot = envelope()
    observed = matching(snapshot, "ssgrep_measured_service_latency_milliseconds")
    assert len(observed) == 1 and observed[0].value == 2000
    assert dict(observed[0].labels)["method"] == "monotonic_v1"
    assert matching(snapshot, "ssgrep_estimated_net_work_seconds") == []


def test_authenticated_loopback_endpoint_preserves_json_uncertainty_and_metric_missingness():
    fx.configure("http://127.0.0.1:4318", enabled=())
    snapshot = envelope()
    snapshots.publish(http.PROFILE, snapshot)
    with http.start() as server:
        assert http.get(server, "/v1/snapshot", token=False).status == 401
        assert http.get(server, "/v1/snapshot", token="wrong-token").status == 401
        response = http.get(server, "/v1/snapshot")
        assert response.status == 200
        estimate = response.json()["windows"]["7d"]["estimates"]["work_seconds"]["conservative"]
        assert estimate["net"]["value"] is None and estimate["known_net"] == 10
        assert estimate["bounds"]["net_lower"]["value"] is None
        assert estimate["unit"]["method"] == "capped_gap_v1"
        body = http.get(server, "/metrics").body.decode()
        assert body.endswith("# EOF\n") and "ssgrep_known_net_work_seconds" in body
        assert "ssgrep_estimated_net_work_seconds{" not in body


def test_otlp_loopback_delivery_preserves_known_bounds_and_distinct_ack_generations():
    with LabReceiver.start() as receiver:
        fx.configure(receiver.url)
        snapshots.publish(fx.PROFILE, envelope(1))
        receiver.queue(Respond(503), path="/v1/metrics")
        first = fx.cycle()
        assert first.sent == 1 and first.acked == 0
        state = fx.state()
        assert state.last_collected_generation == state.last_attempted_generation == 1
        assert state.last_acked_generation is None
        second = fx.cycle(seconds=400)
        assert second.acked == 1
        assert receiver.requests[0].body == receiver.requests[1].body
        wire = json.loads(receiver.requests[1].body)["resourceMetrics"][0]["scopeMetrics"][0][
            "metrics"
        ]
        names = {item["name"] for item in wire}
        assert "ssgrep_known_net_work_seconds" in names
        assert "ssgrep_estimated_net_work_seconds" not in names
        assert fx.state().last_acked_generation == 1
        snapshots.publish(fx.PROFILE, envelope(2, missing=False))
        assert fx.cycle(seconds=800).acked == 1
        assert fx.state().last_collected_generation == fx.state().last_acked_generation == 2


def test_export_authorization_scope_and_pseudonym_keys_are_independent():
    with LabReceiver.start() as receiver:
        fx.configure(receiver.url, enabled=())
        snapshot = envelope()
        snapshots.publish(fx.PROFILE, snapshot)
        assert fx.cycle().sent == 0 and receiver.requests == []
        fx.set_enabled(Signal.METRICS, True)
        assert fx.cycle(seconds=1).acked == 1
        old = envelope(2)
        fx.amend(roots=("/approved/new-root",), policy_version=2)
        snapshots.publish(fx.PROFILE, old)
        assert fx.cycle(seconds=2).sent == 0
        assert len(receiver.requests) == 1
    one, _ = export_records.project_candidates(rows(), key=KEY, destination=DEST)
    other, _ = export_records.project_candidates(rows(), key=KEY[::-1], destination=DEST)
    tenant, _ = export_records.project_candidates(rows(), key=KEY, destination="other-org")
    assert {c.record_id for c in one}.isdisjoint(c.record_id for c in other)
    assert {c.record_id for c in one}.isdisjoint(c.record_id for c in tenant)


def test_log_projection_carries_sanitized_bounds_and_unknowns_without_private_ids():
    candidates, omitted = export_records.project_candidates(rows(), key=KEY, destination=DEST)
    assert omitted == 0
    candidate = next(c for c in candidates if c.unit[0] == "work_seconds")
    scenario = candidate.core["scenarios"]["conservative"]
    assert scenario["known_net"] == 10 and not scenario["complete"]
    assert scenario["net"]["value"] is None
    assert scenario["bounds"]["net_lower"]["value"] is None
    assert scenario["debit_missing"] == {"unspecified": 1}
    assert "service-1" not in json.dumps(candidate.core) and "claude:consumer" not in json.dumps(
        candidate.core
    )
    planned = export_registry.plan_revisions(Registry(), (candidate,))
    key = Key("authenticated-tenant", "authenticated-device", candidate.record_id)
    state, outcome = apply(ReceiverState(), key, planned[0].body)
    assert outcome is Outcome.APPLIED_NEW
    assert state.current[key].body["scenarios"]["conservative"]["known_net"] == 10
    hostile = copy.deepcopy(planned[0].body)
    hostile["scenarios"]["conservative"]["complete"] = True
    assert apply(ReceiverState(), key, hostile)[1] is Outcome.REJECTED_INVALID
    hostile = copy.deepcopy(planned[0].body)
    hostile["tenant"] = "payload-asserted-tenant"
    assert apply(ReceiverState(), key, hostile)[1] is Outcome.REJECTED_INVALID


def test_namespace_replacement_retires_old_method_record_even_when_queued_unacked():
    candidates, _ = export_records.project_candidates(
        rows(missing=False), key=KEY, destination=DEST
    )
    current = next(c for c in candidates if c.unit[0] == "work_seconds")
    old_id = "f" * 32
    old = Entry(
        current.task_id,
        current.root_key,
        current.period,
        ("work_seconds", "monotonic_v1", None, None),
        "e" * 32,
        queued=(State(3, "a" * 64, "b" * 64, "adjustment", "queued-old"),),
    )
    planned = export_registry.plan_revisions(
        Registry({old_id: old}), (current,), current_namespace=METHOD.hash
    )
    assert {p.record_type for p in planned} == {"contribution", "deletion"}
    deletion = next(p for p in planned if p.record_type == "deletion")
    assert deletion.record_id == old_id and deletion.revision == 4 and deletion.supersedes == 3
    key = Key("tenant", "device", old_id)
    old_body = {
        **deletion.body,
        "revision": 3,
        "supersedes_revision": 2,
        "record_type": "adjustment",
        "scenarios": current.core["scenarios"],
        "known_costs": current.core["known_costs"],
        "coverage": current.core["coverage"],
    }
    for sequence in ((old_body, deletion.body, old_body), (deletion.body, old_body, deletion.body)):
        state = ReceiverState()
        for body in sequence:
            state, _ = apply(state, key, body)
        assert state.current == {} and state.tombstones[key] == 4


def test_duplicate_reordered_points_are_deduplicated_and_negative_monotonic_sum_rejected():
    point = Point(
        "known_net", "gauge", "s", "Known subtotal", (("method", "capped_gap_v1"),), 10, -7
    )
    resource = Resource("test", "a" * 32, 1)
    assert otlp.split_metrics(resource, (point, point)) == otlp.split_metrics(resource, (point,))
    bad = replace(point, kind="sum", start_unix_nano=1)
    with pytest.raises(ValueError):
        otlp.split_metrics(resource, (bad,))


def test_sealed_measured_latency_preserves_known_subtotal_and_unknown_coverage(tmp_path):
    from ssgrep.accounting.ledger.contracts import Quantity
    from ssgrep.accounting.views.history import SealedDay, _observations, seal_days
    from tests.accounting.ledger.reduce_builders import claude_evidence
    from tests.accounting.views.test_lookup_time import records, reduce_timing, timed_events

    evidence = claude_evidence(tmp_path, {"-work/consumer.jsonl": records()})
    observed = reduce_timing(tmp_path, evidence, timed_events()).executions[0]
    unavailable = replace(
        observed,
        execution_id="missing-second",
        measured_service_ns=Quantity.unavailable("work_ns", "no_journal_timing"),
    )
    sealed, _ = seal_days((observed, unavailable), before_day="2026-10-13", already={})
    restored = SealedDay.from_dict(sealed["2026-10-12"].to_dict())
    quantities = _observations((restored,))
    assert quantities["measured_service_ms"]["value"] is None
    assert quantities["measured_service_ms_known"]["value"] == 2000
    assert quantities["measured_service_missing"]["value"] == 1
    legacy = restored.to_dict()
    for field in ("measured_service_known_ns", "measured_service_missing"):
        legacy.pop(field)
    assert _observations((SealedDay.from_dict(legacy),))["measured_service_ms"]["value"] is None


def test_legacy_enabled_policy_requires_explicit_v2_field_reapproval_without_automatic_send():
    from ssgrep.accounting.export import consent as export_consent
    from ssgrep.accounting.export.consent import approval_reason
    from ssgrep.accounting.export.otlp import FIELDS_VERSION
    from ssgrep.accounting.privacy import policy

    with LabReceiver.start() as receiver:
        fx.configure(receiver.url, enabled=())
        current = policy.load_profile_policy(fx.PROFILE)
        policy.save_profile(replace(current, enabled_exports=((fx.DEST, "metrics"),)))
        snapshots.publish(fx.PROFILE, envelope())
        assert FIELDS_VERSION["metrics"] == "metrics-v2"
        assert approval_reason(fx.PROFILE, fx.DEST, Signal.METRICS) == "consent_reapproval_required"
        assert fx.cycle().sent == 0 and receiver.requests == []
        plan = export_consent.plan_enable(fx.PROFILE, fx.DEST, Signal.METRICS)
        assert plan.fields_version == "metrics-v2" and not plan.already_enabled
        assert any("subtotal" in field for field in plan.fields)
        assert export_consent.apply_enable(fx.PROFILE, fx.DEST, Signal.METRICS)
        assert receiver.requests == []  # confirmation only grants permission; it never sends
        assert fx.cycle(seconds=1).acked == 1


def test_pending_identity_migration_blocks_company_send_and_legacy_metrics():
    from ssgrep.accounting.collect import store as collector_store
    from ssgrep.accounting.privacy import policy

    with LabReceiver.start() as receiver:
        fx.configure(receiver.url)
        snapshot = envelope()
        snapshots.publish(fx.PROFILE, snapshot)
        checkpoint = collector_store.Checkpoint(
            fx.PROFILE,
            methodology=METHOD.hash,
            identity_migration={"target": "native_identity_v2", "status": "pending"},
        )
        collector_store.commit_checkpoint(fx.PROFILE, checkpoint)
        assert fx.cycle().sent == 0 and receiver.requests == []
        current = policy.load_profile_policy(fx.PROFILE)
        with pytest.raises(ValueError, match="identity_migration_pending"):
            export_build.build_metrics(fx.PROFILE, current.destinations[0], current, snapshot, KEY)
        with http.start() as server:
            denied = http.get(server, "/metrics")
            assert denied.status == 503 and denied.json()["error"] == "identity_migration_pending"
            legacy = http.get(server, "/v1/snapshot")
            assert legacy.status == 200
            assert legacy.headers["x-ssgrep-identity-migration"] == "pending"
            assert legacy.headers["x-ssgrep-view"] == "legacy"
            assert legacy.json()["snapshot_id"] == snapshot.snapshot_id
            assert legacy.headers["etag"] == f'"{snapshot.snapshot_id}"'
            checkpoint.identity_migration["status"] = "complete"
            checkpoint.identity_method = "native_identity_v2"
            collector_store.commit_checkpoint(fx.PROFILE, checkpoint)
            assert http.get(server, "/metrics").status == 200
        assert fx.cycle(seconds=1).acked == 1


def test_missing_duration_and_replay_coverage_do_not_manufacture_zero_samples():
    snapshot = envelope()
    snapshot.observations["counters"].pop("duration_unknown", None)
    snapshot.coverage.pop("replay_unavailable_sessions", None)
    snapshot.coverage.pop("frequency_revision_unavailable", None)
    coverage = matching(snapshot, "ssgrep_coverage_records")
    assert not any(dict(s.labels) == {"kind": "durations", "status": "unknown"} for s in coverage)
    assert not any(dict(s.labels).get("kind") == "replay" for s in coverage)
    snapshot.coverage["replay_unavailable_sessions"] = 2
    assert (
        matching(snapshot, "ssgrep_coverage_records", kind="replay", status="sessions_unavailable")[
            0
        ].value
        == 2
    )


def test_malformed_or_cross_target_field_approval_never_authorizes_v2():
    from ssgrep.accounting.export.consent import approval_reason

    with LabReceiver.start() as receiver:
        fx.configure(receiver.url)
        path = fx.profile_root() / "export" / fx.DEST / "consent-metrics.json"
        document = json.loads(path.read_text())
        document["schema"] = True
        path.write_text(json.dumps(document))
        assert approval_reason(fx.PROFILE, fx.DEST, fx.M) == "consent_reapproval_required"
        assert fx.cycle().sent == 0 and receiver.requests == []
        document["schema"] = 1
        document["destination"] = "other-org"
        path.write_text(json.dumps(document))
        assert approval_reason(fx.PROFILE, fx.DEST, fx.M) == "consent_reapproval_required"


def test_queued_retry_remains_held_until_field_version_is_explicitly_reapproved():
    from ssgrep.accounting.export import consent as export_consent

    with LabReceiver.start() as receiver:
        fx.configure(receiver.url)
        snapshots.publish(fx.PROFILE, envelope())
        receiver.queue(Respond(503), path="/v1/metrics")
        assert fx.cycle().sent == 1
        path = fx.profile_root() / "export" / fx.DEST / "consent-metrics.json"
        document = json.loads(path.read_text())
        document["fields_version"] = "metrics-v1"
        path.write_text(json.dumps(document))
        held = fx.cycle(seconds=400)
        assert held.sent == held.acked == 0 and held.state == "paused"
        assert len(receiver.requests) == 1 and fx.state().last_acked_generation is None
        verdict = export_consent.classify_cycle(held, fx.PROFILE)
        assert verdict.next_command is not None
        assert (
            verdict.condition == "consent_reapproval_required"
            and "savings export enable" in verdict.next_command
        )
        export_consent.apply_enable(fx.PROFILE, fx.DEST, fx.M)
        assert fx.cycle(seconds=401).acked == 1
        assert receiver.requests[0].body == receiver.requests[1].body


def test_conflicting_candidate_identity_is_never_selected_by_input_order():
    candidates, _ = export_records.project_candidates(rows(False), key=KEY, destination=DEST)
    first = candidates[0]
    second = replace(first, content_digest="a" * 64)
    for sequence in ((first, second), (second, first)):
        with pytest.raises(ValueError, match="conflicting_export_candidate_identity"):
            export_registry.plan_revisions(Registry(), sequence, current_namespace=METHOD.hash)


def test_negative_company_time_nets_preserve_sign_and_are_not_roi_claims():
    snapshot = envelope(missing=False, milliseconds=25000)
    nets = matching(snapshot, "ssgrep_estimated_net_work_seconds")
    assert [sample.value for sample in nets] == [-15, -5, 5]
    from ssgrep.accounting.export.projection import project_metrics

    wire = project_metrics(snapshot).points
    values = [point.value for point in wire if point.name == "ssgrep_estimated_net_work_seconds"]
    assert all(value is not None for value in values)
    assert sorted(value for value in values if value is not None) == [-15, -5, 5]
    assert all("human" not in point.name and "roi" not in point.name for point in wire)


def test_current_method_without_completed_proof_is_unavailable_not_automatically_repaired():
    from ssgrep.accounting.collect import store as collector_store
    from ssgrep.accounting.export import consent as export_consent

    with LabReceiver.start() as receiver:
        fx.configure(receiver.url)
        snapshots.publish(fx.PROFILE, envelope())
        checkpoint = collector_store.load_checkpoint(fx.PROFILE)
        assert checkpoint is not None
        assert checkpoint.identity_method == "native_identity_v2"
        checkpoint.identity_migration = {}
        collector_store.commit_checkpoint(fx.PROFILE, checkpoint)
        held = fx.cycle()
        assert held.sent == held.acked == 0 and receiver.requests == []
        verdict = export_consent.classify_cycle(held, fx.PROFILE)
        assert verdict.condition == "identity_migration_unavailable"
        assert verdict.next_command is not None
        assert (
            "supported checkpoint" in verdict.message and "savings status" in verdict.next_command
        )
        assert "savings collect" not in verdict.next_command
        assert "savings resume" not in verdict.next_command
