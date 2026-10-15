"""Stage immutable cycle state and bounded-maintenance authority before acceptance."""

from ssgrep.accounting import (
    collector_expiry as expiry,
    collector_published_retention as published_retention,
)
from ssgrep.accounting.collector_frequency_gc import stage as stage_frequency_gc
from ssgrep.accounting.collector_journal import save_journal_cache
from ssgrep.accounting.collector_partitions import save_partitions
from ssgrep.accounting.collector_root_cache import save_cache
from ssgrep.accounting.collector_state import save_state
from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.journal_retention import persist_journal_retention, prepare_journal_retention
from ssgrep.accounting.retention_hooks import aggregate_horizon_day


def stage(
    profile,
    checkpoint,
    *,
    state,
    prior_state,
    methodology,
    clock,
    deadline,
    affected_roots,
    changed,
    partitions,
    ledger,
    expiry_out,
    journal,
    previous_journal_retention,
    horizon,
    now,
    policy,
    evidence_job,
    today,
    rows,
    maintenance_budget,
):
    publication_work = published_retention.prepare(
        profile,
        checkpoint,
        rows=rows,
        before_day=max(
            aggregate_horizon_day(now, policy), expiry_out.coverage.aggregates_expired_before or ""
        ),
        evidence_before_day=horizon,
        clock=clock,
        deadline=deadline,
        byte_budget=maintenance_budget,
    )
    blobs = dict(checkpoint.blobs)
    save_state(
        profile,
        state,
        methodology.hash,
        blobs,
        previous=prior_state,
        clock=clock,
        deadline=deadline,
        dirty_roots=affected_roots,
    )
    if changed or "partitions" not in blobs:
        save_partitions(profile, partitions, blobs)
    if changed or expiry_out.changed or "root-evaluations" not in blobs:
        save_cache(profile, ledger, blobs)
    retention = {
        **checkpoint.retention,
        **expiry.retention_record(
            expiry_out.coverage,
            expiry_out.sessions,
            journal=prepare_journal_retention(
                journal,
                previous=previous_journal_retention,
                horizon_day=horizon,
                aggregate_horizon_day=aggregate_horizon_day(now, policy),
            ),
        ),
        "evidence_maintenance": evidence_job,
    }
    persisted = persist_journal_retention(profile, retention["journal"], blobs=blobs)
    save_journal_cache(
        profile, journal, blobs, retention["journal"], retention_tag=digest_of(persisted)
    )
    retention["journal"] = persisted
    retention["root_state_migration"] = {"pending_roots": state.root_migration_pending}
    frequency_pending = stage_frequency_gc(
        profile,
        checkpoint,
        blobs,
        current_refs={row.view_id for row in state.frequency_assignments},
        before_day=aggregate_horizon_day(now, policy),
        today=today,
    )
    return blobs, retention, frequency_pending, publication_work
