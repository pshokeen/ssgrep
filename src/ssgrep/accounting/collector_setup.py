"""Pre-read native identity cutover: never project mixed legacy and current IDs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_types import CollectResult, CycleReport

if TYPE_CHECKING:
    from ssgrep.accounting.collector_cycle import CycleContext, CycleOutcome
    from ssgrep.accounting.identity_migration import MigrationResult
    from ssgrep.accounting.journal_reader import JournalRead
    from ssgrep.accounting.ledger_contracts import ProfilePolicy


def prepare_identity(
    ctx: CycleContext,
    checkpoint: store.Checkpoint,
    policy: ProfilePolicy,
    journal: JournalRead,
) -> tuple[MigrationResult, CycleOutcome | None]:
    from ssgrep.accounting.identity_migration import prepare

    migration = prepare(
        ctx.profile,
        checkpoint,
        ctx.now,
        env=ctx.env,
        policy=policy,
        journal=journal,
        byte_budget=ctx.options.budget.bytes,
    )
    if migration.complete:
        return migration, None
    # Persist bounded migration progress, but do not import/read/reduce new inputs.
    store.commit_checkpoint(ctx.profile, checkpoint)
    reason = (
        "identity_migration_unavailable"
        if migration.unavailable_sessions
        else "identity_migration_pending"
    )
    report = CycleReport(
        status="failed",
        generation=checkpoint.generation,
        snapshot_id=None,
        complete=False,
        stopped_by="identity_migration",
        bytes_read=migration.bytes_read,
        error=reason,
    )
    from ssgrep.accounting.collector_cycle import CycleOutcome

    return migration, CycleOutcome(
        report,
        CollectResult(
            state="partial",
            mode=ctx.options.mode,
            report=report,
            warnings=migration.warnings,
        ),
    )


def prepare_evidence(
    ctx: CycleContext, checkpoint: store.Checkpoint
) -> tuple[store.Checkpoint, CycleOutcome | None]:
    """One privacy-compaction quantum; never continue into new-native work when pending."""
    from ssgrep.accounting.evidence_migration import migrate_evidence

    checkpoint = migrate_evidence(ctx.profile, checkpoint, byte_budget=ctx.options.budget.bytes)
    if not checkpoint.retention.get("evidence_migration_pending"):
        return checkpoint, None
    report = CycleReport(
        status="failed",
        generation=checkpoint.generation,
        snapshot_id=None,
        complete=False,
        stopped_by="evidence_migration",
        error="evidence_migration_pending",
    )
    from ssgrep.accounting.collector_cycle import CycleOutcome

    return checkpoint, CycleOutcome(
        report,
        CollectResult(
            state="partial",
            mode=ctx.options.mode,
            report=report,
            warnings=("evidence_migration_pending",),
        ),
    )
