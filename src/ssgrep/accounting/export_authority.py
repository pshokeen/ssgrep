"""Read-only company preflight: native identity migration and explicit field approval."""

from dataclasses import dataclass

from ssgrep.accounting import collector_store
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.export_consent_fields import approval_reason
from ssgrep.accounting.ledger_contracts import Signal

NATIVE_IDENTITY = "native_identity_v2"
AUTHORITY_HOLD_REASONS = frozenset(
    {
        "identity_migration_required",
        "identity_migration_pending",
        "identity_migration_unavailable",
        "consent_reapproval_required",
        "consent_state_unavailable",
    }
)


@dataclass(frozen=True)
class MigrationView:
    state: str  # current | legacy | pending | unavailable
    reason: str | None = None


def migration_of(checkpoint) -> MigrationView:
    if checkpoint is None:
        return MigrationView("legacy", "identity_migration_required")
    marker = checkpoint.identity_migration
    if not isinstance(marker, dict):
        return MigrationView("unavailable", "identity_migration_unavailable")
    if not marker and checkpoint.identity_method != NATIVE_IDENTITY:
        return MigrationView("legacy", "identity_migration_required")
    if marker.get("target") != NATIVE_IDENTITY:
        return MigrationView("unavailable", "identity_migration_unavailable")
    status = marker.get("status")
    if status == "complete":
        if checkpoint.identity_method != NATIVE_IDENTITY:
            return MigrationView("unavailable", "identity_migration_unavailable")
        return MigrationView("current")
    if status == "pending":
        return MigrationView("pending", "identity_migration_pending")
    return MigrationView("unavailable", "identity_migration_unavailable")


def migration_view(profile: str) -> MigrationView:
    try:
        return migration_of(collector_store.load_checkpoint(profile))
    except (CollectError, OSError, ValueError, TypeError):
        return MigrationView("unavailable", "identity_migration_unavailable")


def authority_reason(profile: str, destination: str, signal: Signal) -> str | None:
    migration = migration_view(profile)
    return migration.reason or approval_reason(profile, destination, signal)
