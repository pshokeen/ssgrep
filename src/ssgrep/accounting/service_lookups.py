"""Read every accepted lookup, including no-receipt calls, without collection or writes.

Native invocations own their matched executions. Orphan journal lookups remain
separate records, never aliases. Pages are bound to one published snapshot and
committed evidence selection. Unavailable historical generations are explicitly
expired; no cursor silently starts reading a newer generation.
"""

from __future__ import annotations

from ssgrep.accounting.contracts import PERSONAL_PROFILE, InvocationClass, digest_of
from ssgrep.accounting.lookup_cursor import next_cursor, offset, reference
from ssgrep.accounting.lookup_projection import project_lookup
from ssgrep.accounting.lookup_types import LookupsView
from ssgrep.accounting.policy_rules import PolicyError, check_profile_id
from ssgrep.accounting.scope import check_limit
from ssgrep.accounting.service_ledger import load_lookup_data
from ssgrep.accounting.service_select import lookup_records
from ssgrep.accounting.service_types import Scope, command_for


def read_lookups(
    profile: str = PERSONAL_PROFILE,
    scope: Scope | None = None,
    *,
    snapshot_id: str | None = None,
    generation: int | None = None,
    limit: int = 20,
    cursor: str | None = None,
) -> LookupsView:
    from ssgrep.accounting.service import read_snapshot

    check_profile_id(profile)
    check_limit(limit)
    scope = scope or Scope()
    command = command_for(profile, "savings lookups")
    if scope.by is not None:
        raise PolicyError("invalid_filter", "Lookup pages do not group rows with --by.", command, 2)
    token = reference(cursor, command)
    if token is not None:
        if (
            snapshot_id is not None
            and snapshot_id != token["snapshot"]
            or generation is not None
            and generation != token["generation"]
        ):
            raise PolicyError(
                "invalid_cursor", "Cursor conflicts with the requested snapshot.", command, 2
            )
        snapshot_id, generation = token["snapshot"], token["generation"]
    snapshot, failure, why = read_snapshot(profile)
    if snapshot_id is not None or generation is not None:
        if (
            snapshot is None
            or (snapshot_id is not None and snapshot.snapshot_id != snapshot_id)
            or (generation is not None and snapshot.generation != generation)
        ):
            raise PolicyError(
                "lookup_snapshot_expired",
                "That lookup snapshot is no longer available. Start without --cursor/--snapshot.",
                command,
                3,
            )
    if snapshot is None:
        return LookupsView(
            profile, scope, limit=limit, reason=why or (failure.value if failure else "no_snapshot")
        )
    doc, committed_generation, ledger, reason = load_lookup_data(profile)
    if doc is None or committed_generation != snapshot.generation:
        if cursor is not None:
            raise PolicyError(
                "lookup_snapshot_expired",
                "Lookup evidence for that generation is unavailable. Start without --cursor.",
                command,
                3,
            )
        return LookupsView(
            profile,
            scope,
            snapshot_id=snapshot.snapshot_id,
            generation=snapshot.generation,
            limit=limit,
            reason=reason or "lookup_generation_pending",
        )
    selected = lookup_records(scope, doc, snapshot.collected_at)
    unique = {}
    for kind, identity, row in selected:
        held = unique.setdefault((kind, identity), row)
        if held != row:
            return LookupsView(
                profile,
                scope,
                snapshot_id=snapshot.snapshot_id,
                generation=snapshot.generation,
                limit=limit,
                reason="conflicting_lookup_identity",
            )
    selected = sorted(
        ((kind, identity, row) for (kind, identity), row in unique.items()),
        key=lambda item: (item[2].ts_utc or "", item[0], item[1]),
    )
    binding = digest_of(
        [
            profile,
            scope,
            limit,
            snapshot.snapshot_id,
            snapshot.generation,
            doc.to_dict(),
            ledger.rows.digest if ledger else reason,
        ]
    )
    start = offset(token, binding, command)
    if token is not None and start >= len(selected):
        raise PolicyError("invalid_cursor", "Cursor has no remaining lookup rows.", command, 2)
    rows = tuple(
        project_lookup(kind, identity, row, doc, ledger)
        for kind, identity, row in selected[start : start + limit]
    )
    # A receipt link is only offered when the actual published explanation exists.
    from dataclasses import replace

    from ssgrep.accounting.service_records import read_receipt

    visible = []
    for row in rows:
        links = []
        missing_links = False
        for receipt_id in row.receipt_ids:
            try:
                if read_receipt(profile, receipt_id) is not None:
                    links.append(receipt_id)
                else:
                    missing_links = True
            except PolicyError:
                missing_links = True
        missing = row.missing_evidence + (
            ("receipt_explanation_unavailable",) if missing_links else ()
        )
        visible.append(replace(row, receipt_ids=tuple(links), missing_evidence=missing))
    more = start + limit < len(selected)
    warnings = () if reason is None else ("valuation_unavailable:" + reason,)
    if doc.lookup_schema != "lookups_v1":
        warnings += ("lookup_metadata_unavailable",)
    untimed = sum(
        r.ts_utc is None
        for r in doc.invocations
        if r.cls
        in {InvocationClass.MATCHED, InvocationClass.TRANSCRIPT_ONLY, InvocationClass.JOURNAL_GAP}
        and set(r.operations) & {"search", "show"}
    )
    if untimed:
        warnings += (
            f"untimed_lookup_rows:{untimed} retained native calls have no timestamp; "
            "default pages retain them without known window membership; "
            "exact time filters exclude them.",
        )
    return LookupsView(
        profile,
        scope,
        tuple(visible),
        len(selected),
        next_cursor(start + limit, binding, snapshot.snapshot_id, snapshot.generation)
        if more
        else None,
        snapshot.snapshot_id,
        snapshot.generation,
        limit,
        warnings=warnings,
        coverage={
            "accepted_lookup_rows_in_selection": len(selected),
            "native_invocation_rows_retained": len(doc.invocations),
            "lookup_metadata_current": doc.lookup_schema == "lookups_v1",
            "untimed_invocation_rows": untimed,
        },
    )
