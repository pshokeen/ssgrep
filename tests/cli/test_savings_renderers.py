"""Consumer-visible boundary behavior for savings CLI renderers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ssgrep.accounting.export.consent import BatchView, EnablePlan, PreviewView
from ssgrep.accounting.ledger.contracts import Quantity
from ssgrep.accounting.ledger.ledger_contracts import Signal
from ssgrep.accounting.views.types import ReceiptView, Scope
from ssgrep.cli.export_render import (
    preview_payload_lines,
    render_enable_preview,
    render_preview_header,
)
from ssgrep.cli.savings_lookup_render import render_lookups
from ssgrep.cli.savings_receipt_render import render_receipt
from ssgrep.cli.savings_text import Doc, fmt_age, fmt_bytes, fmt_quantity, safe


def test_export_preview_keeps_wire_bytes_and_summarizes_batches_after_cap() -> None:
    first = BatchView("batch-one", 4, 1, "a" * 64, b'{"x":1}')
    second = BatchView("batch-two", 5, 2, "b" * 64, b'{"y":2}')
    view = PreviewView(
        "personal-local",
        "org",
        Signal.METRICS,
        "ok",
        "scope_changed",
        3,
        "collector.example",
        "v1",
        True,
        8,
        "snapshot",
        1,
        (),
        (first, second),
    )

    header = render_preview_header(view, 80).plain
    payload = preview_payload_lines(view, cap=4)

    assert "Endpoint host" in header and "collector.example" in header
    assert "scope changed" in header and "Omitted" in header
    assert payload[1] == '{"x":1}'
    assert payload[2].startswith("-- 1 more batch(es) not printed")
    assert payload[3].startswith("-- batch-two: 5 bytes, 2 items")


def test_export_preview_without_endpoint_reason_or_batches_is_still_explained() -> None:
    view = PreviewView(
        "personal-local",
        "org",
        Signal.LOGS,
        "no_snapshot",
        None,
        1,
        None,
        "v1",
        False,
        None,
        None,
        0,
        (),
        (),
    )

    rendered = render_preview_header(view, 80).plain

    assert "No snapshot exists yet" in rendered
    assert "Endpoint host" not in rendered
    assert "Generation" not in rendered


def test_lookup_unavailable_view_gives_status_command_and_safe_text() -> None:
    view = SimpleNamespace(
        profile="personal-local",
        snapshot_id=None,
        generation=0,
        reason="source\nmissing",
    )

    rendered = render_lookups(view).plain

    assert "Unavailable: source\\nmissing" in rendered
    assert "ssgrep savings status" in rendered


def test_text_helpers_preserve_unavailable_reasons_and_format_boundaries() -> None:
    assert "\\x1b" in safe("\x1b")
    assert fmt_quantity(Quantity.unavailable("tokens", "incomplete:sample")) == (
        "unavailable (incomplete:sample)"
    )
    assert fmt_bytes(1024) == "1.0 KiB"
    assert fmt_age(None) == "age unknown"
    assert fmt_age(119) == "119s ago"
    assert fmt_age(120) == "2m ago"


def test_export_approval_preview_discloses_policy_and_retention_boundaries() -> None:
    plan = EnablePlan(
        profile="personal-local",
        destination="org",
        signal=Signal.METRICS,
        endpoint="https://collector.example",
        proxy="http://proxy.example",
        credential_ref="env:TOKEN",
        fields_version="v1",
        fields=("count",),
        roots=(),
        roots_note="no roots",
        policy_version=2,
        policy_source="local",
        managed=False,
        retention={"evidence_days": 1, "receipt_days": 2, "spool_days": 3, "spool_bytes": 1024},
        already_enabled=True,
        paused=True,
        notices=("notice",),
    )
    preview = render_enable_preview(plan, 80).plain
    assert "1 days" in preview and "1.0 KiB" in preview
    assert "env:TOKEN" in preview and "no roots" in preview
    assert "already enabled" in preview


def test_preview_payload_zero_cap_and_lookup_pagination_render_scope() -> None:
    batch = BatchView("batch", 2, 1, "a" * 64, b"{}")
    view = PreviewView(
        "personal-local",
        "org",
        Signal.METRICS,
        "ok",
        None,
        1,
        None,
        "v1",
        False,
        1,
        "snapshot",
        0,
        (),
        (batch,),
    )
    lines = preview_payload_lines(view, cap=0)
    assert len(lines) == 2
    assert lines[0].startswith("-- 1 more batch(es) not printed")
    assert lines[-1].startswith("-- batch:")
    assert "{}" not in lines
    shown = preview_payload_lines(view, cap=2)
    assert shown == [f"-- request batch: 2 bytes, 1 items, sha256 {'a' * 64}", "{}"]
    lookup = SimpleNamespace(
        profile="personal-local",
        snapshot_id="snap",
        generation=2,
        reason=None,
        rows=(),
        total=0,
        warnings=("partial",),
        next_cursor="cursor",
        scope=Scope(
            since="2026-01-01",
            until=None,
            project="a path",
            runtime=None,
            session=None,
            task=None,
            include_children=True,
        ),
    )
    rendered = render_lookups(lookup).plain
    assert "Showing 0 of 0" in rendered and "Warning: partial" in rendered
    assert "--from 2026-01-01" in rendered and "--project 'a path'" in rendered
    assert "--children" in rendered
    assert "--page \\\ncursor" in rendered


def test_text_folding_rejects_width_too_small_for_wide_character() -> None:
    try:
        Doc(40).line("界" * 50, indent=39)
    except ValueError as exc:
        assert "indentation leaves no room" in str(exc)
    else:
        raise AssertionError("over-indented wide text must not silently disappear")


def test_receipt_expired_content_and_imprecise_credit_are_explicit() -> None:
    view = ReceiptView(
        receipt_id="receipt-1",
        revision=2,
        root="conversation",
        task_turn=None,
        candidate_type="retrieval",
        atoms=(),
        allocations=(),
        exclusions=("atom:unresolved:missing_cost:opaque-id",),
        costs=(),
        methodology="method-v1",
        evidence_available=False,
        execution_ids=(),
        note="retained note",
        period="2026-04-02",
        credit_raw=(("central", 123),),
        unit="work_seconds:capped_gap_v1",
        raw_unit="ticks",
    )

    rendered = render_receipt(view, profile_hint="team", include_content=True).plain

    assert "Credit coverage is unavailable" in rendered
    assert "unavailable (raw_time_precision_unavailable)" in rendered
    assert "Detailed evidence has expired" in rendered and "retained note" in rendered
    assert "evidence has expired, so no content can be shown" in rendered
    assert "Source atom: missing_cost" in rendered
    assert "savings history --from 2026-04-02 --to 2026-04-03 --profile team" in rendered
    assert "Private content is hidden" not in rendered


def test_receipt_credit_boundary_reports_missing_atoms_and_invalid_period() -> None:
    from ssgrep.accounting.views.history import parse_scenarios
    from tests.accounting.views.snapshot_builders import scenarios as build_scenarios

    estimates = parse_scenarios(
        list(build_scenarios((8, 8, 8), (0, 0, 0), (8, 8, 8), "tokens").values())
    )
    view = ReceiptView(
        receipt_id="receipt-2",
        revision=1,
        root="conversation",
        task_turn=None,
        candidate_type="retrieval",
        atoms=(),
        allocations=(),
        exclusions=("atom:unresolved:missing_cost:opaque-id",),
        costs=(),
        methodology="method-v1",
        evidence_available=True,
        execution_ids=(),
        note=None,
        period="not-a-date",
        credit_raw=(("central", 8),),
        unit="context_tokens:chars_div4_v1",
        raw_unit="chars",
        eligibility="eligible",
        scenarios=estimates,
    )

    rendered = render_receipt(view).plain

    assert "Central: no credit" in rendered
    assert "no source atom qualified in this scenario" in rendered
    assert "Conservative: not computed" in rendered
    assert "Optimistic: not computed" in rendered
    assert "savings history --profile" not in rendered


def test_lookup_without_next_page_omits_pagination_command() -> None:
    view = SimpleNamespace(
        profile="personal-local",
        snapshot_id="snapshot-1",
        generation=3,
        reason=None,
        rows=(),
        total=0,
        warnings=(),
        next_cursor=None,
        scope=Scope(),
    )

    rendered = render_lookups(view).plain

    assert "Showing 0 of 0" in rendered
    assert "--page" not in rendered


def test_history_continuation_quotes_filters_and_nondefault_limit() -> None:
    from ssgrep.accounting.views.types import HistoryView
    from ssgrep.cli.savings_render import history_more

    view = HistoryView(
        profile="team",
        entries=(),
        current_total=Quantity.unavailable("tokens", "empty"),
        next_cursor="opaque cursor",
        since="2026-01-01",
        until="2026-01-31",
        limit=7,
    )

    assert history_more(view) == (
        "ssgrep savings history --from 2026-01-01 --to 2026-01-31 --limit 7 "
        "--page 'opaque cursor' --profile team"
    )


def test_history_without_filters_and_default_limit_keeps_short_continuation() -> None:
    from ssgrep.accounting.views.types import HistoryView
    from ssgrep.cli.savings_render import history_more

    view = HistoryView(
        profile="personal-local",
        entries=(),
        current_total=Quantity.unavailable("tokens", "empty"),
        next_cursor="page-2",
    )

    assert history_more(view) == "ssgrep savings history --page page-2"


def _receipt(**changes: Any) -> ReceiptView:
    fields: dict[str, Any] = {
        "receipt_id": "receipt-x",
        "revision": 1,
        "root": "conversation",
        "task_turn": None,
        "candidate_type": "retrieval",
        "atoms": (),
        "allocations": (),
        "exclusions": (),
        "costs": (),
        "methodology": "method-v1",
        "evidence_available": True,
        "execution_ids": (),
    }
    return ReceiptView(**{**fields, **changes})


def test_receipt_two_part_exclusion_has_no_subject_and_unit_fallbacks_name_the_unit() -> None:
    rendered = render_receipt(_receipt(exclusions=("atom:missing_cost",))).plain
    assert "Source atom: missing_cost" in rendered
    assert "Source atom :" not in rendered and "Source atom  " not in rendered

    seconds = render_receipt(_receipt(unit="work_seconds", credit_raw=(("central", 8),))).plain
    assert "Central known credit subtotal" in seconds and "8 s" in seconds
    bare = render_receipt(_receipt(unit=None, credit_raw=(("central", 8),))).plain
    assert "Central known credit subtotal" in bare and "8 units" in bare
    # positive control: a division method rescales the same raw value
    divided = render_receipt(
        _receipt(unit="context_tokens:chars_div4_v1", credit_raw=(("central", 8),))
    ).plain
    assert "2 tokens" in divided and "8 units" not in divided


def test_lookup_row_reports_possible_execution_links_and_join_limits() -> None:
    row = SimpleNamespace(
        lookup_id="lookup-1",
        ts_utc=None,
        runtime=None,
        execution="exec-1",
        outcome="ok",
        exposure="shown",
        attribution="unattributed",
        reuse="none",
        eligibility="ineligible",
        useful_outcome="not observed",
        actor="agent",
        actor_basis="runtime",
        scope_basis="cwd",
        ambiguous_execution_ids=("e1", "e2"),
        join_rejection_reasons=("ambiguous_window",),
        visible_bytes=Quantity.observed(10, "bytes"),
        measured_service_ms=Quantity.unavailable("ms", "not_recorded"),
        known_costs=(),
        missing_evidence=(),
        receipt_ids=(),
    )
    view = SimpleNamespace(
        profile="personal-local",
        snapshot_id="snap",
        generation=1,
        reason=None,
        rows=(row,),
        total=1,
        warnings=(),
        next_cursor=None,
        scope=Scope(),
    )

    rendered = render_lookups(view).plain
    assert "Possible execution links: 2 (not confirmed ownership)" in rendered
    assert "Join limits: ambiguous_window" in rendered
    row.ambiguous_execution_ids = ()
    row.join_rejection_reasons = ()
    plain = render_lookups(view).plain
    assert "Possible execution links" not in plain and "Join limits" not in plain


def test_measured_block_explains_partially_known_counts_and_plain_reasons() -> None:
    from ssgrep.accounting.ledger.contracts import Basis
    from ssgrep.accounting.ledger.ledger_contracts import SnapshotState
    from ssgrep.accounting.views.types import SummaryView
    from ssgrep.cli.savings_render import render_summary

    def view(reason: str) -> SummaryView:
        return SummaryView(
            profile="personal-local",
            state=SnapshotState.FRESH,
            paused=False,
            paused_at=None,
            stale=False,
            scope=Scope(by="task"),
            collected_at="2026-10-12T12:00:00Z",
            age_seconds=5,
            collection_enabled=True,
            exports=(),
            measured=(
                ("searches", Quantity.observed(3, "count")),
                ("shows", Quantity(None, "count", Basis.UNAVAILABLE, reason=reason)),
                ("shows_known", Quantity.observed(2, "count")),
            ),
        )

    partial = render_summary(view("missing_for_some_executions"), 120).plain
    assert "unavailable: 2" in partial and "known, but some executions lack it" in partial
    plain = render_summary(view("clock_skew_detected"), 120).plain
    assert "known, but clock skew detected" in plain
    assert "some executions lack it" not in plain


def test_group_block_is_silent_without_group_by_and_present_with_it() -> None:
    from ssgrep.accounting.ledger.ledger_contracts import SnapshotState
    from ssgrep.accounting.views.types import SummaryView
    from ssgrep.cli.savings_state_render import render_groups

    def view(by: str | None) -> SummaryView:
        return SummaryView(
            profile="personal-local",
            state=SnapshotState.FRESH,
            paused=False,
            paused_at=None,
            stale=False,
            scope=Scope(by=by),
            collected_at=None,
            age_seconds=None,
            collection_enabled=True,
            exports=(),
        )

    from ssgrep.cli.savings_text import Doc

    silent, grouped = Doc(80), Doc(80)
    render_groups(silent, view(None))
    render_groups(grouped, view("task"))
    assert silent.text().plain == ""
    assert "BY TASK" in grouped.text().plain and "No rows match" in grouped.text().plain


def test_compact_conditions_report_measurement_off_and_unknown_codes_safely() -> None:
    from ssgrep.accounting.ledger.ledger_contracts import SnapshotState
    from ssgrep.accounting.views.types import Condition, SummaryView
    from ssgrep.cli.savings_state_render import render_compact_conditions
    from ssgrep.cli.savings_text import Doc

    def view(enabled: bool, *conditions: Condition) -> SummaryView:
        return SummaryView(
            profile="personal-local",
            state=SnapshotState.FRESH,
            paused=False,
            paused_at=None,
            stale=False,
            scope=Scope(),
            collected_at=None,
            age_seconds=None,
            collection_enabled=enabled,
            exports=(),
            conditions=conditions,
        )

    off = Doc(80)
    render_compact_conditions(off, view(False, Condition("future_state\x1b[2J", "needs care")))
    text = off.text().plain
    assert "Measurement off; showing retained history only." in text
    assert "future_state\\x1b[2J: needs care" in text and "\x1b" not in text
    on = Doc(80)
    render_compact_conditions(on, view(True))
    assert on.text().plain == ""


def test_diagnostics_page_shows_children_scope_and_reason_with_or_without_records() -> None:
    from ssgrep.accounting.views.diagnostics import DiagnosticPage, DiagnosticRecord
    from ssgrep.cli.savings_state_render import render_diagnostics

    record = DiagnosticRecord("rec-1", "some_code", "warning", "ledger")

    def page(records: tuple[DiagnosticRecord, ...]) -> DiagnosticPage:
        return DiagnosticPage(
            profile="personal-local",
            scope=Scope(include_children=True),
            snapshot_id="snap",
            generation=1,
            records=records,
            total=len(records),
            alias_duplicates=0,
            next_cursor=None,
            reason="partial_scan",
        )

    empty = render_diagnostics(page(())).plain
    assert "Session children included." in empty and "No records: partial_scan." in empty
    shown = render_diagnostics(page((record,))).plain
    assert "State: partial_scan." in shown and "No records:" not in shown


def test_history_entry_names_namespace_revision_and_adjustment_and_expired_evidence() -> None:
    from ssgrep.accounting.views.types import HistoryEntry, HistoryView
    from ssgrep.cli.savings_render import render_history

    def view(**changes: Any) -> HistoryView:
        entry = HistoryEntry(
            period="2026-04-01",
            revision=2,
            status="closed",
            current=True,
            supersedes=1,
            net=Quantity.observed(5, "tokens"),
            **changes,
        )
        return HistoryView("personal-local", (entry,), Quantity.observed(5, "tokens"), None)

    full = render_history(
        view(namespace="team-a", revision_name="audit-fix", kind="adjustment", recomputable=False)
    ).plain
    assert "Namespace" in full and "team-a" in full
    assert "Revision name" in full and "audit-fix" in full
    assert "adjustment: late evidence replaced the earlier revision" in full
    assert "expired: total kept, cannot be recomputed from source" in full
    bare = render_history(view()).plain
    for absent in ("Namespace", "Revision name", "adjustment:", "expired:"):
        assert absent not in bare


def test_doc_text_drops_trailing_blank_lines() -> None:
    doc = Doc(40)
    doc.line("alpha")
    doc.blank()
    assert doc.text().plain == "alpha"


def test_eligible_receipt_credit_loop_reports_unavailable_raw_time_and_converts_nanoseconds() -> (
    None
):
    from ssgrep.accounting.views.history import parse_scenarios
    from tests.accounting.views.snapshot_builders import scenarios as build_scenarios

    estimates = parse_scenarios(
        list(build_scenarios((8, 8, 8), (0, 0, 0), (8, 8, 8), "tokens").values())
    )

    def view(raw_unit: str, raw: int) -> ReceiptView:
        return _receipt(
            eligibility="eligible",
            scenarios=estimates,
            unit="work_seconds:capped_gap_v1",
            raw_unit=raw_unit,
            credit_raw=(("central", raw),),
            allocations=(("central", ("atom-a",)),),
        )

    unavailable = render_receipt(view("ticks", 7)).plain
    assert "Central: raw allocation unavailable (raw_time_precision_unavailable)" in unavailable
    assert "Central: +" not in unavailable
    converted = render_receipt(view("nanoseconds", 3_000_000_000)).plain
    assert "Central: +3 s" in converted
    assert "raw allocation unavailable" not in converted


def test_headings_wrap_to_the_terminal_width() -> None:
    doc = Doc(40)
    doc.heading("HISTORICAL BACKFILL (not live executions)", "Imported history is its own series.")

    lines = doc.text().plain.splitlines()

    assert lines[:2] == ["HISTORICAL BACKFILL (not live", "executions)"]
    assert max(len(line) for line in lines) <= 40


def test_grouped_summary_keeps_negative_values_and_unknowns_distinct():
    from ssgrep.accounting.ledger.ledger_contracts import SnapshotState
    from ssgrep.accounting.views.history import scenario_row
    from ssgrep.accounting.views.types import GroupRow, SummaryView
    from ssgrep.cli.savings_render import render_summary
    from tests.accounting.views.snapshot_builders import scenarios

    rows = (
        scenario_row(
            "context_tokens",
            scenarios((50, None, 80), (100, 100, 100), (-50, None, -20), "tokens"),
        ),
        scenario_row("work_seconds", scenarios((0, 1, 2), (5, 5, 5), (-5, -4, -3), "s")),
    )
    measured = (
        ("searches", Quantity.observed(3, "count")),
        ("shows", Quantity.unavailable("count", "missing_show_count")),
    )
    view = SummaryView(
        profile="personal-local",
        state=SnapshotState.FRESH,
        paused=False,
        paused_at=None,
        stale=False,
        scope=Scope(by="task"),
        collected_at=None,
        age_seconds=None,
        collection_enabled=True,
        exports=(),
        measured=measured,
        scenarios=rows,
        groups=(GroupRow("task-a", measured, rows, receipts=2),),
        group_total=1,
    )
    text = " ".join(render_summary(view, 120).plain.split())
    assert "task-a" in text and "searches 3" in text
    assert "-50 / unavailable / -20 tokens" in text
    assert "-5 / -4 / -3 s" in text
    assert "shows 0" not in text
    assert "Unknown, not zero" in text and "Complete net unavailable" in text
    assert "2 receipt(s) in this group" in text
