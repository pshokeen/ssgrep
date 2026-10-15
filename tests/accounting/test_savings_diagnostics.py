"""V14: real summaries and complete read-only, snapshot-bound diagnostic pages."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rich.cells import cell_len

from ssgrep.accounting import service, snapshots
from ssgrep.accounting.contracts import Basis, Quantity
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.cli import savings_render
from tests.accounting import test_savings_cli as savings_cli
from tests.accounting.snapshot_builders import enable_policy, snapshot, write_snapshot

NOW = datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)
savings_cli_run = savings_cli.run


def problems() -> list[str]:
    return (
        [f"lineage:missing_parent:omp:session-{i}" for i in range(504)]
        + [f"coverage_gap:source-{i}" for i in range(30)]
        + ["storage_corrupt:SECRET\x1b[2J[red]CANARY[/red]\nFORGED"]
    )


def publish(data: Path, generation: int = 7, warnings: list[str] | None = None):
    raw = snapshot(generation=generation, warnings=warnings or problems())
    raw["snapshot_id"] = snapshots.snapshot_identity(raw)
    path = write_snapshot(data, **raw)
    envelope, _, _ = service.read_snapshot("personal-local")
    assert envelope is not None
    snapshots.stage_generation("personal-local", envelope)
    return path, envelope


@pytest.mark.parametrize("width,budget", [(40, 70), (80, 40), (120, 40)])
def test_hundreds_of_errors_summary_preserves_meaning_within_budget(tmp_path, width, budget):
    enable_policy()
    publish(tmp_path / "data")
    view = service.read_summary(now=NOW)
    assert len(view.diagnostic_groups) == 3 and view.diagnostic_total == len(problems())
    row = view.scenarios[1]
    unknown = Quantity.unavailable("s", "source_work_unavailable")
    view = replace(
        view,
        scenarios=(view.scenarios[0], replace(row, net=((row.net[0][0], unknown), *row.net[1:]))),
    )
    plain = savings_render.render_summary(view, width).plain
    assert len(plain.splitlines()) <= budget
    assert all(cell_len(line) <= width for line in plain.splitlines())
    flattened = " ".join(plain.replace("\\\n", "").split())
    for value in (
        "-12,000 tokens",
        "unavailable",
        "Unknown, not zero",
        "Local only",
        "91/112",
        "8/13",
        "storage_corrupt",
        "504",
        "omitted",
        "savings diagnostics",
        "--snapshot",
    ):
        assert value in flattened
    assert "SECRET" not in plain and "FORGED" not in plain and "\x1b" not in plain
    assert "all projects" in flattened and "Last 7 days" in flattened


def test_all_warning_identities_once_across_pinned_pages_and_new_collection(tmp_path):
    from ssgrep.accounting.service_diagnostics import read_diagnostics

    enable_policy()
    path, envelope = publish(tmp_path / "data", warnings=problems() + [problems()[0]])
    first = read_diagnostics(snapshot_id=envelope.snapshot_id, generation=7, limit=17)
    expected_total = len(problems())
    assert first.total == expected_total and first.alias_duplicates == 1
    ids = [record.record_id for record in first.records]
    publish(tmp_path / "data", generation=8, warnings=["new_problem:later"])
    cursor = first.next_cursor
    while cursor:
        page = read_diagnostics(
            snapshot_id=envelope.snapshot_id, generation=7, limit=17, cursor=cursor
        )
        ids.extend(record.record_id for record in page.records)
        assert len(page.records) <= 17 and page.total == expected_total
        cursor = page.next_cursor
    assert len(ids) == expected_total and len(set(ids)) == expected_total
    assert path.exists()


def test_cursor_rejects_changed_selection_or_limit_and_expired_generation(tmp_path):
    from ssgrep.accounting.service_diagnostics import read_diagnostics

    enable_policy()
    _, envelope = publish(tmp_path / "data")
    first = read_diagnostics(snapshot_id=envelope.snapshot_id, generation=7, limit=5)
    for code, limit in (("lineage.missing_parent", 5), (None, 6)):
        with pytest.raises(PolicyError, match="Cursor"):
            read_diagnostics(
                snapshot_id=envelope.snapshot_id,
                generation=7,
                cursor=first.next_cursor,
                code=code,
                limit=limit,
            )
    generation_file = next(
        (tmp_path / "data/accounting/personal-local/snapshots").glob("gen-*.json")
    )
    generation_file.unlink()
    # current was advanced: missing generation must not substitute current content.
    publish(tmp_path / "data", generation=8, warnings=["later:other"])
    with pytest.raises(PolicyError) as error:
        read_diagnostics(snapshot_id=envelope.snapshot_id, generation=7)
    assert error.value.condition == "diagnostic_snapshot_expired"


def test_default_details_hide_private_text_explicit_text_escapes_controls(tmp_path):
    from ssgrep.accounting.service_diagnostics import diagnostics_document, read_diagnostics
    from ssgrep.cli.savings_state_render import render_diagnostics

    enable_policy()
    _, envelope = publish(tmp_path / "data")
    hidden = read_diagnostics(
        snapshot_id=envelope.snapshot_id, generation=7, code="storage_corrupt"
    )
    assert "SECRET" not in json.dumps(diagnostics_document(hidden))
    shown = read_diagnostics(
        snapshot_id=envelope.snapshot_id, generation=7, code="storage_corrupt", include_text=True
    )
    for width in (40, 80):
        plain = render_diagnostics(shown, width).plain
        assert "SECRET" in plain and "\x1b" not in plain and "\nFORGED" not in plain
        assert all(cell_len(line) <= width for line in plain.splitlines())
    assert len(hidden.records) == len(shown.records) == 1
    assert hidden.records[0].record_id == shown.records[0].record_id


def test_disabled_diagnostics_never_creates_accounting_storage(tmp_path):
    from ssgrep.accounting.service_diagnostics import read_diagnostics

    data = tmp_path / "data"
    before = sorted(data.rglob("*")) if data.exists() else []
    view = read_diagnostics()
    assert not view.records and view.reason == "measurement_off"
    assert (sorted(data.rglob("*")) if data.exists() else []) == before


def test_typed_command_discovery_actual_json_pages(savings_cli_run):
    # Run is the existing isolated executable fixture; no private controls in either mode.
    enable_policy()
    write_snapshot(savings_cli_run.data, warnings=problems())
    first = savings_cli_run("savings", "diagnostics", "--limit", "9", "--json")
    assert first.returncode == 0, first.stderr
    body = json.loads(first.stdout)["data"]
    assert len(body["records"]) == 9 and body["total"] == len(problems())
    second = savings_cli_run(
        "savings",
        "diagnostics",
        "--limit",
        "9",
        "--snapshot",
        body["snapshot_id"],
        "--generation",
        str(body["generation"]),
        "--cursor",
        body["next_cursor"],
        "--json",
    )
    assert second.returncode == 0, second.stderr
    ids = {row["record_id"] for row in body["records"]}
    assert not ids.intersection(
        row["record_id"] for row in json.loads(second.stdout)["data"]["records"]
    )


@pytest.mark.parametrize("unbounded_debit", [False, True])
@pytest.mark.parametrize("width", [40, 80])
def test_uncertainty_shows_subtotal_without_substituting_complete_net(
    tmp_path, unbounded_debit, width
):
    from ssgrep.accounting.contracts import Scenario
    from ssgrep.accounting.ledger_contracts import EstimateBounds, ScenarioEstimate

    enable_policy()
    write_snapshot(tmp_path / "data", warnings=problems())
    view = service.read_summary(now=NOW)
    lower = (
        Quantity.unavailable("tokens", "unbounded_debit")
        if unbounded_debit
        else Quantity(-100, "tokens", Basis.ESTIMATED)
    )
    upper = Quantity.unavailable("tokens", "unbounded_credit")
    credit = Quantity(0, "tokens", Basis.ESTIMATED)
    debit = Quantity(100, "tokens", Basis.ESTIMATED)
    net = Quantity.unavailable("tokens", "incomplete:source_cost_unavailable")
    estimate = ScenarioEstimate(
        Scenario.CONSERVATIVE,
        credit,
        debit,
        net,
        -100,
        False,
        (("source-a", "source_cost_unavailable"),),
        (("lookup-a", "lookup_cost_unavailable"),) if unbounded_debit else (),
        0,
        1,
        EstimateBounds(credit, upper, debit, upper if unbounded_debit else debit, lower, upper),
    )
    row = replace(
        view.scenarios[0],
        credit=((Scenario.CONSERVATIVE, credit),),
        debit=((Scenario.CONSERVATIVE, debit),),
        net=((Scenario.CONSERVATIVE, net),),
        estimates=(estimate,),
    )
    view = replace(view, scenarios=(row,), conditions=())
    text = savings_render.render_summary(view, width).plain
    flat = " ".join(text.split()).lower()
    assert "known net subtotal" in flat and "-100 tokens" in flat
    assert "complete net unavailable" in flat
    assert "lower bound" in flat
    if unbounded_debit:
        assert "lower bound unavailable" in flat and "unbounded_debit" in flat
    else:
        assert "lower bound -100 tokens" in flat
    assert len(text.splitlines()) <= (70 if width == 40 else 40)


def test_warning_categories_are_capped_without_losing_severe_counts(tmp_path):
    from ssgrep.accounting.service_diagnostics import read_diagnostics

    enable_policy()
    warnings = problems() + [f"category_{i}:private-{i}" for i in range(12)]
    _, envelope = publish(tmp_path / "data", warnings=warnings)
    summary = service.read_summary(now=NOW)
    for width, budget in ((40, 70), (80, 40)):
        text = savings_render.render_summary(summary, width).plain
        flat = " ".join(text.split())
        assert "ERROR storage_corrupt: 1" in flat
        assert "other categories" in flat and f"{len(warnings):,} details omitted" in flat
        assert (
            sum(line.lstrip().startswith(("ERROR ", "WARNING ")) for line in text.splitlines()) <= 5
        )
        assert len(text.splitlines()) <= budget
    details = read_diagnostics(
        snapshot_id=envelope.snapshot_id, generation=envelope.generation, code="category_11"
    )
    assert details.total == 1 and len(details.records) == 1


def test_selected_profile_filters_survive_diagnostic_navigation(tmp_path):
    import shlex

    from ssgrep.accounting.service_diagnostics import diagnostics_command
    from ssgrep.accounting.service_types import Scope
    from ssgrep.cli.savings_text import Doc

    scope = Scope(
        since="2026-10-01T00:00:00Z",
        until="2026-10-12T00:00:00Z",
        project="/w/O'Brien team",
        runtime="omp",
        session="native-parent",
        task="omp:parent#turn",
        include_children=True,
    )
    command = diagnostics_command(
        "acme", scope, "a" * 64, 9, limit=7, cursor="diag1.7.1234", include_text=True
    )
    for width in (40, 80):
        doc = Doc(width)
        doc.command("More:", command)
        lines = doc.text().plain.splitlines()
        executable = "\n".join(line for line in lines if not line.startswith("#")).replace(
            "\\\n", ""
        )
        assert shlex.split(executable) == shlex.split(command)
        assert all(cell_len(line) <= width for line in lines)


def test_filter_notes_are_reachable_and_expire_instead_of_rebinding_new_ledger(tmp_path):
    from ssgrep.accounting.scope import build_scope
    from ssgrep.accounting.service_diagnostics import read_diagnostics

    enable_policy()
    _, envelope = publish(tmp_path / "data")
    scope = build_scope(project=str(tmp_path / "project"), runtime="omp")
    page = read_diagnostics(
        scope=scope, snapshot_id=envelope.snapshot_id, generation=7, code="filters_unavailable"
    )
    assert page.total == 1
    assert page.records[0].source == "selection_note"
    assert page.scope.project == str(tmp_path / "project") and page.scope.runtime == "omp"
    publish(tmp_path / "data", generation=8, warnings=["later:new"])
    with pytest.raises(PolicyError) as error:
        read_diagnostics(scope=scope, snapshot_id=envelope.snapshot_id, generation=7)
    assert error.value.condition == "diagnostic_selection_expired"


@pytest.mark.parametrize("width,budget", [(40, 70), (80, 40)])
def test_actual_pipe_no_color_summary_is_bounded_and_keeps_recovery(savings_cli_run, width, budget):
    enable_policy()
    write_snapshot(savings_cli_run.data, warnings=problems())
    result = savings_cli_run("savings", columns=width)
    assert result.returncode == 0, result.stderr
    assert "\x1b" not in result.stdout and "SECRET" not in result.stdout
    flat = " ".join(result.stdout.replace("\\\n", "").split())
    assert "STALE" in flat and "metrics collect" in flat
    assert "savings diagnostics" in flat and "535 details omitted" in flat
    assert "-12,000 tokens" in flat and "Local only" in flat
    assert len(result.stdout.splitlines()) <= budget


def test_unreadable_policy_warning_remains_reachable_without_reading_snapshot(tmp_path):
    from ssgrep.accounting.service_diagnostics import read_diagnostics

    enable_policy()
    policy_path = tmp_path / "data/accounting/policy.json"
    policy_path.write_text("{")
    summary = service.read_summary(now=NOW)
    page = read_diagnostics()
    assert summary.warnings and page.total == len(set(summary.warnings))
    assert len(page.records) == 1 and page.records[0].source == "operational_warning"
    assert page.records[0].severity == "error" and page.snapshot_id is None
    assert page.reason and page.records[0].message is None


@pytest.mark.parametrize("width", [40, 80])
def test_diagnostic_private_text_respects_terminal_cells_not_character_count(width):
    from ssgrep.accounting.diagnostic_types import DiagnosticPage, DiagnosticRecord
    from ssgrep.accounting.service_types import Scope
    from ssgrep.cli.savings_state_render import render_diagnostics

    page = DiagnosticPage(
        "personal-local",
        Scope(),
        "a" * 64,
        7,
        (
            DiagnosticRecord(
                "b" * 64, "source_unavailable", "warning", "snapshot_warning", "京都研发 " * 80
            ),
        ),
        1,
        0,
        None,
        include_text=True,
    )
    assert all(
        cell_len(line) <= width for line in render_diagnostics(page, width).plain.splitlines()
    )


@pytest.mark.parametrize("width,budget", [(40, 70), (80, 40)])
def test_repeated_severe_delivery_conditions_are_counted_not_dumped(tmp_path, width, budget):
    from ssgrep.accounting.service_types import Condition, NextAction

    enable_policy()
    publish(tmp_path / "data")
    summary = service.read_summary(now=NOW)
    summary = replace(
        summary,
        conditions=tuple(
            Condition(
                "blocked",
                f"private-destination-{i}: blocked by policy",
                NextAction("ssgrep metrics status", "Inspect policy"),
            )
            for i in range(16)
        ),
    )
    text = savings_render.render_summary(summary, width).plain
    flat = " ".join(text.replace("\\\n", "").split())
    assert "Export blocked" in flat and "16 conditions" in flat
    assert "ssgrep metrics status" in flat
    assert "private-destination" not in text
    assert len(text.splitlines()) <= budget


def test_nonreplayable_snapshot_counts_survive_scope_projection_and_name_recovery(tmp_path):
    from ssgrep.cli.savings_json import summary_document

    enable_policy()
    publish(tmp_path / "data")
    write_snapshot(
        tmp_path / "data",
        coverage={"replay_unavailable_sessions": 3, "frequency_revision_unavailable": 1},
    )
    view = service.read_summary(now=NOW)
    assert view.coverage["replay_unavailable_sessions"] == 3
    assert view.coverage["frequency_revision_unavailable"] == 1
    document = summary_document(view)
    document_data = document["data"]
    assert isinstance(document_data, dict)
    coverage = document_data["coverage"]
    assert isinstance(coverage, dict)
    assert coverage["replay_unavailable_sessions"] == 3
    assert coverage["frequency_revision_unavailable"] == 1
    # Missing replay must remain partial even if a historical window's flag was false.
    view = replace(view, partial=False)
    for width, budget in ((40, 70), (80, 40)):
        text = savings_render.render_summary(view, width).plain
        flat = " ".join(text.replace("\\\n", "").split())
        assert "partial evidence" in flat and "Non-replayable (snapshot)" in flat
        assert "3 sessions" in flat and "1 frequency revisions" in flat
        assert "ssgrep metrics collect --recompute" in flat
        assert "approved native sources" in flat
        assert len(text.splitlines()) <= budget


def test_unavailable_replay_and_frequency_revision_codes_are_severe():
    from ssgrep.accounting.diagnostics import grouped_warnings

    groups = grouped_warnings(
        (
            "replay_unavailable_sessions:3 retained sessions cannot be replayed",
            "frequency_revision_unavailable:1 frozen rarity revision unavailable",
        )
    )
    assert {group.code for group in groups} == {
        "replay_unavailable_sessions",
        "frequency_revision_unavailable",
    }
    assert all(group.severity == "error" and group.count == 1 for group in groups)


@pytest.mark.parametrize(
    "selection,condition",
    [
        ({"snapshot_id": "../current"}, "invalid_snapshot"),
        ({"generation": True}, "invalid_snapshot"),
        ({"generation": 0}, "invalid_snapshot"),
        ({"generation": 10**10}, "invalid_snapshot"),
        ({"code": "lineage missing parent"}, "invalid_filter"),
        ({"code": "storage_corrupt\nFORGED"}, "invalid_filter"),
        ({"cursor": "diag1.1.0000000000000000"}, "invalid_cursor"),
    ],
)
def test_diagnostic_invalid_selection_rejects_before_storage_and_preserves_bytes(
    tmp_path, selection, condition
):
    from ssgrep.accounting.service_diagnostics import read_diagnostics
    from tests.accounting.test_ux_support import fingerprint

    enable_policy()
    _, envelope = publish(tmp_path / "data", warnings=["storage_corrupt:private-canary"])
    control = read_diagnostics(snapshot_id=envelope.snapshot_id, generation=7)
    assert control.total == 1 and control.records[0].code == "storage_corrupt"
    before = fingerprint(tmp_path / "data")
    with pytest.raises(PolicyError) as caught:
        read_diagnostics(**selection)
    assert caught.value.condition == condition
    assert fingerprint(tmp_path / "data") == before


@pytest.mark.parametrize("corruption", ["wrong_kind", "edited_offset", "mismatched_snapshot"])
def test_diagnostic_cursor_integrity_and_snapshot_pair_are_enforced(tmp_path, corruption):
    from ssgrep.accounting.service_diagnostics import read_diagnostics
    from tests.accounting.test_ux_support import fingerprint

    enable_policy()
    _, envelope = publish(tmp_path / "data", warnings=["one:private", "two:private"])
    first = read_diagnostics(snapshot_id=envelope.snapshot_id, generation=7, limit=1)
    assert first.total == 2 and first.next_cursor
    snapshot_id = envelope.snapshot_id
    cursor = first.next_cursor
    if corruption == "wrong_kind":
        cursor = first.next_cursor.replace("diag1.", "history1.", 1)
    elif corruption == "edited_offset":
        kind, _, tag = first.next_cursor.split(".")
        cursor = f"{kind}.9999999.{tag}"
    else:
        snapshot_id = "a" * 64 if envelope.snapshot_id != "a" * 64 else "b" * 64
    before = fingerprint(tmp_path / "data")
    with pytest.raises(PolicyError) as caught:
        read_diagnostics(snapshot_id=snapshot_id, generation=7, limit=1, cursor=cursor)
    assert caught.value.condition == (
        "invalid_snapshot" if corruption == "mismatched_snapshot" else "invalid_cursor"
    )
    assert fingerprint(tmp_path / "data") == before


def test_diagnostic_unreadable_frozen_generation_never_substitutes_current(tmp_path):
    from ssgrep.accounting.service_diagnostics import read_diagnostics
    from tests.accounting.test_ux_support import fingerprint

    enable_policy()
    data = tmp_path / "data"
    _, frozen = publish(data, warnings=["old_problem:private-old"])
    generation = next((data / "accounting/personal-local/snapshots").glob("gen-0000000007-*"))
    assert read_diagnostics(snapshot_id=frozen.snapshot_id, generation=7).total == 1
    _, current = publish(data, generation=8, warnings=["new_problem:private-new"])
    generation.write_bytes(b'{"incomplete":')
    before = fingerprint(data)
    with pytest.raises(PolicyError) as caught:
        read_diagnostics(snapshot_id=frozen.snapshot_id, generation=7)
    assert caught.value.condition == "diagnostic_snapshot_unreadable"
    current_page = read_diagnostics(snapshot_id=current.snapshot_id, generation=8)
    assert current_page.total == 1 and current_page.records[0].code == "new_problem"
    assert fingerprint(data) == before


@pytest.mark.parametrize("selected", [False, True])
def test_diagnostic_no_snapshot_is_empty_or_explicitly_expired_without_creating_files(
    tmp_path, selected
):
    from ssgrep.accounting.service_diagnostics import read_diagnostics
    from tests.accounting.test_ux_support import fingerprint

    enable_policy()
    data = tmp_path / "data"
    before = fingerprint(data)
    if selected:
        with pytest.raises(PolicyError) as caught:
            read_diagnostics(snapshot_id="a" * 64, generation=7)
        assert caught.value.condition == "diagnostic_snapshot_expired"
    else:
        page = read_diagnostics()
        assert page.total == 0 and page.records == () and page.next_cursor is None
        assert page.snapshot_id is None and page.generation is None and page.reason
    assert fingerprint(data) == before
