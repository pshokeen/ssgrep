"""Renderer behavior at 40/80/120 columns, NO_COLOR, pipes, hostile text, negatives, partials."""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ssgrep.accounting import service
from ssgrep.accounting.contracts import Basis, Quantity
from ssgrep.accounting.service_estimates import parse_scenarios
from ssgrep.accounting.service_types import HistoryEntry, HistoryView, Scope
from ssgrep.cli import savings_render, savings_state_render
from ssgrep.cli.savings_text import Doc, fmt_age, fmt_bytes, fmt_quantity
from tests.accounting.snapshot_builders import (
    RECEIPT_ID,
    enable_policy,
    scenarios,
    window_7d,
    write_receipt,
    write_snapshot,
)

T0 = datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)
ANSI = re.compile(r"\x1b")


def flat(text: str) -> str:
    """Collapse wrapping (and shell ``\\`` continuations) so phrase assertions ignore breaks."""
    return " ".join(text.replace(" \\\n", " ").split())


@pytest.fixture
def data(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def fresh_view(data: Path):
    enable_policy()
    write_snapshot(data)
    return service.read_summary(now=T0 + timedelta(seconds=38))


@pytest.mark.parametrize("width", [40, 80, 120])
def test_summary_fits_width_and_carries_required_facts(fresh_view, width: int) -> None:
    plain = savings_render.render_summary(fresh_view, width).plain
    assert all(len(line) <= width for line in plain.splitlines())
    squashed = flat(plain)
    for required in (
        "-12,000 tokens",  # negative sign preserved
        "+4,000 tokens",
        "-60 s",
        "420.0 KiB",  # units
        "91/112",  # coverage denominator
        "8/13",
        "38s ago",  # freshness
        "Local only",  # privacy
        f"ssgrep savings explain {RECEIPT_ID}",  # real next action
        "-7,000",  # every scenario value on its unit row
        "+9,000",
        "+20,000",
        "16,000",
        "+180 s",
        "+580 s",
        "Conservative",
        "Central",
        "Optimistic",
    ):
        assert required in squashed, required
    assert fresh_view.partial and "partial" in squashed.casefold()
    assert plain.index("MEASURED") < plain.index("ESTIMATED")
    assert plain.index("ESTIMATED") < plain.index("EVIDENCE") < plain.index("PRIVACY")
    assert "<" not in plain.split("Next:")[1] and "placeholder" not in plain


def test_profile_is_preserved_in_the_rendered_command(data: Path) -> None:
    from ssgrep.accounting import policy

    policy.enable_collection("acme", {"profile_id": "acme", "roots": [str(data.parent)]})
    write_snapshot(data, "acme")
    plain = savings_render.render_summary(service.read_summary("acme", now=T0), 80).plain
    assert f"ssgrep savings explain {RECEIPT_ID} --profile acme" in flat(plain)
    assert "Profile acme" in plain


def test_stale_view_shows_age_state_and_collect_but_keeps_values(data: Path) -> None:
    enable_policy()
    write_snapshot(data)
    view = service.read_summary(now=T0 + timedelta(seconds=7300))
    plain = savings_render.render_summary(view, 80).plain
    assert "STALE" in plain and "2h ago" in plain and "-12,000 tokens" in plain
    assert "ssgrep metrics collect" in plain
    assert "last good values" in plain


def test_paused_view_names_timestamp_and_resume(data: Path) -> None:
    enable_policy(paused=True)
    write_snapshot(data)
    view = service.read_summary(now=T0 + timedelta(seconds=10))
    plain = savings_render.render_summary(view, 80).plain
    assert "PAUSED" in plain and "Paused since" in plain and "-12,000 tokens" in plain
    assert "ssgrep metrics resume" in plain


def test_known_no_reuse_keeps_measured_costs_and_negative_nets(data: Path) -> None:
    enable_policy()
    window = window_7d()
    window["coverage"].update(candidate_receipts=0, source_costs_available=0, unavailable_sources=0)
    window["estimates"]["context_tokens"] = scenarios(
        (0,) * 3, (16000,) * 3, (-16000,) * 3, "tokens"
    )
    window["estimates"]["work_seconds"] = scenarios((0,) * 3, (120,) * 3, (-120,) * 3, "s")
    write_snapshot(data, windows={"7d": window})
    view = service.read_summary(now=T0)
    assert view.no_reuse
    assert all(e.complete and e.credit.value == 0 for row in view.scenarios for e in row.estimates)
    plain = savings_render.render_summary(view, 80).plain
    assert "-16,000 tokens" in flat(plain) and "-120 s" in flat(plain)


def test_consistent_current_missing_source_metadata_remains_unavailable(data: Path) -> None:
    enable_policy()
    window = window_7d()
    window["estimates"]["context_tokens"] = scenarios(
        (4000, 9000, 20000), (16000,) * 3, (-12000, None, 4000), "tokens"
    )
    write_snapshot(data, windows={"7d": window})
    view = service.read_summary(now=T0)
    central = next(e for e in view.scenarios[0].estimates if e.scenario.value == "central")
    assert central.net.value is None and central.credit_missing and not central.complete
    assert central.bounds is not None
    assert central.known_net == -7000 and central.bounds.net_lower.value == -7000
    assert central.bounds.net_upper.value is None
    plain = savings_render.render_summary(view, 80).plain
    assert "unavailable" in plain.lower() and "-7,000 tokens" in flat(plain)


@pytest.mark.parametrize("state_setup", ["disabled", "no_snapshot"])
def test_empty_states_show_no_numbers_and_a_real_action(data: Path, state_setup: str) -> None:
    if state_setup == "no_snapshot":
        enable_policy()
    plain = savings_render.render_summary(service.read_summary(now=T0), 40).plain
    command = "ssgrep metrics collect" if state_setup == "no_snapshot" else "ssgrep metrics setup"
    assert command in plain
    assert not re.search(r"\d+ tokens|\d+ s\b|MEASURED|ESTIMATED", plain)


def test_hostile_strings_are_inert_at_40_columns(data: Path) -> None:
    write_receipt(
        data,
        root="/w/\x1b[2J\x1b]0;pwn\x07[bold red]boom[/]\nFORGED: line",
        candidate_type="[link=http://evil]x[/link]\r",
        exclusions=["\x1b[31mred\x1b[0m"],
    )
    view = service.read_receipt("personal-local", RECEIPT_ID)
    assert view is not None
    text = savings_render.render_receipt(view, width=40)
    plain = text.plain
    assert "\x1b" not in plain and "\x07" not in plain and "\r" not in plain
    assert "\\x1b[2J" in plain, "escaped, not silently dropped"
    assert "[bold red]boom[/]" in flat(plain), "markup stays literal text"
    assert not any(line.startswith("FORGED") for line in plain.splitlines())
    assert all(len(line) <= 40 for line in plain.splitlines())
    assert not text.spans or all(span.style in ("bold", "dim", "") for span in text.spans)


def test_markup_like_text_never_changes_styling_when_color_is_forced(
    data: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_receipt(data, root="[red]RED[/red] [bold]B[/bold]")
    view = service.read_receipt("personal-local", RECEIPT_ID)
    assert view is not None
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    savings_render.emit(savings_render.render_receipt(view, width=80))
    out = capsys.readouterr().out
    assert "[red]RED[/red] [bold]B[/bold]" in re.sub(r"\x1b\[[0-9;]*m", "", out)
    assert "\x1b[31m" not in out


def test_receipt_hides_content_by_default_and_shows_it_only_when_present(data: Path) -> None:
    write_receipt(data)
    hidden = service.read_receipt("personal-local", RECEIPT_ID)
    assert hidden is not None
    plain = savings_render.render_receipt(hidden, width=80).plain
    assert "SECRET-CONTENT-CANARY" not in plain and "--include-content" in plain
    assert "unavailable (source_unavailable)" in plain
    assert "prior_known" in plain and "already known before the lookup" in flat(plain)
    shown = service.read_receipt("personal-local", RECEIPT_ID, include_content=True)
    assert shown is not None
    assert "SECRET-CONTENT-CANARY" in savings_render.render_receipt(shown, width=80).plain


def test_expired_receipt_states_unavailability(data: Path) -> None:
    write_receipt(data, evidence_available=False)
    view = service.read_receipt("personal-local", RECEIPT_ID)
    assert view is not None
    plain = savings_render.render_receipt(view, width=80).plain
    squashed = flat(plain)
    assert "EXPIRED" in plain and "Full recomputation is unavailable" in squashed
    assert "Retained totals above are kept" in squashed


def test_history_render_marks_superseded_and_total_label(width: int = 40) -> None:
    net = lambda v: Quantity(v, "tokens", Basis.ESTIMATED)  # noqa: E731
    view = HistoryView(
        "personal-local",
        (
            HistoryEntry("2026-10-01", 1, "sealed", False, None, net(100)),
            HistoryEntry("2026-10-01", 2, "revised", True, 1, net(-70)),
        ),
        net(-70),
        "h1.2.deadbeef",
        next_actions=(),
    )
    plain = savings_render.render_history(view, width).plain
    assert "superseded" in plain and "current" in plain
    assert "-70 tokens" in plain and "--cursor h1.2.deadbeef" in flat(plain)
    assert all(len(line) <= width for line in plain.splitlines())


def test_number_formatting_is_locale_free_and_unknown_never_zero() -> None:
    assert fmt_bytes(0) == "0 B" and fmt_bytes(1536) == "1.5 KiB"
    assert fmt_bytes(5 * 1024**3) == "5.0 GiB"
    assert (
        fmt_quantity(Quantity(-1234567, "tokens", Basis.ESTIMATED), signed=True)
        == "-1,234,567 tokens"
    )
    assert fmt_quantity(Quantity(0, "tokens", Basis.OBSERVED), signed=True) == "+0 tokens"
    unknown = fmt_quantity(Quantity.unavailable("tokens", "why"))
    assert unknown == "unavailable (why)" and "0" not in unknown
    # age precision changes at the whole-day boundary; unknown is spelled out, never 0
    for seconds, expected in (
        (None, "age unknown"),
        (172799, "47h ago"),
        (172800, "2d ago"),
        (259199, "2d ago"),
    ):
        assert fmt_age(seconds) == expected


def test_corrupt_snapshot_view_says_so_and_changes_nothing(data: Path) -> None:
    enable_policy()
    write_snapshot(data).write_text("{")
    view = service.read_summary(now=T0)
    plain = savings_render.render_summary(replace(view), 40).plain
    assert "unreadable" in plain.lower() and "Nothing was changed" in plain


@pytest.mark.parametrize("width", [40, 80, 120])
@pytest.mark.parametrize("label", ["More:", ""])
def test_scope_command_round_trips_shell_metacharacters_and_child_selection(
    width: int,
    label: str,
    tmp_path: Path,
):
    scope = Scope(
        since="2026-10-01",
        until="2026-10-02",
        project="/w/O'Brien team;$(printf unexpected)/" + "space path/" * 14,
        runtime="omp",
        session="session-a",
        task="turn-a",
        by="task",
        include_children=True,
    )
    command = savings_state_render.scope_command("acme", scope, 7, "cursor-1")
    doc = Doc(width)
    doc.command(label, command)
    rendered = doc.text().plain
    assert all(len(line) <= width for line in rendered.splitlines())
    parsed = subprocess.run(
        ["/bin/sh", "-c", "set -e; ssgrep() { printf '%s\\0' ssgrep \"$@\"; };\n" + rendered],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    tokens = [part.decode() for part in parsed.stdout.split(b"\0")[:-1]]
    assert tokens == shlex.split(command)
    assert "\n" not in command
    assert tokens[:2] == ["ssgrep", "savings"]
    assert tokens[tokens.index("--project") + 1] == scope.project
    for flag, value in (
        ("--since", scope.since),
        ("--until", scope.until),
        ("--runtime", "omp"),
        ("--session", "session-a"),
        ("--task", "turn-a"),
        ("--by", "task"),
        ("--limit", "7"),
        ("--cursor", "cursor-1"),
        ("--profile", "acme"),
    ):
        assert tokens[tokens.index(flag) + 1] == value
    assert "--include-children" in tokens


@pytest.mark.parametrize("width", [40, 80, 120])
def test_receipt_displays_current_subtotals_units_and_missing_coverage(data: Path, width: int):
    write_receipt(data)
    base = service.read_receipt("personal-local", RECEIPT_ID)
    assert base is not None
    view = replace(
        base,
        unit="context_tokens:chars_div4_v1",
        period="not-a-day",
        credit_raw=(("conservative", 19), ("central", 31), ("optimistic", 31)),
        allocations=(
            ("conservative", ("atom-1",)),
            ("central", ("atom-1", "atom-2")),
            ("optimistic", ("atom-1", "atom-2")),
        ),
        scenarios=tuple(
            replace(e, known_debits=2, known_atoms=1 if e.scenario.value == "conservative" else 2)
            for e in parse_scenarios(
                list(scenarios((4, 7, 7), (18,) * 3, (None,) * 3, "context_tokens").values())
            )
        ),
        eligibility_reasons=("source_costs_missing",),
        costs=(
            ("first", Quantity(7, "context_tokens", Basis.ESTIMATED, "chars_div4_v1")),
            ("second", Quantity(11, "context_tokens", Basis.ESTIMATED, "chars_div4_v1")),
        ),
        exclusions=(
            "atom:source:path:prior_known",
            "missing:unresolved:source_unresolved:opaque:unavailable",
        ),
    )
    text = flat(savings_render.render_receipt(view, profile_hint="acme", width=width).plain)
    assert "+4 tokens" in text and "+7 tokens" in text
    assert "18 tokens" in text
    assert "source:path" in text and "source_unresolved" in text
    assert "unresolved:source_unresolved:opaque" not in text
    assert "ssgrep savings --profile acme" in text
    assert "--since not-a-day" not in text
    missing = flat(
        savings_render.render_receipt(
            replace(
                view,
                costs=(),
                scenarios=(),
                credit_raw=(),
                allocations=(),
                evidence_available=False,
            ),
            width=width,
        ).plain
    )
    assert "unknown" in missing and "0 tokens" not in missing


def test_empty_and_expired_history_do_not_fabricate_a_recomputable_zero():
    unknown = Quantity.unavailable("context_tokens", "periods_unavailable")
    view = HistoryView("acme", (), unknown, None, reason="retention_boundary")
    text = flat(savings_render.render_history(view).plain)
    assert "periods_unavailable" in text and "retention_boundary" in text
    assert "0 tokens" not in text
    retained = replace(
        view,
        entries=(
            HistoryEntry(
                "2026-10-01",
                2,
                "revised",
                True,
                1,
                Quantity(-37, "context_tokens", Basis.ESTIMATED),
                recomputable=False,
            ),
        ),
        current_total=Quantity(-37, "context_tokens", Basis.ESTIMATED),
    )
    text = flat(savings_render.render_history(retained).plain)
    assert "-37 tokens" in text and "expired" in text


def test_missing_measured_fields_and_empty_groups_stay_absent(data: Path):
    enable_policy()
    write_snapshot(data)
    view = service.read_summary(now=T0)
    view = replace(
        view,
        measured=(("searches", Quantity.observed(17, "count")),),
        scope=Scope(by="task"),
        groups=(),
    )
    text = flat(savings_render.render_summary(view).plain)
    measured = text.split("MEASURED", 1)[1].split("ESTIMATED", 1)[0]
    assert "17" in measured and "Shows" not in measured and "0 B" not in measured
    groups = text.split("BY TASK", 1)[1].split("EVIDENCE", 1)[0]
    assert "0 tokens" not in groups and "No rows" in groups


@pytest.mark.parametrize("width", [40, 80, 120])
@pytest.mark.parametrize("raw_unit", ["", "seconds", "nanoseconds"])
def test_legacy_raw_work_never_invents_precision_or_complete_credit(data, width, raw_unit):
    from rich.cells import cell_len

    from ssgrep.cli.savings_json import receipt_document
    from tests.accounting.test_ux_support import fingerprint

    raw_credit = 2_999_999_999
    write_receipt(
        data,
        format="receipt_v1",
        unit="work_seconds:capped_gap_v1",
        raw_unit=raw_unit,
        credit_raw=[["central", raw_credit]],
        allocations=[["central", ["legacy-atom"]]],
        eligibility="eligible",
        scenarios=[],
        costs=[],
    )
    before = fingerprint(data)
    view = service.read_receipt("personal-local", RECEIPT_ID)
    assert view is not None
    document = receipt_document(view, "personal-local")["data"]
    assert isinstance(document, dict)
    assert document["credit_raw"] == {"central": raw_credit}
    assert document["raw_unit"] == (raw_unit or None)
    assert document["eligibility"] == "unavailable"
    assert document["scenarios"] == [] and document["costs"] == {}
    assert document["content"] is None
    text = savings_render.render_receipt(view, width=width).plain
    compact = "".join(text.split())
    assert all(cell_len(line) <= width for line in text.splitlines())
    assert "legacy-atom" in compact
    assert "+2s" not in compact and "0s" not in compact
    if raw_unit == "nanoseconds":
        assert "2s" in compact
        assert "raw_time_precision_unavailable" not in compact
    else:
        assert "raw_time_precision_unavailable" in compact
        assert "2,999,999,999s" not in compact
    assert "SECRET-CONTENT-CANARY" not in text
    assert fingerprint(data) == before


@pytest.mark.parametrize("width", [40, 80, 120])
def test_empty_allocations_and_multiple_exposures_preserve_exclusion_scope_and_zero_credit(
    data, width
):
    from rich.cells import cell_len

    from ssgrep.cli.savings_json import receipt_document
    from tests.accounting.test_ux_support import fingerprint

    allocations = [[name, []] for name in ("conservative", "central", "optimistic")]
    exclusions = [
        "candidate:exposure-one:user_supplied",
        "atom:source:path:prior_known",
        "missing:unresolved:source_unresolved:opaque:unavailable",
    ]
    estimates = list(scenarios((0,) * 3, (19,) * 3, (-19,) * 3, "tokens").values())
    for estimate in estimates:
        estimate["known_atoms"] = 0
    write_receipt(
        data,
        unit="context_tokens:chars_div4_v1",
        raw_unit="characters",
        credit_raw=[[name, 0] for name, _ in allocations],
        allocations=allocations,
        exclusions=exclusions,
        exposure_ids=["exposure-one", "exposure-two"],
        reuse={"class": "acted", "action_ok": True, "user_supplied": True},
        eligibility="eligible",
        eligibility_reasons=[],
        scenarios=estimates,
    )
    before = fingerprint(data)
    view = service.read_receipt("personal-local", RECEIPT_ID)
    assert view is not None
    document = receipt_document(view, "personal-local")["data"]
    assert isinstance(document, dict)
    assert document["exposure_ids"] == ["exposure-one", "exposure-two"]
    assert document["allocations"] == {name: [] for name, _ in allocations}
    assert document["credit_raw"] == {name: 0 for name, _ in allocations}
    assert view.reuse is not None
    assert dict(view.reuse)["action_ok"] == "yes"
    assert dict(view.reuse)["user_supplied"] == "yes"
    assert document["exclusions"] == exclusions
    scenario_rows = document["scenarios"]
    assert isinstance(scenario_rows, list) and len(scenario_rows) == 3
    for row in scenario_rows:
        assert isinstance(row, dict)
        credit, debit, net = row["credit"], row["debit"], row["net"]
        assert isinstance(credit, dict) and isinstance(debit, dict) and isinstance(net, dict)
        assert credit["value"] == 0 and debit["value"] == 19 and net["value"] == -19
        assert row["complete"] and row["known_net"] == -19
        assert row["known_atoms"] == 0 and row["known_debits"] == 1
        assert row["credit_missing"] == [] and row["debit_missing"] == []
    text = savings_render.render_receipt(view, width=width).plain
    compact = "".join(text.split())
    assert all(cell_len(line) <= width for line in text.splitlines())
    assert "+0tokens" in compact and "19tokens" in compact
    assert "-19tokens" in compact
    for evidence in (
        "exposure-one",
        "user_supplied",
        "source:path",
        "prior_known",
        "source_unresolved",
    ):
        assert evidence in compact
    assert "unresolved:source_unresolved:opaque" not in compact
    assert "SECRET-CONTENT-CANARY" not in text
    assert fingerprint(data) == before
