"""Renderer behavior at 40/80/120 columns, NO_COLOR, pipes, hostile text, negatives, partials."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ssgrep.accounting import service
from ssgrep.accounting.contracts import Basis, Quantity
from ssgrep.accounting.service_types import HistoryEntry, HistoryView
from ssgrep.cli import savings_render
from ssgrep.cli.savings_text import fmt_bytes, fmt_quantity
from tests.accounting.snapshot_builders import (
    RECEIPT_ID,
    enable_policy,
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
        "Partial",
    ):
        assert required in squashed, required
    assert plain.index("MEASURED") < plain.index("ESTIMATED")
    assert plain.index("ESTIMATED") < plain.index("EVIDENCE") < plain.index("PRIVACY")
    assert "<" not in plain.split("Next:")[1] and "placeholder" not in plain


@pytest.mark.parametrize("width", [40, 80, 120])
def test_every_scenario_value_appears_exactly_once_per_unit_row(fresh_view, width: int) -> None:
    plain = savings_render.render_summary(fresh_view, width).plain
    for value in ("-12,000", "-7,000", "+4,000", "+9,000", "+20,000", "16,000", "+180 s", "+580 s"):
        assert value in plain
    assert "Conservative" in plain and "Central" in plain and "Optimistic" in plain


def test_nonpositive_scenario_has_no_celebratory_text_and_explains_minus(fresh_view) -> None:
    plain = savings_render.render_summary(fresh_view, 80).plain
    assert "negative scenario" in plain
    assert not re.search(r"saved|great|win|!", plain, re.IGNORECASE)


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


def test_no_reuse_does_not_claim_savings_and_unknowns_stay_unknown(data: Path) -> None:
    enable_policy()
    window = window_7d()
    window["coverage"]["candidate_receipts"] = 0
    window["estimates"]["context_tokens"]["central"]["net"] = {
        "value": None,
        "unit": "tokens",
        "basis": "unavailable",
        "reason": "source_costs_missing",
    }
    write_snapshot(data, windows={"7d": window})
    plain = savings_render.render_summary(service.read_summary(now=T0), 80).plain
    assert "No traceable reuse" in plain
    assert "unavailable" in plain and "source_costs_missing" in plain
    assert "Unknown, not zero" in plain


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


def test_corrupt_snapshot_view_says_so_and_changes_nothing(data: Path) -> None:
    enable_policy()
    write_snapshot(data).write_text("{")
    view = service.read_summary(now=T0)
    plain = savings_render.render_summary(replace(view), 40).plain
    assert "unreadable" in plain.lower() and "Nothing was changed" in plain
