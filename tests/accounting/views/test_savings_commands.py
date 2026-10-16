"""Real subprocess runs of the installed ``ssgrep`` binary in an isolated HOME/data dir."""

from __future__ import annotations

import json
import os
import pty
import re
import select
import shlex
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from rich.cells import cell_len
from rich.text import Text

from ssgrep.accounting.collect import state as collector_state, store as collector_store
from ssgrep.accounting.collect.collector import CollectOptions
from ssgrep.accounting.export.consent import DisableReport, EnablePlan
from ssgrep.accounting.ledger.contracts import (
    Attribution,
    Basis,
    Exposure,
    InvocationClass,
    Outcome,
    Quantity,
    Reuse,
)
from ssgrep.accounting.ledger.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.privacy import policy
from ssgrep.accounting.privacy.policy import PolicyError
from ssgrep.accounting.views import ledger as service_ledger, service, snapshots
from ssgrep.accounting.views.history import parse_scenarios
from ssgrep.accounting.views.lookups import LookupRow, LookupsView, next_cursor, read_lookups
from ssgrep.accounting.views.types import (
    HistoryEntry,
    HistoryView,
    NextAction,
    Scope,
    build_scope,
    command_for,
)
from ssgrep.cli import metrics_render, savings_render
from ssgrep.cli.accounting_cli import stdin_is_tty
from ssgrep.cli.collect_render import render_collect, render_running, resume_command
from ssgrep.cli.export_render import render_disable, render_enable_result
from ssgrep.cli.metrics_render import render_lifecycle, render_setup_preview, render_setup_result
from ssgrep.cli.savings_lookup_render import render_lookups
from ssgrep.cli.savings_text import fmt_age, fmt_bytes, fmt_quantity
from ssgrep.services import api
from tests.accounting.views.snapshot_builders import (
    RECEIPT_ID,
    enable_policy,
    scenarios,
    snapshot,
    window_7d,
    write_receipt,
    write_snapshot,
)
from tests.accounting.views.test_ux_support import BINARY, Binary, collected_scene, fingerprint
from tests.pipeline.test_app import fake_models as fake_models, write_claude_session


class Run:
    def __init__(self, home: Path, data: Path) -> None:
        self.home, self.data = home, data

    def __call__(
        self, *args: str, columns: int = 80, stdin: str | None = "", env: dict | None = None
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            **os.environ,
            "HOME": str(self.home),
            "SSGREP_DATA_DIR": str(self.data),
            "COLUMNS": str(columns),
            "NO_COLOR": "1",
            **(env or {}),
        }
        return subprocess.run(
            [str(BINARY), *args],
            input=stdin,
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
            check=False,
        )

    def files(self) -> list[str]:
        if not self.data.exists():
            return []
        return sorted(str(p.relative_to(self.data)) for p in self.data.rglob("*"))


@pytest.fixture
def run(tmp_path: Path) -> Run:
    home = tmp_path / "home"
    home.mkdir()
    return Run(home, tmp_path / "data")


def document(result: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_summary_json_envelope_for_hand_built_snapshot(run: Run) -> None:
    write_snapshot(run.data)
    body = document(run("savings", "--json"))["data"]
    # the fixed 2026 timestamp is far in the past, so this is a stale last-good snapshot
    assert body["state"] == "stale"
    inner = body["data"]
    assert inner["freshness"]["stale"] is True and inner["freshness"]["stale_after_seconds"] == 180
    net = inner["estimates"]["context_tokens"]["conservative"]["net"]
    assert net == {
        "value": -12000,
        "unit": "tokens",
        "basis": "estimated",
        "method": "chars_div4_v1",
        "reason": None,
    }
    assert inner["estimates_are_causal"] is False
    assert inner["measured"]["searches"]["value"] == 100
    assert inner["coverage"]["attributed_lookups"] == 91
    assert body["provenance"]["snapshot_id"] == "a" * 16
    assert body["next_actions"][0]["command"] == "ssgrep savings collect"
    assert "\x1b" not in run("savings", "--json").stdout
    assert "." not in json.dumps(inner["estimates"]).replace("chars_div4_v1", "")


def test_setup_without_tty_and_without_yes_exits_2_and_changes_nothing(run: Run) -> None:
    result = run("savings", "setup", stdin="")
    assert result.returncode == 2 and "--yes" in result.stderr
    assert run.files() == []
    as_json = run("savings", "setup", "--json")
    assert as_json.returncode == 2 and document(as_json)["condition"] == "confirmation_required"
    assert run.files() == []


def test_setup_rejects_invalid_policy_without_side_effects(run: Run, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"profile_id": "acme", "roots": ["relative/path"]}')
    result = run("savings", "setup", "--yes", "--policy", str(bad), "--profile", "acme", "--json")
    assert result.returncode == 2 and document(result)["condition"] == "invalid_policy"
    garbled = tmp_path / "garbled.json"
    garbled.write_text("{nope")
    assert run("savings", "setup", "--yes", "--policy", str(garbled)).returncode == 2
    assert (
        run("savings", "setup", "--yes", "--policy", str(tmp_path / "absent.json")).returncode == 2
    )
    assert run.files() == []


def test_cancelled_interactive_setup_leaves_no_files(run: Run) -> None:
    master, slave = pty.openpty()
    environment = {
        **os.environ,
        "HOME": str(run.home),
        "SSGREP_DATA_DIR": str(run.data),
        "NO_COLOR": "1",
    }
    process = subprocess.Popen(
        [str(BINARY), "savings", "setup"],
        stdin=slave,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        text=True,
    )
    os.write(master, b"n\n")
    stdout, stderr = process.communicate(timeout=120)
    os.close(master)
    os.close(slave)
    assert process.returncode == 0
    assert "approved roots".upper() in stdout.upper() and "No export is enabled" in stdout
    assert "Cancelled" in stderr
    assert run.files() == []


def test_first_use_journey_end_to_end(run: Run) -> None:
    assert "savings setup" in run("savings").stdout
    assert run("savings", "setup", stdin="").returncode == 2
    assert run.files() == []
    confirmed = run("savings", "setup", "--yes")
    assert confirmed.returncode == 0 and "Export remains off" in confirmed.stdout
    policy_file = run.data / "accounting" / "policy.json"
    assert policy_file.stat().st_mode & 0o777 == 0o600
    stored = json.loads(policy_file.read_text())["profiles"]["personal-local"]
    assert stored["collection_enabled"] is True and stored["enabled_exports"] == []
    pending = run("savings", "--no-refresh")
    assert "ssgrep savings collect" in pending.stdout
    assert "tokens" not in pending.stdout
    paused = run("savings", "pause")
    assert paused.returncode == 0 and "resume" in paused.stdout
    assert "PAUSED" in run("savings").stdout and "ssgrep savings resume" in run("savings").stdout
    status = document(run("savings", "status", "--json"))["data"]
    assert status["state"] == "paused" and status["data"]["paused_intervals"][0]["until"] is None
    resumed = run("savings", "resume", "--json")
    assert document(resumed)["data"]["state"] == "resumed"
    after = json.loads(policy_file.read_text())["profiles"]["personal-local"]
    assert after["paused"] is False and after["enabled_exports"] == []
    assert document(run("savings", "status", "--json"))["data"]["state"] == "no_snapshot"
    assert not (run.data / "lancedb").exists(), "no index was created by any accounting command"


def test_profile_is_preserved_in_real_next_actions(run: Run, tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    policy_path = tmp_path / "acme.json"
    policy_path.write_text(json.dumps({"profile_id": "acme", "roots": [str(root)]}))
    ok = run(
        "savings", "setup", "--yes", "--policy", str(policy_path), "--profile", "acme", "--json"
    )
    assert ok.returncode == 0, ok.stderr
    assert (
        document(ok)["data"]["next_actions"][0]["command"]
        == "ssgrep savings collect --profile acme"
    )
    summary = document(run("savings", "--profile", "acme", "--no-refresh", "--json"))["data"]
    assert summary["state"] == "no_snapshot"
    assert summary["next_actions"][0]["command"] == "ssgrep savings collect --profile acme"
    personal = document(run("savings", "--json"))["data"]
    assert personal["state"] == "disabled", "an organization profile is never inferred or shared"


def test_hostile_receipt_text_through_the_real_binary_has_no_control_bytes(run: Run) -> None:
    write_receipt(run.data, root="/w/\x1b[2J[bold red]x[/]\x07", candidate_type="\x1b]0;t\x07")
    result = run(
        "savings", "explain", RECEIPT_ID, columns=40, env={"FORCE_COLOR": "1", "NO_COLOR": ""}
    )
    assert result.returncode == 0
    assert "\x1b" not in result.stdout and "\x07" not in result.stdout
    assert "\\x1b[2J" in " ".join(result.stdout.split())


# ------------------------------------------------------------ consent prompts and policy


def environment(run: Run) -> dict[str, str]:
    return {**os.environ, "HOME": str(run.home), "SSGREP_DATA_DIR": str(run.data), "NO_COLOR": "1"}


@contextmanager
def prompt(run: Run, *arguments: str):
    """Wait for an actual consent prompt before sending input or changing its policy."""
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [str(BINARY), *arguments],
        stdin=slave,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment(run),
    )
    os.close(slave)
    preview = bytearray()
    try:
        assert process.stdout is not None
        deadline = time.monotonic() + 30
        while b"[y/N] " not in preview:
            remaining = deadline - time.monotonic()
            assert remaining > 0, bytes(preview)
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            assert ready, bytes(preview)
            chunk = os.read(process.stdout.fileno(), 65536)
            assert chunk, (process.poll(), bytes(preview))
            preview.extend(chunk)
        yield process, master
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
        os.close(master)


def export_arguments() -> tuple[str, ...]:
    from tests.accounting.export.test_consent import DEST

    return ("savings", "export", "enable", DEST)


@pytest.mark.parametrize("command", ["setup", "export"])
@pytest.mark.parametrize("answer", [b"yes\n", b"no\n", b"\x04"], ids=["approve", "refuse", "eof"])
def test_interactive_consent_changes_only_the_approved_state(run: Run, command: str, answer: bytes):
    from tests.accounting.export.test_consent import DEST, configure, enabled_pairs

    if command == "export":
        configure()
    before = policy.policy_path().read_bytes() if policy.policy_path().exists() else None
    args = ("savings", "setup") if command == "setup" else export_arguments()
    with prompt(run, *args) as (process, terminal):
        os.write(terminal, answer)
        stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stderr
    assert b"Traceback" not in stdout + stderr
    approved = answer == b"yes\n"
    if command == "setup":
        assert policy.load_profile_policy().collection_enabled is approved
    else:
        assert enabled_pairs() == (((DEST, "metrics"),) if approved else ())
    if not approved:
        after = policy.policy_path().read_bytes() if policy.policy_path().exists() else None
        assert after == before


@pytest.mark.parametrize("command", ["setup", "export"])
def test_consent_revalidates_policy_changed_while_prompt_is_open(run: Run, command: str):
    from tests.accounting.export.test_consent import configure

    configure()
    args = ("savings", "setup") if command == "setup" else export_arguments()
    with prompt(run, *args) as (process, terminal):
        policy.policy_path().write_text("{")
        os.write(terminal, b"yes\n")
        stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 1
    assert policy.policy_path().read_bytes() == b"{"
    assert b"ssgrep savings setup" in stderr
    assert b"Traceback" not in stdout + stderr


def test_oversized_setup_policy_is_rejected_before_creating_accounting_state(
    run: Run, tmp_path: Path
):
    source = tmp_path / "oversized.json"
    source.write_bytes(b" " * (256 * 1024 + 1))
    result = run("savings", "setup", "--yes", "--policy", str(source), "--json")
    assert result.returncode == 2
    assert document(result)["condition"] == "invalid_policy"
    assert not run.data.exists()


def test_closed_or_absent_stdin_cannot_grant_interactive_consent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    with (tmp_path / "stdin").open("w+") as stream:
        monkeypatch.setattr(sys, "stdin", stream)
        assert stdin_is_tty() is False
    assert stdin_is_tty() is False
    monkeypatch.setattr(sys, "stdin", None)
    assert stdin_is_tty() is False
    master, slave = pty.openpty()
    try:
        with os.fdopen(slave) as terminal:
            monkeypatch.setattr(sys, "stdin", terminal)
            assert stdin_is_tty() is True
    finally:
        os.close(master)


@pytest.mark.parametrize("roots", [(), ("/w/team\x1b[2J",)])
def test_setup_preview_preserves_roots_retention_and_disabled_destinations(roots: tuple[str, ...]):
    from tests.accounting.export.test_consent import DEST, ENDPOINT

    p = ProfilePolicy(
        profile_id="acme",
        roots=roots,
        evidence_retention_days=17,
        receipt_retention_days=43,
        destinations=(DestinationPolicy(DEST, ENDPOINT),),
    )
    text = flat(
        metrics_render.render_setup_preview(
            p, width=80, already_enabled=False, source="managed"
        ).plain
    )
    assert "17 days" in text and "43 days" in text and DEST in text
    assert "disabled" in text
    assert "\x1b" not in text
    if roots:
        assert "/w/team\\x1b[2J" in text
    else:
        assert "none approved" in text and "all working directories" not in text


def test_summary_scope_filters_preserve_hash_prefixed_values(run: Run) -> None:
    result = run(
        "savings",
        "--project",
        "#notes",
        "--runtime",
        "omp",
        "--session",
        "#session",
        "--json",
    )
    assert result.returncode == 0, result.stderr
    selected = document(result)["data"]["data"]["scope"]
    assert {key: selected[key] for key in ("project", "runtime", "session")} == {
        "project": "#notes",
        "runtime": "omp",
        "session": "#session",
    }


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
    from ssgrep.accounting.privacy import policy

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
    assert "ssgrep savings collect" in plain
    assert "last good values" in plain


def test_paused_view_names_timestamp_and_resume(data: Path) -> None:
    enable_policy(paused=True)
    write_snapshot(data)
    view = service.read_summary(now=T0 + timedelta(seconds=10))
    plain = savings_render.render_summary(view, 80).plain
    assert "PAUSED" in plain and "Paused since" in plain and "-12,000 tokens" in plain
    assert "ssgrep savings resume" in plain


def test_paused_view_with_an_old_snapshot_says_both_paused_and_stale(data: Path) -> None:
    enable_policy(paused=True)
    write_snapshot(data)
    fresh = savings_render.render_summary(
        service.read_summary(now=T0 + timedelta(seconds=10)), 80
    ).plain
    assert "PAUSED" in fresh and "snapshot also stale" not in fresh

    view = service.read_summary(now=T0 + timedelta(seconds=7300))
    assert view.paused and view.stale
    plain = savings_render.render_summary(view, 80).plain
    assert "PAUSED (snapshot also stale)" in flat(plain)
    assert "ssgrep savings resume" in plain


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
    command = "ssgrep savings collect" if state_setup == "no_snapshot" else "ssgrep savings setup"
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
    assert "-70 tokens" in plain and "--page h1.2.deadbeef" in flat(plain)
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
def test_grouped_summary_explains_that_continuation_is_unavailable(data: Path, width: int) -> None:
    enable_policy()
    write_snapshot(data)
    view = service.read_summary(now=T0)
    grouped = replace(
        view,
        scope=Scope(by="task"),
        group_total=5,
        next_cursor="cursor-1",
    )
    text = flat(savings_render.render_summary(grouped, width).plain)
    assert "Additional grouped rows exist; this summary does not expose group paging." in text
    assert "More:" not in text and "--cursor" not in text


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
    from tests.accounting.views.test_ux_support import fingerprint

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
    from tests.accounting.views.test_ux_support import fingerprint

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


PROFILE = "p" * 32
DESTINATION = "destination-" + "d" * 20
WIDTHS = (30, 40, 80, 120)


@pytest.fixture
def isolated_rendering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "SSGREP_DATA_DIR"):
        root = tmp_path / key
        root.mkdir()
        monkeypatch.setenv(key, str(root))
    sentinel = tmp_path / "unchanged"
    sentinel.write_bytes(b"rendering must be read-only")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    yield
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before


def copied_commands(rendered: Text, width: int) -> list[list[str]]:
    """Copy shell command lines, including escaped continuations, not prose/headings."""
    assert "\x1b" not in rendered.plain
    lines = rendered.plain.splitlines()
    commands: list[list[str]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith(("Next: ssgrep ", "Or: ssgrep ")):
            line = line.split(": ", 1)[1]
        elif not line.startswith("ssgrep savings "):
            index += 1
            continue
        # Bare heading examples are not copy actions.
        elif index == 0:
            index += 1
            continue
        block = [line]
        while block[-1].endswith("\\"):
            index += 1
            assert index < len(lines)
            block.append(lines[index])
        assert all(len(part) <= min(max(width, 30), 100) for part in block)
        commands.append(shlex.split("\n".join(block).replace("\\\n", "")))
        index += 1
    return commands


def argv(action: str) -> list[str]:
    return ["ssgrep", *shlex.split(action), "--profile", PROFILE]


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.usefixtures("isolated_rendering")
def test_setup_recovery_commands_preserve_selected_profile(width: int) -> None:
    rendered = render_setup_result(ProfilePolicy(profile_id=PROFILE), width)
    assert copied_commands(rendered, width) == [argv("savings collect"), argv("savings status")]


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.usefixtures("isolated_rendering")
def test_setup_cache_revocation_targets_selected_profile(width: int) -> None:
    policy = ProfilePolicy(
        profile_id=PROFILE,
        evidence_cache_enabled=True,
        evidence_cache_retention_days=1,
        evidence_cache_max_bytes=1024,
    )
    rendered = render_setup_preview(policy, width=width, already_enabled=False, source="local")
    assert copied_commands(rendered, width) == [argv("savings setup --no-evidence-cache --yes")]


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("action", ("resume", "status"))
@pytest.mark.usefixtures("isolated_rendering")
def test_pause_resume_recovery_commands_preserve_profile(width: int, action: str) -> None:
    rendered = render_lifecycle(
        "Measurement state changed.", (), command_for(PROFILE, f"savings {action}"), width
    )
    assert copied_commands(rendered, width) == [argv(f"savings {action}")]


def enable_plan(signal: Signal, paused: bool) -> EnablePlan:
    return EnablePlan(
        PROFILE,
        DESTINATION,
        signal,
        "https://collector.example/v1/metrics",
        None,
        None,
        "v1",
        (),
        (),
        None,
        1,
        "local",
        False,
        {"evidence_days": 1, "receipt_days": 1, "spool_days": 1, "spool_bytes": 1024},
        False,
        paused,
        (),
    )


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("signal", (Signal.METRICS, Signal.LOGS))
@pytest.mark.parametrize("changed,paused", ((True, False), (False, True)))
@pytest.mark.usefixtures("isolated_rendering")
def test_export_enable_actions_preserve_target(
    width: int, signal: Signal, changed: bool, paused: bool
) -> None:
    rendered = render_enable_result(enable_plan(signal, paused), changed, width)
    action = " --logs" if signal is Signal.LOGS else ""
    assert copied_commands(rendered, width) == [
        argv(f"savings export check {DESTINATION}{action}"),
        argv(f"savings export preview {DESTINATION}{action}"),
    ]


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("enabled,in_flight", ((True, None), (False, "batch-" + "b" * 64)))
@pytest.mark.usefixtures("isolated_rendering")
def test_export_disable_recovery_preserves_target(
    width: int, enabled: bool, in_flight: str | None
) -> None:
    report = DisableReport(DESTINATION, Signal.LOGS, enabled, 2, in_flight, in_flight)
    rendered = render_disable(report, PROFILE, width)
    assert copied_commands(rendered, width) == [argv(f"savings export enable {DESTINATION} --logs")]


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("mode", ("live", "backfill", "recompute"))
@pytest.mark.parametrize("state", ("partial", "published"))
@pytest.mark.usefixtures("isolated_rendering")
def test_collect_recovery_and_alternative_preserve_mode_interval_profile(
    width: int, mode: str, state: str
) -> None:
    options = CollectOptions(
        backfill=mode == "backfill",
        recompute=mode == "recompute",
        since="2026-01-01T00:00:00Z",
        until="2026-02-01T00:00:00Z",
    )
    actions = (
        (
            NextAction(resume_command(PROFILE, options), "Continue collection."),
            NextAction(command_for(PROFILE, "savings"), "Read saved results."),
        )
        if state == "partial"
        else (NextAction(command_for(PROFILE, "savings"), "Read saved results."),)
    )
    data = {
        "state": state,
        "profile": PROFILE,
        "mode": mode,
        "epoch": 1,
        "generation": 2,
        "snapshot_id": "s" * 64,
        "bytes_read": 1,
        "lag_bytes": 2,
        "stopped_by": "byte_budget",
        "new_epoch": False,
        "interval": {"since": options.since, "until": options.until},
        "backfill": {"complete": False, "counts": []},
        "unavailable_periods": [],
    }
    rendered = render_collect(data, (), actions, width)
    expected = [argv("savings")]
    if state == "partial":
        if mode == "backfill":
            command = f"savings collect --backfill {options.since} {options.until}"
        elif mode == "recompute":
            command = "savings collect --recompute"
        else:
            command = "savings collect"
        expected.insert(0, argv(command))
    assert copied_commands(rendered, width) == expected


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.usefixtures("isolated_rendering")
def test_running_worker_recovery_preserves_profile(width: int) -> None:
    owner = dict.fromkeys(
        (
            "pid",
            "kind",
            "phase",
            "sources_done",
            "sources_total",
            "bytes_read",
            "lag_bytes",
            "updated_at",
        )
    )
    rendered = render_running(
        {"owner": owner},
        (NextAction(command_for(PROFILE, "savings status"), "Inspect worker."),),
        width,
    )
    assert copied_commands(rendered, width) == [argv("savings status")]


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.usefixtures("isolated_rendering")
def test_lookup_unavailable_recovery_preserves_profile(width: int) -> None:
    rendered = render_lookups(LookupsView(PROFILE, Scope(), reason="no_snapshot"), width)
    assert copied_commands(rendered, width) == [argv("savings status")]


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.usefixtures("isolated_rendering")
def test_lookup_continuation_preserves_cursor_and_all_scope_filters(width: int) -> None:
    scope = Scope(
        since="2026-01-01",
        until="2026-02-01",
        project="/approved/project with spaces",
        runtime="claude",
        session="session-" + "s" * 64,
        task="task-" + "t" * 64,
        include_children=True,
    )
    cursor = next_cursor(17, "selection-binding", "s" * 64, 2)
    rendered = render_lookups(LookupsView(PROFILE, scope, next_cursor=cursor, limit=17), width)
    expected = ["ssgrep", "savings", "lookups"]
    for name, flag in (
        ("since", "--from"),
        ("until", "--to"),
        ("project", "--project"),
        ("runtime", "--runtime"),
        ("session", "--session"),
        ("task", "--task"),
    ):
        value = getattr(scope, name)
        assert isinstance(value, str)
        expected.extend((flag, value))
    expected.extend(("--children", "--page", cursor, "--profile", PROFILE))
    assert copied_commands(rendered, width) == [expected]


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.usefixtures("isolated_rendering")
def test_lookup_receipt_command_preserves_complete_receipt_id(width: int) -> None:
    receipt = "receipt-" + "a" * 56
    row = LookupRow(
        "lookup",
        None,
        (),
        (),
        None,
        None,
        None,
        "task",
        "unknown",
        "unknown",
        "local",
        None,
        (),
        True,
        (),
        "executed",
        "unknown",
        "unknown",
        "unknown",
        False,
        "unknown",
        (),
        "unknown",
        "unknown",
        "unknown",
        Quantity.unavailable("ms", "missing"),
        Quantity.unavailable("bytes", "missing"),
        False,
        receipt_ids=(receipt,),
    )
    rendered = render_lookups(LookupsView(PROFILE, Scope(), rows=(row,), total=1), width)
    assert copied_commands(rendered, width) == [argv(f"savings explain {receipt}")]


@pytest.fixture
def scene(tmp_path):
    collected_scene(tmp_path, pairs=(0,))
    data, reason = service_ledger.load_ledger("personal-local")
    assert data is not None and data.doc is not None
    assert reason is None and data.doc.invocations
    invocation = next(
        r
        for r in data.doc.invocations
        if set(r.operations) & {"search", "show"} and r.execution_ids == ("n-0",)
    )
    additions = tuple(
        replace(
            invocation,
            invocation_id=name,
            call_id=name,
            execution_ids=(),
            ambiguous_execution_ids=(),
            outcome=outcome,
            exposure=Exposure.OBSERVED,
            reuse=reuse,
            cls=InvocationClass.TRANSCRIPT_ONLY,
            attribution=Attribution.UNATTRIBUTED,
        )
        for name, outcome, reuse in (
            ("empty-lookup", Outcome.EMPTY, Reuse.NONE_OBSERVED),
            ("error-lookup", Outcome.ERROR, Reuse.NONE_OBSERVED),
            ("inspected-lookup", Outcome.OK, Reuse.INSPECTED),
        )
    )
    doc = replace(data.doc, invocations=(*data.doc.invocations, *additions))
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    checkpoint.blobs[collector_state.REDUCTION] = collector_store.write_blob(
        "personal-local", collector_state.REDUCTION, doc.to_dict()
    )
    collector_store.commit_checkpoint("personal-local", checkpoint)
    return tmp_path, invocation


def test_pages_enumerate_all_lookup_ids_once_including_no_receipt_and_journal_only(scene):
    root, invocation = scene
    before = fingerprint(root / "data")
    page = read_lookups(limit=1)
    rows = list(page.rows)
    while page.next_cursor:
        page = read_lookups(limit=1, cursor=page.next_cursor)
        rows.extend(page.rows)
    assert {r.lookup_id for r in rows} == {
        f"invocation:{invocation.invocation_id}",
        "invocation:empty-lookup",
        "invocation:error-lookup",
        "invocation:inspected-lookup",
        "execution:n-start",
    }
    assert len(rows) == len({r.lookup_id for r in rows}) == 5
    by_id = {r.lookup_id: r for r in rows}
    assert by_id["invocation:empty-lookup"].receipt_ids == ()
    assert by_id["invocation:empty-lookup"].reuse == "none_observed"
    assert by_id["invocation:error-lookup"].outcome == "error"
    assert by_id["invocation:inspected-lookup"].reuse == "inspected"
    assert by_id["execution:n-start"].eligibility == "unavailable"
    assert fingerprint(root / "data") == before


def test_lookup_filters_intersect_and_cursor_is_bound_to_selection_and_rows(scene):
    _, invocation = scene
    page = read_lookups(limit=1)
    with pytest.raises(PolicyError, match="Page token"):
        read_lookups(scope=build_scope(runtime="codex"), limit=1, cursor=page.next_cursor)
    scoped = read_lookups(
        scope=build_scope(
            runtime=invocation.runtime,
            session=invocation.session_id,
            task=invocation.task.root,
            project=invocation.cwd,
        ),
        limit=100,
    )
    assert len(scoped.rows) == 4 and all(r.runtime == invocation.runtime for r in scoped.rows)
    assert read_lookups(scope=build_scope(runtime="codex"), limit=100).rows == ()


def test_lookup_old_generation_is_honestly_expired_not_current_data(scene):
    root, _ = scene
    page = read_lookups(limit=1)
    path = root / "data" / "accounting" / "personal-local" / "snapshots" / "current.json"
    raw = json.loads(path.read_text())
    raw["generation"] += 1
    raw["snapshot_id"] = "new-snapshot"
    path.write_text(json.dumps(raw))
    with pytest.raises(PolicyError) as caught:
        read_lookups(
            snapshot_id=page.snapshot_id,
            generation=page.generation,
            limit=1,
            cursor=page.next_cursor,
        )
    assert caught.value.condition == "lookup_snapshot_expired"


@pytest.mark.parametrize("cursor", ["broken", "lookups1.***", "lookups1.e30"])
def test_lookup_malformed_pages_are_rejected_without_mutating_evidence(scene, cursor: str) -> None:
    root, _ = scene
    before = fingerprint(root / "data")
    with pytest.raises(PolicyError) as caught:
        read_lookups(cursor=cursor)
    assert caught.value.condition == "invalid_cursor"
    assert fingerprint(root / "data") == before


def test_lookup_cursor_cannot_override_an_explicit_snapshot(scene) -> None:
    page = read_lookups(limit=1)
    assert page.next_cursor is not None
    with pytest.raises(PolicyError) as caught:
        read_lookups(snapshot_id="different-snapshot", cursor=page.next_cursor, limit=1)
    assert caught.value.condition == "invalid_cursor"


def commit_reduction_doc(transform) -> None:
    doc, _, _, _ = service_ledger.load_lookup_data("personal-local")
    assert doc is not None
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    checkpoint.blobs[collector_state.REDUCTION] = collector_store.write_blob(
        "personal-local", collector_state.REDUCTION, transform(doc).to_dict()
    )
    collector_store.commit_checkpoint("personal-local", checkpoint)


def test_lookup_conflicting_identity_withholds_rows_instead_of_choosing_one(scene):
    _, invocation = scene
    assert len(read_lookups(limit=100).rows) == 5  # positive control
    commit_reduction_doc(
        lambda doc: replace(
            doc,
            invocations=(
                *doc.invocations,
                replace(invocation, outcome=Outcome.EMPTY, reuse=Reuse.NONE_OBSERVED),
            ),
        )
    )
    view = read_lookups(limit=100)
    assert view.rows == () and view.total == 0 and view.next_cursor is None
    assert view.reason == "conflicting_lookup_identity"
    assert view.snapshot_id is not None and view.generation is not None


def test_lookup_cursor_for_a_pending_generation_is_expired_not_empty(scene):
    root, _ = scene
    path = root / "data/accounting/personal-local/snapshots/current.json"
    raw = json.loads(path.read_text())
    raw["generation"] += 1
    path.write_text(json.dumps(raw))
    cursor = next_cursor(1, "unbound", raw["snapshot_id"], raw["generation"])
    before = fingerprint(root / "data")
    with pytest.raises(PolicyError) as caught:
        read_lookups(limit=1, cursor=cursor)
    assert caught.value.condition == "lookup_snapshot_expired"
    assert fingerprint(root / "data") == before
    assert read_lookups(limit=1).reason == "lookup_generation_pending"  # same state, no cursor


@pytest.mark.parametrize(
    ("field", "value"),
    [("snapshot", "s" * 513), ("snapshot", 7), ("page", 7), ("page", None)],
)
def test_lookup_cursor_rejects_wrongly_typed_or_oversized_fields(scene, field, value):
    import base64

    root, _ = scene
    first = read_lookups(limit=1)
    assert first.next_cursor is not None
    token = first.next_cursor
    raw = json.loads(base64.urlsafe_b64decode(token[9:] + "=" * (-len(token[9:]) % 4)))
    assert read_lookups(cursor=token, limit=1).rows  # positive control: untouched token works
    raw[field] = value
    forged = "lookups1." + base64.urlsafe_b64encode(json.dumps(raw).encode()).decode()
    before = fingerprint(root / "data")
    with pytest.raises(PolicyError) as caught:
        read_lookups(cursor=forged, limit=1)
    assert caught.value.condition == "invalid_cursor"
    assert fingerprint(root / "data") == before


def test_unreadable_receipt_is_not_offered_and_marks_the_explanation_unavailable(scene):
    root, _ = scene
    original = read_lookups(limit=100)
    linked = next(row for row in original.rows if row.receipt_ids)
    receipt_id = linked.receipt_ids[0]
    path = root / "data/accounting/personal-local/receipts" / f"{receipt_id}.json"
    path.write_text("{not json")
    reread = read_lookups(limit=100)
    row = next(row for row in reread.rows if row.lookup_id == linked.lookup_id)
    assert receipt_id not in row.receipt_ids
    assert "receipt_explanation_unavailable" in row.missing_evidence
    assert row.execution == linked.execution
    assert reread.total == original.total


def test_legacy_lookup_metadata_is_flagged_and_never_claims_eligibility(scene):
    assert read_lookups(limit=100).coverage["lookup_metadata_current"] is True
    commit_reduction_doc(lambda doc: replace(doc, lookup_schema="lookups_v0"))
    view = read_lookups(limit=100)
    assert "lookup_metadata_unavailable" in view.warnings
    assert view.coverage["lookup_metadata_current"] is False
    assert view.rows
    assert all(r.eligibility == "unavailable" for r in view.rows)
    assert all("lookup_metadata_unavailable" in r.eligibility_reasons for r in view.rows)


def test_lookup_gate_reasons_for_verifier_actor_and_unobserved_exposure(scene):
    from ssgrep.accounting.views.lookups import project_lookup

    _, invocation = scene
    doc, _, ledger, _ = service_ledger.load_lookup_data("personal-local")
    assert doc is not None and ledger is not None
    base = replace(
        invocation,
        execution_ids=(),
        ambiguous_execution_ids=(),
        outcome=Outcome.OK,
        exposure=Exposure.OBSERVED,
        reuse=Reuse.NONE_OBSERVED,
        reuse_evaluated=True,
        cls=InvocationClass.TRANSCRIPT_ONLY,
        attribution=Attribution.UNATTRIBUTED,
        actor="agent",
        actor_basis="runtime",
    )
    observed = project_lookup("invocation", "plain-lookup", base, doc, ledger)
    assert observed.eligibility == "known_rejected"  # positive control: observed, no reuse
    assert "no_supported_reuse" in observed.eligibility_reasons
    verifier = project_lookup(
        "invocation",
        "verifier-lookup",
        replace(base, actor="verifier", actor_basis="configured_verifier"),
        doc,
        ledger,
    )
    assert verifier.eligibility == "known_rejected"
    assert "excluded_verifier_actor" in verifier.eligibility_reasons
    blind = project_lookup(
        "invocation",
        "blind-lookup",
        replace(base, exposure=Exposure.UNAVAILABLE),
        doc,
        ledger,
    )
    assert blind.eligibility == "unavailable"
    assert "downstream_exposure_unobserved" in blind.eligibility_reasons


def test_actual_cli_lookup_command_is_read_only_bounded_and_json_complete(scene):
    root, _ = scene
    before = fingerprint(root / "data")
    binary = Binary(root)
    status, body = binary.json("savings", "lookups")
    assert status == 0, body
    data = body["data"]["data"]
    assert len(data["rows"]) == 5 and data["total"] == 5 and data["next_cursor"] is None
    for row in data["rows"]:
        assert {
            "execution",
            "exposure",
            "attribution",
            "reuse",
            "eligibility",
            "outcome",
            "scenarios",
        } <= set(row)
        assert row["causal_usefulness"] is None
    assert fingerprint(root / "data") == before


def test_default_lookup_page_keeps_unplaced_calls_without_inventing_time_membership(scene):
    _, invocation = scene
    data, _ = service_ledger.load_ledger("personal-local")
    assert data is not None and data.doc is not None
    doc = replace(
        data.doc,
        invocations=tuple(
            replace(r, ts_utc=None, timing_known=False)
            if r.invocation_id == invocation.invocation_id
            else r
            for r in data.doc.invocations
        ),
    )
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    checkpoint.blobs[collector_state.REDUCTION] = collector_store.write_blob(
        "personal-local", collector_state.REDUCTION, doc.to_dict()
    )
    collector_store.commit_checkpoint("personal-local", checkpoint)
    view = read_lookups(limit=100)
    row = next(r for r in view.rows if r.invocation_id == invocation.invocation_id)
    assert row.ts_utc is None and not row.timing_known
    assert any(w.startswith("untimed_lookup_rows:") for w in view.warnings)
    exact = read_lookups(scope=build_scope(since="2026-10-12", until="2026-10-13"), limit=100)
    assert all(r.invocation_id != invocation.invocation_id for r in exact.rows)


def test_skipped_reuse_evaluation_is_not_observed_no_reuse_or_known_zero_eligibility(scene):
    _, invocation = scene
    data, _ = service_ledger.load_ledger("personal-local")
    assert data is not None and data.doc is not None
    skipped = replace(
        invocation,
        invocation_id="not-evaluated",
        execution_ids=(),
        ambiguous_execution_ids=(),
        reuse=Reuse.NONE_OBSERVED,
        reuse_evaluated=False,
        outcome=Outcome.OK,
        exposure=Exposure.PARTIAL,
        cls=InvocationClass.TRANSCRIPT_ONLY,
        attribution=Attribution.UNATTRIBUTED,
    )
    doc = replace(data.doc, invocations=(*data.doc.invocations, skipped))
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    checkpoint.blobs[collector_state.REDUCTION] = collector_store.write_blob(
        "personal-local", collector_state.REDUCTION, doc.to_dict()
    )
    collector_store.commit_checkpoint("personal-local", checkpoint)
    row = next(r for r in read_lookups(limit=100).rows if r.invocation_id == "not-evaluated")
    assert row.reuse == "unavailable" and row.useful_outcome == "not_evaluated"
    assert row.eligibility == "unavailable" and "reuse_not_evaluated" in row.missing_evidence


@pytest.mark.parametrize("width", [40, 80, 120])
def test_lookup_binary_text_and_json_preserve_evidence_and_receipt_actions(scene, width):
    import subprocess

    from rich.cells import cell_len

    root, _ = scene
    before = fingerprint(root / "data")
    binary = Binary(root)
    status, envelope = binary.json("savings", "lookups")
    assert status == 0, envelope
    rows = envelope["data"]["data"]["rows"]
    assert len(rows) == 5
    result = binary("savings", "lookups", columns=width)
    assert result.returncode == 0, result.stderr
    assert "\x1b" not in result.stdout
    assert all(cell_len(line) <= width for line in result.stdout.splitlines())
    lines = result.stdout.splitlines()
    shown_commands = []
    for position, line in enumerate(lines):
        offset = line.find("ssgrep savings explain")
        if offset < 0:
            continue
        block = [line[offset:]]
        while block[-1].endswith("\\"):
            position += 1
            block.append(lines[position])
        parsed = subprocess.run(
            [
                "/bin/sh",
                "-c",
                "set -e; ssgrep() { printf '%s\\0' ssgrep \"$@\"; };\n" + "\n".join(block),
            ],
            cwd=root,
            capture_output=True,
            check=True,
        )
        shown_commands.append([part.decode() for part in parsed.stdout.split(b"\0")[:-1]])
    compact = "".join(result.stdout.split())
    links = []
    known_costs = []
    unknown = []
    for row in rows:
        assert row["lookup_id"] in compact
        for field in ("execution", "outcome", "exposure", "attribution", "reuse", "eligibility"):
            assert row[field] in compact
        for name, quantity in row["known_costs"].items():
            known_costs.append(quantity)
            assert name in compact
            if quantity["value"] is not None:
                unit = {"context_tokens": "tokens", "work_seconds": "s"}.get(
                    quantity["unit"], quantity["unit"]
                )
                assert f"{quantity['value']:,}{unit}" in compact
        unknown.extend(row["missing_evidence"])
        for reason in row["missing_evidence"]:
            assert reason in compact
        for receipt_id in row["receipt_ids"]:
            links.append(receipt_id)
            copied = [tokens for tokens in shown_commands if tokens[-1] == receipt_id]
            assert copied == [["ssgrep", "savings", "explain", receipt_id]]
            code, receipt = binary.json(*copied[0][1:])
            assert code == 0 and receipt["data"]["data"]["receipt_id"] == receipt_id
        assert row["causal_usefulness"] is None
    assert links and known_costs and unknown
    assert sorted(tokens[-1] for tokens in shown_commands) == sorted(links)
    assert fingerprint(root / "data") == before


@pytest.mark.parametrize("width", [40, 80, 120])
def test_lookup_next_command_roundtrips_quoted_scope_and_children_without_writes(scene, width):
    import subprocess

    from rich.cells import cell_len

    from ssgrep.cli.savings_lookup_render import render_lookups

    root, invocation = scene
    data, _ = service_ledger.load_ledger("personal-local")
    assert data is not None and data.doc is not None
    project = str(root / ("project " + "'`" * 40 + " ; $not_a_command"))
    base = tuple(replace(row, cwd=project) for row in data.doc.invocations)
    extra = tuple(
        replace(
            invocation,
            invocation_id=f"lookup-extra-{i}",
            call_id=f"lookup-extra-{i}",
            execution_ids=(),
            ambiguous_execution_ids=(),
            cwd=project,
        )
        for i in range(20)
    )
    doc = replace(data.doc, invocations=(*base, *extra))
    checkpoint = collector_store.load_checkpoint("personal-local")
    assert checkpoint is not None
    checkpoint.blobs[collector_state.REDUCTION] = collector_store.write_blob(
        "personal-local", collector_state.REDUCTION, doc.to_dict()
    )
    collector_store.commit_checkpoint("personal-local", checkpoint)
    scope = build_scope(
        project=project,
        session=invocation.session_id,
        runtime=invocation.runtime,
        include_children=True,
    )
    first = read_lookups(scope=scope, limit=20)
    assert first.rows and first.next_cursor
    binary = Binary(root)
    before = fingerprint(root / "data")
    text = render_lookups(first, width)
    lines = text.plain.splitlines()
    assert all(cell_len(line) <= width for line in lines)
    start = next(i for i, line in enumerate(lines) if line.startswith("ssgrep savings lookups"))
    command = "\n".join(lines[start:])
    parsed = subprocess.run(
        ["/bin/sh", "-c", "set -e; ssgrep() { printf '%s\\0' ssgrep \"$@\"; };\n" + command],
        cwd=root,
        capture_output=True,
        check=True,
    )
    tokens = [part.decode() for part in parsed.stdout.split(b"\0")[:-1]]
    assert tokens[:3] == ["ssgrep", "savings", "lookups"]
    assert tokens[tokens.index("--project") + 1] == project
    assert tokens[tokens.index("--session") + 1] == invocation.session_id
    assert "--children" in tokens and "--page" in tokens and "--limit" not in tokens
    code, second = binary.json(*tokens[1:])
    assert code == 0, second
    second_data = second["data"]["data"]
    expected = read_lookups(scope=scope, limit=20, cursor=first.next_cursor)
    assert [row["lookup_id"] for row in second_data["rows"]] == [
        row.lookup_id for row in expected.rows
    ]
    assert second_data["rows"][0]["lookup_id"] != first.rows[0].lookup_id
    assert fingerprint(root / "data") == before


@pytest.mark.parametrize("corruption", ["prefix", "bool_generation", "page", "snapshot"])
def test_lookup_cursor_rejects_forged_fields_without_rebinding_or_writes(scene, corruption):
    import base64

    root, _ = scene
    first = read_lookups(limit=1)
    assert first.next_cursor is not None
    token = first.next_cursor
    if corruption == "prefix":
        token = "history1." + token.partition(".")[2]
    else:
        raw = json.loads(base64.urlsafe_b64decode(token[9:] + "=" * (-len(token[9:]) % 4)))
        raw[
            {"bool_generation": "generation", "page": "page", "snapshot": "snapshot"}[corruption]
        ] = {
            "bool_generation": True,
            "page": "diag1.9999999.0000000000000000",
            "snapshot": "not-the-selected-snapshot",
        }[corruption]
        token = "lookups1." + base64.urlsafe_b64encode(json.dumps(raw).encode()).decode()
    before = fingerprint(root / "data")
    with pytest.raises(PolicyError) as caught:
        read_lookups(
            cursor=token, limit=1, snapshot_id=first.snapshot_id, generation=first.generation
        )
    assert caught.value.condition == "invalid_cursor"
    assert fingerprint(root / "data") == before


def test_lookup_authenticated_past_end_cursor_and_grouping_are_typed_errors(scene):
    from ssgrep.accounting.ledger.contracts import digest_of
    from ssgrep.accounting.views.lookups import next_cursor
    from ssgrep.accounting.views.types import Scope

    root, _ = scene
    page = read_lookups(limit=1)
    assert page.snapshot_id is not None and page.generation is not None
    doc, _, ledger, reason = service_ledger.load_lookup_data("personal-local")
    assert doc is not None and ledger is not None
    binding = digest_of(
        [
            "personal-local",
            Scope(),
            1,
            page.snapshot_id,
            page.generation,
            doc.to_dict(),
            ledger.rows.digest if ledger else reason,
        ]
    )
    cursor = next_cursor(page.total, binding, page.snapshot_id, page.generation)
    before = fingerprint(root / "data")
    with pytest.raises(PolicyError) as past_end:
        read_lookups(limit=1, cursor=cursor)
    assert past_end.value.condition == "invalid_cursor"
    with pytest.raises(PolicyError) as grouped:
        read_lookups(scope=Scope(by="runtime"))
    assert grouped.value.condition == "invalid_filter"
    assert fingerprint(root / "data") == before


def test_deleted_receipt_is_not_offered_as_an_explainable_lookup(scene):
    root, _ = scene
    original = read_lookups(limit=100)
    linked = next(row for row in original.rows if row.receipt_ids)
    receipt_id = linked.receipt_ids[0]
    path = root / "data/accounting/personal-local/receipts" / f"{receipt_id}.json"
    assert path.is_file()
    path.unlink()
    before = fingerprint(root / "data")
    reread = read_lookups(limit=100)
    row = next(row for row in reread.rows if row.lookup_id == linked.lookup_id)
    assert receipt_id not in row.receipt_ids
    assert "receipt_explanation_unavailable" in row.missing_evidence
    assert row.execution == linked.execution and row.exposure == linked.exposure
    assert row.known_costs == linked.known_costs
    assert reread.total == original.total
    assert fingerprint(root / "data") == before


@pytest.mark.parametrize("enabled", [False, True])
def test_lookup_unavailable_states_have_reachable_status_and_no_invented_rows(tmp_path, enabled):
    from ssgrep.accounting.privacy import policy

    if enabled:
        policy.enable_collection("personal-local")
    root = tmp_path
    before = fingerprint(root / "data")
    binary = Binary(root)
    code, envelope = binary.json("savings", "lookups")
    assert code == 0, envelope
    payload = envelope["data"]
    assert payload["state"] == "unavailable"
    assert payload["data"]["rows"] == [] and payload["data"]["total"] == 0
    assert payload["data"]["next_cursor"] is None
    assert payload["data"]["reason"] == "no_snapshot"
    result = binary("savings", "lookups")
    assert result.returncode == 0 and "ssgrep savings status" in result.stdout
    status, recovered = binary.json("savings", "status")
    assert status == 0, recovered
    assert fingerprint(root / "data") == before


def test_new_snapshot_generation_without_matching_ledger_withholds_old_lookup_rows(scene):
    root, _ = scene
    first = read_lookups(limit=1)
    assert first.rows and first.next_cursor
    assert first.generation is not None
    path = root / "data/accounting/personal-local/snapshots/current.json"
    raw = json.loads(path.read_text())
    raw["generation"] += 1
    path.write_text(json.dumps(raw))
    before = fingerprint(root / "data")
    pending = read_lookups(limit=1)
    assert pending.rows == () and pending.next_cursor is None
    assert pending.generation == first.generation + 1
    assert pending.snapshot_id == first.snapshot_id
    assert pending.reason == "lookup_generation_pending"
    with pytest.raises(PolicyError) as caught:
        read_lookups(limit=1, cursor=first.next_cursor)
    assert caught.value.condition == "lookup_snapshot_expired"
    assert fingerprint(root / "data") == before


@pytest.mark.parametrize("width", [30, 40, 80, 120])
def test_command_renderer_preserves_hostile_tokens_at_its_supported_width(width):
    import shlex
    import subprocess

    from rich.cells import cell_len

    from ssgrep.cli.savings_text import Doc

    project = "project " + "'`" * 40 + " ; $not_a_command 界"
    expected = ["ssgrep", "savings", "lookups", "--project", project, "--page", "cursor-1"]
    doc = Doc(width)
    doc.command("Explain:", " ".join(shlex.quote(argument) for argument in expected))
    text = doc.text().plain
    assert all(cell_len(line) <= width for line in text.splitlines())
    parsed = subprocess.run(
        ["/bin/sh", "-c", "set -e; ssgrep() { printf '%s\\0' ssgrep \"$@\"; };\n" + text],
        capture_output=True,
        check=True,
    )
    assert [part.decode() for part in parsed.stdout.split(b"\0")[:-1]] == expected


NOW = datetime(2026, 10, 12, 12, 0, 0, tzinfo=UTC)


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
        "savings status --details",
    ):
        assert value in flattened
    assert "SECRET" not in plain and "FORGED" not in plain and "\x1b" not in plain
    assert "all projects" in flattened and "Last 7 days" in flattened


def test_all_warning_identities_once_across_pinned_pages_and_new_collection(tmp_path):
    from ssgrep.accounting.views.diagnostics import read_diagnostics

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
    from ssgrep.accounting.views.diagnostics import read_diagnostics

    enable_policy()
    _, envelope = publish(tmp_path / "data")
    first = read_diagnostics(snapshot_id=envelope.snapshot_id, generation=7, limit=5)
    for code, limit in (("lineage.missing_parent", 5), (None, 6)):
        with pytest.raises(PolicyError, match="Page token"):
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
    from ssgrep.accounting.views.diagnostics import diagnostics_document, read_diagnostics
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
    from ssgrep.accounting.views.diagnostics import read_diagnostics

    data = tmp_path / "data"
    before = sorted(data.rglob("*")) if data.exists() else []
    view = read_diagnostics()
    assert not view.records and view.reason == "measurement_off"
    assert (sorted(data.rglob("*")) if data.exists() else []) == before


def test_typed_command_discovery_actual_json_pages(run):
    # Run is the existing isolated executable fixture; no private controls in either mode.
    enable_policy()
    write_snapshot(run.data, warnings=problems())
    first = run("savings", "status", "--details", "--json")
    assert first.returncode == 0, first.stderr
    body = json.loads(first.stdout)["data"]["data"]["diagnostics"]
    assert len(body["records"]) == 20 and body["total"] == len(problems())
    second = run("savings", "status", "--details", "--page", body["next_cursor"], "--json")
    assert second.returncode == 0, second.stderr
    second_body = json.loads(second.stdout)["data"]["data"]["diagnostics"]
    ids = {row["record_id"] for row in body["records"]}
    assert not ids.intersection(row["record_id"] for row in second_body["records"])


@pytest.mark.parametrize("unbounded_debit", [False, True])
@pytest.mark.parametrize("width", [40, 80])
def test_uncertainty_shows_subtotal_without_substituting_complete_net(
    tmp_path, unbounded_debit, width
):
    from ssgrep.accounting.ledger.contracts import Scenario
    from ssgrep.accounting.ledger.ledger_contracts import EstimateBounds, ScenarioEstimate

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
    from ssgrep.accounting.views.diagnostics import read_diagnostics

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


def test_filter_notes_are_reachable_and_expire_instead_of_rebinding_new_ledger(tmp_path):
    from ssgrep.accounting.views.diagnostics import read_diagnostics
    from ssgrep.accounting.views.types import build_scope

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
def test_actual_pipe_no_color_summary_is_bounded_and_keeps_recovery(run, width, budget):
    enable_policy()
    write_snapshot(run.data, warnings=problems())
    result = run("savings", columns=width)
    assert result.returncode == 0, result.stderr
    assert "\x1b" not in result.stdout and "SECRET" not in result.stdout
    flat = " ".join(result.stdout.replace("\\\n", "").split())
    assert "savings status --details" in flat
    assert "details omitted" in flat
    assert "Profile personal-local" in flat and "DIAGNOSTICS" in flat
    assert len(result.stdout.splitlines()) <= budget


def test_unreadable_policy_warning_remains_reachable_without_reading_snapshot(tmp_path):
    from ssgrep.accounting.views.diagnostics import read_diagnostics

    enable_policy()
    policy_path = tmp_path / "data/accounting/policy.json"
    policy_path.write_text("{")
    summary = service.read_summary(now=NOW)
    page = read_diagnostics()
    assert summary.warnings and page.total == len(set(summary.warnings))
    assert len(page.records) == 1 and page.records[0].source == "operational_warning"
    assert page.records[0].severity == "error" and page.snapshot_id is None
    assert page.reason and page.records[0].message is None


def test_diagnostic_page_expires_when_policy_becomes_unreadable(tmp_path) -> None:
    from ssgrep.accounting.views.diagnostics import read_diagnostics
    from tests.accounting.views.test_ux_support import fingerprint

    enable_policy()
    data = tmp_path / "data"
    _, envelope = publish(data, warnings=problems())
    first = read_diagnostics(
        snapshot_id=envelope.snapshot_id, generation=envelope.generation, limit=1
    )
    assert first.next_cursor is not None
    policy.policy_path().write_text("{")
    before = fingerprint(data)

    with pytest.raises(PolicyError) as caught:
        read_diagnostics(
            snapshot_id=envelope.snapshot_id,
            generation=envelope.generation,
            limit=1,
            cursor=first.next_cursor,
        )

    assert caught.value.condition == "invalid_cursor"
    assert fingerprint(data) == before


@pytest.mark.parametrize("width", [40, 80])
def test_diagnostic_private_text_respects_terminal_cells_not_character_count(width):
    from ssgrep.accounting.views.diagnostics import DiagnosticPage, DiagnosticRecord
    from ssgrep.accounting.views.types import Scope
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
    from ssgrep.accounting.views.types import Condition, NextAction

    enable_policy()
    publish(tmp_path / "data")
    summary = service.read_summary(now=NOW)
    summary = replace(
        summary,
        conditions=tuple(
            Condition(
                "blocked",
                f"private-destination-{i}: blocked by policy",
                NextAction("ssgrep savings status", "Inspect policy"),
            )
            for i in range(16)
        ),
    )
    text = savings_render.render_summary(summary, width).plain
    flat = " ".join(text.replace("\\\n", "").split())
    assert "Export blocked" in flat and "16 conditions" in flat
    assert "ssgrep savings status" in flat
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
        assert "ssgrep savings collect --recompute" in flat
        assert "approved native sources" in flat
        assert len(text.splitlines()) <= budget


def test_unavailable_replay_and_frequency_revision_codes_are_severe():
    from ssgrep.accounting.views.diagnostics import grouped_warnings

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
    from ssgrep.accounting.views.diagnostics import read_diagnostics
    from tests.accounting.views.test_ux_support import fingerprint

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
    from ssgrep.accounting.views.diagnostics import read_diagnostics
    from tests.accounting.views.test_ux_support import fingerprint

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
    from ssgrep.accounting.views.diagnostics import read_diagnostics
    from tests.accounting.views.test_ux_support import fingerprint

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
    from ssgrep.accounting.views.diagnostics import read_diagnostics
    from tests.accounting.views.test_ux_support import fingerprint

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


def test_archived_source_count_is_visible_in_real_cli_status() -> None:
    source = write_claude_session("archived-status")
    api.index()
    assert api.status().archived_source_count == 0
    source.unlink()
    archived = api.index(full_reprocess=True)
    assert archived.archived_source_count == 1
    env = {**os.environ, "NO_COLOR": "1", "COLUMNS": "120"}
    machine = subprocess.run(
        [str(BINARY), "status", "--json"],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert machine.returncode == 0, machine.stderr
    data = json.loads(machine.stdout)["data"]
    assert data["archived_source_count"] == 1
    assert data["session_count"] == 1 and data["episode_count"] == 1
    human = subprocess.run(
        [str(BINARY), "status"],
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert human.returncode == 0, human.stderr
    archive_row = next(line for line in human.stdout.splitlines() if "Archived" in line)
    assert "1 sources" in archive_row
    assert api.status().archived_source_count == 1


def test_status_census_total_is_unknown_when_any_runtime_source_count_is_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from dataclasses import replace

    import typer

    import ssgrep.accounting.views.service as status_service
    import ssgrep.cli.commands.metrics_status_command as status_command
    from ssgrep.accounting.serve.status_worker import RuntimeLag
    from ssgrep.cli.commands.metrics_status_command import MetricsStatusCommand

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    current = status_service.read_status()
    view = replace(
        current,
        lag=replace(
            current.lag,
            by_runtime=(
                RuntimeLag("claude", 2, 4, 4, 0, 0),
                RuntimeLag("codex", None, None, None, None, None),
            ),
        ),
    )
    monkeypatch.setattr(status_service, "read_status", lambda profile: view)
    monkeypatch.setattr(status_command, "is_json_mode", lambda: True)

    result = MetricsStatusCommand(typer.Typer()).handle()
    assert result is not None
    data = result.get("data")
    assert isinstance(data, dict)
    census = data.get("source_census")
    assert isinstance(census, dict)
    assert census.get("sources") is None
    by_runtime = census.get("by_runtime")
    assert isinstance(by_runtime, list)
    assert [row["sources"] for row in by_runtime if isinstance(row, dict)] == [2, None]


def test_status_rejects_invalid_profile_without_creating_state(run: Run) -> None:
    before = run.files()
    result = run("savings", "status", "--profile", "../outside", "--json")

    assert result.returncode == 2
    assert document(result)["condition"] == "invalid_profile"
    assert run.files() == before


def test_status_details_renders_private_diagnostics_safely_in_human_mode(run: Run) -> None:
    enable_policy()
    write_snapshot(run.data, warnings=problems())

    result = run("savings", "status", "--details")

    assert result.returncode == 0, result.stderr
    assert "ssgrep savings diagnostics" in result.stdout and "storage_corrupt" in result.stdout
    assert "SECRET" not in result.stdout and "FORGED" not in result.stdout


@pytest.mark.parametrize(
    "cache_options",
    [
        ("--evidence-cache", "--no-evidence-cache"),
        ("--evidence-cache", "--cache-days", "0"),
    ],
)
def test_setup_rejects_conflicting_or_unbounded_cache_consent_without_writes(
    run: Run, cache_options: tuple[str, ...]
) -> None:
    before = run.files()
    result = run("savings", "setup", "--yes", *cache_options, "--json")

    assert result.returncode == 2
    assert document(result)["condition"] == "invalid_policy"
    assert run.files() == before


def test_savings_help_option_without_subcommand_exits_cleanly(run: Run):
    result = run("savings", "--help")
    assert result.returncode == 0
    assert "Usage: ssgrep savings" in result.stdout


def test_diagnostics_command_quotes_the_page_token_and_keeps_text_opt_in():
    from ssgrep.accounting.views.diagnostics import diagnostics_command

    plain = diagnostics_command("personal-local")
    assert "savings status --details" in plain and "--page" not in plain
    assert "--include-text" not in plain
    paged = diagnostics_command("personal-local", cursor="diag1.2.ab cd;x", include_text=True)
    assert "--page 'diag1.2.ab cd;x'" in paged
    assert paged.endswith("--include-text")


def test_diagnostics_by_snapshot_id_alone_reads_the_older_committed_generation(tmp_path):
    from ssgrep.accounting.views.diagnostics import read_diagnostics

    enable_policy()
    _, old = publish(tmp_path / "data", warnings=["old_problem:private-old"])
    _, current = publish(tmp_path / "data", generation=8, warnings=["new_problem:private-new"])
    older = read_diagnostics(snapshot_id=old.snapshot_id)
    assert older.generation == 7 and older.snapshot_id == old.snapshot_id
    assert [r.code for r in older.records] == ["old_problem"]
    assert [r.code for r in read_diagnostics(snapshot_id=current.snapshot_id).records] == [
        "new_problem"
    ]
    with pytest.raises(PolicyError) as caught:
        read_diagnostics(snapshot_id="a" * 64)
    assert caught.value.condition == "diagnostic_snapshot_expired"


def test_retained_snapshot_diagnostics_say_measurement_is_off(tmp_path):
    from ssgrep.accounting.views.diagnostics import read_diagnostics

    _, envelope = publish(tmp_path / "data", warnings=["old_problem:private-old"])
    page = read_diagnostics(snapshot_id=envelope.snapshot_id, generation=7)
    assert {r.code: r.source for r in page.records} == {
        "old_problem": "snapshot_warning",
        "measurement_off": "operational_warning",
    }
    enable_policy()
    enabled = read_diagnostics(snapshot_id=envelope.snapshot_id, generation=7)
    assert [r.code for r in enabled.records] == ["old_problem"]


def test_filtered_diagnostics_expire_when_collection_advances_during_selection(
    tmp_path, monkeypatch
):
    from ssgrep.accounting.views.diagnostics import read_diagnostics
    from ssgrep.accounting.views.types import build_scope

    enable_policy()
    _, envelope = publish(tmp_path / "data")
    scope = build_scope(runtime="omp")
    assert read_diagnostics(scope=scope, snapshot_id=envelope.snapshot_id, generation=7).total

    real = service.read_summary

    def advancing(profile, selected):
        publish(tmp_path / "data", generation=8, warnings=["later:new"])
        return real(profile, selected)

    monkeypatch.setattr(service, "read_summary", advancing)
    with pytest.raises(PolicyError) as caught:
        read_diagnostics(scope=scope, snapshot_id=envelope.snapshot_id, generation=7)
    assert caught.value.condition == "diagnostic_selection_expired"
    assert "Collection advanced during selection" in str(caught.value)


def test_diagnostics_document_escapes_control_text_only_when_text_is_requested(tmp_path):
    from ssgrep.accounting.views.diagnostics import diagnostics_document, read_diagnostics

    enable_policy()
    _, envelope = publish(tmp_path / "data")
    kwargs: dict[str, Any] = {
        "snapshot_id": envelope.snapshot_id,
        "generation": 7,
        "code": "storage_corrupt",
    }
    hidden = diagnostics_document(read_diagnostics(**kwargs))
    assert all("message" not in row for row in cast("list[dict[str, Any]]", hidden["records"]))
    shown = diagnostics_document(read_diagnostics(include_text=True, **kwargs))
    (row,) = cast("list[dict[str, Any]]", shown["records"])
    assert "SECRET" in row["message"]
    assert "\x1b" not in row["message"] and "\n" not in row["message"]
