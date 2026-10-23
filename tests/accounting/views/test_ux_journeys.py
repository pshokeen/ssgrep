"""10.5: the documented journey through the REAL binary, at 40/80/120 columns.

`collect` -> summary -> explain -> history -> pause -> resume, in an isolated HOME and
data root. The consumer transcripts carry hostile strings (ANSI, OSC, Rich markup, newlines) in
the conversation id. Checks are structural: width, control bytes, navigation (every `Next:`
command is parsed and RUN), sign/unit/action visibility at 40 columns, and JSON parity. Screens
are written to `UX_SCREENS_DIR` when that variable is set (used for the release report).
"""

from __future__ import annotations

import json
import os
import pty
import re
import select
import shlex
import subprocess
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting.export import acks as acks
from ssgrep.accounting.ledger.ledger_contracts import (
    DestinationPolicy,
    Signal,
    SnapshotState,
)
from ssgrep.accounting.privacy import policy
from ssgrep.accounting.privacy.policy import PolicyError
from ssgrep.accounting.views import service
from ssgrep.accounting.views.service import read_status
from ssgrep.accounting.views.types import Condition, NextAction, build_scope
from ssgrep.cli import savings_render
from ssgrep.cli.metrics_render import render_status
from tests.accounting.collect import (
    collector_scene as journey_scene,
    collector_scene as scene,
    verify_scenes as vs,
)
from tests.accounting.ledger.reduce_builders import source_session
from tests.accounting.ledger.test_reduce_evidence import FACT, consumer
from tests.accounting.views.test_savings_commands import BINARY
from tests.accounting.views.test_ux_support import (
    NOW,
    Binary,
    collected_scene,
    current_rarity_index,
    fingerprint,
)

HOSTILE = "bad\x1b[31m[bold red]x[/]\x1b]0;pwn\x07"
ANSI_OTHER = re.compile(r"\x1b\[[0-9;]*[^0-9;m]|\x1b\][^\x07]*\x07|\x1b[^\[\]]|\x9b|\x07|\r")
SGR = re.compile(r"\x1b\[[0-9;]*m")
WIDTHS = (40, 80, 120)


def renamed(session: str, records: list[dict]) -> list[dict]:
    return [{**r, "sessionId": session} if "sessionId" in r else dict(r) for r in records]


def screens_dir() -> Path | None:
    value = os.environ.get("UX_SCREENS_DIR")
    return Path(value) if value else None


def keep(name: str, width: int, text: str) -> None:
    target = screens_dir()
    if target is not None:
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{name}.{width}.txt").write_text(text)


def lines_fit(text: str, width: int) -> None:
    for line in text.splitlines():
        assert len(line) <= width, f"{len(line)} > {width}: {line!r}"


def next_command(text: str) -> list[str]:
    """The parsed argv of the ``Next:`` command (shell continuations joined)."""
    block = text.split("Next:")[1]
    parts = []
    for line in block.splitlines(keepends=True):
        if not parts and not line.strip():
            continue
        parts.append(line)
        if not line.rstrip().endswith("\\"):
            break
    command = "".join(parts).replace("\\\n", "")
    argv = shlex.split(command)
    assert argv[0] == "ssgrep", command
    return argv[1:]


def collect_real(run: Binary) -> None:
    """Run `ssgrep savings collect` with the real clock for a fresh journey snapshot."""
    done = run("savings", "collect", "--json")
    assert done.returncode == 0, (done.stdout, done.stderr[-400:])


@pytest.fixture
def journey(tmp_path: Path) -> Binary:
    return Binary(tmp_path)


def build_dataset(root: Path) -> None:
    """Transcripts the collector will read: one fact source and a hostile-named consumer."""
    vs.projects(root).mkdir(parents=True, exist_ok=True)
    vs.put(
        vs.projects(root),
        "src1",
        source_session("src1", FACT, when_day=11),
        journey_scene.at(hour=10),
    )
    vs.put(vs.projects(root), "hostile", renamed(HOSTILE, consumer()), journey_scene.at(hour=10))


def test_documented_journey_end_to_end_at_each_width(tmp_path: Path) -> None:
    run = Binary(tmp_path)
    vs.enable()
    vs.start_capture(day=11)
    build_dataset(tmp_path)
    assert run("savings", columns=80).returncode == 0
    # setup was done through the policy module by vs.enable; the binary then reads and collects
    assert run("savings", "setup", "--yes").returncode == 0
    search = run("search", "payment retries")  # real execution; no index -> an error is fine
    assert search.returncode != 0 or search.stdout is not None
    collect_real(run)
    for width in WIDTHS:
        summary = run("savings", columns=width)
        assert summary.returncode == 0, summary.stderr
        text = summary.stdout
        keep("summary", width, text)
        lines_fit(text, width)
        assert not ANSI_OTHER.search(text) and "\x1b" not in text
        assert HOSTILE not in text
        compact = " ".join(text.split())
        assert "Next: ssgrep savings explain" in compact.replace("\\ ", "")
        assert "MEASURED" in text and "ESTIMATED NET BENEFIT" in text and "EVIDENCE" in text
        # navigation: the suggested command really runs and shows the receipt
        argv = next_command(text)
        assert argv[:2] == ["savings", "explain"]
        explained = run(*argv, columns=width)
        assert explained.returncode == 0, explained.stderr
        keep("explain", width, explained.stdout)
        lines_fit(explained.stdout, width)
        assert "\x1b" not in explained.stdout and "\x07" not in explained.stdout
        assert "\\x1b" in explained.stdout, "the hostile conversation id is escaped, not dropped"
        assert "[bold red]x[/]" in " ".join(explained.stdout.split())
        history_argv = next_command(explained.stdout)
        assert history_argv[:2] == ["savings", "history"]
        history = run(*history_argv, columns=width)
        assert history.returncode == 0, history.stderr
        keep("history", width, history.stdout)
        lines_fit(history.stdout, width)
        assert "CURRENT TOTAL" in history.stdout and "tokens" in history.stdout
        back = next_command(history.stdout)
        assert back == ["savings"], "history leads back to the summary"
    paused = run("savings", "pause", columns=40)
    keep("pause", 40, paused.stdout)
    assert paused.returncode == 0
    paused_summary = run("savings", columns=40)
    keep("paused-summary", 40, paused_summary.stdout)
    assert "Paused" in paused_summary.stdout and "ssgrep savings resume" in paused_summary.stdout
    paused_data = run.data("savings")["data"]["estimates"]
    assert all(
        estimate["net"]["value"] is None for estimate in paused_data["context_tokens"].values()
    ), "an absent current index stays unknown while paused"
    status = run("savings", "status", columns=40)
    keep("status-paused", 40, status.stdout)
    lines_fit(status.stdout, 40)
    resumed = run("savings", "resume", columns=40)
    keep("resume", 40, resumed.stdout)
    assert resumed.returncode == 0 and "resumed" in resumed.stdout


@pytest.mark.parametrize("width", WIDTHS)
def test_numeric_pipe_values_keep_signs_units_and_actions_at_each_width(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    current_rarity_index(tmp_path, monkeypatch)
    collected_scene(tmp_path)
    before = fingerprint(tmp_path / "data")
    run = Binary(tmp_path)
    text = run("savings", "--no-refresh", columns=width, env={"NO_COLOR": "1"}).stdout
    compact = " ".join(text.replace("\\\n", " ").split())
    body = run.data("savings", "--no-refresh")
    net = body["data"]["estimates"]["context_tokens"]["central"]["net"]["value"]
    assert net is not None and net < 0
    assert f"{net:+,}" in compact and "tokens" in compact
    assert next_command(text) == shlex.split(body["next_actions"][0]["command"])[1:]
    assert "credit" in compact and "cost" in compact
    for line in text.splitlines():
        assert len(line) <= width, line
    assert "\x1b" not in text
    assert fingerprint(tmp_path / "data") == before


def test_color_is_only_decoration_on_a_real_tty_and_hostile_text_stays_inert(
    tmp_path: Path,
) -> None:
    run = Binary(tmp_path)
    vs.enable()
    vs.start_capture(day=11)
    build_dataset(tmp_path)
    collect_real(run)
    receipt = run.data("savings")["data"]["representative_receipt"]
    assert isinstance(receipt, str)
    for no_color in (False, True):
        env: dict[str, str] = {
            **vs.env_for(tmp_path),
            "COLUMNS": "40",
            "TERM": "xterm-256color",
        }
        env.pop("NO_COLOR", None)
        if no_color:
            env["NO_COLOR"] = "1"
        master, slave = pty.openpty()
        process = subprocess.Popen(
            [str(BINARY), "savings", "explain", receipt],
            stdin=subprocess.DEVNULL,
            stdout=slave,
            stderr=subprocess.PIPE,
            env=env,
        )
        os.close(slave)
        chunks = b""
        while True:
            ready, _, _ = select.select([master], [], [], 30)
            if not ready:
                break
            try:
                data = os.read(master, 65536)
            except OSError:
                break
            if not data:
                break
            chunks += data
        process.wait(timeout=60)
        os.close(master)
        screen = chunks.decode(errors="replace").replace("\r\n", "\n")
        assert process.returncode == 0
        stripped = SGR.sub("", screen)
        assert not ANSI_OTHER.search(stripped.replace("\r", "")), repr(stripped[:300])
        assert HOSTILE not in screen and "pwn" not in re.sub(r"\\x1b\]0;pwn\\x07", "", stripped)
        if no_color:
            assert "\x1b" not in screen, "NO_COLOR means no styling at all on a tty"
        else:
            assert SGR.search(screen), "the real terminal is the positive styling control"
        keep("explain-tty-nocolor" if no_color else "explain-tty-color", 40, screen)
        text = " ".join(stripped.split())
        assert "CREDIT BY SCENARIO" in text and "Next:" in text


def test_pipe_output_has_no_cursor_control_and_json_parity_with_text(tmp_path: Path) -> None:
    run = Binary(tmp_path)
    vs.enable()
    vs.start_capture(day=11)
    build_dataset(tmp_path)
    assert scene.collect(NOW).state == "published"
    before = fingerprint(tmp_path / "data")
    text = run(
        "savings", "--no-refresh", columns=80, env={"FORCE_COLOR": "1", "NO_COLOR": ""}
    ).stdout
    assert "\x1b" not in text, "a pipe never receives color or cursor control"
    body = run.data("savings", "--no-refresh")
    estimates = body["data"]["estimates"]["context_tokens"]
    assert all(estimate["net"]["value"] is None for estimate in estimates.values())
    assert all(
        not estimate["complete"] and estimate["credit_missing"] for estimate in estimates.values()
    )
    assert all(estimate["bounds"]["net_upper"]["value"] is None for estimate in estimates.values())
    assert any(condition["code"] == "index_unavailable" for condition in body["data"]["conditions"])
    assert "unavailable" in text.lower()
    assert body["next_actions"][0]["command"] in " ".join(text.replace("\\\n", " ").split())
    for hostile_free in (text, str(body)):
        assert "pwn" not in hostile_free.replace("\\x1b]0;pwn\\x07", "").replace(
            "\\u001b]0;pwn\\u0007", ""
        )
    assert fingerprint(tmp_path / "data") == before


def test_bare_savings_refreshes_missing_snapshot_once_and_supports_read_only_opt_out(
    tmp_path: Path,
) -> None:
    run = Binary(tmp_path)
    vs.enable()
    vs.start_capture()
    build_dataset(tmp_path)
    data_root = tmp_path / "data"

    before = fingerprint(data_root)
    opted_out = run("savings", "--no-refresh", "--json")
    assert opted_out.returncode == 0, opted_out.stderr
    assert fingerprint(data_root) == before
    assert not (data_root / "accounting" / "personal-local" / "snapshots" / "current.json").exists()

    first = run("savings", "--json")
    first_document = json.loads(first.stdout)["data"]
    assert first_document["provenance"]["generation"] == 1
    assert first_document["data"]["freshness"]["stale"] is False

    committed = fingerprint(data_root)
    second = run("savings", "--json")
    assert second.returncode == 0, second.stderr
    second_document = json.loads(second.stdout)["data"]
    assert second_document["provenance"]["generation"] == first_document["provenance"]["generation"]
    assert fingerprint(data_root) == committed


DEST = "org-primary"
ENDPOINT = "https://collector.example.test:4318"
STAMP = "2026-10-12T20:00:00Z"
COMMANDS = {
    "savings",
    "savings explain",
    "savings history",
    "metrics setup",
    "metrics status",
    "metrics collect",
    "metrics resume",
    "metrics export",
    "index",
}


def action_of(item: Any) -> NextAction:
    """The next action a condition or delivery line carries (fails loudly when it has none)."""
    assert item.action is not None, item
    return item.action


def conditions(view: Any) -> dict[str, Condition]:
    return {c.code: c for c in view.conditions}


def assert_executable(command: str, profile: str = "personal-local") -> None:
    """Parses as an ssgrep command whose subcommand exists and whose --profile is preserved."""
    words = shlex.split(command)
    assert words[0] == "ssgrep", command
    name = " ".join(w for w in words[1:3] if not w.startswith("-"))
    assert any(name == c or name.startswith(c.split()[0] + " " + c.split()[-1]) or
               name.split()[0] == c for c in COMMANDS), command  # fmt: skip
    assert not any(w.startswith("<") or w.endswith(">") for w in words), command
    if profile != "personal-local":
        assert words[-2:] == ["--profile", profile], command
    else:
        assert "--profile" not in words, command


def with_destination(*, enabled: bool = True, profile: str = "personal-local") -> None:
    dest = DestinationPolicy(DEST, ENDPOINT, (Signal.METRICS,), credential_ref="env:UX_TOKEN")
    current = policy.load_profile_policy(profile)
    policy.save_profile(
        replace(
            current,
            destinations=(dest,),
            enabled_exports=((DEST, "metrics"),) if enabled else (),
        )
    )


def write_delivery(change) -> None:
    acks.update_delivery("personal-local", DEST, Signal.METRICS, change)


@pytest.fixture
def collected(tmp_path: Path) -> Path:
    collected_scene(tmp_path)
    return tmp_path


# ---- readable states without numbers -----------------------------------------------------------


def test_disabled_names_setup_and_does_not_enable_anything(tmp_path: Path) -> None:
    view = service.read_summary(now=NOW)
    assert view.state is SnapshotState.DISABLED and not view.has_values
    assert set(conditions(view)) == {"disabled"}
    assert view.next_actions[0].command == "ssgrep savings setup"
    assert not policy.load_profile_policy("personal-local").collection_enabled
    assert not (tmp_path / "data").exists(), "reading created no storage"


def test_enabled_without_snapshot_says_capture_on_collection_incomplete_and_shows_no_zero(
    tmp_path: Path,
) -> None:
    vs.enable()
    view = service.read_summary(now=NOW)
    assert view.state is SnapshotState.NO_SNAPSHOT and view.measured == ()
    assert view.next_actions[0].command == "ssgrep savings collect"
    text = savings_render.render_summary(view, 80).plain
    assert "MEASURED" not in text and "ESTIMATED" not in text
    assert "Capture is enabled" in " ".join(text.split())


def test_every_condition_with_an_action_offers_a_parseable_command_with_the_profile(
    tmp_path: Path,
) -> None:
    policy.enable_collection("acme", {"profile_id": "acme", "roots": [str(tmp_path)]})
    view = service.read_summary("acme", now=NOW)
    assert view.conditions and all(c.action is not None for c in view.conditions)
    for condition in view.conditions:
        assert condition.action is not None
        assert_executable(condition.action.command, "acme")
    assert_executable(view.next_actions[0].command, "acme")


# ---- fresh states from a real collection ---------------------------------------------------------


def test_negative_partial_values_keep_their_signs_and_name_receipt_and_coverage_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_rarity_index(tmp_path, monkeypatch)
    collected_scene(tmp_path)
    view = service.read_summary(now=NOW)
    assert view.state is SnapshotState.FRESH and view.has_negative and view.partial
    found = conditions(view)
    assert {"negative", "partial", "index_unavailable"} <= set(found)
    assert (
        view.receipt_id
        and f"savings explain {view.receipt_id}" in action_of(found["negative"]).command
    )
    negatives = [
        q for row in view.scenarios for _, q in row.net if q.value is not None and q.value < 0
    ]
    assert negatives
    for width in (40, 80, 120):
        text = savings_render.render_summary(view, width).plain
        assert max(map(len, text.splitlines())) <= width
        compact = " ".join(text.split())
        assert any(f"{q.value:+,}" in compact for q in negatives)
        assert "tokens" in compact
    assert not any(word in text.lower() for word in ("saved", "great", "congrat", "win", "!"))
    for condition in found.values():
        if condition.action:
            assert_executable(condition.action.command)


def test_index_unavailable_is_separate_from_usage_and_never_asks_for_exit_4(
    collected: Path,
) -> None:
    view = service.read_summary(now=NOW)
    index = conditions(view)["index_unavailable"]
    assert view.index == "unavailable" and dict(view.measured)["searches"].value == 4
    assert index.action is not None
    assert shlex.split(index.action.command) == [
        "ssgrep",
        "index",
        "&&",
        "ssgrep",
        "savings",
        "collect",
        "--recompute",
        "--repin-frequency",
    ]
    status = read_status("personal-local", now=NOW)
    assert status.state is SnapshotState.FRESH, "status stays readable without the index"
    assert not (collected / "data" / "lancedb").exists(), "nothing built an index"


def test_no_traceable_reuse_shows_costs_and_exclusions_but_claims_nothing(tmp_path: Path) -> None:
    vs.enable()
    vs.start_capture(day=12)
    vs.put(
        vs.projects(tmp_path),
        "lonely",
        [vs.claude_prompt("lonely", "hello", uuid="u", ts=vs.ts(12, 9))],
        scene.at(hour=10),
    )
    scene.journal_events(scene.execution("n-1", vs.ts(12, 9, 5), text="anything", transport="cli"))
    assert scene.collect(NOW).state == "published"
    view = service.read_summary(now=NOW)
    assert view.no_reuse and view.coverage["candidate_receipts"] == 0
    assert "no_reuse" in conditions(view)
    claimed = [q.value for r in view.scenarios for _, q in r.net if q.value is not None]
    assert all(v <= 0 for v in claimed), "no positive savings were invented"


# ---- time-dependent states ----------------------------------------------------------------------


def test_stale_keeps_last_good_values_with_age_and_collect_and_reads_do_not_refresh_it(
    collected: Path,
) -> None:
    later = NOW + timedelta(hours=3)
    first = service.read_summary(now=later)
    assert first.state is SnapshotState.STALE and first.stale and first.has_values
    assert first.age_seconds == int((later - scene.at(hour=20)).total_seconds())
    assert action_of(conditions(first)["stale"]).command == "ssgrep savings"
    assert scene.current().collected_at == "2026-10-12T20:00:00Z"
    again = service.read_summary(now=later + timedelta(seconds=30))
    assert again.age_seconds == first.age_seconds + 30, "age keeps growing; a read resets nothing"
    assert dict(first.measured) == dict(again.measured)


def test_paused_keeps_history_and_names_timestamp_and_resume_without_collect(
    collected: Path,
) -> None:
    policy.pause("personal-local")
    view = service.read_summary(now=NOW + timedelta(hours=3))
    assert view.state is SnapshotState.PAUSED and view.has_values and view.stale
    paused = conditions(view)["paused"]
    assert action_of(paused).command == "ssgrep savings resume"
    assert view.paused_at and view.paused_at in paused.cause
    assert conditions(view)["stale"].action is None, "collect would fail while paused"
    assert "ssgrep savings collect" not in [a.command for a in view.next_actions]


def test_evidence_expired_keeps_totals_and_states_the_boundary(collected: Path) -> None:
    later = scene.at(hour=10) + timedelta(days=45)
    assert scene.collect(later).state == "published"
    view = service.read_summary(now=later)
    expired = conditions(view)["evidence_expired"]
    assert view.evidence_expired_before and view.frozen_roots
    assert view.evidence_expired_before in expired.cause
    assert action_of(expired).command == "ssgrep savings history"
    history = service.read_history()
    assert history.entries and not any(e.recomputable for e in history.entries)
    receipt = service.read_receipt("personal-local", view.receipt_id or "")
    assert receipt is None or not receipt.evidence_available


# ---- delivery states ----------------------------------------------------------------------------


def delivery_state(profile: str = "personal-local") -> dict[str, Any]:
    status = read_status(profile, now=NOW)
    assert len(status.delivery_lines) == 1
    return {"line": status.delivery_lines[0], "view": status}


def test_configured_but_disabled_and_enabled_but_unsent_are_distinct(collected: Path) -> None:
    with_destination(enabled=False)
    assert delivery_state()["line"].state == "disabled"
    with_destination(enabled=True)
    line = delivery_state()["line"]
    assert line.state == "waiting" and "nothing has been queued" in line.cause
    assert f"savings export {DEST} --dry-run" in line.action.command


def test_credentials_expired_names_rotation_and_never_prints_a_secret(
    collected: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UX_TOKEN", "sk-SECRET-VALUE-0001")
    with_destination()
    write_delivery(
        lambda d: acks.blocked(
            acks.set_counts(acks.collected(d, 1), pending_batches=2, unknown_batches=0),
            "unauthorized",
            "fp-1",
        )
    )
    view = read_status("personal-local", now=NOW)
    line = view.delivery_lines[0]
    assert line.state == "credentials_expired" and line.blocked_reason == "unauthorized"
    assert "Rotate the credential" in action_of(line).why
    rendered = render_status(view, 80).plain
    assert "sk-SECRET-VALUE-0001" not in rendered and "credentials expired" in rendered.lower()
    summary = service.read_summary(now=NOW)
    assert action_of(conditions(summary)["credentials_expired"]).command == action_of(line).command


def test_partial_export_shows_counts_and_is_never_called_acknowledged(collected: Path) -> None:
    with_destination()

    def partial(d: acks.Delivery) -> acks.Delivery:
        d = acks.collected(d, 1)
        d = acks.attempted(d, 1, STAMP)
        return acks.outcome(
            d,
            generation=1,
            snapshot_id="a1" * 32,
            index=0,
            total=2,
            kind="partial",
            now=STAMP,
            rejected=4,
        )

    write_delivery(partial)
    line = delivery_state()["line"]
    assert line.state == "partial_export" and line.rejected_points == 4
    assert line.acked_generation is None and "Not fully acknowledged" in line.cause


def test_collection_and_delivery_freshness_are_independent(collected: Path) -> None:
    with_destination()

    def acked(d: acks.Delivery) -> acks.Delivery:
        d = acks.collected(d, 1)
        d = acks.attempted(d, 1, "2026-10-12T19:00:00Z")
        return acks.outcome(
            d,
            generation=1,
            snapshot_id="a1" * 32,
            index=0,
            total=1,
            kind="acked",
            now="2026-10-12T19:00:00Z",
        )

    write_delivery(acked)
    status = read_status("personal-local", now=NOW + timedelta(hours=1))
    assert status.collection_age_seconds == 3600 and status.stale
    assert status.delivery_age_seconds == 7200 and status.delivery_lines[0].state == "acknowledged"
    assert status.collection_age_seconds != status.delivery_age_seconds
    from ssgrep.cli.metrics_json import status_document

    document = status_document(status)
    data = document["data"]
    assert isinstance(data, dict)
    freshness = data["freshness"]
    assert freshness["collection"]["age_seconds"] == 3600
    assert freshness["delivery"]["age_seconds"] == 7200


TASKS = ("claude:cons0#turn:u0", "claude:cons1#turn:u1", "claude:cons2#turn:u2")
PER_TASK = (-51, -45, -45)  # conservative / central / optimistic net tokens of every pair


def nets(view) -> list[int | None]:
    row = next(r for r in view.scenarios if r.unit_key == "context_tokens")
    return [q.value for _, q in row.net]


def read(**kwargs):
    limit = kwargs.pop("limit", 20)
    cursor = kwargs.pop("cursor", None)
    return service.read_summary(scope=build_scope(**kwargs), now=NOW, limit=limit, cursor=cursor)


@pytest.fixture
def scene_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    current_rarity_index(tmp_path, monkeypatch)
    collected_scene(tmp_path)
    return tmp_path


def test_unfiltered_total_is_the_sum_of_its_three_tasks(scene_root: Path) -> None:
    total = nets(read())
    # characters are summed BEFORE the single division by 4, so the total may differ from the
    # sum of per-task rounded rows by < 1 token per task, never more
    for got, per_task in zip(total, PER_TASK, strict=True):
        assert got is not None and abs(got - 3 * per_task) <= 3
    snapshot_total = [
        q.value
        for q in (
            service.parse_quantity(
                scene.window(scene.current(), "7d")["estimates"]["context_tokens"][s]["net"],
                "tokens",
                "x",
            )
            for s in ("conservative", "central", "optimistic")
        )
    ]
    assert total == snapshot_total, "unfiltered view agrees with the collector's own window"
    by_task = read(by="task")
    rows = {g.key: g for g in by_task.groups}
    assert set(TASKS) <= set(rows)
    for key in TASKS:
        row = next(r for r in rows[key].scenarios if r.unit_key == "context_tokens")
        assert [q.value for _, q in row.net] == list(PER_TASK)
        assert rows[key].receipts == 1


def test_task_and_session_filters_select_exactly_their_records(scene_root: Path) -> None:
    one = read(task="claude:cons1#turn:u1")
    assert nets(one) == list(PER_TASK)
    assert dict(one.measured)["searches"].value == 1
    by_root = read(task="claude:cons1")  # a root id selects every task of that conversation
    assert nets(by_root) == list(PER_TASK)
    session = read(session="cons2")
    assert nets(session) == list(PER_TASK)
    assert dict(session.measured)["searches"].value == 1
    assert session.coverage["candidate_receipts"] == 1
    none = read(session="no-such-session")
    assert dict(none.measured)["searches"].value == 0
    assert nets(none) == [None, None, None], "an empty selection is unknown, never a zero total"


def test_filters_intersect_never_union(scene_root: Path) -> None:
    both = read(runtime="claude", task="claude:cons0#turn:u0", session="cons0")
    assert nets(both) == list(PER_TASK)
    clash = read(runtime="claude", task="claude:cons0#turn:u0", session="cons1")
    assert dict(clash.measured)["searches"].value == 0
    assert nets(clash) == [None, None, None]
    wrong_runtime = read(runtime="codex", task="claude:cons0#turn:u0")
    assert dict(wrong_runtime.measured)["searches"].value == 0
    assert nets(wrong_runtime) == [None, None, None]


def test_interval_is_utc_half_open_for_measured_counts_and_whole_days_for_estimates(
    scene_root: Path,
) -> None:
    inside = read(since="2026-10-12", until="2026-10-13")
    assert dict(inside.measured)["searches"].value == 3
    assert nets(inside) == nets(read()), "the whole day 12 holds every estimate"
    before = read(since="2026-10-11", until="2026-10-12")  # end exclusive: day 12 excluded
    assert dict(before.measured)["searches"].value == 1, "only the day-11 capture-start search"
    assert nets(before) == [None, None, None], "no estimate period lies in that day"
    exact = read(since="2026-10-12T09:00:00Z", until="2026-10-12T09:00:11Z")
    assert dict(exact.measured)["searches"].value == 0  # event is at 09:00:11, end is exclusive
    covering = read(since="2026-10-12T09:00:11Z", until="2026-10-12T09:00:12Z")
    assert dict(covering.measured)["searches"].value == 1


def test_by_day_runtime_and_project_group_the_same_totals(scene_root: Path) -> None:
    for by in ("day", "runtime", "project"):
        view = read(by=by, limit=100)
        counted = sum(
            dict(g.measured)["searches"].value or 0 for g in view.groups if g.key != "task_unknown"
        )
        assert counted == dict(view.measured)["searches"].value, by
    days = read(by="day", limit=100)
    assert [g.key for g in days.groups] == ["2026-10-11", "2026-10-12"] or "2026-10-12" in [
        g.key for g in days.groups
    ]
    runtime = read(by="runtime", limit=100)
    assert "claude" in {g.key for g in runtime.groups}


def test_project_filter_uses_recorded_working_directories_and_says_when_unknown(
    scene_root: Path,
) -> None:
    other = read(project="/definitely/not/here")
    assert dict(other.measured)["searches"].value == 0
    assert nets(other) == [None, None, None]
    inside = read(project="/work/proj")  # CLI start marker's measured process cwd
    assert dict(inside.measured)["searches"].value == 1
    prefix = read(project="/work")
    assert dict(prefix.measured)["searches"].value == 4
    demo = read(project="/work/demo")  # native MCP caller cwd, NOT the server's /work/proj
    assert dict(demo.measured)["searches"].value == 3
    assert nets(demo) == nets(read()), "estimates follow the recorded caller workspace"
    assert nets(inside) == [None] * 3, "an unrelated process workspace cannot claim consumer credit"
    sibling = read(project="/wor")  # a path prefix is not a directory prefix
    assert dict(sibling.measured)["searches"].value == 0


def test_pages_neither_skip_nor_duplicate_and_every_cursor_is_bound(scene_root: Path) -> None:
    full = read(by="task", limit=100)
    keys = [g.key for g in full.groups]
    assert len(keys) == len(set(keys)) >= 3
    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        page = read(by="task", limit=1, cursor=cursor)
        seen += [g.key for g in page.groups]
        pages += 1
        assert page.group_total == len(keys)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert seen == keys and pages == len(keys)
    first = read(by="task", limit=1)
    assert first.next_cursor is not None
    for bad in (
        read.__name__,  # not even shaped like a cursor
        first.next_cursor.replace("g1.", "g1.9"),  # edited offset
        first.next_cursor[:-1] + ("0" if first.next_cursor[-1] != "0" else "1"),  # edited tag
    ):
        with pytest.raises(PolicyError) as caught:
            read(by="task", limit=1, cursor=bad)
        assert caught.value.condition == "invalid_cursor" and caught.value.exit_code == 2
    for other in (
        {"by": "task", "limit": 2},  # other page size
        {"by": "runtime", "limit": 1},  # other grouping
        {"by": "task", "limit": 1, "runtime": "claude"},  # other selection
        {"by": "task", "limit": 1, "since": "2026-10-12"},
    ):
        with pytest.raises(PolicyError) as caught:
            read(cursor=first.next_cursor, **other)
        assert caught.value.condition == "invalid_cursor"
    with pytest.raises(PolicyError):
        read(cursor=first.next_cursor)  # no grouping: this cursor belongs to no page


def test_a_cursor_dies_when_the_snapshot_changes_instead_of_skipping_rows(
    scene_root: Path,
) -> None:
    first = read(by="task", limit=1)
    cursor = first.next_cursor
    assert cursor is not None
    from tests.accounting.collect import verify_scenes as vs

    vs.pair(vs.projects(scene_root), 3, scene.at(hour=9))  # new evidence -> new generation
    assert scene.collect(scene.at(hour=21)).state == "published"
    with pytest.raises(PolicyError) as caught:
        service.read_summary(
            scope=build_scope(by="task"), now=scene.at(hour=21), limit=1, cursor=cursor
        )
    assert caught.value.condition == "invalid_cursor" and caught.value.exit_code == 2


def test_children_enter_only_with_include_children_and_only_for_a_session(
    scene_root: Path,
) -> None:
    with pytest.raises(PolicyError) as caught:
        build_scope(include_children=True)
    assert caught.value.exit_code == 2 and "--session" in str(caught.value)
    plain = read(session="cons0")
    with_children = read(session="cons0", include_children=True)
    assert nets(plain) == nets(with_children) == list(PER_TASK), "cons0 has no children"
    assert with_children.scope.include_children is True and plain.scope.include_children is False


def test_real_binary_group_summary_applies_filters_without_dead_page_commands(
    scene_root: Path,
) -> None:
    run = Binary(scene_root)
    data = run.data("savings", "--no-refresh", "--group-by", "task", "--runtime", "claude")["data"]
    assert data["scope"]["runtime"] == "claude"
    assert data["groups"]["rows"]
    text = run("savings", "--group-by", "task", "--runtime", "claude", columns=60)
    assert text.returncode == 0
    assert "BY TASK" in text.stdout
    assert "--cursor" not in text.stdout and "--limit" not in text.stdout


def test_numeric_fixture_uses_current_scoped_index_evidence_not_a_stale_pin(scene_root):
    from ssgrep.accounting.collect import frequency as collector_frequency
    from ssgrep.accounting.collect.frequency import BASIS_INDEX
    from ssgrep.accounting.views import ledger as service_ledger

    data, reason = service_ledger.load_ledger("personal-local")
    assert data is not None and reason is None
    assert data.doc is not None
    lookups = [row for row in data.doc.invocations if "search" in row.operations]
    assert len(lookups) == 3
    for row in lookups:
        view = collector_frequency.load_view("personal-local", row.frequency_view_id)
        assert view.basis == BASIS_INDEX and view.profile_scope == "personal-local"
        assert view.generation == "synthetic-ux-arithmetic-v1"
        assert view.document_unit == "indexed_chunk" and not view.calibrated
    assert all(
        candidate.rarity_lower_bound_milli is not None
        and candidate.rarity_lower_bound_milli >= 8000
        for candidate in data.candidates
    )


def test_stale_manual_pin_cannot_make_absent_current_index_complete(tmp_path):
    from tests.accounting.collect import verify_scenes as vs

    vs.enable(pin=True)
    collected_scene(tmp_path, pairs=(0,))
    view = read(task="claude:cons0#turn:u0")
    row = next(row for row in view.scenarios if row.unit_key == "context_tokens")
    assert [q.value for _, q in row.net] == [None, None, None]
    assert all(not estimate.complete and estimate.credit_missing for estimate in row.estimates)
    assert all(
        estimate.debit.value == 70 and estimate.known_net == -70 for estimate in row.estimates
    )
