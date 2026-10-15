"""10.2/10.3/10.1/1.4 over the REAL binary: delivery states, paused ordering, cursors, datasets.

Delivery states come from a real ``ssgrep metrics export`` cycle against a loopback receiver
(or a refused port), never from hand-written delivery documents. Screens are captured at
40/80/120 columns with NO_COLOR and written to ``UX_SCREENS_DIR`` when it is set. Datasets
are the synthetic builders of ``ux_datasets`` (mapping table in its docstring).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ssgrep.accounting import service
from ssgrep.accounting.ledger_contracts import SnapshotState
from tests.accounting import delivery_fixtures as fx, ux_datasets as ds
from tests.accounting.test_ux_journeys import ANSI_OTHER, SGR, WIDTHS, keep
from tests.accounting.test_ux_matrix import parse
from tests.accounting.test_ux_support import Binary, current_rarity_index

SECRET_FIELDS = re.compile(r"sk-CANARY|Bearer ")


def screens(run: Binary, name: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for width in WIDTHS:
        done = run("metrics", "status", columns=width, env=fx.ENV)
        assert done.returncode == 0 and not done.stderr, done.stderr[-300:]
        text = done.stdout
        assert not SGR.search(text) and not ANSI_OTHER.search(text), "NO_COLOR, no controls"
        assert max(len(line) for line in text.splitlines()) <= width
        assert not SECRET_FIELDS.search(text)
        keep(f"{name}-status", width, text)
        summary = run("savings", columns=width, env=fx.ENV)
        assert summary.returncode == 0
        keep(f"{name}-savings", width, summary.stdout)
        out[width] = text
    return out


def delivery_line(run: Binary) -> dict:
    code, body = run.json("metrics", "status")
    assert code == 0
    lines = [ln for d in body["data"]["data"]["destinations"] for ln in d["lines"]]
    assert len(lines) == 1, lines
    return lines[0]


@pytest.fixture
def isolated(tmp_path: Path) -> Path:
    return tmp_path


def test_offline_backlog_through_the_real_binary_is_pending_with_retry_never_synced(
    isolated: Path,
) -> None:
    with ds.delivery("offline", isolated) as (run, _):
        line = delivery_line(run)
        assert line["state"] == "offline", line
        assert line["pending_batches"] >= 1 and line["next_retry_utc"]
        assert line["acked_generation"] is None and line["last_acked_at"] is None
        for width, text in screens(run, "offline").items():
            flat = " ".join(text.split())
            assert "Not synced" in flat and "pending" in flat.lower(), width
            assert "Export offline" in flat
            assert "--check" in flat
            assert "Delivery outcome unknown" not in flat, "a refused connection is offline"
        # the same cause is not spelled out twice (STATE explains, DELIVERY points back)
        wide = screens(run, "offline")[120]
        assert wide.count("Destination unreachable") == 1


def test_credentials_expired_through_the_real_binary_names_rotation_and_no_secret(
    isolated: Path,
) -> None:
    with ds.delivery("auth_error", isolated) as (run, receiver):
        assert receiver is not None and receiver.requests and receiver.requests[0].status == 401
        line = delivery_line(run)
        assert line["state"] == "credentials_expired" and line["blocked_reason"]
        assert line["next_retry_utc"] is None, "a rejected credential is not retried blindly"
        for width, text in screens(run, "auth").items():
            flat = " ".join(text.split())
            assert "Credentials expired" in flat and "--check" in flat, width
            assert fx.SECRET not in text and "token rejected" not in text


def test_partial_rejection_through_the_real_binary_is_never_called_acknowledged(
    isolated: Path,
) -> None:
    with ds.delivery("partial_export", isolated) as (run, receiver):
        assert receiver is not None and receiver.requests
        line = delivery_line(run)
        assert line["state"] == "partial_export", line
        assert line["rejected_points"] == 4 and line["acked_generation"] is None
        for width, text in screens(run, "partial").items():
            flat = " ".join(text.split())
            assert "Partial export" in flat and "rejected points 4" in flat, width
            assert "synthetic rejection" not in text, "receiver text is never surfaced"


def test_paused_profile_with_pending_export_keeps_the_backlog_and_names_resume(
    isolated: Path,
) -> None:
    with ds.delivery("paused_pending", isolated) as (run, _):
        line = delivery_line(run)
        assert line["next_action"]["command"] == "ssgrep metrics resume", line
        assert line["pending_batches"] >= 1, "pause neither drops nor sends the backlog"
        code, body = run.json("savings")
        assert code == 0 and body["data"]["state"] == "paused"
        summary_actions = [a["command"] for a in body["data"]["next_actions"]]
        assert "ssgrep metrics resume" in summary_actions
        for width, text in screens(run, "paused-pending").items():
            flat = " ".join(text.split())
            assert "PAUSED" in flat and "resume" in flat, width
        before = fx.tree()
        send = run("metrics", "export", "--destination", fx.DEST, "--json", env=fx.ENV)
        assert send.returncode == 1 and parse(send.stdout)["condition"] == "export_paused"
        assert fx.tree() == before, "a paused send changed nothing"


@pytest.mark.parametrize("mode", [(), ("--check",)])
def test_paused_export_exits_1_before_any_import_or_network_text_and_json(
    isolated: Path, mode: tuple[str, ...]
) -> None:
    with ds.delivery("offline", isolated) as (run, receiver):
        assert receiver is None
        assert run("metrics", "pause", "--json", env=fx.ENV).returncode == 0
        before = fx.tree()
        base = ("metrics", "export", "--destination", fx.DEST, *mode)
        as_json = run(*base, "--json", env=fx.ENV)
        assert as_json.returncode == 1 and as_json.stderr == ""
        body = parse(as_json.stdout)  # exactly one JSON document
        assert body["condition"] == "export_paused" and body["ok"] is False
        assert body["command"].endswith("metrics resume")
        text = run(*base, env=fx.ENV)
        assert text.returncode == 1 and "paused" in (text.stdout + text.stderr).lower()
        assert "Traceback" not in text.stderr and "ImportError" not in text.stderr
        assert fx.tree() == before


# ---- 1.4: every dataset reproduces its experience state --------------------------------------

EXPECT = {
    "disabled": (SnapshotState.DISABLED, {"disabled"}),
    "no_snapshot": (SnapshotState.NO_SNAPSHOT, set()),
    "fresh": (SnapshotState.FRESH, {"partial", "negative"}),
    "no_reuse": (SnapshotState.FRESH, {"no_reuse"}),
    "stale": (SnapshotState.STALE, {"stale"}),
    "paused": (SnapshotState.PAUSED, {"paused"}),
    "evidence_expired": (SnapshotState.FRESH, {"evidence_expired"}),
    "overlap": (SnapshotState.FRESH, set()),
    "revised": (SnapshotState.FRESH, set()),
}


@pytest.mark.parametrize("name", ds.STATES)
def test_each_dataset_reproduces_its_state_and_conditions(
    name: str, isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if name == "fresh":  # the negative arithmetic dataset explicitly has current rarity evidence
        current_rarity_index(isolated, monkeypatch)
    built = ds.BUILDERS[name](isolated)
    view = service.read_summary(now=built.now)
    expected_state, expected_conditions = EXPECT[name]
    assert view.state is expected_state, (name, view.state)
    assert expected_conditions <= {c.code for c in view.conditions}, name
    for condition in view.conditions:  # every state names a real next action (or none by design)
        assert condition.cause
    assert not any(q.value == 0 and q.basis.value == "unavailable" for _, q in view.measured), (
        "unknown is never rendered as zero"
    )
    json.dumps(
        {"state": view.state.value, "conditions": [c.code for c in view.conditions]}
    )  # JSON-safe, integer-only


def test_policy_discards_clear_active_offline_status_without_relabeling_loss(
    isolated: Path,
) -> None:
    with ds.delivery("offline", isolated) as (run, _):
        before = delivery_line(run)
        assert before["state"] == "offline" and before["pending_batches"] > 0
        fx.amend(roots=(str(isolated / "narrower-workspace"),))
        run("metrics", "export", "--destination", fx.DEST, "--json", env=fx.ENV)
        durable = fx.delivery()
        assert durable is not None
        assert durable.pending_batches == 0 and durable.lost_records > 0
        assert durable.loss_summary[0].reason == "policy_contraction"
        after = delivery_line(run)
        assert after["state"] == "waiting"
        assert after["lost_records"] == durable.lost_records
        assert "policy_contraction" in after["cause"]
        assert "expired" not in after["cause"] and "unreachable" not in after["cause"]
        assert after["next_action"]["command"].endswith("--preview")
