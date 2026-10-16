"""``ssgrep savings export`` through the real installed binary in a subprocess.

Real files under the isolated data dir, a real loopback lab receiver, and assertions on state,
bytes and request counts (never on help text). Nothing here touches the real HOME or network.
"""

from __future__ import annotations

import hashlib
import json
import os
import pty
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting.collect import store as collector_store
from ssgrep.accounting.export import acks, consent as export_consent, ids as export_ids
from ssgrep.accounting.export.consent import approval_reason
from ssgrep.accounting.ledger.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.privacy import policy
from ssgrep.accounting.views import snapshots
from tests.accounting.export.http_builders import make_snapshot, synthetic_current_checkpoint
from tests.accounting.export.lab_receiver import LabReceiver, Respond

BINARY = Path(sys.executable).parent / "ssgrep"
PROFILE = "personal-local"
DEST = "org-primary"
TOKEN = "sk-CANARY-SECRET-0123456789"
TOKEN_ENV = "SSGREP_ORG_TOKEN"
EMPTY_METRICS = b'{"resourceMetrics":[]}'


class Run:
    def __init__(self, home: Path, data: Path) -> None:
        self.home, self.data = home, data

    def environment(self, env: dict[str, str] | None = None, columns: int = 80) -> dict[str, str]:
        base = {k: v for k, v in os.environ.items() if k != TOKEN_ENV}
        return {
            **base,
            "HOME": str(self.home),
            "SSGREP_DATA_DIR": str(self.data),
            "COLUMNS": str(columns),
            "NO_COLOR": "1",
            **(env or {}),
        }

    def __call__(
        self,
        *args: str,
        columns: int = 80,
        stdin: str | None = "",
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(BINARY), *args],
            input=stdin,
            capture_output=True,
            text=True,
            env=self.environment(env, columns),
            timeout=120,
            check=False,
        )

    def tty(self, *args: str, answer: str) -> tuple[int, str, str]:
        master, slave = pty.openpty()
        process = subprocess.Popen(
            [str(BINARY), *args],
            stdin=slave,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.environment(),
            text=True,
        )
        os.write(master, answer.encode())
        stdout, stderr = process.communicate(timeout=120)
        os.close(master)
        os.close(slave)
        return process.returncode, stdout, stderr

    def tree(self) -> dict[str, tuple[int, int, bytes | None]]:
        """Names, sizes, mtimes and bytes of everything under the data dir."""
        if not self.data.exists():
            return {}
        found: dict[str, tuple[int, int, bytes | None]] = {}
        for path in sorted(self.data.rglob("*")):
            info = path.lstat()
            found[str(path.relative_to(self.data))] = (
                info.st_size,
                info.st_mtime_ns,
                path.read_bytes() if path.is_file() else None,
            )
        return found

    def names(self) -> list[str]:
        return sorted(self.tree())


@pytest.fixture
def run(tmp_path: Path) -> Run:
    home = tmp_path / "home"
    home.mkdir()
    return Run(home, tmp_path / "data")


@pytest.fixture
def receiver() -> Iterator[LabReceiver]:
    with LabReceiver.start() as lab:
        yield lab


def document(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return json.loads(result.stdout.strip().splitlines()[-1])


def configure(
    url: str,
    *,
    enabled: tuple[tuple[str, str], ...] = (),
    credential: str | None = f"env:{TOKEN_ENV}",
    logs: bool = True,
) -> None:
    signals = (Signal.METRICS, Signal.LOGS) if logs else (Signal.METRICS,)
    current = policy.save_profile(
        ProfilePolicy(
            destinations=(
                DestinationPolicy(
                    DEST, url, signals, logs_permitted=logs, credential_ref=credential
                ),
            ),
            enabled_exports=enabled,
            collection_enabled=True,
        )
    )
    checkpoint = synthetic_current_checkpoint(PROFILE, current)
    collector_store.commit_checkpoint(PROFILE, checkpoint)
    for destination, signal in enabled:
        export_consent.apply_enable(PROFILE, destination, Signal(signal))


def publish(generation: int = 1) -> None:
    snapshots.publish(PROFILE, make_snapshot(generation))


def enabled_pairs() -> tuple[tuple[str, str], ...]:
    return policy.policy_status(PROFILE).profile.enabled_exports


def export(*extra: str, signal: str | None = None) -> tuple[str, ...]:
    flags = set(extra)
    mode = next(
        (
            name.removeprefix("--")
            for name in ("--preview", "--check", "--enable", "--disable")
            if name in flags
        ),
        "send",
    )
    tail = tuple(
        value for value in extra if value not in {"--preview", "--check", "--enable", "--disable"}
    )
    if signal == "logs":
        tail = (*tail, "--logs")
    return ("savings", "export", mode, DEST, *tail)


def tokened() -> dict[str, str]:
    return {TOKEN_ENV: TOKEN}


# ---- argument validation: exit 2 before any side effect ---------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ("savings", "export", "send", DEST, "--preview"),
        ("savings", "export", "send", DEST, "--signal", "logs"),
        ("savings", "export", "send", DEST, "--profile", "../etc"),
    ],
)
def test_export_mode_flags_and_obsolete_signal_option_are_rejected_before_side_effects(
    run: Run, args: tuple[str, ...]
) -> None:
    result = run(*args, "--json")
    assert result.returncode == 2
    assert document(result)["ok"] is False
    assert run.names() == []


def test_destination_is_required_and_must_be_a_bounded_id(run: Run) -> None:
    before = run.tree()
    missing = run("savings", "export", "send")
    assert missing.returncode == 2
    assert "Usage:" in missing.stdout and "Arguments:" in missing.stdout
    bad = run("savings", "export", "send", "../x", "--json")
    assert bad.returncode == 2 and document(bad)["condition"] == "invalid_destination"
    assert run.tree() == before


def test_unknown_destination_and_unpermitted_signal_exit_2_and_change_nothing(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url, logs=False)
    before = run.tree()
    for args, condition in (
        (("savings", "export", "enable", "ghost", "--yes"), "unknown_destination"),
        (export("--enable", "--yes", signal="logs"), "signal_not_permitted"),
        (("savings", "export", "preview", "ghost"), "unknown_destination"),
        (("savings", "export", "disable", "ghost"), "unknown_destination"),
    ):
        result = run(*args, "--json")
        assert result.returncode == 2, result.stdout
        assert document(result)["condition"] == condition
    assert run.tree() == before and receiver.requests == []


def test_logs_need_separate_permission_before_the_prompt_even_with_yes(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url)
    document_ = json.loads(policy.policy_path().read_text())
    for dest in document_["profiles"][PROFILE]["destinations"]:
        dest["logs_permitted"] = False
    policy.policy_path().write_text(json.dumps(document_))
    before = run.tree()
    result = run(*export("--enable", "--yes", "--json", signal="logs"))
    assert result.returncode == 2 and document(result)["condition"] == "logs_not_permitted"
    assert run.tree() == before and "preview" not in result.stdout
    tty_code, stdout, _ = run.tty(*export("--enable", signal="logs"), answer="y\n")
    assert tty_code == 2 and "[y/N]" not in stdout
    assert run.tree() == before


# ---- enable: always previews, always confirms -------------------------------------------------


def test_enable_without_yes_is_refused_in_text_and_json_before_any_write(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url)
    before = run.tree()
    text = run(*export("--enable"))
    assert text.returncode == 2 and "--yes" in text.stderr
    as_json = run(*export("--enable", "--json"))
    assert as_json.returncode == 2 and document(as_json)["condition"] == "confirmation_required"
    assert run.tree() == before and enabled_pairs() == ()


def test_enable_yes_persists_exactly_one_pair_and_sends_nothing(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url)
    publish()
    result = run(*export("--enable", "--yes"), env=tokened())
    assert result.returncode == 0, result.stderr
    assert enabled_pairs() == ((DEST, "metrics"),)
    assert receiver.requests == [] and receiver.connections == 0
    assert not (run.data / "accounting" / PROFILE / "spool").exists()
    assert not (run.data / "accounting" / PROFILE / "delivery").exists()
    assert DEST in result.stdout and f"127.0.0.1:{receiver.port}" in result.stdout
    assert f"env:{TOKEN_ENV}" in result.stdout
    assert approval_reason(PROFILE, DEST, Signal.METRICS) is None
    assert approval_reason(PROFILE, DEST, Signal.LOGS) == "consent_reapproval_required"
    preview = document(run(*export("--preview", "--json")))["data"]["data"]
    assert preview["signal"] == "metrics" and preview["network"] is False
    net = [
        int(point["asInt"])
        for batch in preview["batches"]
        for resource in batch["payload"]["resourceMetrics"]
        for scope in resource["scopeMetrics"]
        for metric in scope["metrics"]
        if metric["name"] == "ssgrep_estimated_net_tokens"
        for point in metric["gauge"]["dataPoints"]
    ]
    assert sorted(net) == [-12000, -7000, 4000]
    assert receiver.requests == [] and receiver.connections == 0
    assert TOKEN not in result.stdout + result.stderr


def test_enable_json_document_carries_the_preview_and_the_credential_reference_only(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url)
    result = run(*export("--enable", "--yes", "--json", signal="logs"), env=tokened())
    assert result.returncode == 0, result.stderr
    envelope = document(result)["data"]
    assert set(envelope) == {
        "schema_version",
        "state",
        "data",
        "warnings",
        "next_actions",
        "provenance",
    }
    assert envelope["state"] == "enabled" and envelope["data"]["changed"] is True
    shown = envelope["data"]["preview"]
    assert shown["destination"] == DEST and shown["signal"] == "logs"
    assert shown["credential_ref"] == f"env:{TOKEN_ENV}"
    approval = json.loads(
        (run.data / "accounting" / PROFILE / "export" / DEST / "consent-logs.json").read_bytes()
    )
    assert approval == {
        "schema": 1,
        "profile": PROFILE,
        "destination": DEST,
        "signal": "logs",
        "fields_version": shown["fields_version"],
    }
    assert approval_reason(PROFILE, DEST, Signal.LOGS) is None
    assert approval_reason(PROFILE, DEST, Signal.METRICS) == "consent_reapproval_required"
    fields = " ".join(shown["fields"]).lower()
    assert all(term in fields for term in ("revision", "known subtotals", "bounds", "methodology"))
    assert receiver.requests == [] and receiver.connections == 0
    assert shown["policy_version"] == 1 and shown["retention"]["spool_days"] == 7
    assert shown["endpoint"] == receiver.url
    notices = " ".join(shown["notices"]).lower()
    assert "tombstones" in notices and "non-reversible" in notices
    assert enabled_pairs() == ((DEST, "logs"),)
    assert TOKEN not in result.stdout + result.stderr
    again = document(run(*export("--enable", "--yes", "--json", signal="logs")))["data"]
    assert again["data"]["changed"] is False and enabled_pairs() == ((DEST, "logs"),)


def test_interactive_cancel_changes_nothing_and_confirm_enables(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url)
    before = run.tree()
    code, stdout, stderr = run.tty(*export("--enable"), answer="n\n")
    assert code == 0 and "Cancelled. Nothing was changed." in stderr
    assert "WILL AUTHORIZE" in stdout.upper() and "[y/N]" in stdout
    assert run.tree() == before and enabled_pairs() == ()
    code, stdout, _ = run.tty(*export("--enable"), answer="yes\n")
    assert code == 0 and enabled_pairs() == ((DEST, "metrics"),)
    assert receiver.requests == []


def test_text_preview_is_width_aware(run: Run, receiver: LabReceiver) -> None:
    configure(receiver.url)
    for columns in (40, 80, 120):
        result = run(*export("--enable", "--yes"), columns=columns)
        assert result.returncode == 0
        assert max(len(line) for line in result.stdout.splitlines()) <= columns


def test_resume_never_reenables_a_disabled_signal(run: Run, receiver: LabReceiver) -> None:
    configure(receiver.url, enabled=((DEST, "logs"), (DEST, "metrics")))
    assert run(*export("--disable", signal="logs")).returncode == 0
    assert run("savings", "pause").returncode == 0
    assert run("savings", "resume").returncode == 0
    assert enabled_pairs() == ((DEST, "metrics"),)


# ---- disable ----------------------------------------------------------------------------------


def test_disable_stops_one_signal_keeps_everything_and_is_idempotent(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url, enabled=((DEST, "logs"), (DEST, "metrics")))
    publish()
    key = export_ids.load_key(PROFILE, create=True)
    assert key
    sent = run(*export("--json"), env=tokened())
    assert sent.returncode == 0, sent.stdout + sent.stderr
    assert receiver.requests, "the metrics batch must have reached the lab receiver"
    owned = ("accounting", "accounting/.lock", "accounting/policy.json")
    before = {k: v for k, v in run.tree().items() if k not in owned}
    result = run(*export("--disable", "--json"))
    assert result.returncode == 0, result.stderr
    body = document(result)["data"]
    assert body["state"] == "disabled" and body["data"]["changed"] is True
    assert body["data"]["retained"]["spool"] and body["data"]["in_flight_note"]
    assert enabled_pairs() == ((DEST, "logs"),), "only the metrics signal is disabled"
    after = {k: v for k, v in run.tree().items() if k not in owned}
    assert after == before, "delivery history, spool and registry are untouched"
    text = run(*export("--disable"))
    assert text.returncode == 0 and "already off" in text.stdout
    assert "cannot be recalled" in text.stdout
    assert enabled_pairs() == ((DEST, "logs"),)


def test_disable_reports_an_in_flight_request_it_cannot_recall(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),))
    batch_id = "g0000000001-" + "a" * 16 + "-0000"
    acks.update_delivery(
        PROFILE,
        DEST,
        Signal.METRICS,
        lambda d: acks.start_in_flight(acks.collected(d, 1), batch_id, 1, "2026-10-13T10:00:00Z"),
    )
    result = run(*export("--disable", "--json"))
    assert result.returncode == 0
    assert document(result)["data"]["data"]["in_flight"] == batch_id
    assert batch_id in run(*export("--disable")).stdout


# ---- preview ----------------------------------------------------------------------------------


def test_preview_json_shows_every_request_and_has_zero_side_effects(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url)  # NOT enabled: preview is still allowed
    publish()
    before = run.tree()
    result = run(*export("--preview", "--json"), env=tokened())
    assert result.returncode == 0, result.stderr
    assert run.tree() == before, "names, sizes, mtimes and bytes of the data dir are unchanged"
    assert receiver.requests == [] and receiver.connections == 0
    body = document(result)["data"]["data"]
    assert body["state"] == "ok" and body["network"] is False and body["state_changed"] is False
    assert body["destination"] == DEST and body["signal"] == "metrics"
    assert body["policy_version"] == 1
    assert body["endpoint_host"] == f"127.0.0.1:{receiver.port}"
    assert body["batch_count"] == len(body["batches"]) >= 1
    assert body["bytes"] == sum(b["bytes"] for b in body["batches"])
    for batch in body["batches"]:
        assert set(batch) == {"batch_id", "bytes", "items", "sha256", "payload"}
        assert batch["payload"]["resourceMetrics"]
    assert TOKEN not in result.stdout + result.stderr
    lowered = result.stdout.lower()
    assert "authorization" not in lowered and "bearer" not in lowered
    assert "pseudonym.key" not in lowered


def test_preview_text_prints_each_request_verbatim_and_states_the_cap(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url)
    publish()
    assert export_ids.load_key(PROFILE, create=True)  # stable pseudonyms across the two runs
    as_json = document(run(*export("--preview", "--json")))["data"]["data"]
    text = run(*export("--preview"))
    assert text.returncode == 0
    for batch in as_json["batches"]:
        assert batch["sha256"] in text.stdout
        printed = [
            line for line in text.stdout.splitlines() if line.startswith('{"resourceMetrics"')
        ]
        assert printed and hashlib.sha256(printed[0].encode()).hexdigest() == batch["sha256"]
    assert "no network request" in text.stdout.lower()
    assert receiver.requests == []


def test_preview_works_while_paused_and_reports_states_for_empty_candidates(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),))
    absent = document(run(*export("--preview", "--json")))["data"]
    assert absent["state"] == "no_snapshot"
    publish()
    assert run("savings", "pause").returncode == 0
    before = run.tree()
    paused = run(*export("--preview", "--json"))
    assert paused.returncode == 0 and document(paused)["data"]["state"] == "ok"
    assert run.tree() == before and receiver.requests == []
    logs = document(run(*export("--preview", "--json", signal="logs")))["data"]["data"]
    assert logs["state"] in {"no_records", "no_changes"} and logs["batches"] == []


def test_previewed_bytes_equal_what_a_real_send_later_puts_on_the_wire(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),))
    publish()
    assert export_ids.load_key(PROFILE, create=True)  # a persisted key: pseudonyms are stable
    previewed = document(run(*export("--preview", "--json")))["data"]["data"]
    assert previewed["pseudonym_key_persisted"] is True
    sent = run(*export("--json"), env=tokened())
    assert sent.returncode == 0, sent.stdout + sent.stderr
    on_wire = sorted(hashlib.sha256(r.body).hexdigest() for r in receiver.requests)
    assert on_wire == sorted(b["sha256"] for b in previewed["batches"])
    delivery = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    assert delivery is not None and delivery.last_acked_generation == 1


# ---- check ------------------------------------------------------------------------------------


def test_check_sends_one_empty_request_with_the_credential_and_changes_no_state(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),))
    publish()
    before = run.tree()
    result = run(*export("--check", "--json"), env=tokened())
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(receiver.requests) == 1
    request = receiver.requests[0]
    assert (request.method, request.path) == ("POST", "/v1/metrics")
    assert request.body == EMPTY_METRICS
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    body = document(result)["data"]["data"]
    assert body["ok"] is True and body["state"] == "ok" and body["http_status"] == 200
    assert run.tree() == before, "check must not write any delivery state"
    assert TOKEN not in result.stdout + result.stderr
    assert acks.read_delivery(PROFILE, DEST, Signal.METRICS) is None


def test_check_while_paused_exits_1_without_network(run: Run, receiver: LabReceiver) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),))
    assert run("savings", "pause").returncode == 0
    before = run.tree()
    for flags in (("--json",), ()):
        result = run(*export("--check", *flags), env=tokened())
        assert result.returncode == 1
        if flags:
            assert document(result)["condition"] == "export_paused"
    assert receiver.requests == [] and receiver.connections == 0
    assert run.tree() == before


@pytest.mark.parametrize("status", [401, 403])
def test_check_unauthorized_exits_1_with_a_credential_action_and_no_body_leak(
    run: Run, receiver: LabReceiver, status: int
) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),))
    receiver.queue(Respond(status, b'{"error":"RESPONSE-BODY-CANARY"}'))
    before = run.tree()
    result = run(*export("--check", "--json"), env=tokened())
    assert result.returncode == 1, result.stdout
    body = document(result)
    assert body["ok"] is False and body["condition"].startswith("check_")
    assert "credential reference" in body["message"].lower()
    assert "RESPONSE-BODY-CANARY" not in result.stdout + result.stderr
    assert TOKEN not in result.stdout + result.stderr
    assert run.tree() == before


def test_check_reports_unreachable_without_changing_state(run: Run) -> None:
    with LabReceiver.start() as dead:
        url = dead.url
    configure(url, enabled=((DEST, "metrics"),))
    before = run.tree()
    result = run(*export("--check", "--json"), env=tokened())
    assert result.returncode == 1
    assert document(result)["condition"] == "check_unreachable"
    assert run.tree() == before


# ---- manual send ------------------------------------------------------------------------------


def test_send_while_paused_exits_1_export_paused_with_no_network_and_nothing_written(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),))
    publish()
    assert run("savings", "pause").returncode == 0
    before = run.tree()
    result = run(*export("--json"), env=tokened())
    assert result.returncode == 1 and document(result)["condition"] == "export_paused"
    assert receiver.requests == [] and receiver.connections == 0
    assert run.tree() == before


def test_send_when_not_enabled_exits_1_with_the_enable_command(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url)
    publish()
    before = run.tree()
    result = run(*export("--json"), env=tokened())
    assert result.returncode == 1
    body = document(result)
    assert body["condition"] == "export_not_enabled"
    assert body["command"] == f"ssgrep savings export enable {DEST}"
    assert receiver.requests == [] and run.tree() == before


# ---- policy damage ----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["--preview", "--check", "--enable", "--disable", None])
def test_corrupt_policy_is_policy_corrupt_exit_1_for_every_mode_and_changes_nothing(
    run: Run, receiver: LabReceiver, mode: str | None
) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),))
    policy.policy_path().write_bytes(b"{garbage")
    before = run.tree()
    args = export(*([mode] if mode else []), *(("--yes",) if mode == "--enable" else ()), "--json")
    result = run(*args, env=tokened())
    assert result.returncode == 1, result.stdout + result.stderr
    assert document(result)["condition"] == "policy_corrupt"
    assert receiver.requests == [] and run.tree() == before


def test_policy_shrink_after_enable_stops_everything_and_reports_a_usage_refusal(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),))
    publish()
    current = policy.policy_status(PROFILE).profile
    policy.save_profile(
        ProfilePolicy(
            profile_id=PROFILE,
            collection_enabled=True,
            paused=current.paused,
            destinations=(),
            enabled_exports=(),
        )
    )
    for mode in ((), ("--check",), ("--preview",), ("--enable", "--yes")):
        result = run(*export(*mode, "--json"), env=tokened())
        assert result.returncode == 2 and document(result)["condition"] == "unknown_destination"
    assert receiver.requests == []


# ---- canary: no secret ever reaches output ----------------------------------------------------


def test_a_secret_in_the_credential_file_or_env_never_appears_in_any_mode(
    run: Run, receiver: LabReceiver, tmp_path: Path
) -> None:
    secret_file = tmp_path / "secrets" / "ingest.token"
    secret_file.parent.mkdir()
    secret_file.write_text(TOKEN + "\n")
    secret_file.chmod(0o600)
    for reference in (f"env:{TOKEN_ENV}", f"file:{secret_file}"):
        configure(receiver.url, credential=reference)
        publish()
        outputs: list[str] = []
        for json_flag in ((), ("--json",)):
            for mode in (("--preview",), ("--enable", "--yes"), ("--check",), (), ("--disable",)):
                result = run(*export(*mode, *json_flag), env=tokened())
                outputs.append(result.stdout + result.stderr)
        assert any(outputs)
        joined = "\n".join(outputs)
        assert TOKEN not in joined, reference
        assert str(secret_file.parent) not in joined, "only the file's base name may be shown"
        assert "Bearer" not in joined
    assert any(r.headers.get("authorization") == f"Bearer {TOKEN}" for r in receiver.requests)


# ---- the full journey against a real receiver --------------------------------------------------


def test_preview_enable_check_send_disable_journey(run: Run, receiver: LabReceiver) -> None:
    configure(receiver.url)
    publish()
    # preview: nothing exists yet, no request leaves the machine
    before = run.tree()
    preview = run(*export("--preview", "--json"), env=tokened())
    assert preview.returncode == 0 and run.tree() == before and receiver.requests == []
    # a manual send is refused until consent exists
    refused = run(*export("--json"), env=tokened())
    assert refused.returncode == 1 and document(refused)["condition"] == "export_not_enabled"
    assert receiver.requests == []
    # enable: no network, no data sent
    enable = run(*export("--enable", "--yes", "--json"), env=tokened())
    assert enable.returncode == 0 and receiver.requests == []
    assert enabled_pairs() == ((DEST, "metrics"),)
    # check: exactly one EMPTY request carrying the credential, nothing acknowledged
    check = run(*export("--check", "--json"), env=tokened())
    assert check.returncode == 0
    assert [r.body for r in receiver.requests] == [EMPTY_METRICS]
    assert receiver.requests[0].headers["authorization"] == f"Bearer {TOKEN}"
    assert acks.read_delivery(PROFILE, DEST, Signal.METRICS) is None
    # send: data arrives and is acknowledged
    sent = run(*export("--json"), env=tokened())
    assert sent.returncode == 0, sent.stdout + sent.stderr
    assert sent.stdout and document(sent)["data"]["state"] == "sent"
    data_requests = receiver.requests[1:]
    assert data_requests and all(r.body != EMPTY_METRICS for r in data_requests)
    delivery = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    assert delivery is not None and delivery.last_acked_generation == 1
    assert delivery.pending_batches == 0
    # disable: sending stops and the receiver sees nothing more
    count = len(receiver.requests)
    assert run(*export("--disable", "--json")).returncode == 0
    stopped = run(*export("--json"), env=tokened())
    assert stopped.returncode == 1 and document(stopped)["condition"] == "export_not_enabled"
    assert len(receiver.requests) == count
    still = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    assert still is not None and still.last_acked_generation == 1, "history is preserved"


def test_savings_help_is_the_only_accounting_command_group_and_explains_export_modes(
    run: Run,
) -> None:
    root = run("--help")
    assert root.returncode == 0
    assert "savings" in root.stdout
    assert "metrics" not in root.stdout

    group = run("savings", "--help")
    assert group.returncode == 0
    for section in ("Start here", "Everyday", "Organization export", "Maintenance"):
        assert section in group.stdout
    assert "How much ssgrep has saved you" in group.stdout
    assert "ssgrep savings setup" in group.stdout

    export_help = run("savings", "export", "--help")
    assert export_help.returncode == 0
    export_summary = " ".join(export_help.stdout.split())
    assert (
        "Organization export: preview, check, enable, disable, send data, or run its service"
        in export_summary
    )
    for mode in ("preview", "check", "enable", "disable", "send", "serve"):
        assert mode in export_help.stdout
    # The user-facing summary command is read-only when disabled, while still
    # rendering a useful next step in both human and JSON modes.
    before = run.tree()
    text = run("savings", "--no-refresh")
    assert text.returncode == 0 and "measurement is off" in text.stdout.lower()
    assert "ssgrep savings setup" in text.stdout
    assert run.tree() == before

    machine = run("savings", "--no-refresh", "--json")
    assert machine.returncode == 0
    summary = document(machine)
    assert summary["data"]["state"] == "disabled"
    assert run.tree() == before

    old_group = run("metrics", "--help")
    assert old_group.returncode != 0


def test_savings_interactive_dispatches_the_selected_group_command(
    run: Run, tmp_path: Path
) -> None:
    hook = tmp_path / "sitecustomize.py"
    hook.write_text(
        "from usecli.cli.commands.defaults.base.internal import fzf_command\n"
        "def choose(options, prompt=''):\n"
        "    return 'collect' if any('collect' in item for item in options) else None\n"
        "fzf_command._run_fzf_menu = choose\n",
        encoding="utf-8",
    )
    before = run.tree()

    result = run("savings", "--interactive", env={"PYTHONPATH": str(tmp_path)})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ssgrep savings collect" in result.stdout + result.stderr
    assert run.tree() == before


def test_stale_summary_refresh_failure_is_reported_without_raw_error_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ssgrep.accounting.collect import collector
    from ssgrep.accounting.collect.store import CollectError
    from ssgrep.accounting.ledger.ledger_contracts import SnapshotState
    from ssgrep.accounting.views import service, types as view_types
    from ssgrep.cli.commands import savings_command

    view = view_types.SummaryView(
        profile=PROFILE,
        state=SnapshotState.STALE,
        paused=False,
        paused_at=None,
        stale=True,
        scope=view_types.Scope(),
        collected_at=None,
        age_seconds=None,
        collection_enabled=True,
        exports=(),
    )
    summary_reads = 0
    collection_calls: list[tuple[str, object]] = []

    def read_summary(*args: object, **kwargs: object) -> view_types.SummaryView:
        nonlocal summary_reads
        summary_reads += 1
        return view

    def collect_once(profile: str, options: object) -> None:
        collection_calls.append((profile, options))
        raise CollectError("state_corrupt", "SECRET details must not reach output")

    monkeypatch.setattr(view_types, "check_profile", lambda profile: None)
    monkeypatch.setattr(service, "read_summary", read_summary)
    monkeypatch.setattr(collector, "collect_once", collect_once)
    monkeypatch.setattr(savings_command, "is_json_mode", lambda: True)

    result = object.__new__(savings_command.SavingsCommand).handle()

    assert summary_reads == 1
    assert len(collection_calls) == 1 and collection_calls[0][0] == PROFILE
    assert result is not None and result["state"] == "stale"
    assert result["warnings"] == ["refresh_failed: state_corrupt"]
    assert "SECRET details" not in str(result)


def test_interactive_savings_delegates_to_group_listing_without_reading_or_collecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.core import TyperCommand as ClickCommand
    from usecli.cli.core.base_command import ClickContext

    from ssgrep.accounting.collect import collector
    from ssgrep.accounting.views import service
    from ssgrep.cli.commands import savings_command

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("interactive mode must not read the summary or collect")

    monkeypatch.setattr(service, "read_summary", forbidden)
    monkeypatch.setattr(collector, "collect_once", forbidden)
    command = object.__new__(savings_command.SavingsCommand)
    calls: list[tuple[object, ...]] = []

    def listing(*args: object) -> dict[str, object]:
        calls.append(args)
        return {"picked": "collect"}

    command._listing_callback = listing
    ctx = ClickContext(ClickCommand("savings"))

    result = command._group_callback(ctx, False, True, False)

    assert result == {"picked": "collect"}
    assert calls == [(ctx, False, True, False)]


def test_savings_command_paths_match_the_registered_group_hierarchy() -> None:
    from ssgrep.cli.commands import metrics_export_command, metrics_serve_command, savings_command

    paths = {
        "savings": savings_command.SavingsCommand,
        "savings export": metrics_export_command.MetricsExportCommand,
        "savings export serve": metrics_serve_command.MetricsServeCommand,
    }
    for path, cls in paths.items():
        assert object.__new__(cls).signature() == path


def test_status_includes_source_census_and_details_page_without_writes(run: Run) -> None:
    before = run.tree()
    status = run("savings", "status", "--sources", "--json")
    assert status.returncode == 0, status.stderr
    data = document(status)["data"]["data"]
    assert "source_census" in data
    assert data["source_census"]["basis"] == "latest_collected_source_watermarks"
    assert "inventory" in data["source_census"]
    assert isinstance(data["source_census"]["inventory"]["totals"]["sources"], int)
    assert run.tree() == before

    details = run("savings", "status", "--details", "--json")
    assert details.returncode == 0, details.stderr
    detail_data = document(details)["data"]["data"]
    assert "diagnostics" in detail_data
    assert run.tree() == before


def test_status_human_source_census_reports_collected_watermarks_without_scanning(
    run: Run,
) -> None:
    before = run.tree()
    result = run("savings", "status", "--sources")

    assert result.returncode == 0, result.stderr
    assert "SOURCE CENSUS" in result.stdout
    assert "Latest collected source watermarks" in result.stdout
    assert "Sources" in result.stdout and "no separate filesystem scan" in result.stdout
    assert run.tree() == before


def test_required_export_purge_and_receipt_arguments_show_command_help_without_writes(
    run: Run,
) -> None:
    before = run.tree()
    for args in (
        ("savings", "export", "preview"),
        ("savings", "purge"),
        ("savings", "explain"),
    ):
        result = run(*args)
        assert result.returncode == 2
        assert "Usage:" in result.stdout and "Arguments:" in result.stdout

    assert run.tree() == before


def test_savings_options_are_self_describing_shortcut_and_bounded(run: Run) -> None:
    help_text = run("savings", "--help").stdout
    for spelling in (
        "--from, -f",
        "--to, -t",
        "--project, -p",
        "--runtime, -r",
        "--session, -s",
        "--group-by, -g",
        "--no-refresh, -N",
        "--profile",
    ):
        assert spelling in help_text
    for obsolete in ("--since", "--until", "--by", "--cursor", "--include-children"):
        assert obsolete not in help_text

    parsed = run("savings", "--from", "2026-10-01", "--to", "2026-10-02", "--json")
    assert parsed.returncode == 0, parsed.stderr
    legacy = run("savings", "--since", "2026-10-01")
    assert legacy.returncode == 2

    commands = (
        ("savings", "setup"),
        ("savings", "status"),
        ("savings", "collect"),
        ("savings", "lookups"),
        ("savings", "history"),
        ("savings", "export", "send", "destination"),
        ("savings", "export", "serve"),
        ("savings", "purge", "2026-10-01"),
    )
    for command in commands:
        help_result = run(*command, "--help")
        assert help_result.returncode == 0, (command, help_result.stderr)
        option_lines = [
            line
            for line in help_result.stdout.splitlines()
            if line.lstrip().startswith("--")
            and not line.lstrip().startswith(("--help", "--interactive", "--json", "--profile"))
        ]
        assert len(option_lines) <= 8, (command, option_lines)
        assert all("," in line and " -" in line for line in option_lines), (
            command,
            option_lines,
        )


def test_status_details_and_source_census_options_have_shortcuts(run: Run) -> None:
    result = run("savings", "status", "--help")
    assert result.returncode == 0
    assert "--details, -d" in result.stdout
    assert "--page, -n" in result.stdout
    assert "--include-text, -t" in result.stdout
    assert "--sources, -s" in result.stdout


def test_status_source_total_preserves_unknown_runtime_counts() -> None:
    from ssgrep.cli.commands.metrics_status_command import _source_total

    assert _source_total((2, 3)) == 5
    assert _source_total((2, None)) is None
    assert _source_total(()) == 0


def test_preview_reports_corrupt_checkpoint_without_echo_or_mutation(
    run: Run, receiver: LabReceiver
) -> None:
    from ssgrep.cli import exit_codes

    configure(receiver.url, logs=False)
    publish()
    checkpoint = run.data / "accounting" / PROFILE / "collector" / collector_store.CHECKPOINT
    checkpoint.write_bytes(b'{"PRIVATE_CHECKPOINT_CANARY":true')
    before = run.tree()

    result = run("savings", "export", "preview", DEST, "--json")

    assert result.returncode == exit_codes.INTERNAL_FAILURE
    assert document(result)["condition"] == "state_corrupt"
    assert "PRIVATE_CHECKPOINT_CANARY" not in result.stdout + result.stderr
    assert run.tree() == before and receiver.requests == []


def test_preview_treats_damaged_snapshot_as_no_snapshot_without_echo_or_mutation(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url, logs=False)
    publish()
    generation = next((run.data / "accounting" / PROFILE / "snapshots").glob("gen-*.json"))
    generation.write_bytes(b'{"PRIVATE_SNAPSHOT_CANARY":true')
    before = run.tree()

    result = run("savings", "export", "preview", DEST, "--json")

    assert result.returncode == 0
    assert document(result)["data"]["data"]["state"] == "no_snapshot"
    assert "PRIVATE_SNAPSHOT_CANARY" not in result.stdout + result.stderr
    assert run.tree() == before and receiver.requests == []


def test_preview_scrubs_untrusted_data_errors(monkeypatch: pytest.MonkeyPatch, capsys):
    from ssgrep.accounting.export import consent
    from ssgrep.cli import accounting_cli, exit_codes
    from ssgrep.cli.commands.metrics_export_command import _preview

    monkeypatch.setattr(accounting_cli, "is_json_mode", lambda: False)
    monkeypatch.setattr(
        consent, "build_preview", lambda *_args: (_ for _ in ()).throw(ValueError("PRIVATE-CANARY"))
    )
    request = consent.ExportRequest("preview", DEST, Signal.METRICS, PROFILE)

    with pytest.raises(SystemExit) as caught:
        _preview(request)

    message = capsys.readouterr().err
    assert caught.value.code == exit_codes.INTERNAL_FAILURE
    assert "current local data" in message
    assert "PRIVATE-CANARY" not in message


def test_status_census_block_reports_sources_behind_only_when_it_is_an_integer() -> None:
    from ssgrep.accounting.views.service import read_status
    from ssgrep.cli.metrics_render import render_status

    view = read_status("personal-local")
    behind = render_status(view, 80, source_census={"sources": 5, "sources_behind": 2}).plain
    assert "SOURCE CENSUS" in behind
    assert (
        "Sources behind" in behind and "2" in behind.split("Sources behind", 1)[1].splitlines()[0]
    )
    unknown = render_status(view, 80, source_census={"sources_behind": "2"}).plain
    assert "Sources" in unknown and "unknown" in unknown
    assert "Sources behind" not in unknown
