"""``ssgrep metrics export`` through the real installed binary in a subprocess.

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

from ssgrep.accounting import acks, export_ids, policy, snapshots
from ssgrep.accounting.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from tests.accounting.http_builders import make_snapshot
from tests.accounting.lab_receiver import LabReceiver, Respond

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
    policy.save_profile(
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


def publish(generation: int = 1) -> None:
    snapshots.publish(PROFILE, make_snapshot(generation))


def enabled_pairs() -> tuple[tuple[str, str], ...]:
    return policy.policy_status(PROFILE).profile.enabled_exports


def export(*extra: str, signal: str | None = None) -> tuple[str, ...]:
    tail = ("--signal", signal) if signal else ()
    return ("metrics", "export", "--destination", DEST, *tail, *extra)


def tokened() -> dict[str, str]:
    return {TOKEN_ENV: TOKEN}


# ---- argument validation: exit 2 before any side effect ---------------------------------------


@pytest.mark.parametrize(
    ("args", "condition"),
    [
        (("--preview", "--check"), "invalid_mode"),
        (("--preview", "--enable"), "invalid_mode"),
        (("--check", "--disable"), "invalid_mode"),
        (("--enable", "--disable", "--yes"), "invalid_mode"),
        (("--signal", "traces"), "invalid_signal"),
        (("--profile", "../etc"), "invalid_profile"),
    ],
)
def test_invalid_arguments_exit_2_with_an_error_document_and_create_nothing(
    run: Run, args: tuple[str, ...], condition: str
) -> None:
    for json_flag in ((), ("--json",)):
        result = run("metrics", "export", "--destination", DEST, *args, *json_flag)
        assert result.returncode == 2, result.stdout + result.stderr
        if json_flag:
            body = document(result)
            assert body["ok"] is False and body["condition"] == condition
        else:
            assert result.stdout == "" and result.stderr
    assert run.names() == []


def test_destination_is_required_and_must_be_a_bounded_id(run: Run) -> None:
    missing = run("metrics", "export", "--json")
    assert missing.returncode == 2 and document(missing)["condition"] == "destination_required"
    bad = run("metrics", "export", "--destination", "../x", "--json")
    assert bad.returncode == 2 and document(bad)["condition"] == "invalid_destination"
    assert run.names() == []


def test_unknown_destination_and_unpermitted_signal_exit_2_and_change_nothing(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url, logs=False)
    before = run.tree()
    for args, condition in (
        (
            ("metrics", "export", "--destination", "ghost", "--enable", "--yes"),
            "unknown_destination",
        ),
        (export("--enable", "--yes", signal="logs"), "signal_not_permitted"),
        (("metrics", "export", "--destination", "ghost", "--preview"), "unknown_destination"),
        (("metrics", "export", "--destination", "ghost", "--disable"), "unknown_destination"),
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
    upper = result.stdout.upper()
    for needed in (
        "WILL AUTHORIZE",
        "FIELDS THAT LEAVE THIS MACHINE",
        "APPROVED ROOTS",
        "RETENTION",
        "READ BEFORE CONFIRMING",
    ):
        assert needed in upper, needed
    assert DEST in result.stdout and f"127.0.0.1:{receiver.port}" in result.stdout
    assert f"env:{TOKEN_ENV}" in result.stdout and "metrics-v1" in result.stdout
    assert "Nothing was sent" in result.stdout
    flat = " ".join(result.stdout.split()).lower()
    for claim in ("not a connectivity", "collector accepted", "non-reversible"):
        assert claim in flat, claim
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
    assert shown["credential_ref"] == f"env:{TOKEN_ENV}" and shown["fields_version"] == "logs-v1"
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
    assert run("metrics", "pause").returncode == 0
    assert run("metrics", "resume").returncode == 0
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
    assert body["policy_version"] == 1 and body["fields_version"] == "metrics-v1"
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
    assert run("metrics", "pause").returncode == 0
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
    assert run("metrics", "pause").returncode == 0
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
    assert run("metrics", "pause").returncode == 0
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
    assert "--enable" in body["command"] and f"--destination {DEST}" in body["command"]
    assert receiver.requests == [] and run.tree() == before


# ---- policy damage ----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["--preview", "--check", "--enable", "--disable", None])
def test_corrupt_policy_is_policy_corrupt_exit_1_for_every_mode_and_changes_nothing(
    run: Run, receiver: LabReceiver, mode: str | None
) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),))
    policy.policy_path().write_bytes(b"{garbage")
    before = run.tree()
    args = export(*([mode] if mode else []), "--yes", "--json")
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
