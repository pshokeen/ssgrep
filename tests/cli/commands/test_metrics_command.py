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
from tests.conftest import get_ssgrep_binary

BINARY = get_ssgrep_binary()
PROFILE = "personal-local"
DEST = "org-primary"
TOKEN = "sk-CANARY-SECRET-0123456789"
TOKEN_ENV = "SSGREP_ORG_TOKEN"


class Run:
    def __init__(self, home: Path, data: Path) -> None:
        self.home, self.data = home, data

    def environment(self, env: dict[str, str] | None = None, columns: int = 80) -> dict[str, str]:
        base = {k: v for k, v in os.environ.items() if k != TOKEN_ENV}
        return {
            **base,
            "HOME": str(self.home),
            "SSGREP_DATA_DIR": str(self.data),
            "XDG_RUNTIME_DIR": str(self.home / "runtime"),
            "XDG_CONFIG_HOME": str(self.home / "config"),
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

    def closed_pipe(
        self, *args: str, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        reader, writer = os.pipe()
        os.close(reader)
        try:
            return subprocess.run(
                [str(BINARY), *args],
                stdin=subprocess.DEVNULL,
                stdout=writer,
                stderr=subprocess.PIPE,
                text=True,
                env=self.environment(env),
                timeout=120,
                check=False,
            )
        finally:
            os.close(writer)

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


def export(*extra: str) -> tuple[str, ...]:
    return ("savings", "export", DEST, *extra)


def tokened() -> dict[str, str]:
    return {TOKEN_ENV: TOKEN}


@pytest.mark.parametrize(
    "args",
    [
        ("savings", "export", DEST, "--signal", "logs"),
        ("savings", "export", DEST, "--logs"),
        ("savings", "export", DEST, "--json"),
        ("savings", "export", DEST, "--interactive"),
        ("savings", "export", DEST, "--profile", "../etc"),
        *(
            ("savings", "export", mode, DEST)
            for mode in ("preview", "check", "enable", "disable", "send", "serve")
        ),
    ],
)
def test_removed_surface_is_rejected_without_writes(run: Run, args: tuple[str, ...]) -> None:
    result = run("--json", *args)
    assert result.returncode == 2
    assert document(result)["ok"] is False
    assert run.tree() == {}


@pytest.mark.parametrize("option", ["--from", "--to"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_remote_dates_are_rejected_before_writes_or_network(
    run: Run, receiver: LabReceiver, option: str, dry_run: bool
) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),))
    publish()
    before = run.tree()
    result = run(
        "--json", *export(option, "2026-10-01", *(("--dry-run",) if dry_run else ())), env=tokened()
    )
    assert result.returncode == 2
    assert document(result)["ok"] is False
    assert run.tree() == before and receiver.connections == 0


def test_destination_validation_and_local_remote_conflict(run: Run, receiver: LabReceiver) -> None:
    configure(receiver.url)
    before = run.tree()
    for args in (
        ("../x",),
        ("ghost",),
        (DEST, "--output", str(run.home / "report.json")),
    ):
        result = run("--json", "savings", "export", *args)
        assert result.returncode == 2
        assert document(result)["ok"] is False
    assert run.tree() == before and receiver.connections == 0
    assert not (run.home / "report.json").exists()


def test_ambiguous_destination_never_selects_an_organization(
    run: Run, receiver: LabReceiver
) -> None:
    from dataclasses import replace

    configure(receiver.url)
    current = policy.policy_status(PROFILE).profile
    other = replace(current.destinations[0], destination_id="org-secondary")
    policy.save_profile(replace(current, destinations=(*current.destinations, other)))
    before = run.tree()
    result = run("--json", "savings", "export")
    assert result.returncode == 2
    assert DEST in result.stdout and "org-secondary" in result.stdout
    assert run.tree() == before and receiver.connections == 0


def test_send_never_implies_consent_and_hint_is_runnable(run: Run, receiver: LabReceiver) -> None:
    import shlex

    configure(receiver.url)
    publish()
    before = run.tree()
    refused = run("--json", "savings", "export", env=tokened())
    assert refused.returncode == 1
    body = document(refused)
    assert body["condition"] == "export_not_enabled"
    assert body["command"] == f"ssgrep savings setup --export-to {DEST}"
    assert run.tree() == before and receiver.connections == 0
    hinted = run("--json", *shlex.split(body["command"])[1:])
    assert hinted.returncode == 2
    assert document(hinted)["condition"] == "confirmation_required"
    assert run.tree() == before and enabled_pairs() == ()


def test_setup_requires_confirmation_and_approves_every_permitted_signal(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url)
    publish()
    before = run.tree()
    refused = run("--json", "savings", "setup", "--export-to", DEST)
    assert refused.returncode == 2
    assert document(refused)["condition"] == "confirmation_required"
    assert run.tree() == before
    approved = run("--json", "savings", "setup", "--export-to", DEST, "--yes", env=tokened())
    assert approved.returncode == 0, approved.stdout + approved.stderr
    body = document(approved)["data"]
    assert body["state"] == "enabled"
    assert set(body["data"]["signals"]) == {"metrics", "logs"}
    assert body["data"]["changed"] is True and len(body["data"]["approvals"]) == 2
    assert set(enabled_pairs()) == {(DEST, "metrics"), (DEST, "logs")}
    for signal in Signal.METRICS, Signal.LOGS:
        assert approval_reason(PROFILE, DEST, signal) is None
    assert receiver.connections == 0
    assert not (run.data / "accounting" / PROFILE / "spool").exists()
    assert TOKEN not in approved.stdout + approved.stderr
    again = run("--json", "savings", "setup", "--export-to", DEST, "--yes")
    assert document(again)["data"]["data"]["changed"] is False


def test_setup_never_approves_policy_forbidden_logs(run: Run, receiver: LabReceiver) -> None:
    configure(receiver.url, logs=False)
    result = run("--json", "savings", "setup", "--export-to", DEST, "--yes")
    assert result.returncode == 0, result.stdout + result.stderr
    assert document(result)["data"]["data"]["signals"] == ["metrics"]
    assert enabled_pairs() == ((DEST, "metrics"),)
    assert approval_reason(PROFILE, DEST, Signal.LOGS) == "consent_reapproval_required"
    assert receiver.connections == 0


def test_setup_export_cannot_be_combined_with_policy_change(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url)
    before = run.tree()
    for control in ("--export-to", "--no-export"):
        args = (control, DEST) if control == "--export-to" else (control,)
        for change in (
            ("--policy", str(run.home / "missing-policy.json")),
            ("--evidence-cache",),
            ("--no-evidence-cache",),
            ("--cache-days", "7"),
        ):
            result = run("--json", "savings", "setup", *args, "--yes", *change)
            assert result.returncode == 2
    assert run.tree() == before and receiver.connections == 0


def test_interactive_cancel_changes_nothing_and_confirm_enables(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url)
    before = run.tree()
    code, _, _ = run.tty("savings", "setup", "--export-to", DEST, answer="n\n")
    assert code == 0 and run.tree() == before and enabled_pairs() == ()
    code, _, _ = run.tty("savings", "setup", "--export-to", DEST, answer="yes\n")
    assert code == 0 and set(enabled_pairs()) == {(DEST, "metrics"), (DEST, "logs")}
    assert receiver.connections == 0


def test_preview_is_read_only_without_consent_and_discloses_every_candidate(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url)
    publish()
    before = run.tree()
    result = run("--json", *export("--dry-run"), env=tokened())
    assert result.returncode == 0, result.stdout + result.stderr
    body = document(result)["data"]
    assert body["state"] == "preview"
    data = body["data"]
    assert data["network"] is False and data["state_changed"] is False
    assert set(data["signals"]) == {"metrics", "logs"}
    assert {p["signal"] for p in data["previews"]} == {"metrics", "logs"}
    for preview in data["previews"]:
        assert preview["approved"] is False
        assert isinstance(preview["consent_state"], str) and preview["consent_state"]
        assert preview["sending_paused"] is False
    metrics = next(p for p in data["previews"] if p["signal"] == "metrics")
    assert metrics["batch_count"] == len(metrics["batches"]) >= 1
    assert metrics["bytes"] == sum(b["bytes"] for b in metrics["batches"])
    assert run.tree() == before and receiver.connections == 0 and enabled_pairs() == ()
    assert TOKEN not in result.stdout + result.stderr
    export_consent.apply_enable(PROFILE, DEST, Signal.METRICS)
    before = run.tree()
    approved = run("--json", *export("--dry-run"), env=tokened())
    assert approved.returncode == 0, approved.stdout + approved.stderr
    candidates = document(approved)["data"]["data"]["previews"]
    assert {p["signal"]: p["approved"] for p in candidates} == {
        "metrics": True,
        "logs": False,
    }
    for preview in candidates:
        assert isinstance(preview["consent_state"], str) and preview["consent_state"]
        assert preview["sending_paused"] is False
    assert run.tree() == before and receiver.connections == 0 and receiver.requests == []


def test_previewed_bytes_equal_send_and_repeated_send_does_not_double_count(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),))
    publish()
    assert export_ids.load_key(PROFILE, create=True)
    preview = document(run("--json", *export("--dry-run")))["data"]["data"]
    sent = run("--json", "savings", "export", env=tokened())
    assert sent.returncode == 0, sent.stdout + sent.stderr
    body = document(sent)["data"]
    assert body["state"] == "sent" and body["data"]["signals"] == ["metrics"]
    assert len(body["data"]["cycles"]) == 1
    batches = next(p for p in preview["previews"] if p["signal"] == "metrics")["batches"]
    assert sorted(hashlib.sha256(r.body).hexdigest() for r in receiver.requests) == sorted(
        b["sha256"] for b in batches
    )
    assert all(r.path == "/v1/metrics" for r in receiver.requests)
    assert all(r.headers["authorization"] == f"Bearer {TOKEN}" for r in receiver.requests)
    count = len(receiver.requests)
    again = run("--json", *export(), env=tokened())
    assert again.returncode == 0 and len(receiver.requests) == count
    delivery = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    assert delivery is not None and delivery.last_acked_generation == 1


@pytest.mark.parametrize("status", [200, 401])
def test_send_attempts_all_approved_signals(run: Run, receiver: LabReceiver, status: int) -> None:
    from ssgrep.accounting.ledger.valuation_inputs import EvidenceBundle
    from tests.accounting.export.export_canary import build_ledger, persist_ledger
    from tests.accounting.ledger.valuation_builders import METHOD, candidate, debit

    configure(receiver.url, enabled=((DEST, "metrics"), (DEST, "logs")))
    publish()
    persist_ledger(
        PROFILE, build_ledger(EvidenceBundle((candidate("a"),), (debit("x", 640),)), METHOD)
    )
    receiver.queue(
        Respond(status, b'{"error":"PRIVATE-RECEIVER-CANARY"}' if status != 200 else b"{}")
    )
    result = run("--json", *export(), env=tokened())
    assert result.returncode == (0 if status == 200 else 1), result.stdout + result.stderr
    assert "PRIVATE-RECEIVER-CANARY" not in result.stdout + result.stderr
    if status != 200:
        assert document(result)["ok"] is False
        assert document(result).get("data", {}).get("state") != "sent"
        assert {r.path for r in receiver.requests} == {"/v1/metrics", "/v1/logs"}
        return
    body = document(result)["data"]["data"]
    assert set(body["signals"]) == {"metrics", "logs"} and len(body["cycles"]) == 2
    assert {r.path for r in receiver.requests} == {"/v1/metrics", "/v1/logs"}


@pytest.mark.parametrize("json_mode", [False, True], ids=["human", "json"])
def test_revoke_all_preserves_history_and_resume_never_reapproves(
    run: Run, receiver: LabReceiver, json_mode: bool
) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"), (DEST, "logs")))
    publish()
    assert run("--json", *export(), env=tokened()).returncode == 0
    retained = {
        k: v
        for k, v in run.tree().items()
        if "/delivery/" in k or "/spool/" in k or "/snapshots/" in k
    }
    count = len(receiver.requests)
    result = run(*(("--json",) if json_mode else ()), "savings", "setup", "--no-export")
    assert result.returncode == 0
    if json_mode:
        body = document(result)["data"]
        assert body["state"] == "disabled" and len(body["data"]["reports"]) == 2
    assert len(receiver.requests) == count
    assert any("/spool/" in name for name in retained)
    assert any("/snapshots/" in name for name in retained)
    assert enabled_pairs() == ()
    assert {k: v for k, v in run.tree().items() if k in retained} == retained
    assert run("savings", "pause").returncode == 0
    assert run("savings", "resume").returncode == 0
    assert enabled_pairs() == ()
    count = len(receiver.requests)
    assert run("--json", *export(), env=tokened()).returncode == 1
    assert len(receiver.requests) == count


def test_paused_send_refuses_but_preview_remains_read_only(run: Run, receiver: LabReceiver) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),))
    publish()
    assert run("savings", "pause").returncode == 0
    before = run.tree()
    result = run("--json", *export(), env=tokened())
    assert result.returncode == 1 and document(result)["condition"] == "export_paused"
    preview = run("--json", *export("--dry-run"))
    assert preview.returncode == 0 and document(preview)["data"]["state"] == "preview"
    candidates = document(preview)["data"]["data"]["previews"]
    assert candidates and all(p["sending_paused"] is True for p in candidates)
    assert {p["signal"]: p["approved"] for p in candidates} == {
        "metrics": True,
        "logs": False,
    }
    assert run.tree() == before and receiver.connections == 0


@pytest.mark.parametrize(
    "args",
    [
        ("savings", "export", DEST),
        ("savings", "export", DEST, "--dry-run"),
        ("savings", "setup", "--export-to", DEST, "--yes"),
        ("savings", "setup", "--no-export"),
    ],
)
def test_corrupt_policy_refuses_without_mutation(run: Run, receiver: LabReceiver, args) -> None:
    configure(receiver.url)
    policy.policy_path().write_bytes(b"{PRIVATE-POLICY-CANARY")
    before = run.tree()
    result = run("--json", *args, env=tokened())
    assert result.returncode == 1 and document(result)["condition"] == "policy_corrupt"
    assert "PRIVATE-POLICY-CANARY" not in result.stdout + result.stderr
    assert run.tree() == before and receiver.connections == 0


def test_export_has_exactly_five_leaf_options_and_serve_is_a_sibling(run: Run) -> None:
    import re

    result = run("savings", "export", "--help")
    assert result.returncode == 0
    options = re.findall(r"^[\s│┃|]*(-{2}[a-z][a-z-]*)\b", result.stdout, re.MULTILINE)
    assert set(options) == {"--output", "--from", "--to", "--profile", "--dry-run", "--help"}
    assert run("savings", "serve", "--help").returncode == 0
    assert run("savings", "export", "serve", "--port", "0").returncode == 2
    assert run("metrics", "--help").returncode != 0


def test_local_output_preserves_service_unknowns_and_privacy_without_mutation(
    run: Run, tmp_path: Path
) -> None:
    import stat

    from ssgrep.services.export_report import build_report
    from tests.accounting.collect import verify_scenes as vs
    from tests.accounting.views.test_ux_support import collected_scene

    collected_scene(tmp_path, pairs=(0,))
    expected = build_report(PROFILE)
    before = run.tree()
    output = run.home / "report.json"
    result = run("savings", "export", "--output", str(output), env=tokened())
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads(output.read_text())
    assert actual == expected
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert actual["metadata"]["profile"] == PROFILE
    assert actual["estimates"]["context_tokens"]["optimistic"]["net"]["value"] is None
    serialized = output.read_text()
    assert TOKEN not in serialized and vs.fact(0) not in serialized
    assert run.tree() == before
    json_output = run.home / "report-machine.json"
    saved = run("--json", "savings", "export", "--output", str(json_output), env=tokened())
    assert saved.returncode == 0, saved.stdout + saved.stderr
    assert document(saved)["data"]["state"] == "saved"
    assert json.loads(json_output.read_text()) == actual
    assert stat.S_IMODE(json_output.stat().st_mode) == 0o600
    assert TOKEN not in saved.stdout + saved.stderr + json_output.read_text()
    assert run.tree() == before
    stdout = run("savings", "export", "--output", "-")
    assert stdout.returncode == 0 and json.loads(stdout.stdout) == actual
    assert run.tree() == before
    machine_stdout = run("--json", "savings", "export", "--output", "-")
    assert machine_stdout.returncode == 0, machine_stdout.stderr
    assert json.loads(machine_stdout.stdout) == actual
    assert "ok" not in json.loads(machine_stdout.stdout)
    assert run.tree() == before


def test_local_range_and_dry_run_respect_explicit_output_only(run: Run, tmp_path: Path) -> None:
    from ssgrep.services.export_report import build_report
    from tests.accounting.views.test_ux_support import collected_scene

    collected_scene(tmp_path, pairs=(0,))
    before = run.tree()
    output = run.home / "report.json"
    args = (
        "savings",
        "export",
        "--output",
        str(output),
        "--from",
        "2026-10-01",
        "--to",
        "2026-11-01",
    )
    preview = run("--json", *args, "--dry-run")
    assert preview.returncode == 0, preview.stdout + preview.stderr
    assert not output.exists() and run.tree() == before
    result = run(*args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text()) == build_report(PROFILE, "2026-10-01", "2026-11-01")
    assert run.tree() == before
    prior = output.read_bytes()
    prior_stat = output.stat()
    human_preview = run(*args, "--dry-run")
    assert human_preview.returncode == 0, human_preview.stdout + human_preview.stderr
    assert json.loads(human_preview.stdout[human_preview.stdout.index("{") :]) == json.loads(prior)
    assert output.read_bytes() == prior
    assert output.stat().st_mtime_ns == prior_stat.st_mtime_ns
    assert output.stat().st_mode == prior_stat.st_mode
    assert run.tree() == before


@pytest.mark.parametrize("kind", ["symlink", "directory", "missing_parent", "permission"])
def test_local_output_path_refusals_are_usage_errors_and_preserve_prior_files(
    run: Run, tmp_path: Path, kind: str
) -> None:
    from tests.accounting.views.test_ux_support import collected_scene, fingerprint

    collected_scene(tmp_path, pairs=(0,))
    output_directory = tmp_path / "exports"
    output_directory.mkdir()
    prior = output_directory / "prior.json"
    prior.write_text('{"PRIVATE-OUTPUT-CANARY":true}\n')
    output = output_directory / "report.json"
    locked = None
    if kind == "symlink":
        output.symlink_to(prior)
    elif kind == "directory":
        output.mkdir()
    elif kind == "missing_parent":
        output = output_directory / "missing" / "report.json"
    else:
        if os.geteuid() == 0:
            pytest.skip("Directory permissions do not deny writes to root")
        locked = output_directory / "locked"
        locked.mkdir()
        output = locked / "report.json"
        output.write_bytes(prior.read_bytes())
        locked.chmod(0o500)
    before = run.tree()
    output_before = fingerprint(output_directory)
    try:
        result = run("--json", "savings", "export", "--output", str(output))
        assert result.returncode == 2, result.stdout + result.stderr
        body = document(result)
        assert body["ok"] is False and body["condition"] == "invalid_output"
        assert "PRIVATE-OUTPUT-CANARY" not in result.stdout + result.stderr
        assert "Traceback" not in result.stdout + result.stderr
        assert fingerprint(output_directory) == output_before
        assert run.tree() == before
    finally:
        if locked is not None:
            locked.chmod(0o700)


def test_local_export_unavailable_state_does_not_write_a_fake_empty_report(run: Run) -> None:
    output = run.home / "report.json"
    result = run("--json", "savings", "export", "--output", str(output))
    assert result.returncode == 1 and document(result)["ok"] is False
    assert not output.exists() and run.tree() == {}


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
        ("savings", "export"),
        ("savings", "serve"),
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

    result = run("--json", *export("--dry-run"))

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

    result = run("--json", *export("--dry-run"))

    assert result.returncode == 0
    assert document(result)["data"]["data"]["previews"][0]["state"] == "no_snapshot"
    assert "PRIVATE_SNAPSHOT_CANARY" not in result.stdout + result.stderr
    assert run.tree() == before and receiver.requests == []


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


@pytest.mark.parametrize("error_type", ["ValueError", "OSError"])
@pytest.mark.parametrize("json_mode", [False, True], ids=["human", "json"])
def test_unexpected_preview_failure_is_private_and_read_only(
    run: Run, receiver: LabReceiver, tmp_path: Path, error_type: str, json_mode: bool
) -> None:
    configure(receiver.url, logs=False)
    publish()
    before = run.tree()
    control = run("--json", *export("--dry-run"))
    assert control.returncode == 0, control.stdout + control.stderr
    assert document(control)["data"]["data"]["previews"][0]["batch_count"] > 0
    hook = tmp_path / "sitecustomize.py"
    hook.write_text(
        "from ssgrep.accounting.export import projection\n"
        "def unavailable(*args, **kwargs):\n"
        f"    raise {error_type}('PRIVATE-PREVIEW-FAILURE /secret/source sk-private')\n"
        "projection.preview_candidate = unavailable\n",
        encoding="utf-8",
    )
    result = run(
        *(("--json",) if json_mode else ()),
        *export("--dry-run"),
        env={"PYTHONPATH": str(tmp_path)},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    if json_mode:
        assert document(result)["condition"] == "preview_unavailable"
        assert document(result)["ok"] is False
    else:
        output = " ".join((result.stdout + result.stderr).split())
        assert "Cannot build the current candidate. Nothing was changed." in output
    for private in ("PRIVATE-PREVIEW-FAILURE", "/secret/source", "sk-private", "Traceback"):
        assert private not in result.stdout + result.stderr
    assert run.tree() == before and receiver.connections == 0


@pytest.mark.parametrize("json_mode", [False, True], ids=["human", "json"])
def test_receiver_rejection_retains_each_signal_outcome(
    run: Run, receiver: LabReceiver, json_mode: bool
) -> None:
    from ssgrep.accounting.ledger.valuation_inputs import EvidenceBundle
    from tests.accounting.export.export_canary import build_ledger, persist_ledger
    from tests.accounting.ledger.valuation_builders import METHOD, candidate, debit

    configure(receiver.url, enabled=((DEST, "metrics"), (DEST, "logs")))
    publish()
    persist_ledger(
        PROFILE, build_ledger(EvidenceBundle((candidate("a"),), (debit("x", 640),)), METHOD)
    )
    receiver.queue(Respond(401, b'{"error":"PRIVATE-REJECTION"}'), path="/v1/metrics")
    result = run(*(("--json",) if json_mode else ()), *export(), env=tokened())
    assert result.returncode == 1, result.stdout + result.stderr
    assert {request.path for request in receiver.requests} == {"/v1/metrics", "/v1/logs"}
    assert "PRIVATE-REJECTION" not in result.stdout + result.stderr
    assert TOKEN not in result.stdout + result.stderr
    metrics = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    logs = acks.read_delivery(PROFILE, DEST, Signal.LOGS)
    assert metrics is not None and metrics.last_acked_generation != 1
    assert logs is not None and logs.last_acked_generation == 1
    if json_mode:
        body = document(result)
        assert body["ok"] is False and body["data"]["state"] == "partial"
        cycles = {cycle["signal"]: cycle for cycle in body["data"]["data"]["cycles"]}
        assert set(cycles) == {"metrics", "logs"}
        assert cycles["metrics"]["acked"] == 0
        assert cycles["logs"]["acked"] > 0 and cycles["logs"]["state"] == "sent"
    else:
        output = " ".join((result.stdout + result.stderr).split())
        assert "metrics:" in output and "logs:" in output
        assert "acknowledged 0" in output
        accepted_logs = sum(request.path == "/v1/logs" for request in receiver.requests)
        assert f"acknowledged {accepted_logs}" in output
        assert "Acknowledged means collector acceptance" not in output


def test_json_send_failure_with_closed_stdout_is_quiet_and_still_fails(
    run: Run, receiver: LabReceiver
) -> None:
    configure(receiver.url, enabled=((DEST, "metrics"),), logs=False)
    publish()
    receiver.queue(Respond(401, b'{"error":"PRIVATE-CLOSED-PIPE"}'))
    result = run.closed_pipe("--json", *export(), env=tokened())
    assert result.returncode == 1
    assert result.stderr == ""
    assert len(receiver.requests) == 1 and receiver.requests[0].status == 401
    delivery = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    assert delivery is not None and delivery.last_acked_generation != 1


@pytest.mark.parametrize("json_mode", [False, True], ids=["human", "json"])
def test_local_export_closed_stdout_is_quiet_and_read_only(
    run: Run, tmp_path: Path, json_mode: bool
) -> None:
    from tests.accounting.views.test_ux_support import collected_scene

    collected_scene(tmp_path, pairs=(0,))
    before = run.tree()
    args = (*(("--json",) if json_mode else ()), "savings", "export", "--output", "-")
    control = run(*args)
    assert control.returncode == 0, control.stdout + control.stderr
    assert json.loads(control.stdout)["metadata"]["profile"] == PROFILE
    result = run.closed_pipe(*args)
    assert result.returncode == 1  # The requested report was not delivered to its consumer.
    assert result.stderr == ""
    assert run.tree() == before


@pytest.mark.parametrize("operation", ["write", "fsync", "replace"])
def test_local_disk_failure_preserves_target_and_removes_temporary_report(
    run: Run, tmp_path: Path, operation: str
) -> None:
    from tests.accounting.views.test_ux_support import collected_scene, fingerprint

    collected_scene(tmp_path, pairs=(0,))
    output = run.home / "report.json"
    control = run("--json", "savings", "export", "--output", str(output))
    assert control.returncode == 0, control.stdout + control.stderr
    assert json.loads(output.read_text())["metadata"]["profile"] == PROFILE
    output.write_text('{"PRIVATE-PRIOR-TARGET":true}\n')
    before = run.tree()
    home_before = fingerprint(run.home)
    injections = {
        "write": (
            "from ssgrep.accounting.privacy import storage\n"
            "def fail(fd, payload):\n"
            "    os.write(fd, payload[:17])\n"
            "    raise OSError(errno.ENOSPC, 'PRIVATE-DISK-FAILURE')\n"
            "storage.write_all = fail\n"
        ),
        "fsync": (
            "def fail(fd):\n    raise OSError(errno.EIO, 'PRIVATE-DISK-FAILURE')\nos.fsync = fail\n"
        ),
        "replace": (
            "def fail(*args, **kwargs):\n"
            "    raise OSError(errno.EIO, 'PRIVATE-DISK-FAILURE')\n"
            "os.replace = fail\n"
        ),
    }
    (tmp_path / "sitecustomize.py").write_text(
        "import errno, os\n" + injections[operation], encoding="utf-8"
    )
    result = run(
        "--json",
        "savings",
        "export",
        "--output",
        str(output),
        env={"PYTHONPATH": str(tmp_path)},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert document(result)["ok"] is False
    assert document(result)["condition"] == "export_unavailable"
    for private in ("PRIVATE-DISK-FAILURE", "PRIVATE-PRIOR-TARGET", "Traceback"):
        assert private not in result.stdout + result.stderr
    assert fingerprint(run.home) == home_before
    assert run.tree() == before


def test_pending_retry_is_a_failure_until_the_receiver_can_accept_the_same_payload(
    run: Run, receiver: LabReceiver, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timedelta

    from ssgrep.accounting.export import delivery

    configure(receiver.url, enabled=((DEST, "metrics"),), logs=False)
    publish()
    receiver.queue(Respond(429, b"PRIVATE-RETRY-RESPONSE", {"Retry-After": "60"}))
    first = run("--json", *export(), env=tokened())
    assert len(receiver.requests) == 1 and receiver.requests[0].status == 429
    held = acks.read_delivery(PROFILE, DEST, Signal.METRICS)
    assert held is not None and held.next_retry_utc is not None
    assert held.last_acked_generation is None and held.pending_batches > 0
    retry = run("--json", *export(), env=tokened())
    assert retry.returncode == 1 and document(retry)["condition"] == "retry_pending"
    assert len(receiver.requests) == 1
    assert "PRIVATE-RETRY-RESPONSE" not in first.stdout + first.stderr + retry.stdout + retry.stderr
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    due = datetime.fromisoformat(held.next_retry_utc.replace("Z", "+00:00")) + timedelta(seconds=1)
    accepted = delivery.run_destination(PROFILE, DEST, Signal.METRICS, now=due)
    assert accepted.acked == 1
    assert receiver.requests[1].body == receiver.requests[0].body
