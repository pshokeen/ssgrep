"""Cross-module canary: private text planted in snapshot + ledger never leaves the machine.

A real journey over real files and real loopback sockets: a policy is saved with
``policy.save_profile``, a canary-laden snapshot and ledger are published through the collector's
own writers, ``delivery.run_destination`` sends both signals to the lab receiver, and every byte
the receiver saw plus every file the export module wrote (spool, delivery docs, registry, scope
files) is searched for each canary shape. Forced failures (receiver echoes the canary in a 400 /
500 / garbage-200 / partial-success message) prove a raw response body is never persisted, logged
or surfaced. Credential secrets (env value, secret file contents) are searched for as well; the
single place they may appear is the ``Authorization`` header the receiver is meant to get.

Local read surfaces (``savings status``, ``savings``) show the user's OWN snapshot to the user and
legitimately echo ``warnings`` / ``collected_at`` / ``coverage_through`` (owned by an earlier
verified wave), so those three positions are excluded ONLY for those surfaces; the export wire,
spool and delivery state are held to the full standard below.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting.export import (
    acks,
    consent as export_consent,
    projection as export_build,
    registry as export_registry,
    spool as export_spool,
)
from ssgrep.accounting.export.ids import load_key
from ssgrep.accounting.ledger.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.privacy import policy
from ssgrep.accounting.privacy.policy import PolicyError
from ssgrep.accounting.views import service as service_status, snapshots
from tests.accounting.export.export_canary import SHAPES, laden_envelope, leaks, publish_laden
from tests.accounting.export.lab_receiver import LabReceiver, Partial, Respond
from tests.conftest import get_ssgrep_binary

PROFILE = "personal-local"
DEST = "acme-org"
ENV_SECRET = "ingest-ENVSECRET-5d1c9e77a3b04f68"
FILE_SECRET = "ingest-FILESECRET-91be44d07c2a6f35"
ENV_NAME = "ACME_INGEST_TOKEN"
BINARY = get_ssgrep_binary()
# Directories that legitimately hold LOCAL evidence (the source of the canaries).
LOCAL_EVIDENCE = {"snapshots", "collector", "journal", "receipts", "history"}
ALL_CANARIES = {**SHAPES, "env_secret": ENV_SECRET, "file_secret": FILE_SECRET}


def delivery() -> Any:
    """The delivery module; a missing module is a FAILURE here, never a skip."""
    from ssgrep.accounting.export import delivery as module

    return module


def save_policy(
    endpoint: str,
    *,
    credential_ref: str | None,
    logs: bool = True,
    enabled: tuple[Signal, ...] | None = None,
) -> ProfilePolicy:
    """Confirm current fields separately for each explicitly enabled journey signal."""
    dest = DestinationPolicy(
        DEST,
        endpoint,
        (Signal.METRICS, Signal.LOGS) if logs else (Signal.METRICS,),
        logs_permitted=logs,
        credential_ref=credential_ref,
    )
    selected = enabled if enabled is not None else dest.signals
    exports = tuple((DEST, signal.value) for signal in selected)
    current = policy.save_profile(
        ProfilePolicy(destinations=(dest,), enabled_exports=exports, collection_enabled=True)
    )
    for destination, signal in exports:
        export_consent.apply_enable(PROFILE, destination, Signal(signal))
    return current


def accounting_root() -> Path:
    return Path(os.environ["SSGREP_DATA_DIR"]) / "accounting"


def export_files() -> list[Path]:
    """Every file under the data dir EXCEPT local evidence directories of each profile."""
    root = Path(os.environ["SSGREP_DATA_DIR"])
    found = []
    for path in sorted(root.rglob("*")):
        parts = path.relative_to(root).parts
        if len(parts) >= 3 and parts[0] == "accounting" and parts[2] in LOCAL_EVIDENCE:
            continue
        found.append(path)
    return found


def assert_tree_clean(canaries: dict[str, str] | None = None) -> list[Path]:
    """No canary in any export-side file CONTENT or any path NAME; returns the files scanned."""
    canaries = canaries or ALL_CANARIES
    files = export_files()
    for path in files:
        data = path.read_bytes() if path.is_file() else b""
        for shape, canary in canaries.items():
            assert not leaks(canary, data), f"{shape} in {path}"
            assert not leaks(canary, str(path)), f"{shape} in path {path}"
    return files


def clean(canaries: dict[str, str], *blobs: object, what: str) -> None:
    for shape, canary in canaries.items():
        assert not leaks(canary, *blobs), f"{shape} leaked into {what}"


@pytest.fixture
def env_secret(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.setenv(ENV_NAME, ENV_SECRET)
    return {ENV_NAME: ENV_SECRET}


@pytest.fixture
def receiver() -> Any:
    with LabReceiver.start() as lab:
        yield lab


# ---- local status surfaces (3 positions excluded: see the module docstring) -------------------


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(BINARY), *args],
        input="",
        capture_output=True,
        text=True,
        env={**os.environ, "COLUMNS": "100", "NO_COLOR": "1"},
        timeout=120,
        check=False,
    )


def test_status_surfaces_hold_no_canary_except_the_three_documented_local_echoes(
    receiver: LabReceiver,
) -> None:
    save_policy(receiver.url, credential_ref=f"env:{ENV_NAME}")
    publish_laden(PROFILE, SHAPES, ledger=True)
    # Local status shows the user's OWN snapshot: ``warnings`` / ``collected_at`` /
    # ``coverage_through`` and the representative receipt id (a ``savings explain <id>`` next
    # action) are legitimately echoed, so those positions are made clean here; every other
    # planted position (coverage/windows/observations keys and values, manifest digest ...) stays.
    fields_ = {
        k: v
        for k, v in vars(laden_envelope(SHAPES, generation=2, warnings=False)).items()
        if k not in ("schema_version", "snapshot_id")
    }
    fields_["estimates"] = {"representative_receipt": "rcpt-7f3a91"}
    snapshots.publish(PROFILE, snapshots.build_envelope(**fields_))
    local = {k: v for k, v in ALL_CANARIES.items() if k != "receipt_id"}
    status = service_status.read_status(PROFILE)
    clean(local, repr(status), what="read_status")
    for args in (("savings", "status", "--json"), ("savings", "status")):
        result = run_cli(*args)
        assert result.returncode == 0, result.stderr
        clean(local, result.stdout, result.stderr, what=" ".join(args))
    assert "acme-org" in run_cli("savings", "status", "--json").stdout  # the surface is real


def test_the_export_wire_never_contains_warnings_collected_at_or_coverage_through(
    receiver: LabReceiver,
) -> None:
    """Even when local status may echo them, none of the three can reach a wire payload."""
    save_policy(receiver.url, credential_ref=None)
    envelope = publish_laden(PROFILE, SHAPES)
    assert SHAPES["error_body"] in envelope.warnings  # planted for real
    current = policy.load_profile_policy(PROFILE)
    dest = current.destinations[0]
    key = load_key(PROFILE, create=True)
    assert key is not None
    built = [
        export_build.build_metrics(PROFILE, dest, current, envelope, key),
        export_build.build_logs(PROFILE, dest, current, envelope, key),
    ]
    assert all(c.batches for c in built)
    for candidate in built:
        for batch in candidate.batches:
            clean(SHAPES, batch.body, what=f"{candidate.signal.value} wire")
        clean(SHAPES, candidate.scope, candidate.notes, what="scope/notes")
    hostile = replace(envelope, collected_at=SHAPES["secret"])
    projection = export_build.build_metrics(PROFILE, dest, current, hostile, key)
    assert projection.batches == () and projection.omitted == 1
    clean(SHAPES, projection.notes, repr(projection), what="malformed collected_at")
    with pytest.raises(ValueError) as raised:
        export_build.build_logs(PROFILE, dest, current, hostile, key)
    clean(SHAPES, str(raised.value), repr(raised.value), what="unix_nano exception")


def test_an_endpoint_with_embedded_credentials_is_refused_without_echoing_them() -> None:
    for endpoint in (
        f"http://user:{FILE_SECRET}@127.0.0.1:9/v1",
        f"http://127.0.0.1:9/v1?token={FILE_SECRET}",
    ):
        with pytest.raises(PolicyError) as raised:
            save_policy(endpoint, credential_ref=None)
        clean(ALL_CANARIES, str(raised.value), repr(raised.value), what="policy error")
    assert (
        not (accounting_root() / "policy.json").exists()
        or FILE_SECRET not in (accounting_root() / "policy.json").read_text()
    )


def test_acks_describe_is_closed_vocabulary_even_for_hostile_inputs() -> None:
    d = acks.Delivery(DEST, Signal.METRICS)
    for hostile in (
        lambda x: acks.retry_scheduled(x, "2026-10-13T12:00:00Z", SHAPES["error_body"]),
        lambda x: acks.blocked(x, SHAPES["secret"], "f" * 16),
        lambda x: acks.record_loss(x, SHAPES["fact"], 1, 1, "2026-10-13T12:00:00Z"),
        lambda x: acks.attempted(x, 1, SHAPES["path"]),
    ):
        with pytest.raises(ValueError) as raised:
            hostile(d)
        clean(SHAPES, str(raised.value), what="validator error")
    documented = acks.describe(
        acks.record_loss(d, "policy_contraction", 1, 3, "2026-10-13T12:00:00Z")
    )
    clean(ALL_CANARIES, json.dumps(documented), what="acks.describe")
    assert (
        documented["lost_records"] == 3 and documented["accepted_means"] == "collector_acceptance"
    )


# Delivery journeys (real receiver, real spool) use `ssgrep.accounting.export.delivery`.

NOW_FMT = "%Y-%m-%dT%H:%M:%SZ"
SIGNALS = (Signal.METRICS, Signal.LOGS)
ECHO = json.dumps(
    {"error": SHAPES["error_body"], "token": SHAPES["secret"], "path": SHAPES["path"]}
)
FAILURES: dict[str, Any] = {
    "400": Respond(400, ECHO.encode()),
    "401": Respond(401, ECHO.encode()),
    "403": Respond(403, ECHO.encode()),
    "429": Respond(429, ECHO.encode(), {"Retry-After": "7"}),
    "500": Respond(500, ECHO.encode()),
    "503": Respond(503, ECHO.encode(), {"Retry-After": "9"}),
    "200_garbage": Respond(200, ECHO.encode()),
    "200_html": Respond(200, b"<html>" + ECHO.encode() + b"</html>"),
    "200_warning": Respond(200, json.dumps({"partialSuccess": {"errorMessage": ECHO}}).encode()),
    "partial": Partial(2, "metrics", ECHO),
}


def scan_everything(
    results: list[Any],
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
    *,
    secrets: bool = True,
) -> None:
    """Results, delivery docs, files, logs and process output carry no canary."""
    canaries = ALL_CANARIES if secrets else SHAPES
    clean(canaries, *[repr(r) for r in results], what="cycle results")
    clean(canaries, caplog.text, *[r.getMessage() for r in caplog.records], what="logging")
    seen = capfd.readouterr()
    clean(canaries, seen.out, seen.err, what="process stdout/stderr")
    assert_tree_clean(canaries)
    for item in acks.list_deliveries(PROFILE):
        clean(canaries, json.dumps(acks.describe(item)), what="acks.describe")
    clean(canaries, repr(service_status.read_status(PROFILE).destinations), what="status")


def wire(lab: LabReceiver) -> list[Any]:
    return list(lab.requests)


@pytest.fixture
def journey(
    receiver: LabReceiver,
    env_secret: dict[str, str],
    caplog: pytest.LogCaptureFixture,
) -> LabReceiver:
    caplog.set_level(logging.DEBUG)
    save_policy(receiver.url, credential_ref=f"env:{ENV_NAME}")
    publish_laden(PROFILE, SHAPES)
    return receiver


def test_metrics_consent_does_not_authorize_logs_and_later_logs_match_preview(
    receiver: LabReceiver,
    env_secret: dict[str, str],
) -> None:
    save_policy(receiver.url, credential_ref=f"env:{ENV_NAME}", enabled=(Signal.METRICS,))
    publish_laden(PROFILE, SHAPES)
    module = delivery()
    metrics = module.run_destination(PROFILE, DEST, Signal.METRICS)
    assert metrics.acked > 0 and {r.path for r in receiver.requests} == {"/v1/metrics"}
    blocked = module.run_destination(PROFILE, DEST, Signal.LOGS)
    assert blocked.state == "not_enabled" and blocked.sent == 0
    assert export_registry.load_registry(PROFILE, DEST).entries == {}
    assert export_spool.list_batches(PROFILE, DEST, Signal.LOGS) == []
    before = list(receiver.requests)
    preview = export_build.preview_candidate(PROFILE, DEST, Signal.LOGS)
    assert preview.candidate is not None and preview.candidate.batches
    preview_candidate = preview.candidate
    assert list(receiver.requests) == before
    assert export_consent.apply_enable(PROFILE, DEST, Signal.LOGS) is True
    assert list(receiver.requests) == before
    result = module.run_destination(PROFILE, DEST, Signal.LOGS)
    assert result.acked > 0
    received = [r.body for r in receiver.requests if r.path == "/v1/logs"]
    assert received == [batch.body for batch in preview_candidate.batches]
    for body in received:
        clean(ALL_CANARIES, body, what="separately approved logs")


def test_both_signals_are_delivered_and_no_canary_reaches_the_receiver_or_the_disk(
    journey: LabReceiver, caplog: pytest.LogCaptureFixture, capfd: pytest.CaptureFixture[str]
) -> None:
    module = delivery()
    results = [module.run_destination(PROFILE, DEST, signal) for signal in SIGNALS]
    assert [r.state for r in results] == ["sent", "sent"], results
    assert all(r.acked >= 1 and r.partial == 0 and r.quarantined == 0 for r in results)
    requests = wire(journey)
    assert {r.path for r in requests} == {"/v1/metrics", "/v1/logs"}
    assert all(r.status == 200 for r in requests)
    for request in requests:
        for shape, canary in SHAPES.items():
            assert not leaks(canary, request.body, request.path), (shape, request.path)
        for name, value in request.headers.items():
            assert not leaks(FILE_SECRET, value), name
            for shape, canary in SHAPES.items():
                assert not leaks(canary, value), (shape, name)
            if ENV_SECRET in value:  # the one place the credential may travel: its own header
                assert name == "authorization" and value == f"Bearer {ENV_SECRET}"
    assert all(r.headers["authorization"] == f"Bearer {ENV_SECRET}" for r in requests)
    assert all(parse_ok(r.body) for r in requests)
    state = {s: acks.read_delivery(PROFILE, DEST, s) for s in SIGNALS}
    assert all(d is not None and d.last_acked_generation == 1 for d in state.values())
    scan_everything(results, caplog, capfd)
    files = assert_tree_clean()
    assert any("spool" in f.parts for f in files) and any("delivery" in f.parts for f in files)
    assert any(f.name == "records.json" for f in files)  # the registry was written and is clean


def parse_ok(body: bytes) -> bool:
    from tests.accounting.export.export_canary import parse_strict

    return bool(parse_strict(body))


def test_preview_bytes_equal_the_spooled_and_received_bytes(
    journey: LabReceiver,
) -> None:
    module = delivery()
    key = load_key(PROFILE, create=True)  # the real send path mints this; a preview never does
    assert key is not None
    previews = {s: export_build.preview_candidate(PROFILE, DEST, s) for s in SIGNALS}
    before = sorted(p for p in accounting_root().rglob("*"))
    again = {s: export_build.preview_candidate(PROFILE, DEST, s) for s in SIGNALS}
    assert sorted(p for p in accounting_root().rglob("*")) == before  # a preview writes nothing
    for signal in SIGNALS:
        assert previews[signal].state == "ok"
        first_candidate = previews[signal].candidate
        second_candidate = again[signal].candidate
        assert first_candidate is not None and second_candidate is not None
        assert [b.body for b in first_candidate.batches] == [
            b.body for b in second_candidate.batches
        ]
    for signal in SIGNALS:
        result = module.run_destination(PROFILE, DEST, signal)
        assert result.state == "sent"
    previewed = {}
    for signal in SIGNALS:
        candidate = previews[signal].candidate
        assert candidate is not None
        previewed[signal] = [batch.body for batch in candidate.batches]
    for signal in SIGNALS:
        spooled = [
            export_spool.read_body(PROFILE, DEST, signal, meta.batch_id)
            for meta in export_spool.list_batches(PROFILE, DEST, signal)
        ]
        received = [r.body for r in journey.requests if r.path == f"/v1/{signal.value}"]
        assert previewed[signal] == spooled == received and received
    tree = sorted(p for p in accounting_root().rglob("*"))
    (accounting_root() / PROFILE / "snapshots" / "current.json").unlink()
    absent = export_build.preview_candidate(PROFILE, DEST, Signal.METRICS)
    assert (absent.state, absent.candidate, absent.envelope) == ("no_snapshot", None, None)
    assert sorted(p for p in accounting_root().rglob("*")) == [
        p for p in tree if p.name != "current.json"
    ]


def test_the_cli_preview_is_clean_and_equals_what_the_send_puts_on_the_wire(
    journey: LabReceiver,
) -> None:
    assert load_key(PROFILE, create=True) is not None
    result = run_cli("--json", "savings", "export", DEST, "--dry-run")
    assert result.returncode == 0, result.stderr
    clean(ALL_CANARIES, result.stdout, result.stderr, what="preview")
    document = json.loads(result.stdout)["data"]["data"]
    shown = {
        Signal(item["signal"]): [hash_of(batch["payload"]) for batch in item["batches"]]
        for item in document["previews"]
    }
    assert set(shown) == set(SIGNALS)
    human = run_cli("savings", "export", DEST, "--dry-run")
    assert human.returncode == 0, human.stderr
    clean(ALL_CANARIES, human.stdout, human.stderr, what="human preview")
    sent = run_cli("savings", "export", DEST)
    assert sent.returncode == 0, sent.stderr
    clean(ALL_CANARIES, sent.stdout, sent.stderr, what="send")
    for signal in SIGNALS:
        received = [
            hash_of(json.loads(r.body)) for r in journey.requests if r.path == f"/v1/{signal.value}"
        ]
        assert received == shown[signal] and received


def hash_of(document: Any) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize("failure", sorted(FAILURES))
def test_a_receiver_echoing_the_canary_is_never_persisted_logged_or_surfaced(
    failure: str,
    journey: LabReceiver,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    module = delivery()
    journey.queue(FAILURES[failure])
    results = [module.run_destination(PROFILE, DEST, Signal.METRICS)]
    results.append(module.run_destination(PROFILE, DEST, Signal.METRICS))  # a retry cycle too
    results += module.run_export(PROFILE)
    assert any(r.state != "sent" or r.warnings or r.partial or r.quarantined for r in results)
    scan_everything(results, caplog, capfd)
    echoed = [r for r in journey.requests if r.status is not None and r.status != 200]
    assert echoed or failure.startswith(("200", "partial"))  # the failure was really delivered
    cli = run_cli("--json", "savings", "export", DEST)
    assert cli.returncode in (0, 1), cli.stderr
    assert "ok" in json.loads(cli.stdout)  # a real export attempt was parsed
    clean(ALL_CANARIES, cli.stdout, cli.stderr, what=f"cli after {failure}")


def test_a_hostile_collected_at_never_surfaces_through_the_delivery_cycle(
    journey: LabReceiver,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    module = delivery()
    publish_laden(PROFILE, SHAPES, generation=2, ledger=False, collected_at=SHAPES["secret"])
    results = [module.run_destination(PROFILE, DEST, s) for s in SIGNALS]
    results += module.run_export(PROFILE)
    scan_everything(results, caplog, capfd)
    for request in journey.requests:
        clean(SHAPES, request.body, what="wire after hostile collected_at")


def test_a_credential_file_secret_is_sent_only_in_its_header(
    receiver: LabReceiver,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(logging.DEBUG)
    secret_file = tmp_path / "outside-data" / "ingest.token"
    secret_file.parent.mkdir()
    secret_file.write_text(FILE_SECRET + "\n")
    secret_file.chmod(0o600)
    save_policy(receiver.url, credential_ref=f"file:{secret_file}")
    publish_laden(PROFILE, SHAPES)
    module = delivery()
    results = [module.run_destination(PROFILE, DEST, s) for s in SIGNALS]
    assert [r.state for r in results] == ["sent", "sent"]
    assert {r.headers["authorization"] for r in receiver.requests} == {f"Bearer {FILE_SECRET}"}
    for request in receiver.requests:
        assert not leaks(FILE_SECRET, request.body, request.path)
    scan_everything(results, caplog, capfd)
    secret_file.chmod(0o644)  # an unsafe credential file is refused without printing it
    receiver.queue()  # nothing queued: any request would still be recorded
    before = len(receiver.requests)
    publish_laden(PROFILE, SHAPES, generation=2, ledger=False)
    results = [module.run_destination(PROFILE, DEST, Signal.METRICS)]
    assert len(receiver.requests) == before  # no network without a usable credential
    scan_everything(results, caplog, capfd)


def test_an_env_secret_that_is_missing_or_malformed_is_never_echoed(
    receiver: LabReceiver,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    caplog.set_level(logging.DEBUG)
    save_policy(receiver.url, credential_ref=f"env:{ENV_NAME}")
    publish_laden(PROFILE, SHAPES)
    module = delivery()
    monkeypatch.setenv(ENV_NAME, f"{ENV_SECRET} {SHAPES['secret']}")  # whitespace: malformed
    results = [module.run_destination(PROFILE, DEST, Signal.METRICS)]
    assert receiver.requests == []
    clean(
        {**ALL_CANARIES, "joined": f"{ENV_SECRET} {SHAPES['secret']}"},
        *[repr(r) for r in results],
        caplog.text,
        what="malformed credential",
    )
    scan_everything(results, caplog, capfd)
    cli = run_cli("--json", "savings", "export", DEST)
    assert cli.returncode in (0, 1), cli.stderr
    assert "ok" in json.loads(cli.stdout)  # a real export attempt was parsed
    clean(ALL_CANARIES, cli.stdout, cli.stderr, what="cli with malformed credential")


def test_status_and_the_registry_after_a_journey_stay_closed_vocabulary(
    journey: LabReceiver,
) -> None:
    module = delivery()
    for signal in SIGNALS:
        module.run_destination(PROFILE, DEST, signal)
    registry = export_registry.load_registry(PROFILE, DEST)
    assert registry.entries
    for entry in registry.entries.values():
        clean(SHAPES, repr(entry), what="registry entry")
        assert len(entry.root_key) == 16 and entry.root_key == entry.root_key.lower()
    for meta in export_spool.list_batches(PROFILE, DEST, Signal.LOGS):
        clean(SHAPES, json.dumps(meta.scope, default=str), meta.warning or "", what="batch meta")
