"""Shared rig for the delivery tests: real policy file, real snapshots, real ledger, real sockets.

Nothing here mocks the delivery code. Policy is saved through ``policy.save_profile``, snapshots go
through ``snapshots.publish``, the ledger state through the collector's own writers, and every
request travels over a loopback socket to ``lab_receiver``. ``Deps.environ`` carries the ingest
credential so the test process environment is never mutated.

``python -m tests.accounting.delivery_fixtures <mode> <args>`` is the subprocess driver for the
tests that need a REAL second process: ``crash`` dies with ``os._exit`` right after the POST
returned (before any acknowledgment is persisted), ``run`` runs one cycle and prints its verdict,
and ``ready`` waits for a stdin command after announcing that process startup is complete.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ssgrep.accounting import (
    acks,
    collector_state,
    collector_store,
    export_http,
    export_records,
    export_spool,
    policy,
    snapshots,
)
from ssgrep.accounting.contracts import TaskId
from ssgrep.accounting.delivery import CycleResult, Deps, run_destination
from ssgrep.accounting.export_policy import with_enabled
from ssgrep.accounting.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.valuation_inputs import EvidenceBundle
from tests.accounting.http_builders import make_snapshot
from tests.accounting.lab_receiver import LabReceiver, _Handler, _Server
from tests.accounting.valuation_builders import METHOD, atom, candidate, debit

PROFILE = "personal-local"
DEST = "org-primary"
M, L = Signal.METRICS, Signal.LOGS
SECRET = "sk-CANARY-ingest-9f2c71d04b"
ENV_NAME = "SSGREP_TEST_INGEST"
ENV = {ENV_NAME: SECRET}
CLOCK0 = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)
CRASH_CODE = 9
LOG_DAY = "2026-11-01"
SEAL_TODAY = "2026-11-02"


def at(seconds: int = 0) -> datetime:
    """A deterministic cycle clock: ``CLOCK0`` + ``seconds``."""
    return CLOCK0 + timedelta(seconds=seconds)


def stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def data_root() -> Path:
    return Path(os.environ["SSGREP_DATA_DIR"])


def profile_root() -> Path:
    return data_root() / "accounting" / PROFILE


def deps(**overrides: Any) -> Deps:
    """Production defaults + the injected credential environment and a deterministic jitter."""
    options: dict[str, Any] = {
        "environ": ENV,
        "rand": lambda: 0.0,
        "connect_timeout": 2.0,
        "read_timeout": 2.0,
    }
    options.update(overrides)
    return Deps(**options)


# --- policy --------------------------------------------------------------------------------


def configure(
    endpoint: str,
    *,
    dest: str = DEST,
    signals: tuple[Signal, ...] = (M,),
    logs_permitted: bool = False,
    enabled: tuple[Signal, ...] = (M,),
    credential_ref: str | None = f"env:{ENV_NAME}",
    proxy: str | None = None,
    roots: tuple[str, ...] = (),
    policy_version: int = 1,
) -> ProfilePolicy:
    """Save a personal-local profile with one destination and the given signals enabled."""
    destination = DestinationPolicy(dest, endpoint, signals, logs_permitted, credential_ref, proxy)
    return policy.save_profile(
        ProfilePolicy(
            roots=roots,
            destinations=(destination,),
            enabled_exports=tuple(sorted((dest, s.value) for s in enabled)),
            collection_enabled=True,
            policy_version=policy_version,
        )
    )


def amend(**changes: Any) -> ProfilePolicy:
    """Re-save the current profile with ``changes`` (narrowing/widening is the caller's call)."""
    return policy.save_profile(replace(policy.load_profile_policy(PROFILE), **changes))


def amend_destination(**changes: Any) -> ProfilePolicy:
    current = policy.load_profile_policy(PROFILE)
    first = replace(current.destinations[0], **changes)
    return policy.save_profile(replace(current, destinations=(first, *current.destinations[1:])))


def set_enabled(signal: Signal, on: bool, dest: str = DEST) -> ProfilePolicy:
    current = policy.load_profile_policy(PROFILE)
    return policy.save_profile(with_enabled(current, dest, signal, on))


def plant_policy(change: Callable[[dict[str, Any]], None]) -> None:
    """Edit the persisted ``policy.json`` in place (states ``save_profile`` refuses to write)."""
    path = policy.policy_path()
    document = json.loads(path.read_text())
    change(document["profiles"][PROFILE])
    path.write_text(json.dumps(document))


class SpyEnviron(Mapping[str, str]):
    """An environment mapping that records every key looked up (credential read detector)."""

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = dict(values)
        self.looked_up: list[str] = []

    def __getitem__(self, key: str) -> str:
        self.looked_up.append(key)
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def offline_post(*args: Any, **kwargs: Any) -> export_http.Outcome:
    """A transport that fails like a dead network WITHOUT touching any socket."""
    return export_http.Outcome("network_error", category="network")


# --- snapshots, ledger ---------------------------------------------------------------------


def publish(
    generation: int,
    *,
    collected_at: str | None = None,
    policy_version: int | None = None,
    **kwargs: Any,
) -> str:
    """Publish a complete snapshot (content differs per generation); returns its snapshot id.

    ``collected_at`` defaults to a time after the scope was first observed by the tests'
    deterministic clock; ``policy_version`` defaults to the saved profile's current version.
    """
    when = collected_at or f"2026-10-13T11:{generation % 60:02d}:00Z"
    version = policy_version or policy.load_profile_policy(PROFILE).policy_version
    base = make_snapshot(generation, extra_searches=generation, collected_at=when, **kwargs)
    envelope = snapshots.build_envelope(
        generation=generation,
        counter_epoch=base.counter_epoch,
        methodology_id=base.methodology_id,
        policy_version=version,
        collected_at=when,
        coverage_through=when,
        source_manifest_digest=base.source_manifest_digest,
        windows=base.windows,
        coverage=base.coverage,
        observations=base.observations,
        estimates=base.estimates,
        warnings=base.warnings,
    )
    snapshots.publish(PROFILE, envelope)
    return envelope.snapshot_id


def commit_ledger(records: int, *, extra_for: tuple[int, ...] = (), generation: int = 1) -> None:
    """Persist a real collector ledger with ``records`` task contributions (one record each)."""
    from ssgrep.accounting.collector_ledger import VersionedLedger

    ledger = VersionedLedger(METHOD)
    for number in range(1, records + 1):
        task = TaskId(f"root-{number}", "turn-1", "main")
        cands = [candidate(f"c{number}", task=task)]
        if number in extra_for:
            cands.append(
                candidate(f"x{number}", task=task, immediate=(atom(f"s{number}", 200, 0, "ev-2"),))
            )
        ledger.reconcile(
            {
                f"root-{number}": EvidenceBundle(
                    tuple(cands), (debit(f"d{number}", 640, task=task),)
                )
            },
            today=SEAL_TODAY,
        )
    blobs: dict[str, str] = {}
    scope = policy.collection_scope(policy.load_profile_policy(PROFILE))
    state = collector_state.State(
        ledger=ledger.state(), superseded=dict(ledger.superseded), collection_scope=scope
    )
    collector_state.save_state(PROFILE, state, METHOD.hash, blobs)
    checkpoint = collector_store.Checkpoint(PROFILE, generation=generation, methodology=METHOD.hash)
    checkpoint.blobs = blobs
    checkpoint.collection_scope = scope
    collector_store.commit_checkpoint(PROFILE, checkpoint)


def split_small(monkeypatch: Any, max_bytes: int = 4000) -> None:
    """Make the REAL splitters chunk at ``max_bytes`` so a small snapshot yields many batches."""
    from ssgrep.accounting import otlp

    real_metrics, real_logs = otlp.split_metrics, otlp.split_logs
    monkeypatch.setattr(
        otlp, "split_metrics", lambda r, p, **kw: real_metrics(r, p, max_bytes=max_bytes)
    )
    monkeypatch.setattr(
        otlp, "split_logs", lambda r, rec, **kw: real_logs(r, rec, max_bytes=max_bytes)
    )


# --- state readers -------------------------------------------------------------------------


def delivery(signal: Signal = M, dest: str = DEST) -> acks.Delivery | None:
    return acks.read_delivery(PROFILE, dest, signal)


def state(signal: Signal = M, dest: str = DEST) -> acks.Delivery:
    """The delivery document, which the test asserts exists."""
    found = acks.read_delivery(PROFILE, dest, signal)
    assert found is not None, "no delivery document was written"
    return found


def metas(signal: Signal = M, dest: str = DEST) -> list[export_spool.Meta]:
    return export_spool.list_batches(PROFILE, dest, signal)


def body_of(meta: export_spool.Meta) -> bytes:
    return export_spool.read_body(PROFILE, meta.destination_id, meta.signal, meta.batch_id)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def registry(dest: str = DEST) -> Any:
    return export_records.load_registry(PROFILE, dest)


def tree(root: Path | None = None) -> dict[str, bytes | None]:
    """Every entry under ``root`` (default: the data dir) mapped to its bytes (None: a dir)."""
    base = root or data_root()
    if not base.exists():
        return {}
    return {
        str(p.relative_to(base)): (p.read_bytes() if p.is_file() else None)
        for p in sorted(base.rglob("*"))
    }


def everything_on_disk() -> str:
    """The text of every file under the data dir the EXPORT side wrote (for leak greps).

    ``policy.json`` is excluded (operator configuration that legitimately names the credential
    REFERENCE) and so is ``snapshots/`` (collector-owned input: a hostile snapshot's own text is
    the thing the export side must not copy elsewhere).
    """
    chunks = []
    for path in sorted(data_root().rglob("*")):
        if path.is_file() and path.name != "policy.json" and "snapshots" not in path.parts:
            chunks.append(path.read_bytes().decode("utf-8", "replace"))
    return "\n".join(chunks)


def cycle(signal: Signal = M, *, seconds: int = 0, **kwargs: Any) -> CycleResult:
    """One ``run_destination`` cycle at the deterministic clock."""
    chosen: Deps = kwargs.pop("deps", None) or deps()
    return run_destination(PROFILE, DEST, signal, now=at(seconds), deps=chosen, **kwargs)


def drain(signal: Signal = M, *, start: int = 0, step: int = 400, limit: int = 12) -> list:
    """Cycle with an advancing clock until nothing is queued (or ``limit`` cycles ran)."""
    results = []
    for index in range(limit):
        results.append(cycle(signal, seconds=start + index * step))
        if not any(m.state in ("queued", "in_flight") for m in metas(signal)):
            break
    return results


# --- a receiver that comes up late on a chosen port ----------------------------------------


def free_port() -> int:
    """A loopback port nothing listens on right now (connections to it are refused)."""
    probe = LabReceiver.start()
    port = probe.port
    probe.stop()
    return port


def receiver_on(port: int) -> LabReceiver:
    """A started lab receiver bound to ``port`` (lab_receiver itself only binds an ephemeral one).

    The endpoint is part of a queued batch's authorized scope, so an offline backlog can only be
    delivered by the SAME endpoint coming back; changing it would (correctly) invalidate it.
    """
    receiver = LabReceiver()
    receiver._server.server_close()
    receiver._server = _Server(("127.0.0.1", port), _Handler)
    receiver._server.lab = receiver
    receiver._thread = threading.Thread(
        target=receiver._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    receiver._thread.start()
    return receiver


# --- subprocess driver ---------------------------------------------------------------------


@contextmanager
def ready_driver(
    signal: str, seconds: str = "0", *, timeout: float = 60.0
) -> Iterator[subprocess.Popen[str]]:
    """Start a REAL process, ready to run a cycle when stdin receives ``run\\n``."""
    with subprocess.Popen(  # noqa: S603 - fixed argv, test-owned module
        [sys.executable, "-m", "tests.accounting.delivery_fixtures", "ready", signal, seconds],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    ) as process:
        try:
            assert process.stdout is not None
            readable, _, _ = select.select([process.stdout], [], [], timeout)
            if not readable:
                raise TimeoutError("delivery driver startup timed out")
            if process.stdout.readline() != "ready\n":
                raise RuntimeError("delivery driver exited before becoming ready")
            yield process
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()


def run_driver(mode: str, *args: str, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
    """Run the driver below as a REAL process with this test's isolated environment."""
    return subprocess.run(  # noqa: S603 - fixed argv, test-owned module
        [sys.executable, "-m", "tests.accounting.delivery_fixtures", mode, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=os.environ.copy(),
    )


def _crash_post(real: Callable[..., export_http.Outcome]) -> Callable[..., export_http.Outcome]:
    def post(*args: Any, **kwargs: Any) -> export_http.Outcome:
        real(*args, **kwargs)  # the receiver really got (and answered) the request
        os._exit(CRASH_CODE)  # ... and we die before ANY acknowledgment is persisted

    return post


def _driver(argv: list[str]) -> None:
    mode, signal = argv[0], Signal(argv[1])
    seconds = int(argv[2]) if len(argv) > 2 else 0
    if mode == "ready":
        print("ready", flush=True)
        if sys.stdin.readline() != "run\n":
            return
    options = deps(post=_crash_post(export_http.post)) if mode == "crash" else deps()
    result = run_destination(PROFILE, DEST, signal, now=at(seconds), deps=options)
    print(json.dumps({"state": result.state, "reason": result.reason, "sent": result.sent}))


if __name__ == "__main__":
    _driver(sys.argv[1:])
