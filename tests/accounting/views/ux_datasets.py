"""Acceptance datasets (task 1.4) for every experience state of the savings surface.

EVERYTHING here is SYNTHETIC: transcripts, journal events, clocks, endpoints and credentials
are made up for tests (values such as ``127.0.0.1`` ports, ``sk-CANARY-...`` tokens and the
fixed 2026-10 clock are illustrative, not measurements of any real person). Builders drive the
REAL collector, policy module and delivery cycle; nothing here hand-writes accounting files.
Experience states are asserted from structured fields (state, condition codes, counts, exit
codes), never from pinned prose.

Each builder takes the isolated ``root`` (pytest ``tmp_path``) whose layout ``Binary`` and the
autouse environment fixture share, and returns a ``Dataset`` naming the reading clock.

===================  =====================================  ====================================
dataset              what it contains                       spec scenarios (savings-experience)
===================  =====================================  ====================================
``disabled``         nothing enabled, nothing on disk       Disabled first run
``no_snapshot``      capture on, no collection yet          Enabled without a completed collection
``fresh``            collected; partial denominators,       Summary leads to a real explanation;
                     unknown net without a current index; Partial and negative result (when
                     numeric tests explicitly install the  explicitly given synthetic rarity);
                     current-index arithmetic fixture      Machine-readable partial data
``no_reuse``         one lookup, no traceable reuse         No traceable reuse
``stale``            ``fresh`` read 3 h later               Stale last good snapshot
``paused``           ``fresh`` then user pause              Paused history remains readable;
                                                            Paused manual operations
``evidence_expired`` ``fresh`` collected again 45 days      Evidence has expired
                     later (retention boundary)
``overlap``          two consumers of ONE source fact       Summary leads to a real explanation
                                                            (overlap removed within a task)
``revised``          ``overlap`` + late evidence for a      Revised period history
                     sealed day (superseding revision)
``offline``          export to a refused port, backlog      Offline or expired credentials
                     pending, retry scheduled
``auth_error``       receiver answers 401                   Offline or expired credentials
``partial_export``   receiver accepts with rejections       Partial and negative result (delivery
                                                            side); Machine-readable partial data
``paused_pending``   ``offline`` backlog, then pause        Paused manual operations
===================  =====================================  ====================================

Delivery datasets (``offline`` ... ``paused_pending``) run the real ``ssgrep savings export send``
binary against a loopback ``LabReceiver`` (the collector helper in ``otel_collector_lab`` needs
Docker; the stdlib receiver is the same wire contract and is what the export tests use).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from ssgrep.accounting.privacy import policy
from tests.accounting.collect import collector_scene as scene, verify_scenes as vs
from tests.accounting.export import delivery_fixtures as fx
from tests.accounting.export.lab_receiver import Behavior, LabReceiver, Partial, Respond
from tests.accounting.ledger.reduce_builders import source_session
from tests.accounting.views.test_ux_support import (
    NOW,
    Binary,
    collected_scene,
    current_fact_consumer,
)

REVISED_AT = NOW + timedelta(days=10)
EXPIRED_AT = scene.at(hour=10) + timedelta(days=45)
STATES = (
    "disabled",
    "no_snapshot",
    "fresh",
    "no_reuse",
    "stale",
    "paused",
    "evidence_expired",
    "overlap",
    "revised",
)
DELIVERY_STATES = ("offline", "auth_error", "partial_export", "paused_pending")


@dataclass(frozen=True)
class Dataset:
    name: str
    now: datetime
    root: Path


def disabled(root: Path) -> Dataset:
    return Dataset("disabled", NOW, root)


def no_snapshot(root: Path) -> Dataset:
    vs.enable()
    return Dataset("no_snapshot", NOW, root)


def fresh(root: Path) -> Dataset:
    collected_scene(root)
    return Dataset("fresh", NOW, root)


def stale(root: Path) -> Dataset:
    collected_scene(root)
    return Dataset("stale", NOW + timedelta(hours=3), root)


def paused(root: Path) -> Dataset:
    collected_scene(root)
    policy.pause("personal-local")
    return Dataset("paused", NOW + timedelta(hours=3), root)


def evidence_expired(root: Path) -> Dataset:
    collected_scene(root)
    assert scene.collect(EXPIRED_AT).state == "published"
    return Dataset("evidence_expired", EXPIRED_AT, root)


def no_reuse(root: Path) -> Dataset:
    vs.enable()
    vs.start_capture(day=12)
    vs.put(
        vs.projects(root),
        "lonely",
        [vs.claude_prompt("lonely", "hello", uuid="u", ts=vs.ts(12, 9))],
        scene.at(hour=10),
    )
    scene.journal_events(scene.execution("n-1", vs.ts(12, 9, 5), text="anything", transport="cli"))
    assert scene.collect(NOW).state == "published"
    return Dataset("no_reuse", NOW, root)


def _renamed(session: str, records: list[dict]) -> list[dict]:
    return [{**r, "sessionId": session} if "sessionId" in r else dict(r) for r in records]


def _put(root: Path, name: str, records: list[dict]) -> None:
    vs.put(vs.projects(root), name, _renamed(name, records), scene.at(hour=10))


def overlap(root: Path) -> Dataset:
    vs.enable()
    vs.start_capture(day=11)
    _put(root, "src1", source_session("src1", vs.fact(0), when_day=11))
    _put(root, "first", current_fact_consumer())
    _put(root, "second", current_fact_consumer())
    assert scene.collect(NOW).state == "published"
    return Dataset("overlap", NOW, root)


def revised(root: Path) -> Dataset:
    overlap(root)
    _put(root, "late", current_fact_consumer())
    assert scene.collect(REVISED_AT).state == "published"
    return Dataset("revised", REVISED_AT, root)


BUILDERS = {
    "disabled": disabled,
    "no_snapshot": no_snapshot,
    "fresh": fresh,
    "no_reuse": no_reuse,
    "stale": stale,
    "paused": paused,
    "evidence_expired": evidence_expired,
    "overlap": overlap,
    "revised": revised,
}


# ---- delivery datasets: real binary against a loopback receiver ------------------------------

#: How the receiver behaves for each delivery dataset (None: nothing listens at the endpoint).
RECEIVER: dict[str, Behavior | None] = {
    "offline": None,
    "auth_error": Respond(401, b'{"message":"token rejected"}'),
    "partial_export": Partial(rejected=4, message="synthetic rejection"),
    "paused_pending": None,
}


@contextmanager
def delivery(kind: str, root: Path) -> Iterator[tuple[Binary, LabReceiver | None]]:
    """Collect, point a destination at the receiver, run one real ``savings export send`` cycle.

    ``paused_pending`` additionally pauses the profile afterwards (backlog stays pending).
    """
    collected_scene(root)
    behavior = RECEIVER[kind]
    receiver = LabReceiver.start() if behavior is not None else None
    try:
        if receiver is not None and behavior is not None:
            receiver.queue(behavior)
            endpoint = receiver.url
        else:
            endpoint = f"http://127.0.0.1:{fx.free_port()}"
        fx.configure(endpoint)
        run = Binary(root)
        sent = run("savings", "export", "send", fx.DEST, "--json", env=fx.ENV)
        assert sent.returncode in (0, 1), sent.stderr[-300:]
        if kind == "paused_pending":
            assert run("savings", "pause", "--json", env=fx.ENV).returncode == 0
        yield run, receiver
    finally:
        if receiver is not None:
            receiver.stop()
