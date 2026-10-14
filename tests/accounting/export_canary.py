"""Canary planting helpers shared by the projection and the cross-module canary tests.

A *canary* is a string that must never leave the machine. ``SHAPES`` holds one per private
content kind (secret, path, query, fact, title, source ref, raw error body, receipt id,
execution id, token-like hex). ``envelope_variants`` visits EVERY key and leaf of a snapshot
envelope and yields a copy with exactly that position replaced by the canary;
``ledger_variants`` plants the canary into every free-text position the ledger builders accept.
Nothing here is mocked: the variants are real ``SnapshotEnvelope`` / ``VersionedLedger`` values.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import fields, replace
from typing import Any

from ssgrep.accounting import collector_state, collector_store, policy
from ssgrep.accounting.collector_ledger import VersionedLedger
from ssgrep.accounting.contracts import Basis, Quantity, TaskId
from ssgrep.accounting.ledger_contracts import SnapshotEnvelope, SourceAtom
from ssgrep.accounting.units import CHARS_DIV4_V1, CONTEXT_TOKENS
from ssgrep.accounting.valuation_inputs import (
    EvidenceBundle,
    Methodology,
    ValuationAtom,
)
from tests.accounting.valuation_builders import CHARS, DAY, METHOD, TASK, atom, candidate, debit

SHAPES: dict[str, str] = {
    "secret": "sk-CANARY-7f3a91c2e5d84b06",
    "secret_lower": "sk-canary-7f3a91c2e5d84b06",
    "path": "/Users/x/secret-project/file-canary7f3a91.py",
    "query": "SELECT secret_col FROM t WHERE canary_id = 7f3a91",
    "fact": "The deploy key for canary7f3a91 lives in the vault.",
    "title": "Quarterly canary7f3a91 reorganisation plan",
    "source_ref": "claude:session:canary7f3a91#L10",
    "error_body": '{"error":"canary7f3a91 denied","trace":"/srv/app.py"}',
    "receipt_id": "rcpt-canary7f3a91",
    "execution_id": "n1-canary7f3a91-0001",
    "token_hex": "7F3A91C2E5D84B067F3A91C2E5D84B067F3A91C2",
}
# Lowercase hex IS the documented shape of a methodology digest; kept apart from SHAPES.
DIGEST_SHAPED = "7f3a91c2e5d84b067f3a91c2e5d84b067f3a91c2"


def needle(canary: str) -> bytes:
    """The bytes a leak would contain: the JSON-escaped canary (ASCII, as ``otlp.dumps`` writes)."""
    return json.dumps(canary, ensure_ascii=True).encode("ascii")[1:-1]


def leaks(canary: str, *blobs: object) -> bool:
    """True when the canary (raw, JSON-escaped or ``repr``-escaped) is inside any blob."""
    forms = {canary.encode(), needle(canary), canary.replace("/", "\\/").encode()}
    for blob in blobs:
        data = blob if isinstance(blob, bytes) else str(blob).encode("utf-8", "replace")
        if any(form in data for form in forms):
            return True
    return False


# ---- envelope walker --------------------------------------------------------------------------


def _leaf(node: object) -> bool:
    return not isinstance(node, (dict, list, tuple))


def _variants(node: Any, canary: str) -> Iterator[tuple[tuple[str, ...], Any]]:
    """(path, copy of ``node`` with that one key/leaf replaced by ``canary``)."""
    if isinstance(node, dict):
        for key in list(node):
            swapped = {(canary if k == key else k): v for k, v in node.items()}
            yield (str(key), "<key>"), swapped
            for path, new in _variants(node[key], canary):
                yield (str(key), *path), {k: (new if k == key else v) for k, v in node.items()}
    elif isinstance(node, (list, tuple)):
        for index, item in enumerate(node):
            for path, new in _variants(item, canary):
                rebuilt = [new if i == index else v for i, v in enumerate(node)]
                yield (f"[{index}]", *path), type(node)(rebuilt)
    else:
        yield (), canary


def envelope_fields(envelope: SnapshotEnvelope) -> dict[str, Any]:
    return {f.name: getattr(envelope, f.name) for f in fields(envelope)}


def envelope_variants(
    envelope: SnapshotEnvelope, canary: str
) -> Iterator[tuple[tuple[str, ...], SnapshotEnvelope]]:
    """One envelope per key/leaf position (all of them), the position replaced by ``canary``."""
    base = envelope_fields(envelope)
    for name, value in base.items():
        for path, new in _variants(value, canary) if not _leaf(value) else [((), canary)]:
            changed = {**base, name: new}
            yield (name, *path), SnapshotEnvelope(**changed)  # ty: ignore[invalid-argument-type]


def leaf_count(envelope: SnapshotEnvelope) -> int:
    return sum(1 for _ in envelope_variants(envelope, "x"))


# ---- float-rejecting payload parser -----------------------------------------------------------


class Decimal(str):
    """A JSON fraction literal kept verbatim (OTLP histogram bounds/sums are the only ones)."""


def parse_strict(body: bytes) -> Any:
    """``json.loads`` where fractions become :class:`Decimal` text and NaN/Infinity are errors."""

    def refuse(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant {token}")

    return json.loads(body, parse_float=Decimal, parse_constant=refuse)


def decimals(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    """Paths of every fraction literal in a parsed document."""
    if isinstance(node, Decimal):
        yield path
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from decimals(value, (*path, key))
    elif isinstance(node, list):
        for value in node:
            yield from decimals(value, (*path, "[]"))


def number_paths(node: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], str]]:
    """(path, "int"|"decimal") of every bare JSON number in a ``parse_strict`` document."""
    if isinstance(node, Decimal):
        yield path, "decimal"
    elif isinstance(node, bool) or node is None or isinstance(node, str):
        return
    elif isinstance(node, int):
        yield path, "int"
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from number_paths(value, (*path, key))
    elif isinstance(node, list):
        for value in node:
            yield from number_paths(value, (*path, "[]"))


# ---- ledger planting --------------------------------------------------------------------------

Plant = tuple[str, Callable[[str], tuple[EvidenceBundle, Methodology]]]


def _bundle(
    cands: list[Any] | None = None, debits: list[Any] | None = None, method: Methodology = METHOD
) -> tuple[EvidenceBundle, Methodology]:
    return (
        EvidenceBundle(tuple(cands or [candidate("a")]), tuple(debits or [debit("x", 640)])),
        method,
    )


def _with_task(task: TaskId) -> tuple[EvidenceBundle, Methodology]:
    return _bundle([candidate("a", task=task)], [debit("x", 640, task=task)])


def _unit_with(**change: Any) -> tuple[EvidenceBundle, Methodology]:
    unit = replace(CHARS, **change)
    method = Methodology("test-v1", (unit,))
    cand = candidate("a")
    return EvidenceBundle((cand,), (replace(debit("x", 640), unit=unit),)), method


def _atom(**change: Any) -> ValuationAtom:
    base = atom("step-1", 400)
    source: SourceAtom = base.atom
    return replace(base, atom=replace(source, **change))


PLANTS: tuple[Plant, ...] = (
    ("task.root", lambda c: _with_task(TaskId(c, "turn-1", "main"))),
    ("task.turn", lambda c: _with_task(TaskId("root-1", c, "main"))),
    ("task.branch", lambda c: _with_task(TaskId("root-1", "turn-1", c))),
    ("candidate_id", lambda c: _bundle([candidate(c)])),
    ("candidate_type", lambda c: _bundle([candidate("a", candidate_type=c)])),
    ("atom_id", lambda c: _bundle([candidate("a", immediate=(_atom(atom_id=c),))])),
    ("origin_event", lambda c: _bundle([candidate("a", immediate=(_atom(origin_event=c),))])),
    ("atom_digest", lambda c: _bundle([candidate("a", immediate=(_atom(digest=c),))])),
    ("exposure_ids", lambda c: _bundle([candidate("a", exposure_ids=(c,))])),
    ("execution_ids", lambda c: _bundle([candidate("a", execution_ids=(c,))])),
    ("witness_locators", lambda c: _bundle([candidate("a", witness_locators=(c,))])),
    ("action_locators", lambda c: _bundle([candidate("a", action_locators=(c,))])),
    ("evidence_id", lambda c: _bundle([candidate("a", discovery=(_atom(atom_id=c),))])),
    ("debit_id", lambda c: _bundle(debits=[debit(c, 640)])),
    ("debit_kind", lambda c: _bundle(debits=[debit("x", 640, kind=c)])),
    ("debit_exposure_id", lambda c: _bundle(debits=[debit("x", 640, exposure_id=c)])),
    (
        "debit_amount_method",
        lambda c: _bundle(
            debits=[
                replace(
                    debit("x", 640),
                    amount=Quantity(160, CONTEXT_TOKENS, Basis.ESTIMATED, c),
                )
            ]
        ),
    ),
    (
        "debit_unavailable_reason",
        lambda c: _bundle(
            debits=[
                replace(debit("x", None), amount=Quantity.unavailable(CONTEXT_TOKENS, c)),
            ]
        ),
    ),
    (
        "debit_incomplete_member",
        lambda c: _bundle(
            debits=[
                replace(
                    debit("x", None),
                    amount=Quantity.unavailable(
                        CONTEXT_TOKENS, f"downstream_exposure_unobserved:{c}"
                    ),
                )
            ]
        ),
    ),
    ("unit.tokenizer", lambda c: _unit_with(tokenizer=c)),
    ("unit.currency", lambda c: _unit_with(currency=c)),
    ("unit.unit", lambda c: _unit_with(unit=c)),
    ("unit.method", lambda c: _unit_with(method=c)),
    ("methodology.label", lambda c: _bundle(method=Methodology(c, (CHARS,)))),
    (
        "methodology.tokenizer",
        lambda c: _bundle(method=Methodology("test-v1", (CHARS,), tokenizer=c)),
    ),
    ("methodology.pricing", lambda c: _bundle(method=Methodology("test-v1", (CHARS,), pricing=c))),
    (
        "methodology.frequency_view",
        lambda c: _bundle(method=Methodology("test-v1", (CHARS,), frequency_view=c)),
    ),
)


def build_ledger(
    bundle: EvidenceBundle, method: Methodology, *, today: str = "2026-11-02"
) -> VersionedLedger:
    """A real ``VersionedLedger`` reconciled root by root from ``bundle``."""
    roots = sorted({c.task.root for c in bundle.candidates} | {d.task.root for d in bundle.debits})
    ledger = VersionedLedger(method)
    ledger.reconcile(
        {
            root: EvidenceBundle(
                tuple(c for c in bundle.candidates if c.task.root == root),
                tuple(d for d in bundle.debits if d.task.root == root),
            )
            for root in roots
        },
        today=today,
    )
    return ledger


def persist_ledger(profile: str, ledger: VersionedLedger) -> None:
    """Commit ledger state + checkpoint through the collector's own writers."""
    blobs: dict[str, str] = {}
    scope = policy.collection_scope(policy.load_profile_policy(profile))
    state = collector_state.State(
        ledger=ledger.state(), superseded=dict(ledger.superseded), collection_scope=scope
    )
    collector_state.save_state(profile, state, ledger.methodology.hash, blobs)
    checkpoint = collector_store.Checkpoint(
        profile, generation=1, methodology=ledger.methodology.hash
    )
    checkpoint.blobs = blobs
    checkpoint.collection_scope = scope
    collector_store.commit_checkpoint(profile, checkpoint)


def laden_bundle(shapes: dict[str, str]) -> EvidenceBundle:
    """Real evidence with a DIFFERENT canary in every free-text position at once."""
    root, turn, branch = shapes["path"], shapes["title"], shapes["fact"]
    task = TaskId(root, turn, branch)
    step = replace(
        atom(shapes["token_hex"], 400),
        atom=SourceAtom(shapes["token_hex"], shapes["source_ref"], shapes["secret_lower"], 400),
    )
    cand = candidate(
        shapes["receipt_id"],
        task=task,
        candidate_type=shapes["query"],
        immediate=(step,),
        exposure_ids=(shapes["execution_id"],),
        execution_ids=(shapes["execution_id"],),
        witness_locators=(shapes["source_ref"],),
        action_locators=(shapes["path"],),
    )
    spent = debit(shapes["error_body"], 640, task=task, kind=shapes["query"])
    return EvidenceBundle((cand,), (replace(spent, exposure_id=shapes["secret"]),))


def laden_envelope(
    shapes: dict[str, str], *, generation: int = 1, warnings: bool = True
) -> SnapshotEnvelope:
    """A complete real envelope whose free-text positions all carry canaries (valid times)."""
    from tests.accounting.http_builders import make_snapshot, window_with_unknown_optimistic

    base = make_snapshot(
        generation,
        methodology_id="19b54b416561f300e32b9e793a8ef5ce",
        window=window_with_unknown_optimistic(),
    )
    fields_ = envelope_fields(base)
    fields_["warnings"] = (
        (shapes["error_body"], f"lineage:{shapes['source_ref']}") if warnings else ()
    )
    fields_["coverage"] = {
        **base.coverage,
        shapes["path"]: shapes["query"],
        "sources": {**dict(base.coverage["sources"]), shapes["title"]: 1},  # ty: ignore[no-matching-overload]
    }
    fields_["observations"] = {
        **base.observations,
        "backfill": [{"kind": shapes["fact"], "counts": {shapes["source_ref"]: 2}}],
        "epoch": {"counter_epoch": 1, "started_at": "2026-10-01T00:00:00Z", shapes["secret"]: 1},
    }
    window = json.loads(json.dumps(fields_["windows"]["7d"]))
    # a REAL method label the catalog accepts (lowercase id shape) but the export allowlist must not
    window["estimates"]["context_tokens"]["central"]["credit"]["method"] = shapes["secret_lower"]
    window["estimates"] = {
        **window["estimates"],
        shapes["title"]: {shapes["fact"]: {"net": {"value": 5, "method": shapes["secret"]}}},
    }
    fields_["windows"] = {"7d": window, shapes["path"]: {shapes["query"]: shapes["fact"]}}
    fields_["estimates"] = {
        "representative_receipt": shapes["receipt_id"],
        "method": shapes["execution_id"],
        "reason": shapes["token_hex"],
    }
    fields_["source_manifest_digest"] = shapes["secret_lower"]
    return SnapshotEnvelope(**{k: v for k, v in fields_.items()})


def publish_laden(
    profile: str,
    shapes: dict[str, str],
    *,
    generation: int = 1,
    ledger: bool = True,
    **overrides: Any,
) -> SnapshotEnvelope:
    """Publish a canary-laden snapshot (real ``snapshots.publish``) plus a laden real ledger."""
    from ssgrep.accounting import snapshots

    base = laden_envelope(shapes, generation=generation)
    fields_ = {
        k: v for k, v in envelope_fields(base).items() if k not in ("schema_version", "snapshot_id")
    }
    fields_.update(overrides)
    envelope = snapshots.build_envelope(**fields_)
    snapshots.publish(profile, envelope)
    if ledger:
        persist_ledger(profile, build_ledger(laden_bundle(shapes), METHOD))
    return envelope


def ledger_variants(canary: str) -> Iterator[tuple[str, EvidenceBundle, Methodology]]:
    for label, build in PLANTS:
        bundle, method = build(canary)
        yield label, bundle, method


__all__ = [
    "CHARS_DIV4_V1",
    "DAY",
    "DIGEST_SHAPED",
    "PLANTS",
    "SHAPES",
    "TASK",
    "Decimal",
    "envelope_fields",
    "envelope_variants",
    "build_ledger",
    "ledger_variants",
    "persist_ledger",
    "leaks",
    "leaf_count",
    "laden_bundle",
    "laden_envelope",
    "needle",
    "number_paths",
    "publish_laden",
    "parse_strict",
]
