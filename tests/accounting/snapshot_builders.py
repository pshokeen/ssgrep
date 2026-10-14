"""Hand-built snapshot/receipt/history files for read-surface tests.

Values are synthetic and exist only to exercise states; nothing here ships.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ssgrep.accounting.contracts import SCHEMA_VERSION

COLLECTED_AT = "2026-10-12T12:00:00Z"
RECEIPT_ID = "rcpt-7f3a91"


def q(value: int | None, unit: str, basis: str = "estimated", reason: str = "") -> dict[str, Any]:
    if value is None:
        return {"value": None, "unit": unit, "basis": "unavailable", "reason": reason}
    return {"value": value, "unit": unit, "basis": basis, "method": "chars_div4_v1"}


def scenarios(credit: tuple, debit: tuple, net: tuple, unit: str) -> dict[str, Any]:
    names = ("conservative", "central", "optimistic")
    return {
        name: {
            "credit": q(c, unit),
            "debit": q(d, unit),
            "net": q(n, unit, reason="source_costs_missing"),
        }
        for name, c, d, n in zip(names, credit, debit, net, strict=True)
    }


def window_7d(**overrides: Any) -> dict[str, Any]:
    window: dict[str, Any] = {
        "observations": {
            "searches": q(100, "count", "observed"),
            "shows": q(12, "count", "observed"),
            "failed": q(8, "count", "observed"),
            "visible_bytes": q(430080, "bytes", "observed"),
        },
        "estimates": {
            "context_tokens": scenarios(
                (4000, 9000, 20000), (16000, 16000, 16000), (-12000, -7000, 4000), "tokens"
            ),
            "work_seconds": scenarios((60, 300, 700), (120, 120, 120), (-60, 180, 580), "s"),
        },
        "coverage": {
            "recorded_lookups": 112,
            "attributed_lookups": 91,
            "candidate_receipts": 13,
            "source_costs_available": 8,
            "ambiguous_joins": 3,
            "unavailable_sources": 2,
            "partial": True,
            "index": "available",
            "exclusions": {"prior_known": 4, "user_supplied": 2},
        },
    }
    window.update(overrides)
    return window


def snapshot(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": "a" * 16,
        "generation": 7,
        "counter_epoch": 1,
        "methodology_id": "m-test-v1",
        "policy_version": 1,
        "collected_at": COLLECTED_AT,
        "coverage_through": COLLECTED_AT,
        "source_manifest_digest": "b" * 16,
        "windows": {"7d": window_7d()},
        "coverage": {},
        "observations": {},
        "estimates": {"representative_receipt": RECEIPT_ID},
        "warnings": [],
    }
    document.update(overrides)
    return document


def write_snapshot(data_dir: Path, profile: str = "personal-local", **overrides: Any) -> Path:
    target = data_dir / "accounting" / profile / "snapshots" / "current.json"
    _private_tree(data_dir, target)
    target.write_text(json.dumps(snapshot(**overrides)))
    target.chmod(0o600)
    return target


def _private_tree(data_dir: Path, target: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    data_dir.chmod(0o700)
    current = data_dir
    for part in target.relative_to(data_dir).parts[:-1]:
        current = current / part
        current.mkdir(exist_ok=True)
        current.chmod(0o700)


def write_receipt(data_dir: Path, profile: str = "personal-local", **overrides: Any) -> Path:
    receipt: dict[str, Any] = {
        "receipt_id": RECEIPT_ID,
        "revision": 2,
        "root": "/work/project",
        "task_turn": "turn-4",
        "candidate_type": "path",
        "atoms": ["atom-1", "atom-2"],
        "allocations": [["conservative", ["atom-1"]], ["central", ["atom-1", "atom-2"]]],
        "exclusions": ["prior_known:atom-9"],
        "costs": [
            ["lookup_debit", q(16000, "tokens")],
            ["source_cost", q(None, "tokens", reason="source_unavailable")],
        ],
        "methodology": "m-test-v1",
        "evidence_available": True,
        "execution_ids": ["n1-0001"],
        "content": "SECRET-CONTENT-CANARY",
    }
    receipt.update(overrides)
    target = data_dir / "accounting" / profile / "receipts" / f"{receipt['receipt_id']}.json"
    _private_tree(data_dir, target)
    target.write_text(json.dumps(receipt))
    target.chmod(0o600)
    return target


def write_history(
    data_dir: Path, rows: list[dict[str, Any]], profile: str = "personal-local"
) -> Path:
    target = data_dir / "accounting" / profile / "history" / "periods.json"
    _private_tree(data_dir, target)
    target.write_text(json.dumps(rows))
    target.chmod(0o600)
    return target


def enable_policy(profile: str = "personal-local", *, paused: bool = False) -> None:
    from ssgrep.accounting import policy

    policy.enable_collection(profile)
    if paused:
        policy.pause(profile)
