"""Strict JSON codec for the persisted policy file (``accounting/policy.json``).

Integers and strings only. Any shape violation raises ``PolicyError`` with the
stable condition ``policy_corrupt`` so readers never act on half-understood
state and never fall back to permissive defaults.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from ssgrep.accounting.contracts import SCHEMA_VERSION
from ssgrep.accounting.ledger_contracts import (
    DestinationPolicy,
    ProfilePolicy,
    Signal,
    valid_profile_id,
)
from ssgrep.accounting.policy_rules import PolicyError, PolicyFile

MAX_POLICY_BYTES = 1024 * 1024
MAX_GAPS = 32


def _corrupt(detail: str) -> PolicyError:
    return PolicyError(
        "policy_corrupt",
        f"Accounting policy file is unreadable ({detail}); sending is paused.",
        "ssgrep metrics setup",
        exit_code=1,
    )


def _str(value: Any, name: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise _corrupt(f"{name} must be a string")
    return value


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _corrupt(f"{name} must be an integer")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise _corrupt(f"{name} must be a boolean")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise _corrupt(f"{name} must be a list")
    return value


def destination_from_dict(raw: Any) -> DestinationPolicy:
    if not isinstance(raw, dict):
        raise _corrupt("destination must be an object")
    try:
        signals = tuple(Signal(s) for s in _list(raw.get("signals", ["metrics"]), "signals"))
    except ValueError as error:
        raise _corrupt("unknown signal") from error
    return DestinationPolicy(
        destination_id=str(_str(raw.get("destination_id"), "destination_id")),
        endpoint=str(_str(raw.get("endpoint"), "endpoint")),
        signals=signals,
        logs_permitted=_bool(raw.get("logs_permitted", False), "logs_permitted"),
        credential_ref=_str(raw.get("credential_ref"), "credential_ref", nullable=True),
        proxy=_str(raw.get("proxy"), "proxy", nullable=True),
    )


def profile_from_dict(raw: Any) -> ProfilePolicy:
    if not isinstance(raw, dict):
        raise _corrupt("profile must be an object")
    exports = []
    for pair in _list(raw.get("enabled_exports", []), "enabled_exports"):
        if not (
            isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)
        ):
            raise _corrupt("enabled_exports entries must be [destination, signal]")
        exports.append((pair[0], pair[1]))
    return ProfilePolicy(
        profile_id=str(_str(raw.get("profile_id"), "profile_id")),
        roots=tuple(str(_str(r, "root")) for r in _list(raw.get("roots", []), "roots")),
        managed=_bool(raw.get("managed", False), "managed"),
        policy_version=_int(raw.get("policy_version", 1), "policy_version"),
        policy_source=str(_str(raw.get("policy_source", "local"), "policy_source")),
        evidence_retention_days=_int(raw.get("evidence_retention_days"), "evidence_retention_days"),
        receipt_retention_days=_int(raw.get("receipt_retention_days"), "receipt_retention_days"),
        destinations=tuple(
            destination_from_dict(d) for d in _list(raw.get("destinations", []), "destinations")
        ),
        collection_enabled=_bool(raw.get("collection_enabled", False), "collection_enabled"),
        paused=_bool(raw.get("paused", False), "paused"),
        paused_at=_str(raw.get("paused_at"), "paused_at", nullable=True),
        enabled_exports=tuple(exports),
        evidence_cache_enabled=_bool(
            raw.get("evidence_cache_enabled", False), "evidence_cache_enabled"
        ),
        evidence_cache_retention_days=_int(
            raw.get("evidence_cache_retention_days", 0), "evidence_cache_retention_days"
        ),
        evidence_cache_max_bytes=_int(
            raw.get("evidence_cache_max_bytes", 0), "evidence_cache_max_bytes"
        ),
    )


def profile_to_dict(profile: ProfilePolicy) -> dict[str, Any]:
    plain = asdict(profile)
    plain["roots"] = list(profile.roots)
    plain["enabled_exports"] = [list(pair) for pair in profile.enabled_exports]
    plain["destinations"] = [
        {**asdict(d), "signals": [s.value for s in d.signals]} for d in profile.destinations
    ]
    return plain


def _profiles(raw: Any, name: str) -> dict[str, ProfilePolicy]:
    if not isinstance(raw, dict):
        raise _corrupt(f"{name} must be an object")
    profiles = {str(k): profile_from_dict(v) for k, v in raw.items()}
    if any(not valid_profile_id(k) or k != v.profile_id for k, v in profiles.items()):
        raise _corrupt(f"{name} keys must match their profile identities")
    return profiles


def _gaps(raw: dict[str, Any]) -> dict[str, list[list[str | None]]]:
    gaps: dict[str, list[list[str | None]]] = {}
    for profile, rows in raw.items():
        if not isinstance(rows, list) or any(
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[0], str)
            or (row[1] is not None and not isinstance(row[1], str))
            for row in rows
        ):
            raise _corrupt("gaps must contain [start, end-or-null] pairs")
        gaps[profile] = rows[-MAX_GAPS:]
    return gaps


def parse(data: bytes) -> PolicyFile:
    try:
        raw = json.loads(data)
    except (ValueError, UnicodeDecodeError) as error:
        raise _corrupt("not valid JSON") from error
    if not isinstance(raw, dict):
        raise _corrupt("top level must be an object")
    if type(raw.get("schema")) is not int or raw["schema"] != SCHEMA_VERSION:
        raise _corrupt("unsupported schema")
    assignments = raw.get("assignments", {})
    paused = raw.get("sending_paused", {})
    gaps = raw.get("gaps", {})
    if not (isinstance(assignments, dict) and isinstance(paused, dict) and isinstance(gaps, dict)):
        raise _corrupt("assignments, sending_paused and gaps must be objects")
    return PolicyFile(
        profiles=_profiles(raw.get("profiles", {}), "profiles"),
        ceilings=_profiles(raw.get("ceilings", {}), "ceilings"),
        assignments={str(k): str(v) for k, v in assignments.items()},
        sending_paused={str(k): str(v) for k, v in paused.items()},
        gaps=_gaps(gaps),
    )


def dump(file: PolicyFile) -> bytes:
    document = {
        "schema": SCHEMA_VERSION,
        "profiles": {k: profile_to_dict(v) for k, v in sorted(file.profiles.items())},
        "ceilings": {k: profile_to_dict(v) for k, v in sorted(file.ceilings.items())},
        "assignments": dict(sorted(file.assignments.items())),
        "sending_paused": dict(sorted(file.sending_paused.items())),
        "gaps": {k: v[-MAX_GAPS:] for k, v in sorted(file.gaps.items())},
    }
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
