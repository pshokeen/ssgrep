"""Optional investigation metadata, independent of notes and index schemas.

A binding names one shared investigation, not a copy of its usage. Its latest
immutable revision supersedes prior estimates; revisions must never be summed.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from ssgrep.accounting.privacy.storage import read_file
from ssgrep.sessions import _work_storage as storage

if TYPE_CHECKING:
    from ssgrep.utilities.types import EpisodeDetail

_MAX_WORK = 64 * 1024
_MAX_RECORD = _MAX_WORK + 4096
_BASES = {"runtime_reported", "agent_estimated", "transcript_inferred"}
_TOKENS = {
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
}
_FIELDS = {
    "work_id",
    "provider",
    "model",
    "elapsed_seconds",
    "usage",
    "basis",
    "contributions",
    "source_ref",
}


def _encode(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\0" in value:
        raise ValueError(f"{name} must be a nonempty string of at most {limit} characters")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    return value


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _contribution(raw: dict, basis: str) -> dict:
    result = {}
    for field in ("provider", "model"):
        if field in raw:
            result[field] = _text(raw[field], field, 256)
    if "usage" in raw:
        usage = raw["usage"]
        if not isinstance(usage, dict) or usage.keys() - _TOKENS:
            raise ValueError("usage must contain only supported token counts")
        result["usage"] = {key: _count(value, key) for key, value in usage.items()}
    actual_basis = raw.get("basis", basis)
    if not isinstance(actual_basis, str) or actual_basis not in _BASES:
        raise ValueError("unsupported work basis")
    result["basis"] = actual_basis
    return result


def validate_work(raw: dict) -> dict:
    """Return detached canonical metadata; unknown quantities remain absent."""
    if not isinstance(raw, dict) or raw.keys() - _FIELDS:
        raise ValueError("work must be an object with supported metadata fields")
    try:
        if len(_encode(raw)) > _MAX_WORK:
            raise ValueError("work metadata exceeds 64 KiB")
    except (TypeError, OverflowError, RecursionError, UnicodeError) as error:
        raise ValueError("work must contain JSON-compatible values") from error
    result: dict[str, Any] = {"work_id": _text(raw.get("work_id"), "work_id", 128)}
    result.update(_contribution(raw, "agent_estimated"))
    if "elapsed_seconds" in raw:
        result["elapsed_seconds"] = _count(raw["elapsed_seconds"], "elapsed_seconds")
    if "source_ref" in raw:
        result["source_ref"] = _text(raw["source_ref"], "source_ref", 2048)
    if "contributions" in raw:
        if {"provider", "model", "usage"} & raw.keys():
            raise ValueError("contributions cannot be mixed with top-level provider/model/usage")
        items = raw["contributions"]
        if not isinstance(items, list) or not items:
            raise ValueError("contributions must be a nonempty list")
        result["contributions"] = []
        for item in items:
            if not isinstance(item, dict) or item.keys() - {"provider", "model", "usage", "basis"}:
                raise ValueError("unsupported contribution fields")
            if not {"provider", "model"} & item.keys():
                raise ValueError("each contribution needs a provider or model")
            result["contributions"].append(_contribution(item, result["basis"]))
    if result["basis"] == "transcript_inferred" and "source_ref" not in result:
        raise ValueError("transcript-inferred work requires source_ref")
    if len(_encode(result)) > _MAX_WORK:
        raise ValueError("canonical work metadata exceeds 64 KiB")
    return result


def subject_digest(detail: EpisodeDetail) -> str:
    """Bind metadata to the actual indexed episode, not just a reusable ref."""
    content = {
        key: getattr(detail, key)
        for key in ("episode_id", "runtime", "prompt_text", "response_text")
    }
    return hashlib.sha256(_encode(content)).hexdigest()


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(ref: str, subject: str) -> None:
    _text(ref, "ref", 4096)
    if (
        not isinstance(subject, str)
        or len(subject) != 64
        or any(c not in "0123456789abcdef" for c in subject)
    ):
        raise ValueError("subject must be a SHA256 digest")


def _load(descriptor: int, name: str) -> dict | None:
    data = read_file(descriptor, name, _MAX_RECORD)
    if data is None:
        return None
    value = json.loads(data)
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("unsupported or corrupt work metadata schema")
    return value


def _binding(descriptor: int, ref: str) -> dict | None:
    value = _load(descriptor, f"binding-{_key(ref)}.json")
    if value is not None:
        if value.keys() != {"schema_version", "ref", "subject", "work_id"} or value["ref"] != ref:
            raise ValueError("invalid work binding")
        _identity(value["ref"], value["subject"])
        _text(value["work_id"], "work_id", 128)
    return value


def _head(descriptor: int, work_id: str) -> int:
    value = _load(descriptor, "head.json")
    if value is None:
        return 0
    if value.keys() != {"schema_version", "work_id", "revision"} or value["work_id"] != work_id:
        raise ValueError("invalid work revision head")
    revision = _count(value["revision"], "revision")
    if not revision:
        raise ValueError("invalid work revision number")
    return revision


def _revision(descriptor: int, work_id: str, revision: int) -> dict:
    value = _load(descriptor, f"revision-{revision}.json")
    if value is None or value.keys() != {"schema_version", "work_id", "revision", "work"}:
        raise ValueError("missing or corrupt work revision")
    if (
        value["work_id"] != work_id
        or type(value["revision"]) is not int
        or value["revision"] != revision
    ):
        raise ValueError("work revision identity mismatch")
    canonical = validate_work(value["work"])
    if canonical != value["work"] or canonical["work_id"] != work_id:
        raise ValueError("noncanonical work revision")
    return value


def attach_work(ref: str, subject: str, raw: dict) -> dict:
    """Publish a revision and bind a subject; identical current work is idempotent.

    Validation precedes storage creation. Interrupted publication retains its
    immutable revisions; validated orphans are retained when retrying or correcting.
    """
    work = validate_work(raw)
    _identity(ref, subject)
    work_id = work["work_id"]
    with storage.store(create=True) as root:
        previous_binding = _binding(root, ref)
        with storage.directory(root, f"work-{_key(work_id)}", create=True) as directory:
            revision = _head(directory, work_id)
            published = revision
            latest = _revision(directory, work_id, revision) if revision else None
            while read_file(directory, f"revision-{revision + 1}.json", _MAX_RECORD) is not None:
                revision += 1
                latest = _revision(directory, work_id, revision)
            if latest is None or latest["work"] != work:
                revision += 1
                latest = {
                    "schema_version": 1,
                    "work_id": work_id,
                    "revision": revision,
                    "work": work,
                }
                storage.immutable(directory, f"revision-{revision}.json", _encode(latest))
            if revision != published:
                head = {key: latest[key] for key in ("schema_version", "work_id", "revision")}
                storage.replace(directory, "head.json", _encode(head), limit=_MAX_RECORD)
            binding = {"schema_version": 1, "ref": ref, "subject": subject, "work_id": work_id}
            if previous_binding != binding:
                storage.replace(
                    root, f"binding-{_key(ref)}.json", _encode(binding), limit=_MAX_RECORD
                )
            return latest


def read_work(ref: str, subject: str) -> dict | None:
    """Read matching metadata without writes; optional-store damage is nonfatal."""
    try:
        _identity(ref, subject)
        with storage.store() as root:
            binding = _binding(root, ref)
            if binding is None or binding["subject"] != subject:
                return None
            work_id = binding["work_id"]
            with storage.directory(root, f"work-{_key(work_id)}", create=False) as directory:
                revision = _head(directory, work_id)
                return _revision(directory, work_id, revision) if revision else None
    except (OSError, ValueError, TypeError, RecursionError):
        return None


def work_history(work_id: str) -> list[dict]:
    """Read every retained revision in order; malformed history is an error."""
    _text(work_id, "work_id", 128)
    try:
        with storage.store() as root:
            with storage.directory(root, f"work-{_key(work_id)}", create=False) as directory:
                return [
                    _revision(directory, work_id, n)
                    for n in range(1, _head(directory, work_id) + 1)
                ]
    except FileNotFoundError:
        return []
