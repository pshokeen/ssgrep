"""Immutable snapshot generations and the atomic ``current.json`` pointer.

Layout (``<data>/accounting/<profile>/snapshots/``):
``gen-<generation:010d>-<snapshot_id>.json`` are immutable, complete envelopes
(at most one file per generation number); ``current.json`` holds the exact bytes of the
newest published generation. A publish is ordered so that no reader and no
crash can ever observe a partial document under a final name:

1. generation temporary written + fsynced           (``gen_tmp_written``)
2. temporary hard-linked to its ``gen-...`` name (exclusive; an existing name is never
   replaced), temporary unlinked, directory fsynced   (``gen_linked``)
3. ``current.json`` temporary written + fsynced     (``current_tmp_written``)
4. temporary renamed over ``current.json``           (``current_renamed``)
5. directory fsynced                                 (``published``)

Only ``.*.tmp`` names may ever be partial. ``sweep_temporaries``, ``prune`` and
``repair_current`` assume the caller holds the per-profile worker lease, so no
other publisher is mid-flight.

``snapshot_id`` is a content digest of everything except ``collected_at``,
``generation`` and ``snapshot_id`` itself: recollecting identical ledger content
keeps its id while the generation and collection time move on. Acknowledged
snapshot ids recorded under ``delivery/<destination>/<signal>.json`` pin their
generation file against ``prune``.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

from ssgrep.accounting.ledger.contracts import SCHEMA_VERSION, canonical_bytes, digest_of
from ssgrep.accounting.ledger.ledger_contracts import SnapshotEnvelope
from ssgrep.accounting.privacy import storage

SNAPSHOT_DIR = "snapshots"
CURRENT_NAME = "current.json"
MAX_SNAPSHOT_BYTES = (
    16 * 1024 * 1024
)  # matches privacy.storage.MAX_SNAPSHOT_BYTES; limit identity stays pinned by test.
MAX_GENERATION = 10**10 - 1
STEPS = ("gen_tmp_written", "gen_linked", "current_tmp_written", "current_renamed", "published")

Hook = Callable[[str], None]

_ID_PATTERN = "[0-9a-f]{16,64}"
_ID_RE = re.compile(_ID_PATTERN)
_GEN_RE = re.compile(rf"gen-(\d{{10}})-({_ID_PATTERN})\.json")
_TEMP_RE = re.compile(r"\..+\.tmp")
# Ledger content (identity) fields; ``collected_at``/``generation``/``snapshot_id`` are excluded.
_IDENTITY_FIELDS = (
    "schema_version",
    "counter_epoch",
    "methodology_id",
    "policy_version",
    "coverage_through",
    "source_manifest_digest",
    "windows",
    "coverage",
    "observations",
    "estimates",
    "warnings",
)


class GenerationExists(ValueError):
    """A generation file with different bytes already exists; generations are immutable."""


def snapshot_identity(envelope_fields: Mapping[str, object]) -> str:
    """Hex digest over the canonical ledger content (no collected_at/generation/snapshot_id)."""
    try:
        content = {name: envelope_fields[name] for name in _IDENTITY_FIELDS}
        return digest_of(content)
    except KeyError as error:
        raise ValueError(f"snapshot content is missing {error.args[0]!r}") from error
    except (TypeError, ValueError) as error:
        raise ValueError(f"snapshot content is not canonical accounting data: {error}") from error


def build_envelope(
    *,
    generation: int,
    counter_epoch: int,
    methodology_id: str,
    policy_version: int,
    collected_at: str,
    coverage_through: str,
    source_manifest_digest: str,
    windows: dict[str, Any],
    coverage: dict[str, Any],
    observations: dict[str, Any],
    estimates: dict[str, Any],
    warnings: Sequence[str] = (),
) -> SnapshotEnvelope:
    fields: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "counter_epoch": counter_epoch,
        "methodology_id": methodology_id,
        "policy_version": policy_version,
        "coverage_through": coverage_through,
        "source_manifest_digest": source_manifest_digest,
        "windows": windows,
        "coverage": coverage,
        "observations": observations,
        "estimates": estimates,
        "warnings": tuple(warnings),
    }
    return SnapshotEnvelope(
        snapshot_id=snapshot_identity(fields),
        generation=generation,
        collected_at=collected_at,
        **fields,
    )


def encode(envelope: SnapshotEnvelope) -> bytes:
    """Canonical JSON plus a newline; floats and oversize documents are rejected."""
    try:
        data = canonical_bytes(envelope) + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"snapshot is not canonical accounting data: {error}") from error
    if len(data) > MAX_SNAPSHOT_BYTES:
        raise ValueError(f"snapshot exceeds {MAX_SNAPSHOT_BYTES} bytes")
    return data


def _reject_float(text: str) -> Any:
    raise ValueError("floats are not allowed in accounting data")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _typed(raw: dict[str, Any], name: str, kind: type) -> Any:
    if name not in raw:
        raise ValueError(f"snapshot field {name!r} is missing")
    value = raw[name]
    ok = _is_int(value) if kind is int else isinstance(value, kind)
    if not ok:
        raise ValueError(f"snapshot field {name!r} has the wrong type")
    return value


def decode(data: bytes) -> SnapshotEnvelope:
    """Parse and fully validate; the stored ``snapshot_id`` must equal the recomputed identity."""
    try:
        raw = json.loads(data, parse_float=_reject_float, parse_constant=_reject_float)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"snapshot is not valid JSON: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("snapshot is not an object")
    if _typed(raw, "schema_version", int) != SCHEMA_VERSION:
        raise ValueError(f"unsupported snapshot schema {raw['schema_version']}")
    names = {*_IDENTITY_FIELDS, "snapshot_id", "generation", "collected_at"}
    if set(raw) != names:
        raise ValueError(f"snapshot fields differ: {sorted(set(raw) ^ names)}")
    generation = _typed(raw, "generation", int)
    if not 1 <= generation <= MAX_GENERATION:
        raise ValueError("snapshot generation out of range")
    for name in ("counter_epoch", "policy_version"):
        _typed(raw, name, int)
    for name in ("snapshot_id", "methodology_id", "collected_at", "coverage_through"):
        _typed(raw, name, str)
    _typed(raw, "source_manifest_digest", str)
    for name in ("windows", "coverage", "observations", "estimates"):
        _typed(raw, name, dict)
    warnings = _typed(raw, "warnings", list)
    if not all(isinstance(item, str) for item in warnings):
        raise ValueError("snapshot warnings must be strings")
    if not all(isinstance(window, dict) for window in raw["windows"].values()):
        raise ValueError("snapshot windows must be objects")
    if raw["snapshot_id"] != snapshot_identity(raw):
        raise ValueError("snapshot_id does not match snapshot content")
    raw["warnings"] = tuple(warnings)
    return SnapshotEnvelope(**raw)


def _checked(generation: int) -> int:
    if not _is_int(generation) or not 1 <= generation <= MAX_GENERATION:
        raise ValueError(f"invalid snapshot generation {generation!r}")
    return generation


def _file_name(generation: int, snapshot_id: str) -> str:
    return f"gen-{_checked(generation):010d}-{snapshot_id}.json"


def _entries(dir_fd: int) -> list[tuple[int, str, str]]:
    """(generation, snapshot_id, file name) of every generation file, ascending."""
    found = (_GEN_RE.fullmatch(name) for name in storage.list_names(dir_fd))
    return sorted((int(m.group(1)), m.group(2), m.group(0)) for m in found if m)


def _names_for(dir_fd: int, generation: int) -> list[str]:
    _checked(generation)
    return [name for number, _, name in _entries(dir_fd) if number == generation]


def _generation_name(dir_fd: int, generation: int) -> str | None:
    """The single file holding ``generation`` (None when absent; ValueError if duplicated)."""
    names = _names_for(dir_fd, generation)
    if len(names) > 1:
        raise ValueError(f"generation {generation} has more than one file")
    return names[0] if names else None


def _step(on_step: Hook | None, name: str) -> None:
    if on_step is not None:
        on_step(name)


def _temporary(dir_fd: int, name: str, data: bytes) -> str:
    """Write+fsync ``data`` to a fresh private ``.<name>.<pid>.<hex>.tmp``; return its name."""
    temporary = f".{name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    descriptor = storage.create_exclusive(dir_fd, temporary)
    try:
        storage.write_all(descriptor, data)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        storage.delete_file(dir_fd, temporary)
        raise
    os.close(descriptor)
    return temporary


def _discard(dir_fd: int, temporary: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        storage.delete_file(dir_fd, temporary)


def _existing(profile: str) -> int | None:
    """Read-only snapshots directory descriptor, or None when nothing was ever published."""
    try:
        return storage.open_profile_dir(profile, SNAPSHOT_DIR, create=False)
    except FileNotFoundError:
        return None


def stage_generation(profile: str, envelope: SnapshotEnvelope, on_step: Hook | None = None) -> None:
    """Write the generation file once; identical bytes already present is a no-op."""
    data = encode(envelope)
    decode(data)  # refuse to publish an envelope whose snapshot_id does not verify
    name = _file_name(envelope.generation, envelope.snapshot_id)
    dir_fd = storage.open_profile_dir(profile, SNAPSHOT_DIR)
    try:
        taken = _names_for(dir_fd, envelope.generation)
        if taken:
            if taken == [name] and storage.read_file(dir_fd, name, MAX_SNAPSHOT_BYTES) == data:
                return
            raise GenerationExists(f"generation {envelope.generation} already holds other bytes")
        temporary = _temporary(dir_fd, name, data)
        try:
            _step(on_step, "gen_tmp_written")
            try:
                os.link(
                    temporary, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd, follow_symlinks=False
                )
            except FileExistsError:
                if storage.read_file(dir_fd, name, MAX_SNAPSHOT_BYTES) != data:
                    raise GenerationExists(
                        f"generation {envelope.generation} already holds other bytes"
                    ) from None
            if _names_for(dir_fd, envelope.generation) != [name]:  # lost a race for the number
                os.unlink(name, dir_fd=dir_fd)
                raise GenerationExists(f"generation {envelope.generation} was staged concurrently")
        finally:
            _discard(dir_fd, temporary)
        storage.fsync_dir(dir_fd)
        _step(on_step, "gen_linked")
    finally:
        os.close(dir_fd)


def make_current(profile: str, generation: int, on_step: Hook | None = None) -> None:
    """Replace ``current.json`` with the exact verified bytes of ``generation``."""
    dir_fd = storage.open_profile_dir(profile, SNAPSHOT_DIR)
    try:
        name = _generation_name(dir_fd, generation)
        data = None if name is None else storage.read_file(dir_fd, name, MAX_SNAPSHOT_BYTES)
        if name is None or data is None:
            raise FileNotFoundError(f"generation {generation} is not staged")
        _verified(name, data, generation)
        temporary = _temporary(dir_fd, CURRENT_NAME, data)
        try:
            _step(on_step, "current_tmp_written")
            os.rename(temporary, CURRENT_NAME, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        except BaseException:
            _discard(dir_fd, temporary)
            raise
        _step(on_step, "current_renamed")
        storage.fsync_dir(dir_fd)
        _step(on_step, "published")
    finally:
        os.close(dir_fd)


def publish(profile: str, envelope: SnapshotEnvelope, on_step: Hook | None = None) -> None:
    """Stage the generation, then atomically make it current (never goes backwards)."""
    current = current_generation(profile)
    if current is not None and envelope.generation < current:
        raise ValueError(f"generation {envelope.generation} is older than current {current}")
    stage_generation(profile, envelope, on_step)
    make_current(profile, envelope.generation, on_step)


def _verified(name: str, data: bytes, generation: int) -> SnapshotEnvelope:
    """Decode ``data`` and insist it is what its file name says it is."""
    envelope = decode(data)
    match = _GEN_RE.fullmatch(name)
    if envelope.generation != generation or match is None or match.group(2) != envelope.snapshot_id:
        raise ValueError(f"{name} holds a different generation")
    return envelope


def generations(profile: str) -> list[int]:
    """Generation numbers of every generation file present, ascending."""
    dir_fd = _existing(profile)
    if dir_fd is None:
        return []
    try:
        return _generations(dir_fd)
    finally:
        os.close(dir_fd)


def _generations(dir_fd: int) -> list[int]:
    return sorted({number for number, _, _ in _entries(dir_fd)})


def _current_envelope(dir_fd: int) -> SnapshotEnvelope | None:
    """Verified current envelope; None when absent or unusable (damaged pointers are repairable)."""
    data = storage.read_file(dir_fd, CURRENT_NAME, MAX_SNAPSHOT_BYTES)
    if data is None:
        return None
    try:
        return decode(data)
    except ValueError:
        return None


def current_generation(profile: str) -> int | None:
    dir_fd = _existing(profile)
    if dir_fd is None:
        return None
    try:
        envelope = _current_envelope(dir_fd)
    finally:
        os.close(dir_fd)
    return None if envelope is None else envelope.generation


def current_generation_strict(profile: str) -> int | None:
    """Like ``current_generation`` but a damaged pointer raises ``ValueError``.

    Publication tolerates a damaged pointer because it rewrites it; a consumer that must not
    mistake damage for absence (export) uses this reader.
    """
    dir_fd = _existing(profile)
    if dir_fd is None:
        return None
    try:
        data = storage.read_file(dir_fd, CURRENT_NAME, MAX_SNAPSHOT_BYTES)
    finally:
        os.close(dir_fd)
    return None if data is None else decode(data).generation


def next_generation(profile: str) -> int:
    """One past the highest generation file or current pointer (1 when none)."""
    highest = max(generations(profile), default=0)
    highest = max(highest, current_generation(profile) or 0)
    if highest >= MAX_GENERATION:
        raise ValueError("snapshot generation space exhausted")
    return highest + 1


def read_generation(profile: str, generation: int) -> SnapshotEnvelope | None:
    """Verified envelope of ``generation``; None when absent; ValueError when invalid or damaged."""
    _checked(generation)
    dir_fd = _existing(profile)
    if dir_fd is None:
        return None
    try:
        name = _generation_name(dir_fd, generation)
        data = None if name is None else storage.read_file(dir_fd, name, MAX_SNAPSHOT_BYTES)
    finally:
        os.close(dir_fd)
    return None if name is None or data is None else _verified(name, data, generation)


def read_by_snapshot_id(profile: str, snapshot_id: str) -> SnapshotEnvelope | None:
    """Newest generation whose file carries ``snapshot_id`` (verified); None when absent."""
    if not isinstance(snapshot_id, str) or not _ID_RE.fullmatch(snapshot_id):
        raise ValueError("snapshot_id must be 16-64 lowercase hex characters")
    dir_fd = _existing(profile)
    if dir_fd is None:
        return None
    try:
        matches = [(n, name) for n, found, name in _entries(dir_fd) if found == snapshot_id]
        if not matches:
            return None
        number, name = matches[-1]
        data = storage.read_file(dir_fd, name, MAX_SNAPSHOT_BYTES)
    finally:
        os.close(dir_fd)
    return None if data is None else _verified(name, data, number)


def repair_current(profile: str, committed_generation: int) -> bool:
    """Make ``current.json`` the committed generation when it is older, missing or damaged."""
    dir_fd = _existing(profile)
    if dir_fd is None:
        raise FileNotFoundError(f"generation {committed_generation} is not staged")
    try:
        name = _generation_name(dir_fd, committed_generation)
        if name is None:
            raise FileNotFoundError(f"generation {committed_generation} is not staged")
        current = _current_envelope(dir_fd)
        if current is not None and current.generation > committed_generation:
            return False
        if current is not None and current.generation == committed_generation:
            stored = storage.read_file(dir_fd, name, MAX_SNAPSHOT_BYTES)
            if stored == storage.read_file(dir_fd, CURRENT_NAME, MAX_SNAPSHOT_BYTES):
                return False
    finally:
        os.close(dir_fd)
    make_current(profile, committed_generation)
    return True


def sweep_temporaries(profile: str) -> int:
    """Delete abandoned ``.*.tmp`` files; generation files and ``current.json`` are untouched."""
    dir_fd = _existing(profile)
    if dir_fd is None:
        return 0
    try:
        removed = sum(
            storage.delete_file(dir_fd, name)
            for name in storage.list_names(dir_fd)
            if _TEMP_RE.fullmatch(name)
        )
        if removed:
            storage.fsync_dir(dir_fd)
        return removed
    finally:
        os.close(dir_fd)


def discard_uncommitted(profile: str, committed_generation: int) -> list[int]:
    """Lease-holder recovery: drop generations staged after ``committed_generation``.

    Deletes generations above ``committed_generation``. A logical-ID pin may
    keep committed bytes, but not an uncommitted duplicate of verified committed
    bytes. A uniquely pinned uncommitted generation blocks recovery untouched.
    Never touches ``current.json`` or a generation ``<=`` the committed one. When
    ``current.json`` is already ahead of the checkpoint nothing is deleted and
    ``ValueError('current_ahead_of_checkpoint')`` is raised. Returns the deleted
    generation numbers, ascending; an absent snapshot directory yields ``[]``.
    With no committed checkpoint, every generation goes unless a pin blocks.
    """
    if committed_generation != 0:  # 0 means nothing is committed yet
        _checked(committed_generation)
    pinned = pinned_snapshot_ids(profile)
    dir_fd = _existing(profile)
    if dir_fd is None:
        return []
    try:
        current = _current_envelope(dir_fd)
        if current is not None and current.generation > committed_generation:
            raise ValueError("current_ahead_of_checkpoint")
        entries = _entries(dir_fd)
        future_pins = {
            identity
            for number, identity, _ in entries
            if number > committed_generation and identity in pinned
        }
        retained_pins = set()
        for number, identity, name in entries:
            if number <= committed_generation and identity in future_pins:
                data = storage.read_file(dir_fd, name, MAX_SNAPSHOT_BYTES)
                if data is not None:
                    _verified(name, data, number)
                    retained_pins.add(identity)
        if future_pins - retained_pins:
            raise ValueError("uncommitted_pinned_generation")
        doomed = [entry for entry in entries if entry[0] > committed_generation]
        removed = sum(storage.delete_file(dir_fd, name) for _, _, name in doomed)
        removed += sum(
            storage.delete_file(dir_fd, name)
            for name in storage.list_names(dir_fd)
            if _TEMP_RE.fullmatch(name)
        )
        if removed:
            storage.fsync_dir(dir_fd)
        return sorted({number for number, _, _ in doomed})
    finally:
        os.close(dir_fd)


def prune(profile: str, *, keep: int = 64, protect: Collection[int] = ()) -> list[int]:
    """Delete the oldest committed generations beyond the newest ``keep``.

    Committed means ``<=`` the current generation. The current generation,
    ``protect``-ed ones, files whose snapshot id is pinned by a delivery state
    (``pinned_snapshot_ids``) and every uncommitted newer orphan are never
    deleted; nothing is deleted while no usable ``current.json`` exists.
    Returns the deleted generation numbers, ascending.
    """
    if not _is_int(keep) or keep < 1:
        raise ValueError("keep must be a positive integer")
    protected = {_checked(number) for number in protect}
    pinned = pinned_snapshot_ids(profile)
    dir_fd = _existing(profile)
    if dir_fd is None:
        return []
    try:
        current = _current_envelope(dir_fd)
        if current is None:
            return []
        entries = [e for e in _entries(dir_fd) if e[0] <= current.generation]
        survivors = set(sorted({number for number, _, _ in entries})[-keep:]) | protected
        doomed = [e for e in entries if e[0] not in survivors and e[1] not in pinned]
        for _, _, name in doomed:
            storage.delete_file(dir_fd, name)
        if doomed:
            storage.fsync_dir(dir_fd)
        return sorted({number for number, _, _ in doomed})
    finally:
        os.close(dir_fd)


DELIVERY_DIR = "delivery"
MAX_DELIVERY_BYTES = 1024 * 1024


def pinned_snapshot_ids(profile: str) -> frozenset[str]:
    """Every ``last_acked_snapshot_id`` under ``delivery/*/*.json``; tolerant, read-only.

    A missing directory, an unreadable or garbage file, or a foreign shape pins
    nothing and never raises or creates anything.
    """
    pinned: set[str] = set()
    try:
        root = storage.open_profile_dir(profile, DELIVERY_DIR, create=False)
    except (OSError, ValueError):
        return frozenset()
    try:
        destinations = storage.list_names(root)
    except OSError:
        destinations = []
    finally:
        os.close(root)
    for destination in destinations:
        pinned.update(_pins_of(profile, destination))
    return frozenset(pinned)


def _pins_of(profile: str, destination: str) -> set[str]:
    found: set[str] = set()
    try:
        dir_fd = storage.open_profile_dir(profile, DELIVERY_DIR, destination, create=False)
    except (OSError, ValueError):
        return found
    try:
        for name in storage.list_names(dir_fd):
            if not name.endswith(".json") or name.startswith("."):
                continue
            try:
                raw = storage.read_file(dir_fd, name, MAX_DELIVERY_BYTES)
                state = None if raw is None else json.loads(raw)
            except (OSError, ValueError, RecursionError):
                continue
            acked = state.get("last_acked_snapshot_id") if isinstance(state, dict) else None
            if isinstance(acked, str) and 0 < len(acked) <= 128:
                found.add(acked)
    except OSError:
        pass
    finally:
        os.close(dir_fd)
    return found
