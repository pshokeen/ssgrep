"""Snapshot envelope, identity, immutable generations, current pointer, prune and sweep."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting import json_io, plainfile, service, snapshots
from ssgrep.accounting.collector_types import CollectError
from ssgrep.accounting.contracts import SCHEMA_VERSION
from ssgrep.accounting.ledger_contracts import SnapshotState
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.snapshot_builders import enable_policy
from tests.accounting.snapshots_driver import gen_name, make_envelope

PROFILE = "personal-local"
NOW = datetime(2026, 10, 12, 12, 1, 30, tzinfo=UTC)


def snap_dir(tmp_path: Path) -> Path:
    return tmp_path / "data" / "accounting" / PROFILE / "snapshots"


def names(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in snap_dir(tmp_path).iterdir())


def fields(**overrides: Any) -> dict[str, Any]:
    base = make_envelope(1)
    raw = {name: getattr(base, name) for name in snapshots._IDENTITY_FIELDS}
    raw.update(overrides)
    return raw


def kwargs(**overrides: Any) -> dict[str, Any]:
    base = make_envelope(1)
    raw = {
        name: getattr(base, name)
        for name in (
            "generation",
            "counter_epoch",
            "methodology_id",
            "policy_version",
            "collected_at",
            "coverage_through",
            "source_manifest_digest",
            "windows",
            "coverage",
            "observations",
            "estimates",
            "warnings",
        )
    }
    raw.update(overrides)
    return raw


# --- identity -----------------------------------------------------------------------------


def test_identity_ignores_collected_at_and_generation_but_keeps_content() -> None:
    first = snapshots.build_envelope(**kwargs(generation=3, collected_at="2026-10-12T12:00:00Z"))
    later = snapshots.build_envelope(**kwargs(generation=99, collected_at="2027-01-01T00:00:00Z"))
    assert first.snapshot_id == later.snapshot_id
    assert (first.generation, later.generation) == (3, 99)
    assert first.collected_at != later.collected_at
    assert len(first.snapshot_id) == 64 and first.schema_version == SCHEMA_VERSION


@pytest.mark.parametrize(
    "change",
    [
        {"windows": {"7d": {"seq": 2}}},
        {"coverage": {"partial": True}},
        {"observations": {"seq": 77}},
        {"estimates": {"representative_receipt": "rcpt-other"}},
        {"warnings": ("late_data",)},
        {"methodology_id": "m-test-v2"},
        {"policy_version": 2},
        {"counter_epoch": 2},
        {"coverage_through": "2026-10-12T11:00:00Z"},
        {"source_manifest_digest": "c" * 16},
    ],
    ids=lambda change: next(iter(change)),
)
def test_every_ledger_field_changes_the_identity(change: dict[str, Any]) -> None:
    baseline = snapshots.build_envelope(**kwargs())
    changed = snapshots.build_envelope(**kwargs(**change))
    assert changed.snapshot_id != baseline.snapshot_id


def test_identity_function_matches_built_envelope_and_is_key_order_independent() -> None:
    envelope = snapshots.build_envelope(**kwargs())
    assert snapshots.snapshot_identity(fields()) == envelope.snapshot_id
    reordered = dict(reversed(list(fields().items())))
    assert snapshots.snapshot_identity(reordered) == envelope.snapshot_id


def test_floats_are_rejected_as_value_error_everywhere() -> None:
    with pytest.raises(ValueError, match="canonical"):
        snapshots.build_envelope(**kwargs(observations={"ratio": 0.5}))
    with pytest.raises(ValueError, match="canonical"):
        snapshots.snapshot_identity(fields(windows={"7d": {"x": [1, 2.5]}}))
    sneaky = replace(make_envelope(1), estimates={"cost": 1.5})
    with pytest.raises(ValueError):
        snapshots.encode(sneaky)


def test_identity_requires_every_content_field() -> None:
    partial = fields()
    del partial["warnings"]
    with pytest.raises(ValueError, match="warnings"):
        snapshots.snapshot_identity(partial)


# --- encode / decode ----------------------------------------------------------------------


def test_encode_is_canonical_newline_terminated_and_roundtrips() -> None:
    envelope = make_envelope(5, bulk=10, warnings=("w1", "w2"))
    data = snapshots.encode(envelope)
    assert data.endswith(b"}\n") and b"\n" not in data[:-1]
    assert json.loads(data)["generation"] == 5
    assert snapshots.decode(data) == envelope
    assert snapshots.decode(data).warnings == ("w1", "w2")
    assert snapshots.encode(snapshots.decode(data)) == data


def _raw(**overrides: Any) -> dict[str, Any]:
    raw = json.loads(snapshots.encode(make_envelope(2)))
    raw.update(overrides)
    return raw


def _bytes(raw: dict[str, Any]) -> bytes:
    return json.dumps(raw).encode()


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": SCHEMA_VERSION + 1},
        {"schema_version": "1"},
        {"schema_version": True},
        {"generation": "2"},
        {"generation": 0},
        {"generation": True},
        {"counter_epoch": None},
        {"policy_version": 1.5},
        {"methodology_id": 7},
        {"collected_at": None},
        {"windows": []},
        {"windows": {"7d": "text"}},
        {"coverage": "x"},
        {"warnings": "late"},
        {"warnings": [1]},
        {"snapshot_id": "0" * 64},
        {"snapshot_id": 5},
        {"surprise": 1},
    ],
    ids=lambda m: "-".join(f"{k}={v!r}" for k, v in m.items()),
)
def test_decode_rejects_invalid_documents(mutation: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        snapshots.decode(_bytes(_raw(**mutation)))


@pytest.mark.parametrize(
    "missing", sorted(snapshots._IDENTITY_FIELDS) + ["snapshot_id", "generation", "collected_at"]
)
def test_decode_rejects_each_missing_field(missing: str) -> None:
    raw = _raw()
    del raw[missing]
    with pytest.raises(ValueError):
        snapshots.decode(_bytes(raw))


@pytest.mark.parametrize("garbage", [b"", b"[]", b"null", b"{", b"\xff\xfe", b'{"a": NaN}'])
def test_decode_rejects_non_documents(garbage: bytes) -> None:
    with pytest.raises(ValueError):
        snapshots.decode(garbage)


def test_decode_rejects_a_float_anywhere() -> None:
    text = snapshots.encode(make_envelope(2)).decode().replace('"seq":2', '"seq":2.0', 1)
    assert '"seq":2.0' in text
    with pytest.raises(ValueError):
        snapshots.decode(text.encode())


def test_tamper_flipping_a_byte_inside_windows_is_detected() -> None:
    data = snapshots.encode(make_envelope(2, bulk=64))
    index = data.index(b"xxxx")
    tampered = data[:index] + b"y" + data[index + 1 :]
    assert tampered != data and json.loads(tampered)  # still valid JSON of the right shape
    with pytest.raises(ValueError, match="snapshot_id"):
        snapshots.decode(tampered)


def test_generation_and_collected_at_edits_stay_valid_by_design() -> None:
    raw = _raw(generation=40, collected_at="2030-01-01T00:00:00Z")
    decoded = snapshots.decode(_bytes(raw))
    assert (decoded.generation, decoded.collected_at) == (40, "2030-01-01T00:00:00Z")


# --- stage / make_current / publish ---------------------------------------------------------


def test_publish_writes_generation_and_identical_current_with_private_modes(
    tmp_path: Path,
) -> None:
    envelope = make_envelope(1)
    snapshots.publish(PROFILE, envelope)
    expected = snapshots.encode(envelope)
    directory = snap_dir(tmp_path)
    assert names(tmp_path) == ["current.json", gen_name(1)]
    assert (directory / "current.json").read_bytes() == expected
    assert (directory / gen_name(1)).read_bytes() == expected
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for entry in directory.iterdir():
        assert stat.S_IMODE(entry.stat().st_mode) == 0o600
    assert snapshots.read_generation(PROFILE, 1) == envelope
    assert snapshots.current_generation(PROFILE) == 1


def test_stage_identical_bytes_is_a_noop_but_different_bytes_raise(tmp_path: Path) -> None:
    envelope = make_envelope(1)
    snapshots.stage_generation(PROFILE, envelope)
    path = snap_dir(tmp_path) / gen_name(1)
    before = (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
    snapshots.stage_generation(PROFILE, envelope)
    assert (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) == before
    assert names(tmp_path) == [gen_name(1)]  # no leftover temporary
    different = make_envelope(1, observations={"seq": 999})
    with pytest.raises(snapshots.GenerationExists):
        snapshots.stage_generation(PROFILE, different)
    assert path.read_bytes() == before[0]
    assert names(tmp_path) == [gen_name(1)]
    assert issubclass(snapshots.GenerationExists, ValueError)


def test_a_staged_generation_is_not_current_until_made_current(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    snapshots.stage_generation(PROFILE, make_envelope(2))
    assert snapshots.current_generation(PROFILE) == 1
    assert snapshots.next_generation(PROFILE) == 3
    snapshots.make_current(PROFILE, 2)
    assert (snap_dir(tmp_path) / "current.json").read_bytes() == snapshots.encode(make_envelope(2))


def test_make_current_refuses_missing_and_mislabelled_generations(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    with pytest.raises(FileNotFoundError):
        snapshots.make_current(PROFILE, 2)
    mislabelled = snap_dir(tmp_path) / f"gen-{9:010d}-{make_envelope(3).snapshot_id}.json"
    mislabelled.write_bytes(snapshots.encode(make_envelope(3)))  # a gen-9 name holding gen 3
    mislabelled.chmod(0o600)
    with pytest.raises(ValueError, match="different generation"):
        snapshots.make_current(PROFILE, 9)
    with pytest.raises(ValueError, match="different generation"):
        snapshots.read_generation(PROFILE, 9)
    assert snapshots.current_generation(PROFILE) == 1


def test_stage_refuses_an_envelope_whose_identity_does_not_verify(tmp_path: Path) -> None:
    forged = replace(make_envelope(1), snapshot_id="f" * 64)
    with pytest.raises(ValueError, match="snapshot_id"):
        snapshots.stage_generation(PROFILE, forged)
    assert not snap_dir(tmp_path).exists() or names(tmp_path) == []


def test_publish_refuses_to_go_backwards(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(5))
    with pytest.raises(ValueError, match="older"):
        snapshots.publish(PROFILE, make_envelope(4))
    assert snapshots.current_generation(PROFILE) == 5
    assert gen_name(4) not in names(tmp_path)


def test_republishing_the_current_generation_is_idempotent(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    snapshots.publish(PROFILE, make_envelope(1))
    assert names(tmp_path) == ["current.json", gen_name(1)]


def test_oversize_snapshot_is_rejected_before_any_write(tmp_path: Path) -> None:
    assert snapshots.MAX_SNAPSHOT_BYTES == json_io.MAX_SNAPSHOT_BYTES
    huge = make_envelope(1, bulk=json_io.MAX_SNAPSHOT_BYTES)
    with pytest.raises(ValueError, match="exceeds"):
        snapshots.encode(huge)
    with pytest.raises(ValueError):
        snapshots.publish(PROFILE, huge)
    assert not snap_dir(tmp_path).exists()


# --- generation numbering ---------------------------------------------------------------------


def test_numbering_starts_at_one_and_never_reuses_numbers(tmp_path: Path) -> None:
    assert snapshots.next_generation(PROFILE) == 1
    assert snapshots.generations(PROFILE) == [] and snapshots.current_generation(PROFILE) is None
    assert not (tmp_path / "data").exists(), "reads must never create storage"
    for number in (1, 2, 3):
        assert snapshots.next_generation(PROFILE) == number
        snapshots.publish(PROFILE, make_envelope(number))
    assert snapshots.generations(PROFILE) == [1, 2, 3]
    (snap_dir(tmp_path) / "current.json").unlink()
    assert snapshots.next_generation(PROFILE) == 4, "missing pointer must not recycle numbers"


def test_numbering_follows_current_when_it_is_ahead_of_every_file(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(2))
    (snap_dir(tmp_path) / gen_name(2)).unlink()
    assert snapshots.generations(PROFILE) == []
    assert snapshots.next_generation(PROFILE) == 3


def test_numbering_counts_uncommitted_orphans_and_ignores_unrelated_names(
    tmp_path: Path,
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    snapshots.stage_generation(PROFILE, make_envelope(7))
    sid = make_envelope(1).snapshot_id
    for noise in (
        "gen-1.json",
        "gen-000000000099.json",  # the retired 12-digit layout
        f"gen-{99:010d}-{sid}.json.bak",
        f"gen-{98:010d}-{sid.upper()}.json",
        f"gen-{97:010d}-short.json",
        "notes.txt",
        "gen-000000000a-" + sid + ".json",
    ):
        (snap_dir(tmp_path) / noise).write_text("x")
    assert snapshots.generations(PROFILE) == [1, 7]
    assert snapshots.next_generation(PROFILE) == 8


def test_damaged_current_does_not_stop_numbering_or_repair(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    snapshots.publish(PROFILE, make_envelope(2))
    (snap_dir(tmp_path) / "current.json").write_bytes(b'{"half": ')
    assert snapshots.current_generation(PROFILE) is None
    assert snapshots.next_generation(PROFILE) == 3
    assert snapshots.repair_current(PROFILE, 2) is True
    assert snapshots.current_generation(PROFILE) == 2


# --- read_generation ---------------------------------------------------------------------------


def test_read_generation_missing_is_none_and_hostile_numbers_never_touch_files(
    tmp_path: Path,
) -> None:
    assert snapshots.read_generation(PROFILE, 1) is None  # nothing published, nothing created
    assert not (tmp_path / "data").exists()
    snapshots.publish(PROFILE, make_envelope(1))
    assert snapshots.read_generation(PROFILE, 2) is None
    secret = tmp_path / "data" / "accounting" / PROFILE / "secret.json"
    secret.write_text("TOP-SECRET")
    for hostile in (-1, 0, 10**10, 10**12, 10**30, "../secret", "1/../../x", 1.5, True, None):
        with pytest.raises(ValueError):
            snapshots.read_generation(PROFILE, hostile)  # ty: ignore[invalid-argument-type]
    assert names(tmp_path) == ["current.json", gen_name(1)]


def test_read_generation_rejects_a_damaged_file(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    (snap_dir(tmp_path) / gen_name(1)).write_bytes(b"{}")
    with pytest.raises(ValueError):
        snapshots.read_generation(PROFILE, 1)


def test_symlinked_generation_is_refused(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    target = tmp_path / "elsewhere.json"
    target.write_bytes(snapshots.encode(make_envelope(2)))
    os.symlink(target, snap_dir(tmp_path) / gen_name(2))
    with pytest.raises(OSError):
        snapshots.read_generation(PROFILE, 2)
    with pytest.raises(OSError):
        snapshots.make_current(PROFILE, 2)


# --- repair_current ----------------------------------------------------------------------------


def test_repair_current_swaps_an_older_pointer_once(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    snapshots.stage_generation(PROFILE, make_envelope(2))
    assert snapshots.repair_current(PROFILE, 2) is True
    assert (snap_dir(tmp_path) / "current.json").read_bytes() == snapshots.encode(make_envelope(2))
    assert snapshots.repair_current(PROFILE, 2) is False
    assert snapshots.repair_current(PROFILE, 1) is False, "never moves the pointer backwards"
    assert snapshots.current_generation(PROFILE) == 2


def test_repair_current_recreates_a_missing_pointer_and_rejects_unknown_generation() -> None:
    snapshots.stage_generation(PROFILE, make_envelope(1))
    assert snapshots.current_generation(PROFILE) is None
    assert snapshots.repair_current(PROFILE, 1) is True
    with pytest.raises(FileNotFoundError):
        snapshots.repair_current(PROFILE, 5)


def test_repair_current_replaces_a_same_generation_pointer_with_different_bytes(
    tmp_path: Path,
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    pointer = snap_dir(tmp_path) / "current.json"
    other = snapshots.encode(make_envelope(1, observations={"seq": 5}))
    pointer.write_bytes(other)  # same generation number, other content: not the committed file
    assert snapshots.repair_current(PROFILE, 1) is True
    assert pointer.read_bytes() == snapshots.encode(make_envelope(1))


def test_repair_current_without_any_storage_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        snapshots.repair_current(PROFILE, 1)
    assert not (tmp_path / "data").exists()


# --- sweep -------------------------------------------------------------------------------------


def test_sweep_removes_only_abandoned_temporaries(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    directory = snap_dir(tmp_path)
    leftovers = [".gen-000000000002.json.1.ab12.tmp", ".current.json.9.cd34.tmp"]
    for name in leftovers:
        (directory / name).write_bytes(b'{"partial')
        (directory / name).chmod(0o600)
    keep = {"notes.txt": b"n", ".hidden": b"h", "tmp": b"t", "x.tmp": b"y"}
    for name, content in keep.items():
        (directory / name).write_bytes(content)
        (directory / name).chmod(0o600)
    before_current = (directory / "current.json").read_bytes()
    assert snapshots.sweep_temporaries(PROFILE) == 2
    assert names(tmp_path) == sorted(["current.json", gen_name(1), *keep])
    assert (directory / "current.json").read_bytes() == before_current
    assert snapshots.sweep_temporaries(PROFILE) == 0
    assert snapshots.sweep_temporaries("never-written") == 0
    assert not (tmp_path / "data" / "accounting" / "never-written").exists()


# --- prune -------------------------------------------------------------------------------------


def seed(count: int) -> None:
    for number in range(1, count + 1):
        snapshots.publish(PROFILE, make_envelope(number))


def test_prune_deletes_only_the_oldest_beyond_keep(tmp_path: Path) -> None:
    seed(6)
    assert snapshots.prune(PROFILE, keep=2) == [1, 2, 3, 4]
    assert snapshots.generations(PROFILE) == [5, 6]
    assert "current.json" in names(tmp_path)
    assert snapshots.next_generation(PROFILE) == 7
    assert snapshots.prune(PROFILE, keep=2) == []


def test_prune_never_deletes_protected_or_current(tmp_path: Path) -> None:
    seed(6)
    assert snapshots.prune(PROFILE, keep=1, protect={2, 4}) == [1, 3, 5]
    assert snapshots.generations(PROFILE) == [2, 4, 6]
    assert snapshots.current_generation(PROFILE) == 6
    assert snapshots.read_generation(PROFILE, 6) is not None


def test_prune_never_deletes_uncommitted_orphans_newer_than_current() -> None:
    seed(4)
    snapshots.stage_generation(PROFILE, make_envelope(5))
    snapshots.stage_generation(PROFILE, make_envelope(6))
    assert snapshots.prune(PROFILE, keep=1) == [1, 2, 3]
    assert snapshots.generations(PROFILE) == [4, 5, 6]
    assert snapshots.current_generation(PROFILE) == 4


def test_prune_with_current_behind_the_newest_keeps_the_pointer_target() -> None:
    seed(3)
    snapshots.stage_generation(PROFILE, make_envelope(4))
    assert snapshots.prune(PROFILE, keep=1) == [1, 2]
    assert snapshots.generations(PROFILE) == [3, 4]
    snapshots.make_current(PROFILE, 4)
    assert snapshots.current_generation(PROFILE) == 4


def test_prune_does_nothing_without_a_usable_current_pointer(tmp_path: Path) -> None:
    seed(3)
    (snap_dir(tmp_path) / "current.json").unlink()
    assert snapshots.prune(PROFILE, keep=1) == []
    assert snapshots.generations(PROFILE) == [1, 2, 3]
    assert snapshots.prune("never-written", keep=1) == []


@pytest.mark.parametrize("keep", [0, -1, True, 1.5])
def test_prune_rejects_nonsense_keep_and_protect(keep: Any) -> None:
    seed(2)
    with pytest.raises(ValueError):
        snapshots.prune(PROFILE, keep=keep)
    with pytest.raises(ValueError):
        snapshots.prune(PROFILE, keep=1, protect={-5})
    assert snapshots.generations(PROFILE) == [1, 2]


# --- compatibility with the read facade ---------------------------------------------


def test_published_files_are_readable_by_the_summary_without_service_changes() -> None:
    enable_policy()
    envelope = make_envelope(1, collected_at="2026-10-12T12:00:00Z")
    snapshots.publish(PROFILE, envelope)
    second = make_envelope(2, collected_at="2026-10-12T12:01:00Z")
    snapshots.publish(PROFILE, second)
    view = service.read_summary(PROFILE, now=NOW)
    assert view.state is SnapshotState.FRESH
    assert view.generation == 2 and view.snapshot_id == second.snapshot_id
    assert view.has_values
    assert dict(view.measured)["searches"].value == 100
    read, failure, why = service.read_snapshot(PROFILE)
    assert (failure, why) == (None, None) and read == second


# --- file layout, lookup by snapshot id, delivery pins -------------------------------------------


def same_as_first(generation: int) -> Any:
    first = make_envelope(1)
    return make_envelope(generation, windows=first.windows, observations=first.observations)


def delivery_file(tmp_path: Path, destination: str, signal: str, content: bytes | str) -> Path:
    plainfile.write_private(
        tmp_path / "data",
        "accounting",
        PROFILE,
        "delivery",
        destination,
        f"{signal}.json",
        data=content if isinstance(content, bytes) else content.encode(),
    )
    return tmp_path / "data" / "accounting" / PROFILE / "delivery" / destination / f"{signal}.json"


def acked(snapshot_id: str | None, **extra: Any) -> str:
    return json.dumps({"destination_id": "d", "last_acked_snapshot_id": snapshot_id, **extra})


def test_generation_file_name_is_ten_digit_generation_then_snapshot_id(tmp_path: Path) -> None:
    envelope = make_envelope(37)
    snapshots.publish(PROFILE, envelope)
    assert f"gen-0000000037-{envelope.snapshot_id}.json" in names(tmp_path)
    assert gen_name(37) == f"gen-0000000037-{envelope.snapshot_id}.json"


def test_two_files_for_one_generation_number_never_exist(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    other = make_envelope(1, observations={"seq": 424242})  # same number, other content/id
    assert other.snapshot_id != make_envelope(1).snapshot_id
    with pytest.raises(snapshots.GenerationExists):
        snapshots.stage_generation(PROFILE, other)
    with pytest.raises(snapshots.GenerationExists):
        snapshots.publish(PROFILE, other)
    assert [n for n in names(tmp_path) if n.startswith("gen-")] == [gen_name(1)]
    assert snapshots.read_generation(PROFILE, 1) == make_envelope(1)


def test_duplicate_files_for_one_number_are_reported_not_guessed(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    twin = snap_dir(tmp_path) / f"gen-{1:010d}-{'a' * 16}.json"
    twin.write_bytes(snapshots.encode(make_envelope(1)))
    twin.chmod(0o600)
    with pytest.raises(ValueError, match="more than one file"):
        snapshots.read_generation(PROFILE, 1)
    with pytest.raises(ValueError, match="more than one file"):
        snapshots.make_current(PROFILE, 1)
    with pytest.raises(snapshots.GenerationExists):
        snapshots.stage_generation(PROFILE, make_envelope(1))
    assert snapshots.generations(PROFILE) == [1]


def test_a_file_whose_name_id_differs_from_its_content_is_refused(tmp_path: Path) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    (snap_dir(tmp_path) / gen_name(1)).rename(snap_dir(tmp_path) / f"gen-{1:010d}-{'c' * 32}.json")
    with pytest.raises(ValueError, match="different generation"):
        snapshots.read_generation(PROFILE, 1)
    with pytest.raises(ValueError):
        snapshots.read_by_snapshot_id(PROFILE, "c" * 32)


def test_read_by_snapshot_id_returns_the_newest_generation_with_that_content(
    tmp_path: Path,
) -> None:
    for number in (1, 2):
        snapshots.publish(PROFILE, make_envelope(number))
    same_content = same_as_first(3)  # generation 3 holding generation 1's content
    assert same_content.snapshot_id == make_envelope(1).snapshot_id
    snapshots.publish(PROFILE, same_content)
    found = snapshots.read_by_snapshot_id(PROFILE, make_envelope(1).snapshot_id)
    assert found == same_content and found is not None and found.generation == 3
    assert snapshots.read_by_snapshot_id(PROFILE, make_envelope(2).snapshot_id) == make_envelope(2)
    assert snapshots.read_by_snapshot_id(PROFILE, "0" * 64) is None


def test_read_by_snapshot_id_without_storage_creates_nothing(tmp_path: Path) -> None:
    assert snapshots.read_by_snapshot_id(PROFILE, "a" * 16) is None
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "hostile",
    [
        "",
        "a" * 15,
        "a" * 65,
        "A" * 32,
        "../current",
        "../../etc/passwd",
        "a" * 16 + "/../x",
        "a" * 16 + "\n",
        "a" * 16 + ".json",
        "g" * 32,
        None,
        12345678901234567,
    ],
)
def test_read_by_snapshot_id_rejects_ids_that_could_escape(tmp_path: Path, hostile: Any) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    with pytest.raises(ValueError):
        snapshots.read_by_snapshot_id(PROFILE, hostile)
    assert names(tmp_path) == ["current.json", gen_name(1)]


def test_pinned_ids_collect_every_acked_snapshot_id_across_destinations_and_signals(
    tmp_path: Path,
) -> None:
    delivery_file(tmp_path, "ops", "metrics", acked("a" * 32))
    delivery_file(tmp_path, "ops", "logs", acked("b" * 32, pending_batches=2))
    delivery_file(tmp_path, "lake", "metrics", acked("a" * 32))
    delivery_file(tmp_path, "lake", "events", acked(None))
    assert snapshots.pinned_snapshot_ids(PROFILE) == frozenset({"a" * 32, "b" * 32})


def test_pinned_ids_ignore_garbage_and_never_create_or_raise(tmp_path: Path) -> None:
    assert snapshots.pinned_snapshot_ids(PROFILE) == frozenset()
    assert not (tmp_path / "data").exists()
    delivery_file(tmp_path, "d1", "metrics", b"\xff\xfe not json")
    delivery_file(tmp_path, "d1", "logs", "[1, 2]")
    delivery_file(tmp_path, "d1", "events", '{"last_acked_snapshot_id": 7}')
    delivery_file(tmp_path, "d2", "metrics", '{"last_acked_snapshot_id": ["x"]}')
    delivery_file(tmp_path, "d2", "logs", '{"last_acked_snapshot_id": 1.5}')
    (tmp_path / "data" / "accounting" / PROFILE / "delivery" / "d2" / "spool.txt").write_text(
        acked("d" * 32)
    )  # not a .json state file
    delivery_file(tmp_path, "d3", "metrics", acked("e" * 32))  # the one good state
    stray = tmp_path / "data" / "accounting" / PROFILE / "delivery" / "stray.json"
    stray.write_text(acked("f" * 32))  # a file where a destination directory belongs
    before = sorted(str(p) for p in (tmp_path / "data").rglob("*"))
    assert snapshots.pinned_snapshot_ids(PROFILE) == frozenset({"e" * 32})
    assert sorted(str(p) for p in (tmp_path / "data").rglob("*")) == before


def test_pinned_ids_survive_a_symlinked_delivery_directory(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "metrics.json").write_text(acked("a" * 32))
    profile_dir = tmp_path / "data" / "accounting" / PROFILE
    profile_dir.mkdir(parents=True)
    os.symlink(real, profile_dir / "delivery")
    assert snapshots.pinned_snapshot_ids(PROFILE) == frozenset()


def test_prune_never_deletes_a_pinned_generation_even_beyond_keep(tmp_path: Path) -> None:
    seed(6)
    pinned_id = make_envelope(2).snapshot_id
    delivery_file(tmp_path, "ops", "metrics", acked(pinned_id))
    assert snapshots.prune(PROFILE, keep=1) == [1, 3, 4, 5]
    assert snapshots.generations(PROFILE) == [2, 6]
    assert snapshots.read_by_snapshot_id(PROFILE, pinned_id) == make_envelope(2)
    # once the destination acknowledges something newer, the old pin is released
    delivery_file(tmp_path, "ops", "metrics", acked(make_envelope(6).snapshot_id))
    assert snapshots.prune(PROFILE, keep=1) == [2]
    assert snapshots.generations(PROFILE) == [6]


def test_prune_pin_applies_by_snapshot_id_to_every_generation_with_that_content(
    tmp_path: Path,
) -> None:
    snapshots.publish(PROFILE, make_envelope(1))
    snapshots.publish(PROFILE, make_envelope(2))
    snapshots.publish(PROFILE, same_as_first(3))  # content of gen 1
    snapshots.publish(PROFILE, make_envelope(4))
    snapshots.publish(PROFILE, make_envelope(5))
    # generation 3 shares generation 1's id: pinning it keeps both generations
    delivery_file(tmp_path, "ops", "metrics", acked(make_envelope(1).snapshot_id))
    assert snapshots.prune(PROFILE, keep=1) == [2, 4]
    assert snapshots.generations(PROFILE) == [1, 3, 5]


def test_prune_with_garbage_delivery_state_deletes_as_if_nothing_were_pinned(
    tmp_path: Path,
) -> None:
    seed(4)
    delivery_file(tmp_path, "ops", "metrics", b"{not json")
    assert snapshots.prune(PROFILE, keep=1) == [1, 2, 3]
    assert snapshots.generations(PROFILE) == [4]


def test_newer_snapshot_schema_is_not_repaired_or_overwritten(tmp_path: Path) -> None:
    collect_at = scene.at(hour=20)
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
    assert scene.report(scene.collect(collect_at)).generation == 1
    assert scene.counter_total(scene.current()) == 2
    path = vs.accounting_dir() / "snapshots" / "current.json"
    original = path.read_bytes()
    newer = json.loads(original)
    newer["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(newer))
    before = path.read_bytes()
    with pytest.raises(CollectError) as failed:
        scene.collect(collect_at)
    assert failed.value.condition == "unsupported_schema"
    assert path.read_bytes() == before and scene.committed().generation == 1
    path.write_bytes(original)
    assert scene.report(scene.collect(collect_at)).generation == 2
