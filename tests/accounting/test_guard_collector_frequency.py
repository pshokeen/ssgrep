"""A corrupt pin is not an invitation to silently select a different methodology."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ssgrep.accounting import collector_frequency as frequency
from ssgrep.accounting.collector_types import CollectError, CollectOptions
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE

NOW = scene.at(hour=20)


@pytest.mark.parametrize(
    "damage",
    [
        "pointer-json",
        "pointer-key",
        "pointer-id",
        "missing-view",
        "view-json",
        "view-name",
        "view-digest",
    ],
)
def test_corrupt_pin_keeps_the_last_generation_and_namespace(tmp_path: Path, damage: str) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, scene.at(hour=9))
    scene.collect(NOW)
    old = scene.current()
    directory = vs.accounting_dir() / "collector"
    pointer = directory / frequency.POINTER
    view = frequency.load_pinned(PROFILE)
    assert view is not None
    path = directory / frequency.view_file_name(view.version_id)
    original = {pointer: pointer.read_bytes(), path: path.read_bytes()}
    if damage.startswith("pointer-"):
        broken = {
            "pointer-json": b"{",
            "pointer-key": b"{}",
            "pointer-id": b'{"view_id":"../other"}',
        }
        pointer.write_bytes(broken[damage])
    elif damage == "missing-view":
        path.unlink()
    elif damage == "view-json":
        path.write_bytes(b"[]")
    else:
        raw = json.loads(path.read_bytes())
        if damage == "view-name":
            raw["version_id"] = "freq-" + "0" * 24
        else:
            raw["doc_count"] += 1
        path.write_text(json.dumps(raw))
    corrupted = {p.name: p.read_bytes() for p in directory.glob("frequency*.json")}
    with pytest.raises(CollectError) as failed:
        scene.collect(NOW, CollectOptions(repin_frequency=True))
    assert failed.value.condition == "state_corrupt"
    assert scene.current() == old
    assert scene.committed().methodology == old.methodology_id
    assert {p.name: p.read_bytes() for p in directory.glob("frequency*.json")} == corrupted
    for file, content in original.items():
        file.write_bytes(content)
        file.chmod(0o600)
    scene.collect(NOW)
    assert scene.current().methodology_id == old.methodology_id
    assert scene.counter_total(scene.current()) == scene.counter_total(old)


def test_frequency_size_limits_fail_closed_without_losing_the_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vs.enable(pin=True)
    view = frequency.load_pinned(PROFILE)
    assert view is not None
    directory = vs.accounting_dir() / "collector"
    for name, path in (
        ("MAX_POINTER_BYTES", directory / frequency.POINTER),
        ("MAX_VIEW_BYTES", directory / frequency.view_file_name(view.version_id)),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(frequency, name, path.stat().st_size - 1)
            with pytest.raises(CollectError) as failed:
                frequency.load_pinned(PROFILE)
            assert failed.value.condition == "state_corrupt"
        assert frequency.load_pinned(PROFILE) == view


def test_repin_cannot_overwrite_a_damaged_content_addressed_view() -> None:
    first = vs.pinned_view(1)
    second = vs.pinned_view(2)
    frequency.pin(PROFILE, first)
    directory = vs.accounting_dir() / "collector"
    pointer = (directory / frequency.POINTER).read_bytes()
    target = directory / frequency.view_file_name(second.version_id)
    target.write_bytes(b"damaged but already named")
    target.chmod(0o600)
    with pytest.raises(CollectError) as failed:
        frequency.pin(PROFILE, second)
    assert failed.value.condition == "state_corrupt"
    assert (directory / frequency.POINTER).read_bytes() == pointer
    assert target.read_bytes() == b"damaged but already named"
    target.unlink()
    frequency.pin(PROFILE, second)
    assert frequency.load_pinned(PROFILE) == second


@pytest.mark.parametrize("damage", ["basis", "empty", "identity"])
def test_invalid_candidate_view_does_not_replace_a_valid_pin(damage: str) -> None:
    valid = vs.pinned_view()
    frequency.pin(PROFILE, valid)
    invalid = {
        "basis": replace(valid, basis="words"),
        "empty": replace(valid, doc_count=0),
        "identity": replace(valid, version_id="freq-" + "0" * 24),
    }[damage]
    with pytest.raises(ValueError):
        frequency.pin(PROFILE, invalid)
    assert frequency.load_pinned(PROFILE) == valid
