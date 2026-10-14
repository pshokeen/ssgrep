"""Frequency publication interruptions recover without changing namespaces or counted events."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.accounting import collector_frequency as frequency
from ssgrep.accounting.collector_types import CollectError, CollectOptions
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE
from tests.accounting.test_verify_frequency import CLOCK, FLOOR, NOW, corpus


@pytest.mark.parametrize("step", ["frequency_view_written", "frequency_pinned"])
def test_interrupted_first_pin_recovers_to_one_published_namespace(
    tmp_path: Path, step: str
) -> None:
    vs.enable()
    vs.start_capture()
    corpus(tmp_path, FLOOR + 5)

    def interrupt(current: str) -> None:
        if current == step:
            raise RuntimeError("interrupted frequency publication")

    with pytest.raises(CollectError) as failed:
        scene.collect(NOW, on_step=interrupt)
    assert failed.value.condition == "collection_failed"
    assert vs.checkpoint() is None
    pin = frequency.load_pinned(PROFILE)
    assert (pin is not None) == (step == "frequency_pinned")
    result = scene.collect(NOW)
    assert scene.report(result).generation == 1
    assert scene.counter_total(scene.current()) == 2
    current = frequency.load_pinned(PROFILE)
    assert current is not None and current.doc_count == FLOOR + 5
    assert len([key for key in scene.committed().blobs if key.startswith("ledger-")]) == 1
    scene.collect(NOW)
    assert frequency.load_pinned(PROFILE) == current
    assert scene.counter_total(scene.current()) == 2


def test_repin_of_unchanged_corpus_is_not_a_new_methodology_or_duplicate_ledger(
    tmp_path: Path,
) -> None:
    vs.enable()
    vs.start_capture()
    corpus(tmp_path, FLOOR + 5)
    scene.collect(NOW)
    before = scene.current()
    directory = vs.accounting_dir() / "collector"
    files = {path.name: path.read_bytes() for path in directory.glob("frequency*.json")}
    result = scene.collect(NOW, CollectOptions(repin_frequency=True))
    assert not any(warning.startswith("frequency_repinned:") for warning in result.warnings)
    assert scene.current().methodology_id == before.methodology_id
    assert scene.current().windows == before.windows
    assert {path.name: path.read_bytes() for path in directory.glob("frequency*.json")} == files
    assert len([key for key in scene.committed().blobs if key.startswith("ledger-")]) == 1
    # The same operation really does repin after retained evidence changes.
    vs.filler(vs.projects(tmp_path), "grown", CLOCK, 20)
    scene.collect(NOW)
    changed = scene.collect(NOW, CollectOptions(repin_frequency=True))
    assert any(warning.startswith("frequency_repinned:") for warning in changed.warnings)
    assert scene.current().methodology_id != before.methodology_id
