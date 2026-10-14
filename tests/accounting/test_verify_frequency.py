"""The pinned candidate-frequency view: unknown below the floor, pinned at 200, stable, repinnable.

Documents are real prompts in real transcripts counted by the extractor's own normalization
(``reduce_io.corpus_view``); nothing here builds a view by hand except where a view is pinned
ahead of time on purpose. A repin is judged on methodology identity, retained namespaces and the
snapshot, never on a summed total.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ssgrep.accounting import collector_frequency as frequency
from ssgrep.accounting.collector_types import CollectOptions
from ssgrep.accounting.frequency import FrequencyView
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import PROFILE

CLOCK = scene.at(hour=9)
NOW = scene.at(hour=20)
FLOOR = frequency.FREQUENCY_MIN_DOCS


def lifetime_credit(envelope: Any) -> dict[str, dict[str, Any]]:
    """Central credit quantity per estimate unit for the lifetime window."""
    out = {}
    for unit, scenarios in scene.window(envelope, "lifetime")["estimates"].items():
        out[unit] = scenarios["central"]["credit"]
    return out


def corpus(tmp_path: Path, documents: int) -> None:
    """The scene's six real documents plus filler so the retained corpus has ``documents``."""
    vs.pair(vs.projects(tmp_path), 0, CLOCK)
    have = 5  # the pair's counted texts: two prompts and three tool results
    vs.filler(vs.projects(tmp_path), "fill", CLOCK, max(documents - have, 0))


def pinned_view() -> FrequencyView:
    """The pinned frequency view; a missing pin fails the test with a clear message."""
    view = frequency.load_pinned(PROFILE)
    assert view is not None, "no frequency view is pinned"
    return view


def pinned_files() -> list[str]:
    folder = vs.accounting_dir() / "collector"
    return sorted(p.name for p in folder.glob("frequency*.json"))


def test_below_the_floor_the_view_is_unknown_with_a_reason_and_credit_is_not_zero(
    tmp_path: Path,
) -> None:
    vs.enable(pin=False)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)  # a perfectly creditable scene, tiny corpus
    result = scene.collect(NOW)
    assert result.state == "published"
    envelope = scene.current()
    block = scene.coverage(envelope)["frequency"]
    assert block["state"] == "insufficient_corpus" and block["floor"] == FLOOR
    assert block["reason"] == "frequency_insufficient_corpus"
    assert 0 < block["documents"] < FLOOR  # the real, counted number is stated
    assert "view_id" not in block
    assert any(w.startswith("frequency_insufficient_corpus") for w in envelope.warnings)
    assert any(w.startswith("frequency_insufficient_corpus") for w in result.warnings)
    assert pinned_files() == []  # nothing was pinned
    # no candidate earned credit, and the snapshot does NOT present that as a measured zero
    estimates = scene.window(envelope, "lifetime")["estimates"]
    for scenarios in estimates.values():
        for scenario in scenarios.values():
            assert scenario["credit"]["value"] in (None, 0)
    assert scene.coverage(envelope)["candidate_receipts"] == 0
    assert (
        scene.coverage(envelope)["exclusions"].get("rarity_unknown", 0) >= 1
    )  # unknown, with its reason
    assert scene.coverage(envelope)["partial"] is True


def test_exactly_199_documents_stay_unknown_and_exactly_200_pin(tmp_path: Path) -> None:
    vs.enable(pin=False)
    vs.start_capture()
    corpus(tmp_path, FLOOR - 1)
    scene.collect(NOW)
    block = scene.coverage(scene.current())["frequency"]
    assert (block["state"], block["documents"]) == ("insufficient_corpus", FLOOR - 1)
    assert pinned_files() == []

    vs.filler(vs.projects(tmp_path), "one-more", CLOCK, 1)  # the 200th document
    scene.collect(NOW)
    block = scene.coverage(scene.current())["frequency"]
    assert (block["state"], block["documents"]) == ("pinned", FLOOR)
    assert block["view_id"].startswith("freq-") and "reason" not in block
    assert len(pinned_files()) == 2  # the immutable view file and its pointer
    assert not any(w.startswith("frequency_insufficient") for w in scene.current().warnings)


def test_the_pinned_view_is_stable_while_the_corpus_keeps_growing(tmp_path: Path) -> None:
    vs.enable(pin=False)
    vs.start_capture()
    corpus(tmp_path, FLOOR + 20)
    first = scene.collect(NOW)
    assert any(w.startswith("frequency_pinned") for w in first.warnings)
    pinned = scene.coverage(scene.current())["frequency"]
    assert pinned["state"] == "pinned"
    view_bytes = {
        p.name: p.read_bytes() for p in (vs.accounting_dir() / "collector").glob("frequency*.json")
    }
    methodology = scene.current().methodology_id
    ledger = vs.ledger_digest()

    for batch in range(3):
        vs.filler(vs.projects(tmp_path), f"grow{batch}", CLOCK, 150)  # corpus keeps growing
        result = scene.collect(NOW)
        envelope = scene.current()
        assert scene.coverage(envelope)["frequency"]["view_id"] == pinned["view_id"]  # unchanged
        assert (
            scene.coverage(envelope)["frequency"]["documents"] == pinned["documents"]
        )  # its own count
        assert envelope.methodology_id == methodology  # same namespace: estimates never drift
        assert not any(w.startswith("frequency_repinned") for w in result.warnings)
    assert {
        p.name: p.read_bytes() for p in (vs.accounting_dir() / "collector").glob("frequency*.json")
    } == view_bytes  # the pinned view files are byte-for-byte immutable
    assert vs.ledger_digest() == ledger


def test_pinning_survives_a_restart_and_is_not_re_derived(tmp_path: Path) -> None:
    vs.enable(pin=False)
    vs.start_capture()
    corpus(tmp_path, FLOOR + 5)
    scene.collect(NOW)
    pinned = scene.coverage(scene.current())["frequency"]["view_id"]
    # delete a chunk of the retained corpus's source: a re-derived view would differ
    for path in vs.projects(tmp_path).rglob("fill.jsonl"):
        path.write_text(path.read_text().splitlines()[0] + "\n")
    scene.collect(NOW)
    assert scene.coverage(scene.current())["frequency"]["view_id"] == pinned
    assert pinned_view().version_id == pinned


def test_repin_creates_a_new_namespace_retains_the_old_and_never_sums_them(
    tmp_path: Path,
) -> None:
    vs.enable(pin=False)
    vs.start_capture()
    corpus(tmp_path, FLOOR + 5)
    scene.collect(NOW)
    old = scene.current()
    old_view = scene.coverage(old)["frequency"]["view_id"]
    old_method = old.methodology_id
    old_ledger_blob = next(
        name
        for name in scene.committed().blobs
        if name.startswith("ledger-")  # type: ignore[union-attr]
    )
    old_credit = lifetime_credit(old)
    state_dir = vs.accounting_dir() / "collector" / "state"
    old_file = scene.committed().blobs[old_ledger_blob]
    old_bytes = (state_dir / old_file).read_bytes()

    vs.filler(vs.projects(tmp_path), "growth", CLOCK, 300)  # a different corpus to pin
    ordinary = scene.collect(NOW)  # an ordinary collect keeps the old pin ...
    assert scene.coverage(scene.current())["frequency"]["view_id"] == old_view
    assert not any(w.startswith("frequency_repinned") for w in ordinary.warnings)

    result = scene.collect(NOW, CollectOptions(repin_frequency=True))  # ... only repin changes it
    new = scene.current()
    new_view = scene.coverage(new)["frequency"]["view_id"]
    assert new_view != old_view and new.methodology_id != old_method  # a NEW namespace
    assert any(w.startswith("frequency_repinned") for w in result.warnings)
    assert pinned_view().version_id == new_view
    names = pinned_files()
    assert frequency.view_file_name(old_view) in names  # the old immutable view is retained
    assert frequency.view_file_name(new_view) in names

    committed = vs.checkpoint()
    assert committed is not None and committed.methodology == new.methodology_id
    selected = f"ledger-{new.methodology_id[:12]}"
    assert selected != old_ledger_blob and selected in committed.blobs  # the new namespace ...
    assert old_ledger_blob in committed.blobs  # ... and the old one, both named by the checkpoint
    on_disk = sorted(p.name for p in (vs.accounting_dir() / "collector" / "state").iterdir())
    kept = {n.rsplit("-", 1)[0] for n in on_disk if n.startswith("ledger-")}
    assert kept == {old_ledger_blob, selected}  # the old ledger file is retained, not rewritten
    assert (state_dir / old_file).read_bytes() == old_bytes  # byte-for-byte what it was

    # totals are never summed across namespaces: the window shows ONE methodology's estimate
    new_credit = lifetime_credit(new)
    assert (
        scene.window(new, "lifetime")["estimates"].keys()
        == scene.window(old, "lifetime")["estimates"].keys()
    )
    for unit, amount in new_credit.items():
        assert amount["method"] != old_credit[unit]["method"]  # different estimate identity
    summed = {
        unit: (old_credit[unit]["value"] or 0) + (new_credit[unit]["value"] or 0)
        for unit in new_credit
    }
    for unit, total in summed.items():
        if old_credit[unit]["value"]:
            assert new_credit[unit]["value"] != total  # not old + new


def test_repin_with_too_small_a_corpus_keeps_the_pin_and_says_so(tmp_path: Path) -> None:
    vs.enable(pin=True)
    vs.start_capture()
    vs.pair(vs.projects(tmp_path), 0, CLOCK)
    scene.collect(NOW)
    pinned = scene.coverage(scene.current())["frequency"]["view_id"]
    result = scene.collect(NOW, CollectOptions(repin_frequency=True))
    block = scene.coverage(scene.current())["frequency"]
    assert block["view_id"] == pinned
    assert result.state == "published"
    assert any(w.startswith("frequency_repin_refused: 5 of 200") for w in result.warnings)


def test_a_repin_flag_without_a_pin_pins_when_the_corpus_allows(tmp_path: Path) -> None:
    vs.enable(pin=False)
    vs.start_capture()
    corpus(tmp_path, FLOOR + 5)
    result = scene.collect(NOW, CollectOptions(repin_frequency=True))
    assert scene.coverage(scene.current())["frequency"]["state"] == "pinned"
    assert any(w.startswith("frequency_pinned") for w in result.warnings)
