"""Behavior of the frozen contracts: unknown-vs-zero and deterministic bytes."""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.accounting.contracts import (
    Basis,
    Quantity,
    TaskId,
    canonical_bytes,
    digest_of,
)
from ssgrep.accounting.ledger_contracts import valid_profile_id
from ssgrep.store.paths import data_dir
from tests.accounting.conftest import REAL_HOME, build_event, build_evidence


def test_unavailable_quantity_cannot_carry_a_value_or_hide_its_reason() -> None:
    with pytest.raises(ValueError):
        Quantity(0, "bytes", Basis.UNAVAILABLE, reason="no usage record")
    with pytest.raises(ValueError):
        Quantity(None, "bytes", Basis.OBSERVED)
    with pytest.raises(ValueError):
        Quantity(None, "bytes", Basis.UNAVAILABLE)

    missing = Quantity.unavailable("bytes", "usecli owns serialization")
    assert missing.value is None
    assert missing.reason == "usecli owns serialization"


def test_zero_observed_is_distinct_from_unavailable_in_canonical_form() -> None:
    zero = canonical_bytes(Quantity.observed(0, "bytes"))
    unknown = canonical_bytes(Quantity.unavailable("bytes", "n/a"))

    assert zero != unknown
    assert b'"value":0' in zero
    assert b'"value":null' in unknown


def test_canonical_bytes_ignore_construction_order_and_reject_floats() -> None:
    assert canonical_bytes({"b": 1, "a": 2}) == canonical_bytes({"a": 2, "b": 1})
    assert canonical_bytes(build_event()) == canonical_bytes(build_event())
    with pytest.raises(TypeError):
        canonical_bytes({"score": 0.5})


def test_event_digest_changes_with_any_semantic_field() -> None:
    base = digest_of(build_event())

    assert digest_of(build_event(truncated=True)) != base
    assert digest_of(build_event(execution_id="n1-0002")) != base


def test_identical_native_events_share_identity_and_edited_ones_do_not() -> None:
    original = build_evidence()
    copy = build_evidence()
    edited = build_evidence(payload={"name": "bash", "arguments": {"command": "rm"}})

    assert original.identity == copy.identity
    assert original.identity != edited.identity


def test_task_without_turn_is_unresolved_not_a_guessed_task() -> None:
    assert not TaskId(root="r1", turn=None).resolved
    assert TaskId(root="r1", turn="t4").resolved


@pytest.mark.parametrize("bad", ["", "../etc", "A", "-x", "a" * 33, "a/b", "a b"])
def test_profile_ids_cannot_act_as_paths(bad: str) -> None:
    assert not valid_profile_id(bad)


def test_profile_id_accepts_bounded_identifiers() -> None:
    assert valid_profile_id("personal-local")
    assert valid_profile_id("acme_eu-1")


def test_tests_cannot_reach_the_real_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Positive evidence: with BOTH isolation layers removed (explicit override
    # and XDG redirect) the resolver points outside tmp_path, so the autouse
    # fixture is what protects the user's real data.
    assert data_dir().is_relative_to(tmp_path)

    monkeypatch.delenv("SSGREP_DATA_DIR")
    assert data_dir().is_relative_to(tmp_path), "XDG_DATA_HOME alone must still isolate"

    monkeypatch.delenv("XDG_DATA_HOME")
    assert not data_dir().is_relative_to(tmp_path)
    assert data_dir().is_relative_to(REAL_HOME)
