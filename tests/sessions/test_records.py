"""Unit tests for shared record validation and counters."""

from __future__ import annotations

import pytest

from ssgrep.sessions import records


@pytest.mark.parametrize("record_type", sorted(records._KNOWN_TYPES))
def test_known_record_types_are_accepted(record_type: str):
    assert records.is_known_record({"type": record_type})


@pytest.mark.parametrize("record", [{}, {"type": None}, {"type": "future-record"}])
def test_missing_and_unknown_record_types_are_rejected(record: dict):
    assert not records.is_known_record(record)


def test_stats_defaults_and_independent_values():
    assert records.MAX_LINE_BYTES == 500_000
    assert records.Stats() == records.Stats(0, 0, 0)
    assert records.Stats(skipped_oversized=2, malformed_lines=3, unknown_types=4) == records.Stats(
        2, 3, 4
    )
