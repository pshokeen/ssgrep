"""Logical source coverage is separate from settlement and publication."""

from ssgrep.accounting.collector_progress import Progress, source_progress
from ssgrep.accounting.collector_store import SourceRecord


def test_source_progress_does_not_turn_settlement_into_completion():
    records = [
        SourceRecord("opencode", "opencode:unvisited"),
        SourceRecord(
            "opencode",
            "opencode:pending",
            state={
                "extra": {
                    "opencode": {
                        "status": "pending",
                        "rows_visited": {"session": 1, "part": 3},
                        "pending_records": None,
                        "known_pending_lower_bound": 3,
                        "pending_count_complete": False,
                        "abandoned_records": 1,
                    }
                }
            },
        ),
        SourceRecord(
            "opencode",
            "opencode:unsupported",
            status="unsupported",
            state={
                "extra": {"opencode": {"status": "unsupported"}},
            },
        ),
        SourceRecord("claude", "claude:done", complete=True),
    ]
    counts = source_progress(records)
    assert counts == {
        "sources_complete": 1,
        "sources_unvisited": 1,
        "sources_pending": 1,
        "rows_visited": 4,
        "pending_records": None,
        "abandoned_records": 1,
        "unsupported_sources": 1,
        "known_pending_lower_bound": 3,
        "pending_count_complete": False,
    }
    document = Progress("isolated", "collect", sources_done=4, **counts).to_dict()
    assert document["sources_done"] == 4
    assert document["sources_complete"] == 1
