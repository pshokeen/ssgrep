"""The collector idle cache cannot authorize a commit the reader never visited."""

from __future__ import annotations

from tests.accounting.readers_fixtures import T0, opencode_add_session, opencode_connect
from tests.accounting.test_collect_read import Cycle, oc_add_call, oc_db


def test_append_between_reader_return_and_settlement_is_visited_next_cycle(tmp_path, monkeypatch):
    connection = opencode_connect(oc_db(tmp_path))
    try:
        opencode_add_session(connection, "ses_1")
        oc_add_call(connection, 1, T0 + 100)
        cycle = Cycle(tmp_path)
        cycle.drain()
        assert any(item.payload.get("call_id") == "call_1" for item in cycle.evidence)
        reader = cycle.readers["opencode"]
        original = reader.read
        appended = False

        def append_after_read(*args, **kwargs):
            nonlocal appended
            result = original(*args, **kwargs)
            if not appended:
                appended = True
                oc_add_call(connection, 3, T0 + 300)
            return result

        monkeypatch.setattr(reader, "read", append_after_read)
        # Force a legitimate new snapshot, then commit another row after it returns.
        oc_add_call(connection, 2, T0 + 200)
        first = cycle.run()
        assert appended and first.bytes_read > 0
        assert not any(item.payload.get("call_id") == "call_3" for item in first.evidence)
        cycle.drain()
        assert (
            sum(
                item.kind == "tool_call" and item.payload.get("call_id") == "call_3"
                for item in cycle.evidence
            )
            == 1
        )
        assert all(record.complete for record in cycle.known.values())
    finally:
        connection.close()
