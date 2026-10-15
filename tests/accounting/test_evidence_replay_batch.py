"""Selected native replay is exact, authorized and byte-linear per affected source."""

from dataclasses import replace

from ssgrep.accounting import collector_store as store
from ssgrep.accounting.collector_read import discover, make_readers
from ssgrep.accounting.evidence_replay import EvidenceReplay
from ssgrep.accounting.policy import enable_collection
from tests.accounting import collector_scene as scene
from tests.accounting.readers_fixtures import claude_prompt
from tests.accounting.reduce_builders import PROFILE, ts


def native_items(root):
    enable_collection(PROFILE)
    records = [claude_prompt("batch", "investigate", ts=ts())]
    for n in range(200):
        records.extend(scene.search_records("batch", f"c-{n}", f"query-{n}", (10, 0, 0)))
    path = scene.write_transcript(root, "batch", records, scene.at())
    found = discover(make_readers())
    reader, source = next(iter(found.values()))
    decoded = reader.read(source, None).evidence
    refs = store.write_segments(PROFILE, decoded)
    compact = list(store.iter_evidence(PROFILE, refs))
    return path, decoded, compact


def test_batch_replays_every_selected_identity_with_linear_native_bytes(tmp_path):
    path, expected, compact = native_items(tmp_path)
    replay = EvidenceReplay(PROFILE)
    actual = replay.hydrate_many(compact)
    assert not replay.unavailable
    assert [(row.identity, row.ts_utc) for row in actual] == [
        (row.identity, row.ts_utc) for row in expected
    ]
    assert [row.payload for row in actual] == [row.payload for row in expected]
    assert replay.adapter.bytes_read <= path.stat().st_size + 64 * 1024 + len(compact)


def test_batch_preserves_per_record_drift_and_missingness(tmp_path):
    path, expected, compact = native_items(tmp_path)
    changed = replace(compact[-1], digest="0" * 64)
    replay = EvidenceReplay(PROFILE)
    actual = replay.hydrate_many([*compact[:-1], changed])
    assert actual[-1].payload["_replay_unavailable"] == "source_digest_drift"
    assert actual[0].identity == expected[0].identity
    assert replay.unavailable[("claude", "batch")]
    path.unlink()
    missing = EvidenceReplay(PROFILE)
    missing.hydrate_many(compact)
    assert missing.unavailable[("claude", "batch")] == "source_missing"


def test_a_started_native_record_finishes_without_starting_the_next_over_budget(tmp_path):
    from ssgrep.accounting.policy import load_profile_policy
    from ssgrep.accounting.reduce_io import native_source_adapter

    path, expected, compact = native_items(tmp_path)
    lines = path.read_bytes().splitlines(keepends=True)
    budget = len(lines[0]) + len(lines[1]) // 2
    for selected_count in (2, 3):
        adapter = native_source_adapter(load_profile_policy(PROFILE), now_ms=0)
        adapter.byte_budget = budget
        replay = EvidenceReplay(PROFILE, adapter=adapter)
        actual = replay.hydrate_many(compact[:selected_count])
        assert [row.identity for row in actual[:2]] == [row.identity for row in expected[:2]]
        assert [row.payload for row in actual[:2]] == [row.payload for row in expected[:2]]
        if selected_count == 3:
            assert actual[2].payload["_replay_unavailable"] == "source_read_bound"
        else:
            assert not replay.unavailable
        assert adapter.bytes_read == len(lines[0]) + len(lines[1])
        assert budget < adapter.bytes_read <= budget + max(map(len, lines))
