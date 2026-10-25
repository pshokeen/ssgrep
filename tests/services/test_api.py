"""Unit contract tests for :mod:`ssgrep.services.api`."""

from __future__ import annotations

import pytest

from ssgrep.services import api
from ssgrep.utilities.types import ContentType, IndexNotReadyError, SearchFilters


def test_index_is_a_lossless_adapter(monkeypatch, sample_stats):
    calls = []

    def fake_index(**kwargs):
        calls.append(kwargs)
        return sample_stats

    monkeypatch.setattr(api.indexer, "index", fake_index)

    result = api.index(
        rebuild=True,
        no_subagents=True,
        allow_shrink=True,
        scope="/work/repo",
        live=True,
        full_reprocess=True,
    )

    assert result is sample_stats
    assert calls == [
        {
            "rebuild": True,
            "no_subagents": True,
            "allow_shrink": True,
            "scope": "/work/repo",
            "quiet": False,
            "live": True,
            "full_reprocess": True,
        }
    ]


def test_index_forwards_defaults(monkeypatch, sample_stats):
    calls = []
    monkeypatch.setattr(api.indexer, "index", lambda **kwargs: calls.append(kwargs) or sample_stats)

    assert api.index() is sample_stats
    assert calls == [
        {
            "rebuild": False,
            "no_subagents": False,
            "allow_shrink": False,
            "scope": None,
            "quiet": False,
            "live": False,
            "full_reprocess": False,
        }
    ]


def test_search_is_a_lossless_adapter(monkeypatch, sample_response):
    calls = []
    filters = SearchFilters(content_type=ContentType.RESPONSE, project="acme")

    def fake_search(query, **kwargs):
        calls.append((query, kwargs))
        return sample_response

    monkeypatch.setattr(api.search_module, "search", fake_search)

    result = api.search(
        "how was this fixed?",
        limit=4,
        token_budget=900,
        filters=filters,
        where="git_branch = 'main'",
    )

    assert result is sample_response
    assert calls == [
        (
            "how was this fixed?",
            {
                "limit": 4,
                "token_budget": 900,
                "filters": filters,
                "where": "git_branch = 'main'",
            },
        )
    ]


def test_show_and_status_delegate_without_transforming(monkeypatch, sample_detail, sample_stats):
    monkeypatch.setattr(api.detail, "show", lambda ref: sample_detail if ref == "known" else None)
    monkeypatch.setattr(api.observability, "status", lambda: sample_stats)

    assert api.show("known") is sample_detail
    assert api.show("missing") is None
    assert api.status() is sample_stats


LANCE_DAMAGE = RuntimeError("lance error: Not found: chunks.lance/data/x.lance")


@pytest.mark.parametrize("operation", ["index", "status"])
def test_storage_engine_failure_maps_to_the_unusable_index_contract(monkeypatch, operation):
    def damaged(**_kwargs):
        raise LANCE_DAMAGE

    monkeypatch.setattr(api.indexer, "index", damaged)
    monkeypatch.setattr(api.observability, "status", damaged)

    with pytest.raises(IndexNotReadyError, match="--rebuild") as caught:
        getattr(api, operation)()

    assert caught.value.condition == "corrupt_index"
    assert caught.value.command == "ssgrep index --rebuild"
    assert "lance error" not in str(caught.value)  # engine text (may carry paths) never shown
    assert caught.value.__cause__ is LANCE_DAMAGE


@pytest.mark.parametrize("operation", ["index", "status"])
def test_other_runtime_errors_are_not_reported_as_a_damaged_index(monkeypatch, operation):
    class PipelineFailure(RuntimeError):
        """Stand-in for ssgrep's own RuntimeError subclasses."""

    def broken(**_kwargs):
        raise PipelineFailure("lance error: but raised by ssgrep, not the engine")

    monkeypatch.setattr(api.indexer, "index", broken)
    monkeypatch.setattr(api.observability, "status", broken)

    with pytest.raises(PipelineFailure):
        getattr(api, operation)()

    def bug(**_kwargs):
        raise RuntimeError("engine died")

    monkeypatch.setattr(api.indexer, "index", bug)
    monkeypatch.setattr(api.observability, "status", bug)

    with pytest.raises(RuntimeError, match="engine died"):
        getattr(api, operation)()


def test_rebuild_storage_failure_is_not_blamed_on_a_damaged_index(monkeypatch):
    def damaged(**_kwargs):
        raise LANCE_DAMAGE

    monkeypatch.setattr(api.indexer, "index", damaged)

    with pytest.raises(RuntimeError) as caught:
        api.index(rebuild=True)

    assert caught.value is LANCE_DAMAGE


def test_sources_forwards_scope_and_no_subagents(monkeypatch) -> None:
    calls = []
    discovered = [object(), object()]

    def fake_discover(*, scope, no_subagents):
        calls.append((scope, no_subagents))
        return discovered

    monkeypatch.setattr(api.transcript_adapters, "discover_sources", fake_discover)
    monkeypatch.setattr(
        api.transcript_adapters,
        "source_counts",
        lambda sources: (("pi", len(sources)),),
    )

    assert api.sources(scope="/work", no_subagents=True) == (("pi", 2),)
    assert calls == [("/work", True)]


def _evidence_detail(tmp_path, sample_detail, records, runtime="claude"):
    import json
    from dataclasses import replace

    path = tmp_path / "evidence.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    session_id = "session-1" if runtime == "claude" else f"{runtime}:session-1"
    return replace(
        sample_detail,
        source_path=str(path),
        runtime=runtime,
        session_id=session_id,
        episode_id=f"{session_id}:ep:0",
    )


def _claude_evidence():
    return [
        {
            "type": "user",
            "sessionId": "session-1",
            "timestamp": "2025-01-02T03:04:00Z",
            "message": {"content": "How do I test this?"},
        },
        {
            "type": "assistant",
            "sessionId": "session-1",
            "timestamp": "2025-01-02T03:04:09Z",
            "message": {
                "id": "m1",
                "model": "legacy-model",
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "Use pytest."}],
                "usage": {"input_tokens": 11, "output_tokens": 7},
            },
        },
    ]


def test_infer_legacy_claude_is_readonly_and_provider_unknown(tmp_path, sample_detail):
    from pathlib import Path

    from ssgrep.sessions.work_evidence import infer_work

    detail = _evidence_detail(tmp_path, sample_detail, _claude_evidence())
    before = Path(detail.source_path).read_bytes()
    work = infer_work(detail)
    assert work is not None
    assert work["model"] == "legacy-model"
    assert "provider" not in work
    assert work["usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert work["elapsed_seconds"] == 9
    assert work["basis"] == "transcript_inferred"
    assert work["source_ref"] == detail.episode_id
    repeated = infer_work(detail)
    assert repeated is not None
    assert work["work_id"] == repeated["work_id"]
    assert Path(detail.source_path).read_bytes() == before


@pytest.mark.parametrize("runtime", ["pi", "omp"])
def test_infer_pi_family_preserves_mixed_model_usage(tmp_path, sample_detail, runtime):
    from dataclasses import replace

    from ssgrep.sessions.work_evidence import infer_work

    records = [
        {"type": "session", "id": "session-1", "version": 3},
        {
            "type": "message",
            "id": "u1",
            "timestamp": "2025-01-02T03:04:00Z",
            "message": {"role": "user", "content": "How do I test this?"},
        },
        {
            "type": "message",
            "id": "a1",
            "timestamp": "2025-01-02T03:04:02Z",
            "message": {
                "role": "assistant",
                "content": "First.",
                "model": "one",
                "provider": "explicit-provider",
                "usage": {"input": 5, "output": 2},
            },
        },
        {
            "type": "message",
            "id": "a2",
            "timestamp": "2025-01-02T03:04:09Z",
            "message": {
                "role": "assistant",
                "content": "Use pytest.",
                "model": "two",
                "provider": "explicit-provider",
                "stopReason": "stop",
                "usage": {"input": 8, "output": 3, "cacheRead": 4},
            },
        },
    ]
    detail = replace(
        _evidence_detail(tmp_path, sample_detail, records, runtime),
        response_text="First.\nUse pytest.",
    )
    work = infer_work(detail)
    assert work is not None
    assert "model" not in work
    assert work["contributions"] == [
        {
            "model": "one",
            "provider": "explicit-provider",
            "basis": "transcript_inferred",
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
        {
            "model": "two",
            "provider": "explicit-provider",
            "basis": "transcript_inferred",
            "usage": {"input_tokens": 8, "output_tokens": 3, "cache_read_tokens": 4},
        },
    ]
    assert work["elapsed_seconds"] == 9


def test_infer_codex_ignores_session_cumulative_counts(tmp_path, sample_detail):
    from ssgrep.sessions.work_evidence import infer_work

    records = [
        {"type": "session_meta", "payload": {"id": "session-1"}},
        {"type": "turn_context", "payload": {"model": "old-codex"}},
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": "How do I test this?"},
        },
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "assistant", "content": "Use pytest."},
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {"total_token_usage": {"input_tokens": 1000, "output_tokens": 200}},
            },
        },
    ]
    work = infer_work(_evidence_detail(tmp_path, sample_detail, records, "codex"))
    assert work is not None
    assert work["model"] == "old-codex"
    assert not {"provider", "usage", "elapsed_seconds"} & work.keys()


def test_infer_duplicate_message_usage_is_not_summed(tmp_path, sample_detail):
    from copy import deepcopy
    from dataclasses import replace

    from ssgrep.sessions.work_evidence import infer_work

    records = _claude_evidence()
    final = deepcopy(records[-1])
    records[-1]["message"]["usage"]["output_tokens"] = 2
    records[-1]["message"].pop("stop_reason")
    records.append(final)
    detail = replace(
        _evidence_detail(tmp_path, sample_detail, records), response_text="Use pytest.\nUse pytest."
    )
    work = infer_work(detail)
    assert work is not None
    assert work["usage"] == {"input_tokens": 11, "output_tokens": 7}
    records[-1]["message"]["usage"]["input_tokens"] = 1
    detail = replace(
        _evidence_detail(tmp_path, sample_detail, records), response_text="Use pytest.\nUse pytest."
    )
    work = infer_work(detail)
    assert work is not None
    assert "usage" not in work


@pytest.mark.parametrize("timestamp", [None, "invalid", "2025-01-02T03:03:59Z"])
def test_infer_invalid_duration_stays_unknown(tmp_path, sample_detail, timestamp):
    from ssgrep.sessions.work_evidence import infer_work

    records = _claude_evidence()
    records[-1]["timestamp"] = timestamp
    work = infer_work(_evidence_detail(tmp_path, sample_detail, records))
    assert work is not None
    assert work["model"] == "legacy-model"
    assert "elapsed_seconds" not in work


def test_infer_rejects_changed_identity_and_unsafe_sources(tmp_path, sample_detail):
    from dataclasses import replace
    from pathlib import Path

    from ssgrep.sessions.work_evidence import infer_work

    detail = _evidence_detail(tmp_path, sample_detail, _claude_evidence())
    assert infer_work(replace(detail, response_text="different")) is None
    assert infer_work(replace(detail, session_id="other", episode_id="other:ep:0")) is None
    assert infer_work(replace(detail, source_path=str(tmp_path))) is None
    link = tmp_path / "link.jsonl"
    link.symlink_to(Path(detail.source_path))
    assert infer_work(replace(detail, source_path=str(link))) is None
    assert infer_work(replace(detail, source_path=str(tmp_path / "missing"))) is None


def test_infer_unfinished_turn_does_not_claim_elapsed(tmp_path, sample_detail):
    from ssgrep.sessions.work_evidence import infer_work

    records = _claude_evidence()
    records[-1]["message"].pop("stop_reason")
    work = infer_work(_evidence_detail(tmp_path, sample_detail, records))
    assert work is not None
    assert "elapsed_seconds" not in work


def test_infer_selected_episode_excludes_later_work(tmp_path, sample_detail):
    from copy import deepcopy

    from ssgrep.sessions.work_evidence import infer_work

    records = _claude_evidence()
    later = deepcopy(records)
    later[0]["timestamp"] = "2025-01-02T04:00:00Z"
    later[1]["timestamp"] = "2025-01-02T05:00:00Z"
    later[1]["message"].update(id="m2", model="later-model")
    later[1]["message"]["usage"] = {"input_tokens": 900, "output_tokens": 700}
    work = infer_work(_evidence_detail(tmp_path, sample_detail, records + later))
    assert work is not None
    assert work["model"] == "legacy-model"
    assert work["elapsed_seconds"] == 9
    assert work["usage"] == {"input_tokens": 11, "output_tokens": 7}


def test_infer_missing_message_identity_leaves_usage_unknown(tmp_path, sample_detail):
    from ssgrep.sessions.work_evidence import infer_work

    records = _claude_evidence()
    records[-1]["message"].pop("id")
    work = infer_work(_evidence_detail(tmp_path, sample_detail, records))
    assert work is not None
    assert "usage" not in work
    assert work["model"] == "legacy-model"


def test_infer_refuses_incomplete_or_oversized_sources(tmp_path, sample_detail):
    from pathlib import Path

    from ssgrep.sessions.work_evidence import infer_work

    detail = _evidence_detail(tmp_path, sample_detail, _claude_evidence())
    path = Path(detail.source_path)
    path.write_bytes(path.read_bytes() + b'{"type":')
    assert infer_work(detail) is None
    path.write_bytes(b" " * (8 * 1024 * 1024 + 1))
    assert infer_work(detail) is None


def test_synthetic_harness_error_is_not_a_model_or_a_completed_investigation(
    tmp_path, sample_detail
):
    from dataclasses import replace

    from ssgrep.sessions.work_evidence import infer_work

    records = _claude_evidence()
    records[-1]["message"].pop("stop_reason")
    records.append(
        {
            "type": "assistant",
            "sessionId": "session-1",
            "timestamp": "2025-01-02T03:04:12Z",
            "message": {
                "id": "synthetic-error",
                "model": "<synthetic>",
                "stop_reason": "stop_sequence",
                "content": [{"type": "text", "text": "API Error: Connection lost"}],
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
    )
    detail = _evidence_detail(
        tmp_path,
        replace(sample_detail, response_text="Use pytest.\nAPI Error: Connection lost"),
        records,
    )
    recovered = infer_work(detail)
    assert recovered is not None
    assert recovered["model"] == "legacy-model" and "contributions" not in recovered
    assert recovered["usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert "elapsed_seconds" not in recovered


@pytest.fixture
def infer_readonly():
    from pathlib import Path

    from ssgrep.sessions.work_evidence import infer_work

    def recover(detail):
        path = Path(detail.source_path)
        before = path.read_bytes()
        stat = path.stat()
        work = infer_work(detail)
        assert path.read_bytes() == before
        after = path.stat()
        assert (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        return work

    return recover


def _pi_evidence():
    return [
        {"type": "session", "id": "session-1", "version": 3},
        {
            "type": "model_change",
            "modelId": "legacy-model",
            "provider": "configured-provider",
        },
        {
            "type": "message",
            "id": "u1",
            "timestamp": "2025-01-02T03:04:00Z",
            "message": {"role": "user", "content": "How do I test this?"},
        },
        {
            "type": "message",
            "id": "a1",
            "timestamp": "2025-01-02T03:04:09Z",
            "message": {
                "role": "assistant",
                "content": "Use pytest.",
                "stopReason": "stop",
                "usage": {"input": 11, "output": 7},
            },
        },
    ]


def test_infer_rejects_source_replacement_after_descriptor_read(
    tmp_path, sample_detail, monkeypatch, infer_readonly
):
    from pathlib import Path

    from ssgrep.sessions import work_evidence

    detail = _evidence_detail(tmp_path, sample_detail, _claude_evidence())
    assert infer_readonly(detail) is not None
    path = Path(detail.source_path)
    before = path.read_bytes()
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_bytes(before)
    original = tmp_path / "original.jsonl"
    replacement_inode = replacement.stat().st_ino
    original_fstat = work_evidence.os.fstat
    calls = 0

    def replace_after_read(fd):
        nonlocal calls
        result = original_fstat(fd)
        calls += 1
        if calls == 2:
            # Both descriptor snapshots describe the accepted original file.
            # Identical replacement bytes ensure only path identity rejects it.
            path.rename(original)
            replacement.rename(path)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(work_evidence.os, "fstat", replace_after_read)
        assert work_evidence.infer_work(detail) is None
    assert calls == 2
    assert path.stat().st_ino == replacement_inode
    assert original.read_bytes() == path.read_bytes() == before
    assert infer_readonly(detail) is not None


def test_infer_accepts_blank_lines_without_losing_valid_records(
    tmp_path, sample_detail, infer_readonly
):
    from pathlib import Path

    detail = _evidence_detail(tmp_path, sample_detail, _claude_evidence())
    expected = infer_readonly(detail)
    assert expected is not None
    path = Path(detail.source_path)
    path.write_bytes(b"\n \t\n" + path.read_bytes().replace(b"\n", b"\n\n \t\n"))
    assert infer_readonly(detail) == expected


def test_infer_refuses_oversized_record_within_source_budget(
    tmp_path, sample_detail, infer_readonly
):
    import json
    from pathlib import Path

    from ssgrep.sessions.work_evidence import _MAX_LINE_BYTES, _MAX_SOURCE_BYTES

    detail = _evidence_detail(tmp_path, sample_detail, _claude_evidence())
    assert infer_readonly(detail) is not None
    path = Path(detail.source_path)
    oversized = json.dumps({"type": "progress", "padding": "x" * _MAX_LINE_BYTES}) + "\n"
    path.write_bytes(path.read_bytes() + oversized.encode())
    assert _MAX_LINE_BYTES < path.stat().st_size < _MAX_SOURCE_BYTES
    assert infer_readonly(detail) is None


@pytest.mark.parametrize("record", [None, [], "text", 7, True])
def test_infer_refuses_nonobject_records_in_otherwise_valid_source(
    tmp_path, sample_detail, infer_readonly, record
):
    import json
    from pathlib import Path

    detail = _evidence_detail(tmp_path, sample_detail, _claude_evidence())
    assert infer_readonly(detail) is not None
    path = Path(detail.source_path)
    path.write_bytes(path.read_bytes() + (json.dumps(record) + "\n").encode())
    assert infer_readonly(detail) is None


def test_infer_refuses_mixed_claude_session_identities(tmp_path, sample_detail, infer_readonly):
    records = _claude_evidence()
    detail = _evidence_detail(tmp_path, sample_detail, records)
    assert infer_readonly(detail) is not None
    records[-1]["sessionId"] = "different-session"
    detail = _evidence_detail(tmp_path, sample_detail, records)
    assert infer_readonly(detail) is None


@pytest.mark.parametrize("runtime", ["pi", "omp", "codex"])
def test_infer_refuses_native_session_identity_mismatch(
    tmp_path, sample_detail, infer_readonly, runtime
):
    from dataclasses import replace

    if runtime == "codex":
        records = [
            {"type": "session_meta", "payload": {"id": "session-1"}},
            {"type": "turn_context", "payload": {"model": "legacy-model"}},
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": "How do I test this?"},
            },
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": "Use pytest."},
            },
        ]
    else:
        records = _pi_evidence()
    detail = _evidence_detail(tmp_path, sample_detail, records, runtime)
    assert infer_readonly(detail) is not None
    mismatched = replace(
        detail, session_id=f"{runtime}:other-session", episode_id=f"{runtime}:other-session:ep:0"
    )
    assert infer_readonly(mismatched) is None


@pytest.mark.parametrize("runtime", ["pi", "omp"])
@pytest.mark.parametrize("message_provider", [None, "message-provider"])
def test_infer_pi_model_change_preserves_explicit_provider_precedence(
    tmp_path, sample_detail, infer_readonly, runtime, message_provider
):
    records = _pi_evidence()
    if message_provider is not None:
        records[-1]["message"]["provider"] = message_provider
    detail = _evidence_detail(tmp_path, sample_detail, records, runtime)
    work = infer_readonly(detail)
    assert work is not None
    assert work["provider"] == (message_provider or "configured-provider")
    assert work["model"] == "legacy-model"
    assert work["usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert work["elapsed_seconds"] == 9
    assert work["source_ref"] == detail.episode_id
    records[1].pop("provider")
    detail = _evidence_detail(tmp_path, sample_detail, records, runtime)
    work = infer_readonly(detail)
    assert work is not None
    if message_provider is None:
        assert "provider" not in work
    else:
        assert work["provider"] == message_provider
    assert work["usage"] == {"input_tokens": 11, "output_tokens": 7}


@pytest.mark.parametrize("runtime", ["claude", "pi", "omp"])
@pytest.mark.parametrize("invalid", [-1, True, 1.5, "11", None])
def test_infer_invalid_usage_discards_whole_measurement_not_model_evidence(
    tmp_path, sample_detail, infer_readonly, runtime, invalid
):
    records = _claude_evidence() if runtime == "claude" else _pi_evidence()
    records[-1]["message"]["provider"] = "explicit-provider"
    field = "input_tokens" if runtime == "claude" else "input"
    detail = _evidence_detail(tmp_path, sample_detail, records, runtime)
    control = infer_readonly(detail)
    assert control is not None
    assert control["usage"] == {"input_tokens": 11, "output_tokens": 7}
    records[-1]["message"]["usage"][field] = invalid
    detail = _evidence_detail(tmp_path, sample_detail, records, runtime)
    work = infer_readonly(detail)
    assert work is not None
    assert work["model"] == "legacy-model"
    assert work["provider"] == "explicit-provider"
    assert work["elapsed_seconds"] == 9
    assert "usage" not in work
    records[-1]["message"]["usage"][field] = 0
    detail = _evidence_detail(tmp_path, sample_detail, records, runtime)
    work = infer_readonly(detail)
    assert work is not None
    assert work["usage"] == {"input_tokens": 0, "output_tokens": 7}


@pytest.mark.parametrize("conflict", ["model", "provider"])
def test_infer_conflicting_message_identity_preserves_attribution_but_not_usage(
    tmp_path, sample_detail, infer_readonly, conflict
):
    from copy import deepcopy
    from dataclasses import replace

    records = _claude_evidence()
    records[-1]["message"]["provider"] = "explicit-provider"
    repeated = deepcopy(records[-1])
    repeated["message"][conflict] = "other"
    records.append(repeated)
    detail = replace(
        _evidence_detail(tmp_path, sample_detail, records),
        response_text="Use pytest.\nUse pytest.",
    )
    work = infer_readonly(detail)
    assert work is not None
    expected = [
        {
            "basis": "transcript_inferred",
            "model": "legacy-model",
            "provider": "explicit-provider",
        },
        {
            "basis": "transcript_inferred",
            "model": "legacy-model",
            "provider": "explicit-provider",
            conflict: "other",
        },
    ]
    assert work["contributions"] == expected
    assert "usage" not in work
    # Distinct provider message IDs prove two independent measurements.
    records[-1]["message"]["id"] = "m2"
    detail = replace(
        _evidence_detail(tmp_path, sample_detail, records),
        response_text="Use pytest.\nUse pytest.",
    )
    control = infer_readonly(detail)
    assert control is not None
    assert control["contributions"] == [
        {**contribution, "usage": {"input_tokens": 11, "output_tokens": 7}}
        for contribution in expected
    ]


@pytest.mark.parametrize("naive_records", [(0,), (1,), (0, 1)])
def test_infer_naive_timestamps_do_not_claim_elapsed(
    tmp_path, sample_detail, infer_readonly, naive_records
):
    records = _claude_evidence()
    detail = _evidence_detail(tmp_path, sample_detail, records)
    control = infer_readonly(detail)
    assert control is not None and control["elapsed_seconds"] == 9
    for index in naive_records:
        records[index]["timestamp"] = records[index]["timestamp"].removesuffix("Z")
    work = infer_readonly(_evidence_detail(tmp_path, sample_detail, records))
    assert work is not None
    assert work["model"] == "legacy-model"
    assert work["usage"] == {"input_tokens": 11, "output_tokens": 7}
    assert "elapsed_seconds" not in work


@pytest.mark.parametrize("field", ["prompt_truncated", "response_truncated", "source_path"])
def test_infer_ineligible_detail_never_opens_source(
    tmp_path, sample_detail, monkeypatch, infer_readonly, field
):
    from dataclasses import replace

    from ssgrep.sessions import work_evidence

    detail = _evidence_detail(tmp_path, sample_detail, _claude_evidence())
    assert infer_readonly(detail) is not None

    def forbidden_read(_path):
        pytest.fail("Ineligible indexed detail must not read the source")

    monkeypatch.setattr(work_evidence, "_read", forbidden_read)
    assert (
        work_evidence.infer_work(
            replace(detail, **{field: None if field == "source_path" else True})
        )
        is None
    )


def test_infer_missing_episode_does_not_substitute_matching_text(
    tmp_path, sample_detail, infer_readonly
):
    from copy import deepcopy
    from dataclasses import replace

    records = _claude_evidence()
    later = deepcopy(records)
    later[-1]["message"].update(id="m2", model="later-model")
    later[-1]["message"]["usage"] = {"input_tokens": 900, "output_tokens": 700}
    detail = _evidence_detail(tmp_path, sample_detail, records + later)
    second = infer_readonly(replace(detail, episode_id="session-1:ep:1"))
    assert second is not None
    assert second["source_ref"] == "session-1:ep:1"
    assert second["model"] == "later-model"
    assert second["usage"] == {"input_tokens": 900, "output_tokens": 700}
    assert infer_readonly(replace(detail, episode_id="session-1:ep:2")) is None
    assert infer_readonly(replace(detail, episode_id="other:ep:0")) is None


def test_infer_evidence_free_records_do_not_fabricate_work(tmp_path, sample_detail, infer_readonly):
    records = _claude_evidence()
    records[-1]["message"].pop("model")
    records[-1]["message"].pop("usage")
    detail = _evidence_detail(tmp_path, sample_detail, records)
    assert infer_readonly(detail) is None
    records[-1]["message"]["provider"] = "explicit-provider"
    work = infer_readonly(_evidence_detail(tmp_path, sample_detail, records))
    assert work is not None
    assert work["provider"] == "explicit-provider"
    assert not {"model", "usage", "contributions"} & work.keys()
    assert work["elapsed_seconds"] == 9


@pytest.mark.parametrize("runtime", ["pi", "omp", "codex"])
@pytest.mark.parametrize("header_position", [0, 1])
def test_infer_refuses_conflicting_native_headers_regardless_of_order(
    tmp_path, sample_detail, infer_readonly, runtime, header_position
):
    from copy import deepcopy

    records: list[dict]
    if runtime == "codex":
        records = [
            {"type": "session_meta", "payload": {"id": "session-1"}},
            {"type": "turn_context", "payload": {"model": "legacy-model"}},
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": "How do I test this?"},
            },
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": "Use pytest."},
            },
        ]
        conflicting = {"type": "session_meta", "payload": {"id": "different-session"}}
    else:
        records = _pi_evidence()
        conflicting = {"type": "session", "id": "different-session", "version": 3}
    detail = _evidence_detail(tmp_path, sample_detail, records, runtime)
    control = infer_readonly(detail)
    assert control is not None
    records.insert(header_position, deepcopy(records[0]))
    detail = _evidence_detail(tmp_path, sample_detail, records, runtime)
    assert infer_readonly(detail) == control
    records[header_position] = conflicting
    detail = _evidence_detail(tmp_path, sample_detail, records, runtime)
    assert infer_readonly(detail) is None


@pytest.mark.parametrize(
    "end,seconds",
    [
        ("2025-01-02T03:04:09.123Z", 9),
        ("2025-01-02T03:04:00.500Z", 0),
    ],
)
def test_inferred_elapsed_rounds_known_subsecond_boundaries_down(
    tmp_path, sample_detail, end, seconds
):
    from ssgrep.sessions.work_evidence import infer_work

    records = _claude_evidence()
    records[-1]["timestamp"] = end
    recovered = infer_work(_evidence_detail(tmp_path, sample_detail, records))
    assert recovered is not None and recovered["elapsed_seconds"] == seconds
    assert recovered["basis"] == "transcript_inferred"
