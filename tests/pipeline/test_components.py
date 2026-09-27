"""Tests for the traced CocoIndex processing components."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from ssgrep.pipeline import components
from ssgrep.pipeline.sources import SourceDescriptor


def test_process_source_with_no_records_declares_no_rows(tmp_path) -> None:
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    descriptor = SourceDescriptor(
        adapter="native",
        key=str(empty),
        path=str(empty),
        size=0,
        mtime=0.0,
        digest="",
    )
    chunk_table = Mock()
    episode_table = Mock()
    session_table = Mock()

    asyncio.run(components.process_source(descriptor, chunk_table, episode_table, session_table))

    session_table.declare_row.assert_called_once()
    episode_table.declare_row.assert_not_called()
    chunk_table.declare_row.assert_not_called()


@pytest.mark.parametrize(
    "failure", [PermissionError("denied"), FileNotFoundError(2, "missing dependency", "/other")]
)
def test_process_source_does_not_archive_unrelated_read_errors(tmp_path, monkeypatch, failure):
    descriptor = SourceDescriptor(
        adapter="native",
        key=str(tmp_path / "source.jsonl"),
        path=str(tmp_path / "source.jsonl"),
        size=1,
        mtime=0.0,
        digest="",
    )
    reader = Mock(side_effect=failure)
    monkeypatch.setattr(components.transcript_adapters, "read_source", reader)
    targets = [Mock(), Mock(), Mock()]
    with pytest.raises(type(failure)) as caught:
        asyncio.run(components.process_source(descriptor, *targets))
    assert caught.value is failure
    reader.assert_called_once()
    for target in targets:
        target.declare_row.assert_not_called()
