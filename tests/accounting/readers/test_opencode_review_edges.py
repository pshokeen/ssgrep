"""Independent regressions for descriptor authority, dirty sweeps and JSON depth."""

from __future__ import annotations

import os

import apsw
import pytest

from ssgrep.accounting.readers import opencode_vfs as native_vfs
from ssgrep.accounting.readers.opencode import OpenCodeReader
from tests.accounting.readers.test_opencode_nonmutating import inventory
from tests.accounting.readers_fixtures import (
    T0,
    opencode_add_message,
    opencode_add_part,
    opencode_add_session,
    opencode_assistant_data,
    opencode_connect,
    opencode_tool_part,
)


def seed(path, call):
    writer = opencode_connect(path)
    opencode_add_session(writer, "owner")
    opencode_add_message(writer, "message", "owner", opencode_assistant_data())
    opencode_add_part(writer, "m", "message", "owner", opencode_tool_part(call, "bash", {}))
    return writer


@pytest.mark.parametrize("kind", [apsw.SQLITE_OPEN_MAIN_DB, apsw.SQLITE_OPEN_WAL])
def test_ancestor_aba_during_native_open_cannot_substitute_an_unapproved_database(
    tmp_path, monkeypatch, kind
):
    approved, outside, parked = tmp_path / "approved", tmp_path / "outside", tmp_path / "parked"
    approved.mkdir()
    outside.mkdir()
    database = approved / "native.db"
    a = seed(database, "approved")
    b = seed(outside / "native.db", "unapproved")
    original = native_vfs.DescriptorFile
    target = (
        database if kind == apsw.SQLITE_OPEN_MAIN_DB else database.with_name(database.name + "-wal")
    )
    expected = target.stat()
    substitutions = []

    def swap_namespace(descriptor):
        actual = os.fstat(descriptor)
        if (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino):
            approved.rename(parked)
            outside.rename(approved)
            try:
                substitutions.append(kind)
                return original(descriptor)
            finally:
                approved.rename(outside)
                parked.rename(approved)
        return original(descriptor)

    reader = OpenCodeReader()
    try:
        before_a, before_b = inventory(database), inventory(outside / "native.db")
        with monkeypatch.context() as patch:
            patch.setattr(native_vfs, "DescriptorFile", swap_namespace)
            result = reader.read(reader.source_for(database))
        assert substitutions
        assert "result:unapproved" not in result.evidence_ids()
        assert "result:approved" in result.evidence_ids()
        assert inventory(database) == before_a
        assert inventory(outside / "native.db") == before_b
    finally:
        a.close()
        b.close()


def test_backdated_insert_behind_inflight_census_is_read_before_complete(tmp_path):
    database = tmp_path / "dirty-sweep.db"
    writer = seed(database, "original")
    opencode_add_part(
        writer, "z", "message", "owner", opencode_tool_part("last", "bash", {}), at=T0 + 10
    )
    reader = OpenCodeReader()
    source = reader.source_for(database)
    try:
        initial = reader.read(source)
        state = initial.state
        for _ in range(30):
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = step.state
            part = state.extra.get("census", {}).get("part", {})
            if part.get("id") == "m" and not part.get("done"):
                break
        else:
            raise AssertionError("fixture did not reach the independent census boundary")
        opencode_add_part(
            writer, "a", "message", "owner", opencode_tool_part("backdated", "bash", {}), at=T0 - 10
        )
        found, saw_dirty, completed = False, False, False
        for _ in range(80):
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = step.state
            found |= "result:backdated" in step.evidence_ids()
            saw_dirty |= state.extra.get("census_epoch", {}).get("dirty", False)
            if state.extra["opencode"]["status"] == "complete":
                assert found, "a dirty prefix must not become the collector's idle-skip proof"
                completed = True
                break
        assert found and saw_dirty and completed
    finally:
        writer.close()


def test_busy_commit_changes_do_not_restart_an_advanced_census_at_first_page(tmp_path):
    database = tmp_path / "busy.db"
    writer = seed(database, "original")
    for index in range(12):
        opencode_add_part(
            writer,
            f"part-{index:02}",
            "message",
            "owner",
            opencode_tool_part(f"call-{index}", "bash", {}),
            at=T0 + index + 1,
        )
    reader = OpenCodeReader()
    source = reader.source_for(database)
    try:
        state = reader.read(source).state
        boundaries = set()
        for index in range(90):
            writer.execute(
                "UPDATE session SET time_updated=? WHERE id='owner'", (T0 + 100 + index,)
            )
            step = reader.read(source, state, budget=1, byte_budget=1, maintenance=True)
            state = step.state
            boundary = state.extra.get("census", {}).get("part", {}).get("id")
            if boundary:
                boundaries.add(boundary)
        assert "part-11" in boundaries
        assert state.extra["opencode"]["status"] != "complete"
    finally:
        writer.close()


@pytest.mark.parametrize("table", ["message", "part"])
def test_deep_subcap_json_is_a_row_local_gap_and_later_native_rows_are_read(tmp_path, table):
    database = tmp_path / "nested.db"
    writer = seed(database, "original")
    nested = "[" * 5000 + "0" + "]" * 5000
    try:
        if table == "message":
            writer.execute(
                "INSERT INTO message VALUES(?,?,?,?,?)", ("deep", "owner", T0 + 1, T0 + 1, nested)
            )
        else:
            opencode_add_part(writer, "deep", "message", "owner", nested, at=T0 + 1)
        opencode_add_part(
            writer, "later", "message", "owner", opencode_tool_part("later", "bash", {}), at=T0 + 2
        )
        reader = OpenCodeReader()
        result = reader.read(reader.source_for(database))
        assert "result:later" in result.evidence_ids()
        (gap,) = [g for g in result.gaps if g.locator == f"{table}:deep"]
        assert gap.size == len(nested.encode())
        assert not any(e.locator == gap.locator for e in result.evidence)
        assert result.bytes_read >= len(nested.encode())
    finally:
        writer.close()


def test_json_decodes_but_completed_tool_normalization_depth_is_contained(tmp_path):
    import json
    import sys

    def reference_walk(value):
        if isinstance(value, dict):
            return {key: reference_walk(item) for key, item in value.items()}
        return value

    raw = None
    # Select from the live recursion limit and independently exercise both
    # decodability and recursive container traversal; no fixed portable depth.
    for numerator in range(12, 40):
        depth = max(1, sys.getrecursionlimit() * numerator // 20)
        nested = '{"node":' * depth + "0" + "}" * depth
        candidate = (
            '{"type":"tool","tool":"bash","callID":"deep-normalization",'
            '"state":{"status":"completed","input":'
            + nested
            + ',"output":"done","time":{"start":1,"end":2}}}'
        )
        try:
            decoded = json.loads(candidate)
        except RecursionError:
            continue
        try:
            reference_walk(decoded["state"]["input"])
        except RecursionError:
            raw = candidate
            break
    assert raw is not None, "fixture needs decodable JSON with recursive traversal overflow"
    assert json.loads(raw)["state"]["status"] == "completed"

    database = tmp_path / "normalization.db"
    writer = seed(database, "original")
    try:
        opencode_add_part(writer, "deep-normalization", "message", "owner", raw, at=T0 + 1)
        opencode_add_part(
            writer,
            "later-normalization",
            "message",
            "owner",
            opencode_tool_part("later-normalization", "bash", {}),
            at=T0 + 2,
        )
        reader = OpenCodeReader()
        result = reader.read(reader.source_for(database))
        assert "result:later-normalization" in result.evidence_ids()
        assert not any(e.locator == "part:deep-normalization" for e in result.evidence)
        assert any(
            g.reason == "normalization_recursion" and g.locator == "part:deep-normalization"
            for g in result.gaps
        )
        assert result.bytes_read >= len(raw.encode())
        assert result.state.extra["wm"]["part"] == [T0 + 2, "later-normalization"]
    finally:
        writer.close()
