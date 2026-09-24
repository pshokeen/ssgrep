"""Unit tests for the provenance metadata recorded in eval result files."""

from __future__ import annotations

from eval import provenance


def test_library_versions_pins_storage_and_encoding_engines():
    versions = provenance.library_versions()

    assert isinstance(versions["lancedb"], str) and versions["lancedb"]
    assert isinstance(versions["pylate"], str) and versions["pylate"]


def test_build_provenance_carries_library_versions(monkeypatch, sample_stats, tmp_path):
    monkeypatch.setattr(provenance, "duplication_check", lambda _path: {"total_chunks": 0})
    monkeypatch.setattr(
        provenance, "git_info", lambda project_dir=None: {"commit": "x", "dirty": False}
    )

    block = provenance.build_provenance(
        date="2026-08-21",
        task="test task",
        label="test label",
        stats=sample_stats,
        db_path=tmp_path,
    )

    assert block["library_versions"] == provenance.library_versions()
    assert block["library_versions"]["lancedb"]
    assert block["library_versions"]["pylate"]
