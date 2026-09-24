"""Tests for pipeline identity and location constants."""

from __future__ import annotations

from ssgrep.pipeline import state
from ssgrep.store.paths import data_dir


def test_app_name_is_stable_identity() -> None:
    assert state.APP_NAME == "SessionIndexV1"


def test_lmdb_path_defaults_inside_data_dir() -> None:
    assert state.lmdb_path() == data_dir() / "cocoindex" / "state.db"


def test_lmdb_path_honors_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(state.LMDB_PATH_ENV, str(tmp_path / "custom" / "state.db"))
    assert state.lmdb_path() == (tmp_path / "custom" / "state.db").absolute()


def test_context_keys_are_unique() -> None:
    assert state.LANCE_DB.key != state.EMBEDDER.key
    assert state.LANCE_DB.key == "ssgrep_lance_v1"
    assert state.EMBEDDER.key == "ssgrep_embedder_v1"
