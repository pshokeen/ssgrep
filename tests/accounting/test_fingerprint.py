"""Keyed request fingerprints (task 3.3): reproducible, keyed, never plaintext."""

from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import pytest

from ssgrep.accounting import fingerprint, storage
from ssgrep.accounting.contracts import Operation

PROFILE = "personal-local"
QUERY = "how did we fix the SECRET_TOKEN_9f3a flake in payments?"
WHERE = "project = '/Users/someone/private-repo'"


@pytest.fixture(autouse=True)
def _forget_keys():
    fingerprint.forget_keys()
    yield
    fingerprint.forget_keys()


def test_key_is_created_once_0600_and_stable_across_loads(tmp_path: Path) -> None:
    assert fingerprint.load_key(PROFILE, create=False) is None
    key = fingerprint.load_key(PROFILE, create=True)
    assert key is not None and len(key) == fingerprint.KEY_BYTES

    path = tmp_path / "data" / "accounting" / PROFILE / fingerprint.KEY_FILE
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    fingerprint.forget_keys()
    assert fingerprint.load_key(PROFILE, create=True) == key == path.read_bytes()


def test_concurrent_first_use_converges_on_one_key() -> None:
    keys: list[bytes | None] = []

    def load() -> None:
        keys.append(fingerprint.load_key(PROFILE, create=True))

    threads = [threading.Thread(target=load) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(set(keys)) == 1 and keys[0] is not None


def test_capture_and_reducer_agree_on_equivalent_arguments_without_plaintext(
    tmp_path: Path,
) -> None:
    key = fingerprint.load_key(PROFILE, create=True)
    assert key is not None
    captured = fingerprint.request_fingerprint(key, Operation.SEARCH, query=QUERY, where=WHERE)
    # The reducer sees the same arguments with transcript-level whitespace/unicode variation.
    reduced = fingerprint.request_fingerprint(
        key,
        Operation.SEARCH,
        query="  " + QUERY.replace("é", "é") + "\n",
        where=" " + WHERE,
    )
    assert captured == reduced and len(captured) == 64

    other = fingerprint.request_fingerprint(key, Operation.SEARCH, query=QUERY + "?", where=WHERE)
    assert other != captured
    assert (
        fingerprint.request_fingerprint(key, Operation.SHOW, query=QUERY, where=WHERE) != captured
    )
    assert fingerprint.request_fingerprint(key, Operation.SEARCH, query=QUERY) != captured

    # No plaintext in the digest, and nothing plaintext is persisted beside the key.
    assert "SECRET_TOKEN" not in captured and "private-repo" not in captured
    for path in (tmp_path / "data").rglob("*"):
        if path.is_file():
            assert b"SECRET_TOKEN" not in path.read_bytes()


def test_fingerprints_are_keyed_per_profile_and_unlinkable_across_keys() -> None:
    first = fingerprint.load_key(PROFILE, create=True)
    second = fingerprint.load_key("org-a", create=True)
    assert first is not None and second is not None and first != second
    assert fingerprint.request_fingerprint(
        first, Operation.SEARCH, query=QUERY
    ) != fingerprint.request_fingerprint(second, Operation.SEARCH, query=QUERY)


def test_blank_where_equals_absent_where() -> None:
    key = fingerprint.load_key(PROFILE, create=True)
    assert key is not None
    assert fingerprint.request_fingerprint(
        key, Operation.SEARCH, query="q", where="   "
    ) == fingerprint.request_fingerprint(key, Operation.SEARCH, query="q", where=None)


def test_key_is_a_separate_file_from_any_endpoint_token(tmp_path: Path) -> None:
    fingerprint.load_key(PROFILE, create=True)
    directory = tmp_path / "data" / "accounting" / PROFILE
    assert fingerprint.KEY_FILE == "fingerprint.key"
    assert sorted(p.name for p in directory.iterdir()) == ["fingerprint.key"]
    assert "token" not in fingerprint.KEY_FILE


def test_damaged_key_is_refused_not_replaced(tmp_path: Path) -> None:
    fingerprint.load_key(PROFILE, create=True)
    path = tmp_path / "data" / "accounting" / PROFILE / fingerprint.KEY_FILE
    path.write_bytes(b"short")
    fingerprint.forget_keys()
    with pytest.raises(storage.UnsafeStorageError):
        fingerprint.load_key(PROFILE, create=True)
    assert path.read_bytes() == b"short"


def test_symlinked_key_file_is_refused(tmp_path: Path) -> None:
    fingerprint.load_key(PROFILE, create=True)
    directory = tmp_path / "data" / "accounting" / PROFILE
    outside = tmp_path / "outside.key"
    outside.write_bytes(b"k" * fingerprint.KEY_BYTES)
    (directory / fingerprint.KEY_FILE).unlink()
    (directory / fingerprint.KEY_FILE).symlink_to(outside)
    fingerprint.forget_keys()
    with pytest.raises(storage.UnsafeStorageError):
        fingerprint.load_key(PROFILE, create=True)
    assert outside.read_bytes() == b"k" * fingerprint.KEY_BYTES
