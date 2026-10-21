"""Unit tests for native note writing and discovery."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.sessions import notes


class _FixedDateTime:
    @classmethod
    def now(cls, timezone):
        assert timezone is UTC
        return datetime(2025, 2, 3, 4, 5, 6, tzinfo=UTC)


class _FixedUuid:
    hex = "123456789abcEXTRA"


def test_notes_dir_uses_isolated_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "private-data"))
    assert notes.notes_dir() == tmp_path / "private-data" / "notes"


@pytest.mark.parametrize(
    ("title", "body", "message"),
    [
        ("", "body", "title"),
        ("  \t", "body", "title"),
        ("title", "", "body"),
        ("title", " \n ", "body"),
    ],
)
def test_write_note_rejects_blank_fields(tmp_path: Path, title: str, body: str, message: str):
    with pytest.raises(ValueError, match=message):
        notes.write_note(tmp_path, title, body)


def test_write_note_creates_and_appends_native_record_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(notes, "datetime", _FixedDateTime)
    monkeypatch.setattr(notes.uuid, "uuid4", lambda: _FixedUuid())
    locks: list[int] = []
    monkeypatch.setattr(notes.fcntl, "flock", lambda _fd, operation: locks.append(operation))

    first = notes.write_note(project, "How is retry configured?", "Use jitter.")
    second = notes.write_note(project, "What is the cap?", "Thirty seconds.")

    assert first == second == notes.notes_dir() / "notes-202502.jsonl"
    assert first.exists()
    assert first.parent.stat().st_mode & 0o777 == 0o700
    assert locks == [notes.fcntl.LOCK_EX, notes.fcntl.LOCK_UN] * 2

    records = [json.loads(line) for line in first.read_text().splitlines()]
    assert len(records) == 4
    user_record, assistant_record = records[:2]
    assert user_record == {
        "parentUuid": None,
        "isSidechain": False,
        "type": "user",
        "message": {"role": "user", "content": "How is retry configured?"},
        "uuid": "note-123456789abc-u",
        "timestamp": "2025-02-03T04:05:06Z",
        "cwd": str(project),
        "sessionId": "note-202502",
        "gitBranch": "",
    }
    assert assistant_record["parentUuid"] == user_record["uuid"]
    assert assistant_record["type"] == "assistant"
    assert assistant_record["message"] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "Use jitter."}],
    }
    assert assistant_record["uuid"] == "note-123456789abc-a"
    assert assistant_record["timestamp"] == user_record["timestamp"]
    assert assistant_record["cwd"] == str(project)
    assert assistant_record["sessionId"] == "note-202502"
    assert assistant_record["gitBranch"] == ""
    assert records[2]["message"]["content"] == "What is the cap?"
    assert records[3]["message"]["content"][0]["text"] == "Thirty seconds."


def test_discover_notes_missing_and_sorted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "note-root"
    monkeypatch.setattr(notes, "notes_dir", lambda: root)
    assert notes.discover_notes() == []

    root.mkdir()
    later = root / "notes-202512.jsonl"
    earlier = root / "notes-202501.jsonl"
    ignored = root / "other.jsonl"
    for path in (later, earlier, ignored):
        path.write_text("{}\n")

    found = notes.discover_notes()
    assert [session.path for session in found] == [earlier, later]
    assert [session.session_id for session in found] == ["notes-202501", "notes-202512"]
    assert all(session.is_main for session in found)
    assert all(session.parent_session_id is None for session in found)


def test_work_metadata_revisions_share_one_investigation(tmp_path, monkeypatch):
    from ssgrep.sessions import work

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    original = {
        "work_id": "investigation-1",
        "provider": "anthropic",
        "model": "exact-model",
        "elapsed_seconds": 840,
        "usage": {"input_tokens": 92000, "output_tokens": 8000},
        "basis": "runtime_reported",
    }
    first = work.attach_work("notes-202610:ep:0", "a" * 64, original)
    duplicate = work.attach_work("notes-202610:ep:1", "b" * 64, original)
    assert first["revision"] == duplicate["revision"] == 1
    updated = work.attach_work("notes-202610:ep:0", "a" * 64, {**original, "elapsed_seconds": 900})
    assert updated["revision"] == 2
    shared = work.read_work("notes-202610:ep:1", "b" * 64)
    assert shared is not None
    assert shared["work"]["elapsed_seconds"] == 900
    assert [row["work"]["elapsed_seconds"] for row in work.work_history("investigation-1")] == [
        840,
        900,
    ]
    assert work.read_work("notes-202610:ep:0", "c" * 64) is None


def test_work_retry_keeps_revision_and_missing_reads_do_not_create(tmp_path, monkeypatch):
    from ssgrep.sessions import work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    assert work.read_work("note:ep:0", "a" * 64) is None
    assert work.work_history("investigation") == []
    assert not root.exists()
    raw = {"work_id": "investigation", "usage": {"input_tokens": 0}}
    first = work.attach_work("note:ep:0", "a" * 64, raw)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert work.attach_work("note:ep:0", "a" * 64, raw) == first
    assert work.work_history("investigation") == [first]
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert first["work"]["usage"] == {"input_tokens": 0}
    assert "provider" not in first["work"]
    assert "elapsed_seconds" not in first["work"]


@pytest.mark.parametrize(
    "extra",
    [
        {"elapsed_seconds": True},
        {"elapsed_seconds": -1},
        {"elapsed_seconds": 1.5},
        {"usage": {"output_tokens": False}},
        {"usage": {"input_tokens": -1}},
        {"usage": {"total_tokens": 12}},
        {"usage": None},
        {"provider": ""},
        {"model": " "},
        {"basis": "guessed"},
        {"source_ref": ""},
        {"saved_dollars": 1},
        {"contributions": []},
        {"contributions": [{"model": "a"}], "usage": {}},
        {"contributions": [{"provider": "x", "elapsed_seconds": 1}]},
        {"model": "m" * 65536},
    ],
)
def test_invalid_work_never_creates_storage(tmp_path, monkeypatch, extra):
    from ssgrep.sessions import work

    root = tmp_path / "absent"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    with pytest.raises(ValueError):
        work.attach_work("note:ep:0", "a" * 64, {"work_id": "one", **extra})
    assert not root.exists()


def test_mixed_work_preserves_unknown_provider_and_independent_usage(tmp_path, monkeypatch):
    from ssgrep.sessions import work

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    raw = {
        "work_id": "mixed",
        "source_ref": "native:ep:1",
        "basis": "transcript_inferred",
        "contributions": [
            {"model": "unclassified", "usage": {"input_tokens": 0}},
            {"provider": "explicit", "model": "other", "usage": {"output_tokens": 7}},
        ],
    }
    saved = work.attach_work("note:ep:0", "a" * 64, raw)
    raw["contributions"][1]["usage"]["output_tokens"] = 99
    stored = work.read_work("note:ep:0", "a" * 64)
    assert stored is not None
    assert stored == saved
    assert stored["work"]["contributions"][0]["usage"] == {"input_tokens": 0}
    assert "provider" not in stored["work"]["contributions"][0]
    assert stored["work"]["contributions"][1]["usage"] == {"output_tokens": 7}
    assert "usage" not in stored["work"]
    assert stored["work"]["source_ref"] == "native:ep:1"


@pytest.mark.parametrize("corruption", [b"{", b'{"schema_version":999}'])
def test_corrupt_work_fails_closed_without_overwriting(tmp_path, monkeypatch, corruption):
    from ssgrep.sessions import work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    raw = {"work_id": "one", "model": "first"}
    first = work.attach_work("note:ep:0", "a" * 64, raw)
    assert work.read_work("note:ep:0", "a" * 64) == first
    revision = next(root.rglob("revision-1.json"))
    revision.write_bytes(corruption)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert work.read_work("note:ep:0", "a" * 64) is None
    with pytest.raises((ValueError, OSError)):
        work.attach_work("note:ep:0", "a" * 64, {**raw, "model": "second"})
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_work_revision_files_are_retained_and_private(tmp_path, monkeypatch):
    import stat

    from ssgrep.sessions import work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    first = work.attach_work("note:ep:0", "a" * 64, {"work_id": "one", "model": "first"})
    revision = next(root.rglob("revision-1.json"))
    original = revision.read_bytes()
    work.attach_work("note:ep:0", "a" * 64, {"work_id": "one", "model": "second"})
    assert revision.read_bytes() == original
    assert work.work_history("one")[0] == first
    assert not (root / "notes").exists()
    for path in root.rglob("*"):
        assert stat.S_IMODE(path.stat().st_mode) == (0o700 if path.is_dir() else 0o600)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "exposed"])
def test_unsafe_work_files_refused_without_touching_target(tmp_path, monkeypatch, kind):
    import os

    from ssgrep.sessions import work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    raw = {"work_id": "one"}
    work.attach_work("note:ep:0", "a" * 64, raw)
    revision = next(root.rglob("revision-1.json"))
    target = tmp_path / "source"
    target.write_text("untouched")
    revision.unlink()
    if kind == "symlink":
        revision.symlink_to(target)
    elif kind == "fifo":
        os.mkfifo(revision, 0o600)
    elif kind == "directory":
        revision.mkdir(mode=0o700)
    else:
        revision.write_text("{}")
        revision.chmod(0o644)
    assert work.read_work("note:ep:0", "a" * 64) is None
    with pytest.raises((ValueError, OSError)):
        work.attach_work("note:ep:0", "a" * 64, raw)
    assert target.read_text() == "untouched"


def test_work_refuses_linked_metadata_directory(tmp_path, monkeypatch):
    from ssgrep.sessions import work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    root.mkdir()
    target = tmp_path / "untouched"
    target.mkdir()
    (root / "work-metadata").symlink_to(target, target_is_directory=True)
    assert work.read_work("note:ep:0", "a" * 64) is None
    with pytest.raises(OSError):
        work.attach_work("note:ep:0", "a" * 64, {"work_id": "one"})
    assert list(target.iterdir()) == []


def test_subject_digest_binds_actual_episode_content(sample_detail):
    from dataclasses import replace

    from ssgrep.sessions.work import subject_digest

    original = subject_digest(sample_detail)
    assert len(original) == 64
    for key in ("episode_id", "runtime", "prompt_text", "response_text"):
        assert subject_digest(replace(sample_detail, **{key: "changed"})) != original


@pytest.mark.parametrize("name", ["head.json", "binding-*.json"])
def test_future_work_control_records_do_not_get_replaced(tmp_path, monkeypatch, name):
    from ssgrep.sessions import work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    first = work.attach_work("note:ep:0", "a" * 64, {"work_id": "one"})
    assert work.read_work("note:ep:0", "a" * 64) == first
    control = next(root.rglob(name))
    future = b'{"schema_version":2,"future_evidence":"preserve"}'
    control.write_bytes(future)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert work.read_work("note:ep:0", "a" * 64) is None
    with pytest.raises(ValueError):
        work.attach_work("note:ep:0", "a" * 64, {"work_id": "one", "model": "new"})
    assert control.read_bytes() == future
    assert len(list(root.rglob("revision-*.json"))) == 1
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("kind", ["fifo", "symlink", "directory", "exposed"])
def test_unsafe_work_lock_refuses_write_without_touching_source(tmp_path, monkeypatch, kind):
    import os

    from ssgrep.sessions import work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    work.attach_work("note:ep:0", "a" * 64, {"work_id": "one"})
    lock = root / "work-metadata" / ".lock"
    lock.unlink()
    source = tmp_path / "source"
    source.write_text("preserve")
    if kind == "fifo":
        os.mkfifo(lock, 0o600)
    elif kind == "symlink":
        lock.symlink_to(source)
    elif kind == "directory":
        lock.mkdir(mode=0o700)
    else:
        lock.write_bytes(b"")
        lock.chmod(0o644)
    with pytest.raises(OSError):
        work.attach_work("note:ep:0", "a" * 64, {"work_id": "one", "model": "new"})
    assert source.read_text() == "preserve"
    assert len(work.work_history("one")) == 1


@pytest.mark.parametrize("field", ["work_id", "ref", "subject"])
def test_invalid_work_identity_never_creates_storage(tmp_path, monkeypatch, field):
    from ssgrep.sessions import work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    args = {"ref": "note:ep:0", "subject": "a" * 64, "raw": {"work_id": "one"}}
    if field == "work_id":
        args["raw"]["work_id"] = "\ud800"
    else:
        args[field] = "\ud800"
    with pytest.raises(ValueError):
        work.attach_work(**args)
    assert not root.exists()


@pytest.mark.parametrize("attempts", [("second",), ("second", "third")])
@pytest.mark.parametrize("identical_retry", [False, True])
def test_work_recovers_interrupted_head_without_losing_revisions(
    tmp_path, monkeypatch, attempts, identical_retry
):
    from ssgrep.sessions import _work_storage, work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    first_work = {"work_id": "shared", "model": "first"}
    first = work.attach_work("note:ep:0", "a" * 64, first_work)
    work.attach_work("note:ep:1", "b" * 64, first_work)
    replace = _work_storage.replace

    def fail_head(directory, name, data, *, limit):
        if name == "head.json":
            raise OSError("injected head publication failure")
        return replace(directory, name, data, limit=limit)

    for model in attempts:
        with monkeypatch.context() as fault:
            fault.setattr(_work_storage, "replace", fail_head)
            with pytest.raises(OSError, match="injected head publication failure"):
                work.attach_work("note:ep:0", "a" * 64, {**first_work, "model": model})
        assert work.read_work("note:ep:0", "a" * 64) == first
        assert work.read_work("note:ep:1", "b" * 64) == first
        assert work.work_history("shared") == [first]

    retained = {path: path.read_bytes() for path in root.rglob("revision-*.json")}
    corrected_model = attempts[-1] if identical_retry else "corrected"
    recovered = work.attach_work("note:ep:0", "a" * 64, {**first_work, "model": corrected_model})
    expected = ["first", *attempts] + ([] if identical_retry else ["corrected"])
    assert recovered["revision"] == len(expected)
    assert [entry["work"]["model"] for entry in work.work_history("shared")] == expected
    assert work.read_work("note:ep:1", "b" * 64) == recovered
    assert all(path.read_bytes() == content for path, content in retained.items())


def test_work_does_not_promote_or_overwrite_corrupt_orphan(tmp_path, monkeypatch):
    from ssgrep.sessions import _work_storage, work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    raw = {"work_id": "shared", "model": "first"}
    first = work.attach_work("note:ep:0", "a" * 64, raw)
    replace = _work_storage.replace

    def fail_head(directory, name, data, *, limit):
        if name == "head.json":
            raise OSError("injected head publication failure")
        return replace(directory, name, data, limit=limit)

    with monkeypatch.context() as fault:
        fault.setattr(_work_storage, "replace", fail_head)
        with pytest.raises(OSError, match="injected head publication failure"):
            work.attach_work("note:ep:0", "a" * 64, {**raw, "model": "orphan"})
    orphan = next(root.rglob("revision-2.json"))
    orphan.write_bytes(b'{"schema_version":999}')
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        work.attach_work("note:ep:0", "a" * 64, {**raw, "model": "correction"})
    assert work.read_work("note:ep:0", "a" * 64) == first
    assert work.work_history("shared") == [first]
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize(
    "extra",
    [
        {"contributions": [{"usage": {"input_tokens": 1}}]},
        {"contributions": [{}]},
        {"contributions": [None]},
        {"contributions": {"model": "m"}},
        {"basis": "transcript_inferred"},
        {"source_ref": " "},
        {"source_ref": "x" * 2049},
        {"source_ref": "\0"},
    ],
)
def test_malformed_contribution_or_source_does_not_replace_valid_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: dict
):
    from ssgrep.sessions import work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    raw = {
        "work_id": "one",
        "basis": "agent_estimated",
        "contributions": [{"model": "m", "usage": {"input_tokens": 1}}],
    }
    first = work.attach_work("note:ep:0", "a" * 64, raw)
    assert work.read_work("note:ep:0", "a" * 64) == first
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        work.attach_work("note:ep:0", "a" * 64, {**raw, **extra})
    assert work.read_work("note:ep:0", "a" * 64) == first
    assert work.work_history("one") == [first]
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    inferred = work.attach_work(
        "note:ep:0",
        "a" * 64,
        {**raw, "basis": "transcript_inferred", "source_ref": "native:ep:0"},
    )
    assert work.read_work("note:ep:0", "a" * 64) == inferred
    assert work.work_history("one") == [first, inferred]


def test_canonical_contribution_expansion_cannot_exceed_storage_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from ssgrep.sessions import work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    raw: dict = {"work_id": "one", "contributions": [{"model": "m"} for _ in range(2000)]}
    assert len(json.dumps(raw).encode()) < 64 * 1024
    with pytest.raises(ValueError, match="canonical work metadata exceeds"):
        work.attach_work("note:ep:0", "a" * 64, raw)
    assert not root.exists()
    accepted = work.attach_work(
        "note:ep:0", "a" * 64, {**raw, "contributions": raw["contributions"][:1000]}
    )
    stored = work.read_work("note:ep:0", "a" * 64)
    assert stored is not None
    assert stored == accepted
    assert len(stored["work"]["contributions"]) == 1000
    assert all(item["basis"] == "agent_estimated" for item in stored["work"]["contributions"])


@pytest.mark.parametrize(
    ("target", "updates"),
    [
        ("binding", {"ref": "note:ep:1"}),
        ("binding", {"unexpected": "preserve"}),
        ("binding", {"subject": "not-a-digest"}),
        ("binding", {"work_id": ""}),
        ("head", {"work_id": "other"}),
        ("head", {"unexpected": "preserve"}),
        ("head", {"revision": 0}),
        ("revision", None),
        ("revision", {"unexpected": "preserve"}),
        ("revision", {"work_id": "other"}),
        ("revision", {"revision": 2}),
        ("revision", {"revision": True}),
        ("revision", {"work": {"work_id": "one", "model": "first"}}),
        (
            "revision",
            {"work": {"work_id": "other", "model": "first", "basis": "agent_estimated"}},
        ),
    ],
)
def test_invalid_stored_identity_cannot_overwrite_or_contaminate_other_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str, updates: dict | None
):
    import hashlib

    from ssgrep.sessions import work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    raw = {"work_id": "one", "model": "first"}
    first = work.attach_work("note:ep:0", "a" * 64, raw)
    other = work.attach_work("note:ep:1", "b" * 64, {"work_id": "other", "model": "unrelated"})
    assert work.read_work("note:ep:0", "a" * 64) == first
    assert work.read_work("note:ep:1", "b" * 64) == other
    assert work.work_history("one") == [first]
    metadata = root / "work-metadata"
    directory = metadata / f"work-{hashlib.sha256(b'one').hexdigest()}"
    paths = {
        "binding": metadata / f"binding-{hashlib.sha256(b'note:ep:0').hexdigest()}.json",
        "head": directory / "head.json",
        "revision": directory / "revision-1.json",
    }
    damaged = paths[target]
    if updates is None:
        damaged.unlink()
    else:
        damaged.write_text(json.dumps({**json.loads(damaged.read_text()), **updates}))
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    assert work.read_work("note:ep:0", "a" * 64) is None
    assert work.read_work("note:ep:1", "b" * 64) == other
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before
    with pytest.raises(ValueError):
        work.attach_work("note:ep:0", "a" * 64, {**raw, "model": "replacement"})
    assert work.read_work("note:ep:1", "b" * 64) == other
    assert work.work_history("other") == [other]
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_competing_revision_publication_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from ssgrep.sessions import _work_storage, work

    root = tmp_path / "data"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    raw = {"work_id": "one", "model": "first"}
    first = work.attach_work("note:ep:0", "a" * 64, raw)
    assert work.read_work("note:ep:0", "a" * 64) == first
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    write_once = _work_storage.write_once
    competing_bytes = json.dumps(
        {
            "schema_version": 1,
            "work_id": "other",
            "revision": 2,
            "work": {"work_id": "other", "model": "competing", "basis": "agent_estimated"},
        }
    ).encode()

    def publish_competitor(directory: int, name: str, data: bytes) -> bool:
        assert name == "revision-2.json"
        # A writer ignoring the advisory lock wins after the absence check.
        assert write_once(directory, name, competing_bytes)
        return write_once(directory, name, data)

    with monkeypatch.context() as fault:
        fault.setattr(_work_storage, "write_once", publish_competitor)
        with pytest.raises(ValueError, match="work revision changed during publication"):
            work.attach_work("note:ep:0", "a" * 64, {**raw, "model": "replacement"})
    orphan = next(root.rglob("revision-2.json"))
    assert orphan.read_bytes() == competing_bytes
    assert all(path.read_bytes() == content for path, content in before.items())
    assert work.read_work("note:ep:0", "a" * 64) == first
    assert work.work_history("one") == [first]
    assert work.work_history("other") == []
    retained = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="work revision identity mismatch"):
        work.attach_work("note:ep:0", "a" * 64, {**raw, "model": "retry"})
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == retained
