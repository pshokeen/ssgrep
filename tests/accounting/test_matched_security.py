"""Real filesystem race/cleanup regressions; no provider or DB calls."""

import hashlib
from types import SimpleNamespace

import pytest

from eval.accounting_airline import matched_native, matched_security
from eval.accounting_airline.live_native import LiveBlocked
from eval.accounting_airline.observer_blob import MAX_SOURCE_BYTES, ObserverError
from eval.accounting_airline.roots import LabRoots


def digest(data):
    return hashlib.sha256(data).hexdigest()


def credential_files(roots):
    paths = []
    for actor in ("solver-a", "solver-b"):
        paths.extend((roots.actor_home(actor) / ".pgpass-lab", roots.verifier / actor / "pgpass"))
    for path in paths:
        path.parent.mkdir(mode=0o700, exist_ok=True)
        path.write_text("synthetic SQL password")
    return paths


def test_frozen_bytes_survive_source_replacement_across_arms(tmp_path):
    source = tmp_path / "verified.jsonl"
    original = b'{"type":"session","id":"a"}\n'
    source.write_bytes(original)
    snapshot = matched_security.freeze_source(source, tmp_path, digest(original))
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"later B history must not enter A corpus\n")
    replacement.replace(source)
    for _ in range(2):
        roots = LabRoots.create(tmp_path)
        installed = matched_security.install_source(roots, snapshot, digest(original))
        assert installed.read_bytes() == original
        assert digest(installed.read_bytes()) == digest(original)


def test_actual_destination_tampering_is_rejected_before_index(tmp_path, monkeypatch):
    roots = LabRoots.create(tmp_path)
    original = b"verified A bytes\n"
    real_freeze = matched_security.freeze_source

    def race(path, root, expected):
        path.chmod(0o600)
        path.write_bytes(b"replaced after copy before verification\n")
        return real_freeze(path, root, expected)

    monkeypatch.setattr(matched_security, "freeze_source", race)
    with pytest.raises(LiveBlocked, match="matched.source_integrity"):
        matched_security.install_source(roots, original, digest(original))


@pytest.mark.parametrize("redirect", ["leaf", "parent", "oversize"])
def test_source_freeze_refuses_unsafe_paths(tmp_path, redirect):
    source = tmp_path / "a.jsonl"
    source.write_bytes(b"original")
    if redirect == "leaf":
        linked = tmp_path / "link.jsonl"
        linked.symlink_to(source)
        source = linked
    elif redirect == "parent":
        real = tmp_path / "real"
        real.mkdir(mode=0o700)
        (real / "a.jsonl").write_bytes(b"original")
        linked = tmp_path / "linked"
        linked.symlink_to(real, target_is_directory=True)
        source = linked / "a.jsonl"
    else:
        with source.open("r+b") as handle:
            handle.truncate(MAX_SOURCE_BYTES + 1)
    with pytest.raises(ObserverError):
        matched_security.freeze_source(source, tmp_path, digest(b"original"))


@pytest.mark.parametrize("failure", [None, "observer", "teardown"])
def test_close_removes_all_actor_credentials_even_when_cleanup_fails(tmp_path, failure):
    roots = LabRoots.create(tmp_path)
    paths = credential_files(roots)
    calls = []

    def finalize():
        calls.append("observer")
        if failure == "observer":
            raise RuntimeError("observer failed")

    def teardown():
        calls.append("teardown")
        if failure == "teardown":
            raise RuntimeError("teardown failed")

    run = SimpleNamespace(
        roots=roots,
        observer=SimpleNamespace(finalize=finalize),
        instance=SimpleNamespace(removed=False, teardown=teardown),
    )
    if failure:
        with pytest.raises(RuntimeError, match="failed"):
            matched_security.close_run(run)
    else:
        matched_security.close_run(run)
    assert calls == ["observer", "teardown"]
    assert all(not path.exists() for path in paths)


def test_partial_build_failure_removes_credentials_without_returned_run(tmp_path, monkeypatch):
    roots = LabRoots.create(tmp_path)
    paths = []

    def fail_build(*_):
        paths.extend(credential_files(roots))
        raise RuntimeError("partial build failed")

    monkeypatch.setattr(matched_native.lab_run, "build_run", fail_build)
    with pytest.raises(RuntimeError, match="partial build failed"):
        matched_native.prepare(roots, lambda _: None)
    assert paths and all(not path.exists() for path in paths)


def test_cleanup_refuses_redirected_home_without_unlinking_outside_canary(tmp_path):
    roots = LabRoots.create(tmp_path)
    credential_files(roots)
    outside = tmp_path / "unrelated-lab"
    outside.mkdir(mode=0o700)
    canary = outside / ".pgpass-lab"
    canary.write_text("unrelated credential must survive")
    home = roots.actor_home("solver-b")
    home.rename(home.with_name("retained-original-home"))
    home.symlink_to(outside, target_is_directory=True)
    with pytest.raises(LiveBlocked, match="matched.credential_cleanup"):
        matched_security.remove_credentials(roots)
    assert canary.read_text() == "unrelated credential must survive"
    assert not (roots.actor_home("solver-a") / ".pgpass-lab").exists()
    assert not (roots.verifier / "solver-b" / "pgpass").exists()
