"""Orchestrator boundary unit tests; no provider calls or empirical comparison."""

import json
from types import SimpleNamespace

import pytest

from eval.accounting_airline import matched
from eval.accounting_airline.matched_bridge import request_args
from eval.accounting_airline.matched_security import freeze_source, install_source


def test_server_refuses_history_even_if_client_is_bypassed():
    with pytest.raises(ValueError, match="matched.history_disabled"):
        request_args("ssgrep", ["search", "rules", "--json"], database="airline", history=False)
    assert request_args("ssgrep", ["search", "rules", "--json"], database="airline", history=True)
    argv = ["--role", "reader", "--database", "airline"]
    assert request_args("sql", argv, database="airline", history=False) == argv
    with pytest.raises(ValueError):
        request_args(
            "sql", ["--role", "reader", "--database", "other"], database="airline", history=True
        )


def test_authorization_and_repetition_bounds_precede_side_effects(tmp_path):
    for authorized, repetitions, gate in [(False, 1, "authorization"), (True, 4, "bound")]:
        with pytest.raises(Exception, match=f"matched.{gate}"):
            matched.run_matched(
                tmp_path,
                tmp_path / "out",
                authorized=authorized,
                model="model",
                repetitions=repetitions,
            )
        assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("interrupted", [False, True])
def test_every_slot_retains_failure_and_common_plan(tmp_path, monkeypatch, interrupted):
    source = tmp_path / "a.jsonl"
    source.write_text("verified source")
    digest = matched.sha(source.read_text())
    native = SimpleNamespace(file_digest=digest, session_id="native-a")
    manifest = {"seed_digest": "seed", "schema_digest": "schema"}
    snapshot = freeze_source(source, tmp_path, digest)
    monkeypatch.setattr(matched, "verified_source", lambda *_: (snapshot, native, manifest))
    monkeypatch.setattr(matched, "verify_protocol", lambda: SimpleNamespace(version="1.0"))
    seen = []

    def prepare(roots, emit):
        source.write_text("changed source during potentially lengthy DB preparation")
        for actor in ("solver-a", "solver-b"):
            (roots.actor_home(actor) / ".pgpass-lab").write_text("synthetic SQL password")
        instance = SimpleNamespace(removed=False)
        instance.teardown = lambda: setattr(instance, "removed", True)
        return SimpleNamespace(
            roots=roots,
            instance=instance,
            observer=SimpleNamespace(finalize=lambda: None),
            manifest=SimpleNamespace(**manifest),
            base=SimpleNamespace(digest="baseline"),
        )

    def execute(run, **kwargs):
        declaration = json.loads((tmp_path / "out/declaration.json").read_text())
        seen.append((run.roots.root, kwargs, declaration))
        pending = json.loads((tmp_path / "out/comparison.json").read_text())
        assert pending["attempts"][-1]["status"] == "unknown"
        assert kwargs["source"] == snapshot
        if kwargs["history"]:
            indexed = install_source(run.roots, kwargs["source"], kwargs["source_digest"])
            assert indexed.read_bytes() == snapshot
        if interrupted:
            raise KeyboardInterrupt()
        raise RuntimeError("controlled test failure, not a provider attempt")

    monkeypatch.setattr(matched.matched_native, "prepare", prepare)
    monkeypatch.setattr(matched.matched_native, "execute", execute)
    report = matched.run_matched(tmp_path, tmp_path / "out", authorized=True, model="model")
    assert not list((tmp_path / "out").glob("*/runtime-*/home/.pgpass-lab"))
    for root, kwargs, _ in seen:
        assert kwargs["source"] == snapshot
        if kwargs["history"]:
            assert (root / "artifacts/collect-view/verified-a.jsonl").read_bytes() == snapshot
    if interrupted:
        assert len(seen) == 1
        assert report.attempts[0]["status"] == "interrupted"
        assert report.pairs_compared == 0
        assert any("incomplete" in p for p in report.plan_problems)
        return
    assert len(seen) == 2
    assert len({row[0] for row in seen}) == 2
    assert {row[1]["history"] for row in seen} == {True, False}
    assert seen[0][1]["prompts"] == seen[1][1]["prompts"]
    assert all(row[2] == seen[0][2] for row in seen)
    assert all(row[0] in seen[1 - i][1]["sealed"] for i, row in enumerate(seen))
    assert [a["status"] for a in report.attempts] == ["failed", "failed"]
    assert report.pairs_compared == 0
    assert not report.plan_problems
    assert len(list((tmp_path / "out").glob("*/artifacts/failure.json"))) == 2


def test_prerequisite_failure_is_named_and_retained(tmp_path, monkeypatch):
    def fail(*_):
        raise matched.LiveBlocked("matched.source_integrity: changed")

    monkeypatch.setattr(matched, "verified_source", fail)
    with pytest.raises(matched.LiveBlocked):
        matched.run_matched(tmp_path, tmp_path / "out", authorized=True, model="model")
    doc = json.loads((tmp_path / "out/blocked.json").read_text())
    assert doc["gate"] == "matched.prerequisite"
    assert "matched.source_integrity" in doc["detail"]
