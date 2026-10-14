"""Configuration/boundary regression checks; never dispatch a provider request."""

import json
import shutil
import sys

import pytest

from eval.accounting_airline.__main__ import build_parser
from eval.accounting_airline.live_prompts import prompt_for
from eval.accounting_airline.live_security import live_profile
from eval.accounting_airline.roots import LabRoots
from eval.accounting_airline.sandbox import run_sandboxed


def test_explicit_thinking_preserves_fresh_default_and_resume_pin():
    parser = build_parser()
    assert parser.parse_args(["--mode", "live"]).thinking is None
    assert parser.parse_args(["--mode", "live", "--thinking", "high"]).thinking == "high"
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "live", "--thinking", "unsupported"])


def test_resume_refuses_effort_change_before_any_native_attempt(tmp_path):
    from eval.accounting_airline.live_native import LiveBlocked
    from eval.accounting_airline.live_resume import resume_live

    roots = LabRoots.create(tmp_path)
    (roots.artifacts / "live-evidence.json").write_text(
        json.dumps({"turns": {"L1": {}, "L2": {}}, "native_ids": {"b": None}})
    )
    (roots.artifacts / "manifest.json").write_text(
        json.dumps({"model": "openai-codex/gpt-6-luna", "sampling": {"thinking": "low"}})
    )
    (roots.artifacts / "report.json").write_text("{}")
    with pytest.raises(LiveBlocked, match="thinking"):
        resume_live(
            roots.root, print, authorized=True, model="openai-codex/gpt-6-luna", thinking="high"
        )


def test_native_business_audit_requires_history_keys_without_answers():
    text = prompt_for("L3", "owned")
    assert "by those retrieved keys" in text
    assert "untrusted cache" in text
    assert "descending service_priority" in text
    assert "ssgrep show <returned-ref>." in text
    assert "separate bash tool call" in text
    for forbidden in ("MCT-", "F5", "B1", "B2", "airline-lab-arithmetic", "rarity", "eligible"):
        assert forbidden not in text


@pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="requires actual macOS sandbox")
@pytest.mark.parametrize("actor", ["solver-a", "solver-b"])
def test_live_profile_denies_future_temp_artifacts_but_own_tools_work(tmp_path, actor):
    roots = LabRoots.create(tmp_path)
    profile = live_profile(roots, actor)
    # Created after the profile: enumeration of existing siblings would not protect it.
    future = tmp_path / "pytest-of-future" / "ssgrep-airline-other-report"
    future.mkdir(parents=True)
    secret = future / "report.json"
    secret.write_text("private future artifact")
    assert secret.read_text() == "private future artifact"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(roots.actor_home(actor))}
    denied = run_sandboxed(
        profile, ["/bin/cat", str(secret)], env=env, cwd=roots.project, timeout=10
    )
    assert denied.returncode != 0 and not denied.stdout
    scratch = roots.actor_home(actor) / "scratch"
    code = (
        "from pathlib import Path; import sys; p=Path(sys.argv[1]); "
        "p.write_text('own'); print(p.read_text())"
    )
    own = run_sandboxed(
        profile,
        [sys.executable, "-I", "-c", code, str(scratch)],
        env=env,
        cwd=roots.project,
        timeout=10,
    )
    assert own.returncode == 0 and own.stdout.strip() == "own", own.stderr
    clients = roots.tools / actor
    clients.mkdir()
    client = clients / "positive-client"
    client.write_text("#!/bin/sh\nprintf own-client")
    client.chmod(0o700)
    positive = run_sandboxed(profile, [str(client)], env=env, cwd=roots.project, timeout=10)
    assert positive.returncode == 0 and positive.stdout == "own-client", positive.stderr


def test_catalog_preregistration_excludes_account_metadata(tmp_path, monkeypatch):
    from eval.accounting_airline import live_auth

    catalog = tmp_path / ".bun/install/global/node_modules/@oh-my-pi/pi-catalog/src/models.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "test-model": {
                        "id": "test-model",
                        "cost": {"input": 1},
                        "thinking": {"efforts": ["low", "high"]},
                        "accountAccess": {"private-account": {"secret": "must-not-appear"}},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(live_auth, "account_home", lambda: tmp_path)
    value = live_auth.catalog_metadata("openai-codex/test-model")
    assert value["model"]["thinking"]["efforts"] == ["low", "high"]
    assert len(value["sha256"]) == 64
    assert "private-account" not in json.dumps(value)
    assert "must-not-appear" not in json.dumps(value)
