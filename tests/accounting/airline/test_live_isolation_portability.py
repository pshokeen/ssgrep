"""Actual sandbox proof must use shipped sources, not ignored planning artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from eval.accounting_airline import live_security
from eval.accounting_airline.gates import FAIL, PASS
from eval.accounting_airline.roots import LabRoots
from eval.accounting_airline.sandbox import SandboxProfile, sbpl_quote


@pytest.mark.usefixtures("live_sandbox_ready")
@pytest.mark.parametrize("expose_oracle", [False, True])
def test_live_isolation_works_without_private_planning_files_and_detects_read_leaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expose_oracle: bool
) -> None:
    checkout = tmp_path / "checkout"
    sources = checkout / "eval" / "accounting_airline"
    sources.mkdir(parents=True)
    oracle = sources / "domain_oracle.py"
    oracle.write_text("# private verifier oracle canary\n")
    dataset = sources / "dataset.py"
    dataset.write_text("# private fixture answer canary\n")
    assert not (checkout / "openspec").exists()
    account = tmp_path / "account"
    credentials = account / ".omp" / "agent" / "agent.db"
    credentials.parent.mkdir(parents=True)
    credentials.write_bytes(b"isolated credential control; not real provider data")
    credentials.chmod(0o600)
    monkeypatch.setattr(live_security, "REPO_ROOT", checkout)
    monkeypatch.setattr(live_security, "account_home", lambda: account)
    roots = LabRoots.create(tmp_path / "runs")
    a_native = roots.runtime_a / "private-a.jsonl"
    a_native.write_bytes(b'{"type":"session","id":"isolated-a"}\n')
    protected = [oracle, dataset, a_native, roots.marker, credentials]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    profile = live_security.live_profile(roots, "solver-b", real_home=account)
    if expose_oracle:
        profile = SandboxProfile(
            profile.text + f"(allow file-read* (literal {sbpl_quote(oracle)}))\n",
            profile.denied,
            profile.limitations,
        )
    gate = live_security.prove_live_isolation(roots, profile, a_native)
    assert gate.status == (FAIL if expose_oracle else PASS), gate.as_dict()
    probes = gate.evidence["probes"]
    reads = {row["path"]: row["denied"] for row in probes if "path" in row}
    assert reads[str(a_native)] is True
    assert reads[str(oracle)] is (not expose_oracle)
    assert reads[str(dataset)] is True
    assert all(row["correct"] for row in probes if "write_probe" in row)
    assert {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in protected
    } == before
    assert not (roots.project / "isolation-positive.txt").exists()
