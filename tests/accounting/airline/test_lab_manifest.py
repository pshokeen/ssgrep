"""Run manifest pins, secret hygiene, and the named live-lane preflight gates."""

from __future__ import annotations

import dataclasses
import json
import os
import stat
from pathlib import Path

import pytest

from eval.accounting_airline import database, dataset, manifest
from eval.accounting_airline.gates import BLOCKED, NOT_RUN, PASS, Gate
from eval.accounting_airline.manifest import (
    ARITHMETIC_METHODOLOGY,
    PRODUCTION_METHODOLOGY,
    STANDIN_LABEL,
    CredentialRef,
    ManifestError,
    RunManifest,
    build_manifest,
    preflight_live,
    prompt_hashes,
    ssgrep_version,
    verify_pins,
)
from eval.accounting_airline.roots import LabRoots
from eval.accounting_airline.sandbox import Limitation

PROMPTS = {"A-L1": "Reconcile cash for the lab day.", "B-L3": "Plan the reaccommodation."}


@pytest.fixture
def lab(tmp_path: Path):
    roots = LabRoots.create(parent=tmp_path)
    try:
        yield roots
    finally:
        if roots.root.exists():
            roots.remove()


def _standin(lab: LabRoots, **kwargs) -> RunManifest:
    return build_manifest(
        run_id=lab.run_id,
        mode=kwargs.pop("mode", "standin"),
        runtime=kwargs.pop("runtime", STANDIN_LABEL),
        roots=lab,
        prompts=PROMPTS,
        **kwargs,
    )


def _gates(**kwargs) -> dict[str, Gate]:
    gates = preflight_live(**kwargs)
    assert len({gate.name for gate in gates}) == len(gates)
    return {gate.name: gate for gate in gates}


# ---------------------------------------------------------------------------- build


def test_standin_manifest_pins_and_never_counts_toward_the_live_gate(lab: LabRoots) -> None:
    built = _standin(lab)
    assert built.mode == "standin" and built.runtime == STANDIN_LABEL
    assert built.counts_toward_live_gate is False and built.live_authorized is False
    assert built.seed_digest == dataset.overall_digest(dataset.seed_digests())
    assert built.schema_digest == dataset.schema_digest()
    assert (built.image_ref, built.image_digest) == (dataset.IMAGE_REF, dataset.IMAGE_DIGEST)
    assert built.prompt_hashes == prompt_hashes(PROMPTS)
    assert all(len(h) == 64 for h in built.prompt_hashes.values())
    assert built.ssgrep_version == ssgrep_version() != "unknown"
    assert built.methodology["production"] == PRODUCTION_METHODOLOGY
    assert built.methodology["arithmetic"] == ARITHMETIC_METHODOLOGY == "airline-lab-arithmetic-v1"
    assert built.methodology["thresholds_milli"] == {
        "conservative": 8000,
        "central": 6000,
        "optimistic": 4000,
    }
    assert built.methodology["arithmetic_threshold_milli"] == 0
    assert built.timeouts[database.READER] == database.ROLE_LIMITS[database.READER]
    assert built.timeouts[database.WRITER] == database.ROLE_LIMITS[database.WRITER]
    assert built.timeouts[database.VERIFIER]["options"] == database.VERIFIER_OPTIONS
    assert built.roots["root"] == str(lab.root) and built.roots["runtime_a"] == str(lab.runtime_a)
    assert built.private_artifact_dir == str(lab.artifacts)
    payload = built.as_dict()
    json.dumps(payload)  # fully serializable
    assert payload["live_authorized"] is False


def test_prompt_hashes_are_sha256_of_utf8() -> None:
    import hashlib

    assert prompt_hashes({"p": "héllo"}) == {"p": hashlib.sha256("héllo".encode()).hexdigest()}


def test_live_requires_authorization_and_standin_is_never_live(lab: LabRoots) -> None:
    with pytest.raises(ManifestError, match="authorization"):
        _standin(lab, mode="live", runtime="omp")
    with pytest.raises(ManifestError, match="never be recorded as a live run"):
        _standin(lab, mode="live", runtime=STANDIN_LABEL, authorize_live=True)
    with pytest.raises(ManifestError, match="unknown mode"):
        _standin(lab, mode="bogus")


def test_authorized_live_counts_only_with_a_real_runtime(lab: LabRoots) -> None:
    live = _standin(
        lab,
        mode="live",
        runtime="omp",
        authorize_live=True,
        model="m",
        runtime_version="18.10.3",
        provider="p",
        sampling={"temperature": "0"},
    )
    assert live.counts_toward_live_gate is True and live.live_authorized is True
    assert (live.model, live.runtime_version, live.provider) == ("m", "18.10.3", "p")
    assert live.sampling == {"temperature": "0"}
    # positive/negative twins on the flag
    assert _standin(lab, mode="deterministic", runtime="omp").counts_toward_live_gate is False
    assert (
        _standin(
            lab, mode="deterministic", runtime="omp", authorize_live=True
        ).counts_toward_live_gate
        is False
    )


def test_limitations_are_recorded_as_plain_dicts(lab: LabRoots) -> None:
    built = _standin(
        lab,
        limitations=[Limitation("network_not_restricted", "why", "detail"), {"name": "x"}],
    )
    assert built.limitations[0] == {
        "name": "network_not_restricted",
        "reason": "why",
        "detail": "detail",
    }
    assert built.as_dict()["limitations"][1] == {"name": "x"}


# ------------------------------------------------------------------------------ pins


def test_verify_pins_empty_then_detects_prompt_and_seed_drift(
    lab: LabRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _standin(lab)
    assert verify_pins(built, prompts=PROMPTS) == []  # positive twin

    changed = {**PROMPTS, "A-L1": PROMPTS["A-L1"] + " Also peek at the answers."}
    drift = verify_pins(built, prompts=changed)
    assert len(drift) == 1 and "'A-L1' changed" in drift[0]
    assert any("'B-L3' is recorded but no longer" in d for d in verify_pins(built, prompts={}))
    extra = verify_pins(built, prompts={**PROMPTS, "C": "new"})
    assert extra == ["prompt 'C' was not recorded in the manifest"]

    real_digests = dataset.seed_digests

    def drifted() -> dict[str, str]:
        return {**real_digests(), "bookings": "0" * 64}

    monkeypatch.setattr(dataset, "seed_digests", drifted)
    drift = verify_pins(built, prompts=PROMPTS)
    assert len(drift) == 1 and drift[0].startswith("seed_digest drifted")
    monkeypatch.setattr(dataset, "seed_digests", real_digests)
    assert verify_pins(built, prompts=PROMPTS) == []


def test_verify_pins_detects_schema_image_and_version_drift(
    lab: LabRoots, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = _standin(lab)
    monkeypatch.setattr(dataset, "schema_digest", lambda: "f" * 64)
    monkeypatch.setattr(dataset, "IMAGE_DIGEST", "sha256:" + "e" * 64)
    monkeypatch.setattr(manifest, "ssgrep_version", lambda: "0.0.0-other")
    drift = verify_pins(built, prompts=PROMPTS)
    assert {d.split(" ")[0] for d in drift} == {"schema_digest", "image_digest", "ssgrep_version"}


def test_ssgrep_version_comes_from_installed_metadata() -> None:
    import importlib.metadata

    assert ssgrep_version() == importlib.metadata.version("ssgrep")


# --------------------------------------------------------------------- digest and file


def test_digest_is_stable_across_created_utc_and_sensitive_to_content(lab: LabRoots) -> None:
    first = _standin(lab)
    second = dataclasses.replace(first, created_utc="2001-01-01T00:00:00+00:00")
    assert first.created_utc != second.created_utc
    assert first.digest() == second.digest() and len(first.digest()) == 64
    other = dataclasses.replace(first, model="different-model")
    assert other.digest() != first.digest()


def test_written_manifest_is_0600_and_never_contains_credential_content(
    lab: LabRoots, tmp_path: Path
) -> None:
    canary = "sk-CANARY-SECRET-0123456789"
    credential = tmp_path / "agent.db"
    credential.write_text(canary)
    ref = CredentialRef("omp-agent-db", str(credential), "omp-credential-store")
    built = _standin(lab, credentials=[ref])
    target = lab.artifacts / "manifest.json"
    built.write(target)
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
    data = target.read_bytes()
    assert (
        canary.encode() not in data and canary.encode() not in json.dumps(built.as_dict()).encode()
    )
    assert str(credential).encode() in data  # the reference is recorded
    parsed = json.loads(data)
    assert parsed["credentials"][0]["path_sha256"] == ref.path_digest
    assert parsed["run_id"] == lab.run_id and parsed["mode"] == "standin"
    # overwriting an existing looser file tightens it
    target.chmod(0o644)
    built.write(target)
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600


# ---------------------------------------------------------------------------- preflight


def test_preflight_returns_every_named_gate_without_authorization(tmp_path: Path) -> None:
    gates = _gates(runtime="omp", authorize_live=False, model_cache=tmp_path / "none")
    assert set(gates) == {
        "live.authorization",
        "live.runtime_binary",
        "live.database",
        "live.psql_client",
        "live.sandbox",
        "live.model_cache",
        "live.credentials",
    }
    assert gates["live.authorization"].status in (BLOCKED, NOT_RUN)
    assert all(gate.status != PASS or gate.name != "live.authorization" for gate in gates.values())
    authorized = _gates(runtime="omp", authorize_live=True, model_cache=tmp_path / "none")
    assert authorized["live.authorization"].status == PASS  # positive twin


def test_unsupported_runtime_is_blocked(tmp_path: Path) -> None:
    gate = _gates(runtime="claude", authorize_live=True, model_cache=tmp_path)[
        "live.runtime_binary"
    ]
    assert gate.status == BLOCKED and "unsupported runtime" in gate.detail
    assert "claude" in gate.detail


def test_omp_binary_gate_reads_the_real_omp_version() -> None:
    gate = _gates(runtime="omp", authorize_live=True)["live.runtime_binary"]
    assert gate.status == PASS, gate.detail
    assert gate.evidence["version"] and gate.evidence["version"][0].isdigit()
    assert Path(gate.evidence["binary"]).name == "omp"


def test_omp_binary_gate_blocks_when_omp_is_not_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent-dir")
    gate = _gates(runtime="omp", authorize_live=True)["live.runtime_binary"]
    assert gate.status == BLOCKED and "omp binary not found" in gate.detail


def test_model_cache_blocks_naming_the_exact_snapshot_path(tmp_path: Path) -> None:
    model_id, revision = manifest._embed_model_pin()
    assert model_id == "lightonai/answerai-colbert-small-v1" and len(revision) == 40
    cache = tmp_path / "does-not-exist"
    expected = (
        cache / "hub" / "models--lightonai--answerai-colbert-small-v1" / "snapshots" / revision
    )
    gate = _gates(runtime="omp", authorize_live=True, model_cache=cache)["live.model_cache"]
    assert gate.status == BLOCKED
    assert str(expected) in gate.detail and gate.evidence["snapshot"] == str(expected)


def test_model_cache_passes_for_a_complete_snapshot_and_blocks_missing_files(
    tmp_path: Path,
) -> None:
    model_id, revision = manifest._embed_model_pin()
    snapshot = manifest.snapshot_dir(tmp_path, model_id, revision)
    snapshot.mkdir(parents=True)
    (snapshot / "modules.json").write_text("[]")
    gate = _gates(runtime="omp", authorize_live=True, model_cache=tmp_path)["live.model_cache"]
    assert gate.status == BLOCKED and str(snapshot / "model.safetensors") in gate.detail
    assert str(snapshot / "modules.json") not in gate.detail
    (snapshot / "model.safetensors").write_bytes(b"\0")
    ok = _gates(runtime="omp", authorize_live=True, model_cache=tmp_path)["live.model_cache"]
    assert ok.status == PASS and ok.evidence["snapshot"] == str(snapshot)


def test_model_cache_default_location_returns_a_gate_without_crashing() -> None:
    gate = _gates(runtime="omp", authorize_live=False)["live.model_cache"]
    assert isinstance(gate, Gate) and gate.status in (PASS, BLOCKED)
    if gate.status == BLOCKED:
        assert str(Path.home() / ".cache" / "huggingface") in gate.detail


def test_preflight_never_imports_heavy_embedding_modules(tmp_path: Path) -> None:
    import subprocess
    import sys

    code = (
        "import sys;from eval.accounting_airline.manifest import preflight_live;"
        "preflight_live(runtime='claude',authorize_live=False,model_cache=__import__('pathlib')"
        f".Path({str(tmp_path)!r}));"
        "bad=[m for m in ('torch','sentence_transformers','ssgrep.indexing.embed') "
        "if m in sys.modules];print(bad)"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "[]"


def test_credentials_gate_records_paths_only_and_names_problems(tmp_path: Path) -> None:
    canary = "TOP-SECRET-CANARY"
    good = tmp_path / "agent.db"
    good.write_text(canary)
    missing = tmp_path / "absent.db"
    directory = tmp_path / "adir"
    directory.mkdir()
    ok = _gates(runtime="omp", authorize_live=True, credential_paths=[good])["live.credentials"]
    assert ok.status == PASS and ok.evidence == {"paths": [str(good)]}
    assert canary not in json.dumps(ok.as_dict())
    bad = _gates(runtime="omp", authorize_live=True, credential_paths=[good, missing, directory])[
        "live.credentials"
    ]
    assert bad.status == BLOCKED
    assert str(missing) in bad.detail and str(directory) in bad.detail
    assert str(good) not in bad.detail.replace(str(good.parent), "")
    assert canary not in json.dumps(bad.as_dict())
    none = _gates(runtime="omp", authorize_live=True)["live.credentials"]
    assert none.status == NOT_RUN


def test_unreadable_credential_is_blocked(tmp_path: Path) -> None:
    locked = tmp_path / "locked.db"
    locked.write_text("x")
    locked.chmod(0o000)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("running with privileges that ignore file modes")
        gate = _gates(runtime="omp", authorize_live=True, credential_paths=[locked])[
            "live.credentials"
        ]
        assert gate.status == BLOCKED and "not readable" in gate.detail
    finally:
        locked.chmod(0o600)


def test_database_psql_and_sandbox_gates_reflect_the_real_host() -> None:
    gates = _gates(runtime="omp", authorize_live=True)
    import shutil

    assert (gates["live.psql_client"].status == PASS) == (shutil.which("psql") is not None)
    db = database.preflight()
    assert gates["live.database"].status == db.status
    from eval.accounting_airline import sandbox

    assert (gates["live.sandbox"].status == PASS) == (sandbox.availability() is None)
    for gate in gates.values():
        assert isinstance(gate, Gate)
