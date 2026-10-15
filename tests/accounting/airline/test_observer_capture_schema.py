"""Actual shim capture metadata crosses only the closed, independently bound spool grammar."""

import copy

import pytest

from eval.accounting_airline import observer as obs
from tests.accounting.airline.test_capture_failure_semantics import shim
from tests.accounting.airline.test_lab_observer import Rig
from tests.accounting.airline.test_observer_spool_security import ingest, native, spool


def rewrite(path, records):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(b"")
    trace = obs.Trace(path, records[0]["run_id"])
    for record in records:
        trace.append({k: v for k, v in record.items() if k not in ("seq", "prev", "hash")})
    return obs.read_trace(path)  # Establish a valid hash chain, independent of payload validity.


def accept(observer, rig):
    return observer.ingest_spool(
        rig.spool, obs._spool_blob_dir(rig.spool), expected_actor="solver-b", expected_task="L2"
    )


def test_actual_ssgrep_and_sql_shim_captures_ingest_idempotently_without_granting_native_authority(
    tmp_path,
):
    rig = Rig(tmp_path)
    assert rig.run("ssgrep search q").returncode == 0
    assert rig.run("sql --role reader --database airline", stdin="SELECT 1").returncode == 0
    records = rig.events()
    captures = [r["capture"] for r in records if r["kind"] == "result"]
    assert len(captures) == 2 and all(c["capture_status"] == "completed" for c in captures)
    verifier = obs.Observer(tmp_path / "verifier", "run-1")
    assert accept(verifier, rig) == len(records) and accept(verifier, rig) == 0
    retained = [r["capture"] for r in obs.read_trace(verifier.trace.path) if r["kind"] == "result"]
    assert retained == captures
    assert all(c.state == "unknown" and c.native_call_id is None for c in verifier.freeze().calls)


BAD_CAPTURE = [
    ("extra", {"raw_stdout": "CAPTURE_CANARY_PRIVATE"}),
    ("extra", {"artifact_dir": "/untrusted/CAPTURE_CANARY_PRIVATE"}),
    ("value", None),
    ("value", "CAPTURE_CANARY_PRIVATE"),
    ("missing", "stdout_bytes"),
    ("set", ("pid", True)),
    ("set", ("pid", 0)),
    ("set", ("returncode", True)),
    ("set", ("returncode", 9)),
    ("set", ("stdout_bytes", -1)),
    ("set", ("stdout_bytes", 1.25)),
    ("set", ("stdout_bytes", True)),
    ("set", ("stdout_bytes", 2**63)),
    ("set", ("stdout_limit_bytes", -1)),
    ("set", ("stdout_limit_bytes", 1_000_001)),
    ("set", ("stderr_limit_bytes", False)),
    ("set", ("timed_out", 1)),
    ("set", ("cancelled", "false")),
    ("set", ("capture_status", "PASS")),
    ("set", ("termination_cause", "timeout")),
    ("set", ("capture_complete", False)),
    ("set", ("pipes_drained", False)),
    ("set", ("output_limit", True)),
    ("set", ("stdout_truncated", True)),
    ("set", ("observed_over_limit_streams", ["stdout", "stdout"])),
    ("set", ("output_limit_streams", ["CAPTURE_CANARY_PRIVATE"])),
    ("set", ("output_limit_streams", [{"raw": "CAPTURE_CANARY_PRIVATE"}])),
    ("set", ("capture_observation", "CAPTURE_CANARY_PRIVATE")),
    ("set", ("capture_observation", "termination_snapshot")),
]


@pytest.mark.parametrize("operation,change", BAD_CAPTURE)
def test_bad_capture_is_rejected_before_any_record_or_blob_is_copied(tmp_path, operation, change):
    rig = Rig(tmp_path)
    assert rig.run("ssgrep search q").returncode == 0
    baseline = obs.Observer(tmp_path / "baseline", "run-1")
    assert (
        accept(baseline, rig) == 2
    )  # A bad-schema-only implementation cannot satisfy this control.
    records = copy.deepcopy(rig.events())
    result = next(r for r in records if r["kind"] == "result")
    if operation == "extra":
        result["capture"].update(change)
    elif operation == "value":
        result["capture"] = change
    elif operation == "missing":
        result["capture"].pop(change)
    else:
        result["capture"][change[0]] = change[1]
    bad = tmp_path / "mutated" / "spool.ndjson"
    assert len(rewrite(bad, records)) == 2
    verifier = obs.Observer(tmp_path / "rejected", "run-1")
    with pytest.raises(obs.ObserverError, match="capture"):
        verifier.ingest_spool(
            bad, obs._spool_blob_dir(rig.spool), expected_actor="solver-b", expected_task="L2"
        )
    assert obs.read_trace(verifier.trace.path) == []
    assert not any(verifier.blobs.iterdir())
    assert b"CAPTURE_CANARY_PRIVATE" not in verifier.trace.path.read_bytes()


@pytest.mark.parametrize(
    "kind", ["timeout", "interrupted", "output_limit", "invalid_output_encoding"]
)
def test_actual_failed_shim_capture_metadata_ingests_without_success_upgrade(tmp_path, kind):
    bodies = {
        "timeout": "import os,time\nos.write(1,b'partial'); time.sleep(60)\n",
        "interrupted": "import os,signal\nos.write(1,b'partial'); "
        "os.kill(os.getpid(),signal.SIGTERM)\n",
        "output_limit": "import os,time\nos.write(1,b'x'*2_000_000); time.sleep(60)\n",
        "invalid_output_encoding": "import os\nos.write(1,b'\\xff')\n",
    }
    kit, done, records = shim(tmp_path, bodies[kind])
    assert done.returncode != 0 and done.stdout == b""
    result = next(r for r in records if r["kind"] == "result")
    verifier = obs.Observer(tmp_path / "independent-verifier", "bounded-run")
    assert verifier.ingest_spool(
        kit.spool, obs._spool_blob_dir(kit.spool), expected_actor="solver-a", expected_task="A/L1"
    ) == len(records)
    retained = next(r for r in obs.read_trace(verifier.trace.path) if r["kind"] == "result")
    assert retained["capture"] == result["capture"] and retained["reason"] == kind
    assert retained["capture"]["capture_complete"] is False
    assert verifier.freeze().calls[0].state != "verified"


def protected_rig(tmp_path):
    rig = Rig(tmp_path)
    verifier = obs.Observer(tmp_path / "verifier", "run-1")
    rig.spool = verifier.dir / "solver-b" / "spool.ndjson"
    rig.spool.parent.mkdir(mode=0o700)
    rig.env[obs.ENV_SPOOL] = str(rig.spool)
    assert rig.run("ssgrep search q").returncode == 0
    assert accept(verifier, rig) == 2
    return verifier, rig


def test_actual_protected_capture_metadata_participates_in_exact_supervisor_attestation(tmp_path):
    verifier, rig = protected_rig(tmp_path)
    args = dict(expected_actor="solver-b", expected_tasks={"L2"})
    blobs = obs._spool_blob_dir(rig.spool)
    assert verifier.attest_supervisor_spool(rig.spool, blobs, **args) == 1
    assert verifier.attest_supervisor_spool(rig.spool, blobs, **args) == 0
    call = verifier.freeze().calls[0]
    assert call.state == "verified" and call.stdout == "fake ssgrep search q\n"
    assert call.native_call_id is None  # Capture metadata does not manufacture native exposure.


def test_well_typed_changed_capture_cannot_be_attested_against_retained_observation(tmp_path):
    verifier, rig = protected_rig(tmp_path)
    records = rig.events()
    next(r for r in records if r["kind"] == "result")["capture"]["pid"] += 1
    rewrite(rig.spool, records)
    before = verifier.trace.path.read_bytes()
    with pytest.raises(obs.ObserverError, match="protected completion differs"):
        verifier.attest_supervisor_spool(
            rig.spool,
            obs._spool_blob_dir(rig.spool),
            expected_actor="solver-b",
            expected_tasks={"L2"},
        )
    assert verifier.trace.path.read_bytes() == before


def test_independent_native_result_without_subprocess_measurements_does_not_get_synthetic_capture(
    tmp_path,
):
    verifier, trace, result = spool(tmp_path)
    association = native(verifier, tmp_path)
    trace.append(result)
    assert "capture" not in result
    assert ingest(verifier, trace, association) == 2
    call = verifier.freeze().calls[0]
    assert call.state == "verified" and call.native_call_id == "native-call"
    retained = next(r for r in obs.read_trace(verifier.trace.path) if r["kind"] == "result")
    assert "capture" not in retained


def test_real_process_start_failure_has_no_fabricated_capture_metadata(tmp_path):
    rig = Rig(tmp_path)
    rig.env[obs.ENV_REAL + "SSGREP"] = str(tmp_path / "missing-executable")
    assert rig.run("ssgrep search q").returncode != 0
    result = next(r for r in rig.events() if r["kind"] == "result")
    assert (
        result["state"] == "unknown"
        and result["reason"] == "FileNotFoundError"
        and "capture" not in result
    )
    verifier = obs.Observer(tmp_path / "verifier", "run-1")
    assert accept(verifier, rig) == 2
    assert verifier.freeze().calls[0].state == "unknown"
