"""Isolation, ownership-checked teardown, blocked gates and the deterministic entry point."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from eval.accounting_airline import __main__ as entry, database, dataset, domain_lane, faults
from eval.accounting_airline.gates import BLOCKED, FAIL, NOT_RUN, PASS


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=False)


# ------------------------------------------------------------------- offline behaviour


def test_caller_supplied_dsn_is_refused_and_never_a_parameter() -> None:
    with pytest.raises(database.ForeignDatabaseRefused, match="DSN"):
        database.refuse_external_dsn("postgresql://prod:secret@db.example.com/prod")
    database.refuse_external_dsn(None)  # no address supplied: fine
    import inspect

    for fn in (database.LabInstance.start, database.LabInstance.psql, database.LabInstance.popen):
        assert not {"dsn", "host", "port", "url"} & set(inspect.signature(fn).parameters)
    assert "PGHOST" not in database._clean_env({"PGPASSWORD": "x"}) and (
        "PGPASSWORD" in database._clean_env({"PGPASSWORD": "x"})
    )


def test_ambient_pg_environment_is_not_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGHOST", "prod.example.com")
    monkeypatch.setenv("DATABASE_URL", "postgresql://prod/db")
    env = database._clean_env({"PGAPPNAME": "lab"})
    assert "PGHOST" not in env and env["PGAPPNAME"] == "lab"


def test_missing_docker_is_a_blocked_gate_not_an_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(database.shutil, "which", lambda _: None)
    gate = database.preflight()
    assert gate.status == BLOCKED and "docker CLI not found" in gate.detail


def test_unreachable_daemon_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args, **kwargs):
        raise OSError("socket gone")

    monkeypatch.setattr(database, "_docker", broken)
    monkeypatch.setattr(database.shutil, "which", lambda _: "/usr/bin/docker")
    gate = database.preflight()
    assert gate.status == BLOCKED and "unreachable" in gate.detail and "OSError" in gate.detail


def test_missing_pinned_image_is_blocked_and_never_substituted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if database.shutil.which("docker") is None:
        pytest.skip("docker CLI absent: preflight blocking is covered by the test above")
    monkeypatch.setattr(dataset, "IMAGE_REF", "postgres@sha256:" + "0" * 64)
    gate = database.preflight()
    assert gate.status == BLOCKED and "never pulls" in gate.detail


def test_blocked_lane_reports_blocked_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        domain_lane, "preflight", lambda: database.blocked("database.prerequisites", "no docker")
    )
    code = entry.main(["--mode", "deterministic", "--run-root-parent", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 3
    assert "[BLOCKED] database.prerequisites - no docker" in out
    assert "domain=BLOCKED" in out and "PASS" not in out.split("domain=")[0].replace(
        "[BLOCKED]", ""
    )
    report = json.loads(
        next(tmp_path.glob("ssgrep-airline-run-*/artifacts/domain-lane.json")).read_text()
    )
    assert report["domain_status"] == "BLOCKED"
    assert [g["status"] for g in report["gates"]] == [BLOCKED]


def test_live_mode_is_explicitly_not_run_and_never_passes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = entry.main(["--mode", "live", "--runtime", "claude", "--run-root-parent", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 3 and "[NOT_RUN" in out and "live.two_session_lab" in out
    report = json.loads(
        next(tmp_path.glob("ssgrep-airline-run-*/artifacts/live-lane.json")).read_text()
    )
    gate = report["gates"][-1]
    assert gate["name"] == "live.two_session_lab" and gate["status"] == NOT_RUN


def test_run_root_is_owner_only_and_reports_are_private(tmp_path: Path) -> None:
    root, run_id = domain_lane.make_run_root(tmp_path)
    for sub in ("", "verifier", "artifacts"):
        assert stat.S_IMODE((root / sub).stat().st_mode) == 0o700
    domain_lane.write_private(root / "artifacts" / "r.json", {"run": run_id})
    assert stat.S_IMODE((root / "artifacts" / "r.json").stat().st_mode) == 0o600


def test_lane_gate_status_helper_never_hides_a_failure() -> None:
    report = domain_lane.LaneReport("r", Path("."))
    report.add(domain_lane.Gate("a", PASS))
    assert report.domain_status == PASS
    report.add(domain_lane.Gate("b", FAIL, "x"))
    report.add(domain_lane.Gate("c", BLOCKED, "y"))
    assert report.domain_status == FAIL
    live_only = domain_lane.LaneReport("r", Path("."))
    live_only.add(domain_lane.Gate("live.two_session_lab", NOT_RUN, "later"))
    live_only.add(domain_lane.Gate("a", PASS))
    assert live_only.domain_status == PASS  # the live gate is reported separately, never merged


# ------------------------------------------------------------------ real docker lifecycle


@pytest.mark.postgres
def test_teardown_refuses_a_container_without_this_runs_label(lab) -> None:
    foreign = f"ssgrep-airline-foreign{os.getpid()}"
    created = _docker(
        "create", "--name", foreign, "--label", "ssgrep-lab-run=someone-else", dataset.IMAGE_REF
    )
    assert created.returncode == 0, created.stderr
    cid = created.stdout.strip()
    try:
        forged = database.LabInstance(
            run_id="forgedrun", container_id=cid, host_port=1, _passwords={}
        )
        with pytest.raises(database.OwnershipError, match="forgedrun"):
            forged.teardown()
        assert not forged.removed
        assert _docker("inspect", "--format", "{{.Name}}", cid).stdout.strip() == "/" + foreign
        assert database.LabInstance.remove_run("forgedrun") == []  # label lookup finds nothing
        assert _docker("inspect", cid).returncode == 0  # still there
        assert _docker("inspect", lab.instance.container_id).returncode == 0  # ours untouched
    finally:
        _docker("rm", "-f", "-v", cid)


@pytest.mark.postgres
def test_remove_run_requires_label_kind_and_name_to_all_match(lab) -> None:
    run = f"decoy{os.getpid()}"
    wrong_name = _docker(
        "create",
        "--name",
        f"not-ours-{run}",
        "--label",
        f"{database.RUN_LABEL}={run}",
        "--label",
        f"{database.KIND_LABEL}={database.KIND_VALUE}",
        dataset.IMAGE_REF,
    )
    no_kind = _docker("create", "--name", f"{database.NAME_PREFIX}{run}", "--label",
                      f"{database.RUN_LABEL}={run}", dataset.IMAGE_REF)  # fmt: skip
    assert wrong_name.returncode == 0 and no_kind.returncode == 0, (
        wrong_name.stderr,
        no_kind.stderr,
    )
    ids = [wrong_name.stdout.strip(), no_kind.stdout.strip()]
    try:
        # the run label matches both decoys but neither carries the full ownership triple
        assert database.LabInstance.remove_run(run) == []
        assert all(_docker("inspect", cid).returncode == 0 for cid in ids)
        assert (
            _docker("inspect", lab.instance.container_id).returncode == 0
        )  # the live lab is untouched
        assert database.LabInstance.remove_run("no-such-run") == []
    finally:
        for cid in ids:
            _docker("rm", "-f", "-v", cid)


@pytest.mark.postgres
def test_failed_run_keeps_a_bounded_private_manifest_and_removes_only_its_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, run_id = domain_lane.make_run_root(tmp_path)

    def explode(_variant: str = "correct") -> list[str]:
        raise database.LabError("injected provisioning failure " + "x" * 5000)

    monkeypatch.setattr(faults, "baseline_sql", explode)
    report = domain_lane.run_lane(root, run_id)
    statuses = {g.name: g.status for g in report.gates}
    assert statuses["lane.execution"] == FAIL and statuses["cleanup.owned_resources"] == PASS
    manifest_path = root / "artifacts" / "failure-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["error_type"] == "LabError" and manifest["run_id"] == run_id
    assert len(manifest["message"]) <= 2000  # bounded
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert (
        _docker("ps", "-aq", "--filter", f"label={database.RUN_LABEL}={run_id}").stdout.split()
        == []
    )
    cleanup = next(g for g in report.gates if g.name == "cleanup.owned_resources")
    removed = cleanup.evidence["lab_removals"]  # what the lab itself removed, not daemon churn
    assert removed and all(r["run_label"] == run_id and r["succeeded"] for r in removed)
    assert {r["name"] for r in removed} == {f"/{database.NAME_PREFIX}{run_id}"}


@pytest.mark.postgres
def test_clone_template_baseline_is_isolated_per_scenario(lab) -> None:
    first, state_a = lab.prep.clone("iso_a")
    second, state_b = lab.prep.clone("iso_b")
    try:
        lab.instance.psql(
            first,
            f"UPDATE {dataset.SCHEMA}.bookings SET service_priority = 9 WHERE booking_id = 'B6';",
        )
        changed = lab.instance.read_state(first)["bookings"]
        untouched = lab.instance.read_state(second)["bookings"]
        assert next(b for b in changed if b["booking_id"] == "B6")["service_priority"] == 9
        assert untouched == state_b["bookings"] == lab.base.state["bookings"]
        recloned, _ = lab.prep.clone("iso_c")
        assert lab.instance.read_state(recloned) == lab.base.state  # baseline never drifts
        lab.instance.drop_clone("iso_c")
    finally:
        lab.instance.drop_clone("iso_a")
        lab.instance.drop_clone("iso_b")
        lab.prep.created.clear()


@pytest.mark.postgres
def test_deterministic_entry_point_runs_the_whole_domain_lane(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = entry.main(["--mode", "deterministic", "--run-root-parent", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0, out
    lines = [ln for ln in out.splitlines() if ln.startswith("[")]
    assert len(lines) >= 20 and all(ln.startswith("[PASS   ]") for ln in lines)
    names = {ln.split("] ", 1)[1].split(" - ")[0] for ln in lines}
    assert {
        "seed.integrity", "l1.cash", "l1.seats", "l2.affected", "l3.joint_plan",
        "sensitivity.domain", "l4.joint_apply", "l4.race", "l4.replay", "l4.key_conflict",
        "l4.rollback", "l4.stale_plan", "sensitivity.l4", "cleanup.owned_resources",
    } <= names  # fmt: skip
    assert "domain=PASS live=NOT_RUN" in out
    root = next(tmp_path.glob("ssgrep-airline-run-*"))
    report_path = root / "artifacts" / "domain-lane.json"
    report = json.loads(report_path.read_text())
    assert report["domain_status"] == PASS and all(g["status"] == PASS for g in report["gates"])
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    assert (root / "verifier" / "seed.sql").read_text() == dataset.seed_sql()
    secret_material = [report_path.read_text(), (root / "verifier" / "seed.sql").read_text()]
    run_id = report["run_id"]
    assert (
        _docker("ps", "-aq", "--filter", f"label={database.RUN_LABEL}={run_id}").stdout.split()
        == []
    )
    for text in secret_material:
        assert "PGPASSWORD" not in text and "password" not in text.lower()
    # the report carries the ACTUAL result rows that were compared
    cash = next(g for g in report["gates"] if g["name"] == "l1.cash")["evidence"]["actual"]
    assert {(r["booking_id"], r["currency"], r["settled_minor"]) for r in cash} >= {
        ("B1", "USD", 45000)
    }
    race = next(g for g in report["gates"] if g["name"] == "l4.race")["evidence"]
    assert race["final_occupancy"] == {"F3": 2, "F4": 2}
