"""Live wiring tests never spend provider credits."""

import pytest

from eval.accounting_airline.live_bridge import validate_request
from eval.accounting_airline.live_prompts import prompt_for
from eval.accounting_airline.live_security import live_profile
from eval.accounting_airline.roots import LabRoots


def test_bridge_restricts_database_role_and_command():
    assert (
        validate_request(
            "sql", ["--role", "reader", "--database", "lab_a"], database="lab_a", writer=False
        )
        == "sql"
    )
    for tool, args in [
        ("sql", ["--role", "writer", "--database", "lab_a"]),
        ("sql", ["--role", "reader", "--database", "postgres"]),
        ("ssgrep", ["index"]),
        ("ssgrep", ["show", "/etc/passwd"]),
        ("bash", ["cat", "/etc/passwd"]),
    ]:
        with pytest.raises(ValueError):
            validate_request(tool, args, database="lab_a", writer=False)
    assert (
        validate_request(
            "ssgrep", ["show", "omp:abc:ep:0", "--json"], database="lab_b", writer=True
        )
        == "ssgrep"
    )


@pytest.mark.parametrize("suffix", [[], ["--json"]])
def test_bridge_accepts_readable_or_json_show(suffix):
    assert (
        validate_request(
            "ssgrep", ["show", "omp:abc:ep:0", *suffix], database="lab_b", writer=False
        )
        == "ssgrep"
    )


@pytest.mark.parametrize(
    "args",
    [
        ["show", "/etc/passwd"],
        ["show", "/etc/passwd", "--json"],
        ["show", "omp:abc:ep:0", "--unexpected"],
        ["show", "omp:abc:ep:0", "--json", "extra"],
        ["show", "omp:abc:ep:0", "extra"],
        ["show"],
        ["search", "connection rules"],
        ["search", "connection rules", "--json", "extra"],
        [],
    ],
)
def test_bridge_readable_show_keeps_lookup_boundary(args):
    with pytest.raises(ValueError):
        validate_request("ssgrep", args, database="lab_b", writer=False)


def test_live_prompts_only_supply_output_contract_not_answers():
    for task in ("L1", "L2", "L3", "L4"):
        text = prompt_for(task, "lab_test")
        assert "lab_test" in text
        for forbidden in ("MCT-", "45000", "195000", "F5", "B1", "B2"):
            assert forbidden not in text
    assert "-- step:cash" in prompt_for("L1", "lab_test")
    assert "-- step:plan" in prompt_for("L3", "lab_test")
    l1 = prompt_for("L1", "lab_test")
    assert "raw_free = capacity - occupied - live_holds" in l1
    assert "zero for cancelled physical flights and otherwise equals raw_free" in l1
    assert "integrity error to report" in l1


def test_profile_seals_repository_sources_and_credential_writes(tmp_path):
    roots = LabRoots.create(tmp_path)
    profile = live_profile(roots, "solver-b", real_home=tmp_path / "real-home")
    assert str(roots.runtime_a) in profile.text
    assert str(roots.ssgrep_data) in profile.text
    assert "/eval" in profile.text and "/openspec" in profile.text
    assert "deny file-write*" in profile.text
    assert str(tmp_path / "real-home") in profile.text
    a = live_profile(roots, "solver-a", real_home=tmp_path / "real-home")
    assert roots.runtime_a not in a.denied


def test_live_requires_authorization_before_runtime(tmp_path, monkeypatch):
    from eval.accounting_airline.live_run import run_live

    with pytest.raises(ValueError, match="authorization"):
        run_live(tmp_path, print, authorized=False)


def test_supervisor_import_ignores_solver_package(tmp_path):
    import subprocess
    import sys

    from eval.accounting_airline.live_bridge import TRUSTED_SHIM

    package = tmp_path / "eval"
    package.mkdir()
    marker = tmp_path / "escaped"
    (package / "__init__.py").write_text(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('unsafe')"
    )
    code = TRUSTED_SHIM.rsplit("raise SystemExit", 1)[0] + "print('trusted')"
    done = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "trusted"
    assert not marker.exists()


def test_sql_backend_sends_metacommand_as_protocol_sql(monkeypatch):
    import io

    from eval.accounting_airline import live_sql

    class Protocol:
        sent = None
        results = iter((2, None))

        def PQconnectdb(self, _conn):
            return 1

        def PQstatus(self, _conn):
            return 0

        def PQsendQuery(self, _conn, sql):
            self.sent = sql
            return 1

        def PQgetResult(self, _conn):
            return next(self.results)

        def PQresultStatus(self, _result):
            return 7

        def PQresultErrorField(self, _result, _field):
            return b"42601"

        def PQresultErrorMessage(self, _result):
            return b"syntax error"

        def PQclear(self, _result):
            pass

        def PQfinish(self, _conn):
            pass

    protocol = Protocol()
    monkeypatch.setattr(live_sql, "_lib", lambda _path: protocol)
    monkeypatch.setenv("AIRLINE_LIBPQ_PATH", "unit-test")
    monkeypatch.setenv("PGPASSFILE", "/private/run-owned-passfile")
    monkeypatch.setattr("sys.stdin", io.StringIO("\\! echo NEVER_EXECUTE"))
    assert live_sql.main(["-h", "127.0.0.1", "-p", "5432", "-U", "lab_reader", "-d", "lab_a"]) == 1
    assert protocol.sent == b"\\! echo NEVER_EXECUTE"


def test_live_profiles_do_not_allow_shared_scratch_writes(tmp_path):
    roots = LabRoots.create(tmp_path)
    for actor, other in (("solver-a", "solver-b"), ("solver-b", "solver-a")):
        profile = live_profile(roots, actor)
        assert "(deny file-write*)" in profile.text
        assert roots.tmp in profile.denied
        assert roots.tools / other in profile.denied
        assert f'(allow file-write* (subpath "{roots.root}"))' not in profile.text


@pytest.mark.parametrize(
    "status,expected", [(None, 0), ("BLOCKED", 3), ("NOT_RUN", 3), ("FAIL", 1)]
)
def test_standin_exit_keeps_required_gate_failures_visible(status, expected):
    from eval.accounting_airline.__main__ import _standin_report_exit
    from eval.accounting_airline.report import LIVE_GATE, GateEntry, Report

    report = Report("test-run", "standin", "scripted-standin", False, "scripted-standin")
    for group in report.gates:
        if group != LIVE_GATE:
            report.add_gate(group, GateEntry("exercised", "PASS", "controlled report fixture"))
    if status is not None:
        report.add_gate("scrape", GateEntry("C9", status, "required gate outcome"))
    assert report.gate_status(LIVE_GATE) == "NOT_RUN"
    assert _standin_report_exit(report) == expected


def test_sandbox_denies_actual_home_when_home_is_overridden(tmp_path, monkeypatch):
    from eval.accounting_airline.sandbox import account_home, profile_for_solver_b

    actual = account_home()
    monkeypatch.setenv("HOME", str(tmp_path / "not-the-account-home"))
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "configured-future-index"))
    roots = LabRoots.create(tmp_path)
    profile = profile_for_solver_b(roots)
    assert actual / ".omp/agent/sessions" in profile.denied
    assert tmp_path / "configured-future-index" in profile.denied
    import sys

    default = actual / (
        "Library/Application Support/ssgrep" if sys.platform == "darwin" else ".local/share/ssgrep"
    )
    assert default in profile.denied


@pytest.mark.usefixtures("live_sandbox_ready")
def test_live_profile_allows_python_startup_in_own_runtime(tmp_path):
    import sys

    from eval.accounting_airline.sandbox import run_sandboxed

    roots = LabRoots.create(tmp_path)
    target = roots.runtime_b / "own-runtime-positive"
    done = run_sandboxed(
        live_profile(roots, "solver-b"),
        [
            sys.executable,
            "-I",
            "-c",
            "from pathlib import Path;import sys;Path(sys.argv[1]).write_text('permitted')",
            str(target),
        ],
        env={"PATH": "/usr/bin:/bin", "HOME": str(roots.actor_home("solver-b"))},
        cwd=roots.project,
        timeout=10,
    )
    assert done.returncode == 0, (done.returncode, done.stderr)
    assert target.read_text() == "permitted"


def test_live_numeric_grading_accepts_integral_sql_decimal_without_rounding():
    from eval.accounting_airline.live_grading import normalize_tables

    tables = {
        "connections": [
            {"gap_minutes": "35.0000000000000000", "minimum_minutes": "45", "valid": "f"}
        ]
    }
    assert normalize_tables(tables)["connections"][0]["gap_minutes"] == "35"
    for bad in ("35.01", "NaN", "Infinity"):
        with pytest.raises(ValueError):
            normalize_tables({"connections": [{"gap_minutes": bad}]})


def test_source_evidence_ignores_markdown_not_wrong_values():
    from eval.accounting_airline.live_grading import missing_facts

    text = "| B1 | USD | 45000 |\n**B2 / V2**.\n| MCT-DEN-045 |"
    assert missing_facts(text, ["B1 USD 45000", "B2 V2", "MCT-DEN-045"]) == []
    assert missing_facts("| B1 | USD | 450000 |", ["B1 USD 45000"]) == ["B1 USD 45000"]
    assert missing_facts("| B1 | USD | 45000.1 |", ["B1 USD 45000"]) == ["B1 USD 45000"]


def test_continuation_refuses_any_prior_b_attempt(tmp_path, monkeypatch):
    import json
    import subprocess

    from eval.accounting_airline.live_native import LiveBlocked
    from eval.accounting_airline.live_resume import resume_live

    roots = LabRoots.create(tmp_path)
    (roots.artifacts / "live-evidence.json").write_text(
        json.dumps(
            {"turns": {"L1": {}, "L2": {}, "L3": {}}, "native_ids": {"b": "already-started"}}
        )
    )
    (roots.artifacts / "manifest.json").write_text("{}")
    (roots.artifacts / "report.json").write_text("{}")
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch another B")
    )
    with pytest.raises(LiveBlocked, match="no B attempt"):
        resume_live(roots.root, print, authorized=True, model="openai-codex/gpt-6-luna")
