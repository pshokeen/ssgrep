"""Real-subprocess capture behavior (tasks 3.4, 3.6, 3.7).

Every case runs the production ``usecli`` entry point in a child process
with an isolated data root; only the retrieval leaf
(``search_module.search`` / ``detail.show``) is substituted by
``capture_driver.py`` so the cases do not need an embedding model or index.
Output bytes and exit codes are compared byte-for-byte across capture
conditions.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from ssgrep.accounting.capture import journal_reader
from ssgrep.accounting.ledger.contracts import Outcome, Phase
from ssgrep.accounting.privacy import policy

ROOT = Path(__file__).resolve().parents[3]
DRIVER = Path(__file__).with_name("capture_driver.py")
PROFILE = "personal-local"
QUERY = "payments flake SECRET_QUERY_CANARY"
WHERE = "project = '/private/WHERE_CANARY'"


@dataclass(frozen=True)
class Run:
    code: int
    out: bytes
    err: bytes


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT), SSGREP_DATA_DIR=str(tmp_path / "data"))
    env.pop("SSGREP_CORRELATION_ID", None)
    env.update(extra)
    return env


def run(
    tmp_path: Path,
    *args: str,
    driver: str = "ok",
    cwd: Path | None = None,
    journal_fails: bool = False,
    fsize_limit: int | None = None,
    mcp: bool = False,
    render_metadata_fails: bool = False,
) -> Run:
    env = _env(tmp_path, SSGREP_DRIVER=driver)
    if mcp:
        env["SSGREP_DRIVER_TRANSPORT"] = "mcp"
    if journal_fails:
        env["SSGREP_DRIVER_JOURNAL"] = "fail"
    if render_metadata_fails:
        env["SSGREP_DRIVER_RENDER_METADATA"] = "fail"
    if fsize_limit is not None:
        env["SSGREP_DRIVER_FSIZE_LIMIT"] = str(fsize_limit)
        # This process cannot write coverage's SQLite database under RLIMIT_FSIZE.
        # Keep the real pressure/exit behavior; unrestricted twins measure these paths.
        env.pop("COVERAGE_PROCESS_CONFIG", None)
        env.pop("COVERAGE_PROCESS_START", None)

    done = subprocess.run(
        [sys.executable, str(DRIVER), *args],
        env=env,
        cwd=cwd or ROOT,
        capture_output=True,
        timeout=120,
    )
    return Run(done.returncode, done.stdout, done.stderr)


def events(profile: str = PROFILE):
    return journal_reader.read_journal(profile, quarantine=False)


def prepare_condition(condition: str, live: Path, tmp_path: Path) -> None:
    """Create the fault a capture condition names; every one must be UID-independent."""
    profile_dir = live / "data" / "accounting" / PROFILE
    if condition == "unwritable":
        profile_dir.parent.mkdir(parents=True, exist_ok=True)
        if os.geteuid() == 0:  # root ignores mode bits: a plain file blocks every UID
            profile_dir.write_bytes(b"not a directory\n")
        else:
            profile_dir.mkdir(mode=0o500)
            os.chmod(profile_dir, 0o500)
    elif condition == "unsafe_symlink":
        victim = tmp_path / "victim"
        victim.mkdir()
        profile_dir.parent.mkdir(parents=True, exist_ok=True)
        profile_dir.symlink_to(victim)


def assert_fault_took_effect(condition: str, tmp_path: Path) -> None:
    """Equal output is only meaningful when the declared fault really happened."""
    profile_dir = tmp_path / "live" / "data" / "accounting" / PROFILE
    if condition == "unwritable":  # nothing was written into or over the blocked profile
        if profile_dir.is_file():
            assert profile_dir.read_bytes() == b"not a directory\n"
        else:
            assert list(profile_dir.iterdir()) == []
    if condition == "full_disk":
        assert events().events == []
    if condition == "unsafe_symlink":
        assert list((tmp_path / "victim").iterdir()) == []
    if condition == "file_size_limit":
        read = events()
        assert read.incomplete_tails or read.quarantined
        assert all(set(p) != {"started", "completed", "rendered"} for p in phases(read).values())


def phases(read) -> dict[str, list[str]]:
    by_id: dict[str, list[str]] = {}
    for event in read.events:
        by_id.setdefault(event.execution_id, []).append(str(event.phase))
    return {k: sorted(v) for k, v in by_id.items()}


CASES = [
    ("search", ["search", QUERY, "--where", WHERE], "ok"),
    ("search_json", ["--json", "search", QUERY], "ok"),
    ("search_empty", ["search", QUERY], "empty"),
    ("search_empty_json", ["--json", "search", QUERY], "empty"),
    ("search_missing_json", ["--json", "search", QUERY], "missing"),
    ("search_missing", ["search", QUERY], "missing"),
    ("search_crash", ["--json", "search", QUERY], "crash"),
    ("search_crash_plain", ["search", QUERY], "crash"),
    ("show", ["show", "claude:s:ep:1"], "ok"),
    ("show_json", ["--json", "show", "claude:s:ep:1"], "ok"),
    ("show_unknown_json", ["--json", "show", "unknown"], "ok"),
    ("show_missing_json", ["--json", "show", "claude:s:ep:1"], "missing"),
]


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Run]:
    """Output of every case with accounting disabled (no policy file at all)."""
    root = tmp_path_factory.mktemp("baseline")
    return {name: run(root, *args, driver=driver) for name, args, driver in CASES}


def test_baseline_cases_cover_distinct_exit_codes(baseline: dict[str, Run]) -> None:
    assert {r.code for r in baseline.values()} == {0, 1, 3, 4}


@pytest.mark.parametrize(
    "condition", ["enabled", "unwritable", "full_disk", "file_size_limit", "unsafe_symlink"]
)
def test_output_bytes_and_exit_codes_identical_under_every_capture_condition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, baseline: dict[str, Run], condition: str
) -> None:
    live = tmp_path / "live"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(live / "data"))
    policy.enable_collection(PROFILE)
    journal_fails = condition == "full_disk"
    prepare_condition(condition, live, tmp_path)
    try:
        for name, args, driver in CASES:
            got = run(
                live,
                *args,
                driver=driver,
                journal_fails=journal_fails,
                fsize_limit=300 if condition == "file_size_limit" else None,
            )
            assert got == baseline[name], (condition, name)
    finally:
        profile_dir = live / "data" / "accounting" / PROFILE
        if profile_dir.exists() and not profile_dir.is_symlink():
            os.chmod(profile_dir, 0o700)
    assert_fault_took_effect(condition, tmp_path)
    if condition == "enabled":
        read = events()
        assert read.damage == [] and read.events
        assert all(set(p) >= {"started", "completed"} for p in phases(read).values())


MCP_CASES = [
    ("search", ["search_sessions", QUERY], "ok"),
    ("search_empty", ["search_sessions", QUERY], "empty"),
    ("search_missing", ["search_sessions", QUERY], "missing"),
    ("search_crash", ["search_sessions", QUERY], "crash"),
    ("show", ["show_session", "claude:s:ep:1"], "ok"),
    ("show_unknown", ["show_session", "unknown"], "ok"),
    ("show_missing", ["show_session", "claude:s:ep:1"], "missing"),
]


@pytest.fixture(scope="module")
def mcp_baseline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Run]:
    """MCP tool payloads with accounting disabled (no policy file at all)."""
    root = tmp_path_factory.mktemp("mcp-baseline")
    return {name: run(root, *args, driver=driver, mcp=True) for name, args, driver in MCP_CASES}


def test_mcp_baseline_covers_success_empty_and_error_payloads(
    mcp_baseline: dict[str, Run],
) -> None:
    assert all(r.code == 0 for r in mcp_baseline.values())
    assert b'"results": [{' in mcp_baseline["search"].out
    assert b'"total_matches": 0' in mcp_baseline["search_empty"].out
    assert b"No global index found" in mcp_baseline["search_missing"].out
    assert b"service exploded" in mcp_baseline["search_crash"].out
    assert b"search_sessions failed" in mcp_baseline["search_crash"].err
    assert b"A title" in mcp_baseline["show"].out
    assert b"Episode not found: unknown" in mcp_baseline["show_unknown"].out
    # Search and show share one actionable missing-index message; every other case differs.
    assert mcp_baseline["show_missing"].out == mcp_baseline["search_missing"].out
    assert len({r.out for r in mcp_baseline.values()}) == len(MCP_CASES) - 1


@pytest.mark.parametrize(
    "condition", ["enabled", "unwritable", "full_disk", "file_size_limit", "unsafe_symlink"]
)
def test_mcp_payloads_identical_under_every_capture_condition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mcp_baseline: dict[str, Run],
    condition: str,
) -> None:
    live = tmp_path / "live"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(live / "data"))
    policy.enable_collection(PROFILE)
    prepare_condition(condition, live, tmp_path)
    try:
        for name, args, driver in MCP_CASES:
            got = run(
                live,
                *args,
                driver=driver,
                journal_fails=condition == "full_disk",
                fsize_limit=300 if condition == "file_size_limit" else None,
                mcp=True,
            )
            assert got == mcp_baseline[name], (condition, name)
    finally:
        profile_dir = live / "data" / "accounting" / PROFILE
        if profile_dir.exists() and not profile_dir.is_symlink():
            os.chmod(profile_dir, 0o700)
    assert_fault_took_effect(condition, tmp_path)
    if condition == "enabled":
        read = events()
        assert read.damage == [] and read.events
        assert {str(e.transport) for e in read.events} == {"mcp"}
        assert all(set(p) >= {"started", "completed"} for p in phases(read).values())


def test_disabled_by_default_creates_no_accounting_files(tmp_path: Path) -> None:
    got = run(tmp_path, "--json", "search", QUERY)
    assert got.code == 0 and b'"results"' in got.out
    assert not (tmp_path / "data" / "accounting").exists()


def test_paused_profile_records_nothing_and_resume_records_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    policy.enable_collection(PROFILE)
    policy.pause(PROFILE)
    paused = run(tmp_path, "--json", "search", QUERY)
    assert paused.code == 0 and phases(events()) == {}
    policy.resume(PROFILE)
    run(tmp_path, "--json", "search", QUERY)
    (only,) = phases(events()).values()
    assert only == ["completed", "rendered", "started"]


def test_workspace_scope_is_decided_per_event_from_that_events_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    inside = tmp_path / "work" / "team" / "repo"
    outside = tmp_path / "work" / "team-other"
    inside.mkdir(parents=True)
    outside.mkdir(parents=True)
    root = tmp_path / "work" / "team"
    policy.enable_collection(PROFILE)
    policy.save_profile(
        policy.load_profile_policy(PROFILE).__class__(
            profile_id=PROFILE, roots=(str(root),), collection_enabled=True
        )
    )

    run(tmp_path, "--json", "search", QUERY, cwd=inside)
    run(tmp_path, "--json", "search", QUERY, cwd=outside)  # lexical-prefix sibling: excluded
    link = tmp_path / "link-to-outside"
    link.symlink_to(outside)
    run(tmp_path, "--json", "search", QUERY, cwd=link)  # symlink escape: excluded

    read = events()
    assert {e.cwd for e in read.events} == {os.path.realpath(inside)}
    assert len(phases(read)) == 1


def test_journal_contains_phase_evidence_without_plaintext_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    policy.enable_collection(PROFILE)
    run(tmp_path, "search", QUERY, "--where", WHERE)
    run(tmp_path, "--json", "show", "claude:s:ep:1")

    read = events()
    by_op = {(str(e.operation), str(e.phase)): e for e in read.events}
    done = by_op[("search", "completed")]
    assert done.outcome == Outcome.OK and done.refs == ("claude:session-0:episode:0",)
    assert done.ranks_scores == ((1, 1000),) and done.mono_ns and done.mono_ns > 0
    assert done.limit == 10 and done.token_budget == 1500 and done.version != ""
    shown = by_op[("show", "rendered")]
    assert shown.rendered_bytes is not None and shown.rendered_bytes.value is None
    assert shown.rendered_bytes.reason == "usecli_owns_serialization"
    assert shown.truncated is True  # the detail's own response_truncated flag
    assert str(by_op[("search", "rendered")].transport) == "cli"

    for path in (tmp_path / "data").rglob("*"):
        if path.is_file():
            blob = path.read_bytes()
            for canary in (b"SECRET_QUERY_CANARY", b"WHERE_CANARY", b"payments flake"):
                assert canary not in blob, path


@pytest.mark.parametrize(
    ("args", "driver", "outcome"),
    [
        (["--json", "search", QUERY], "empty", Outcome.EMPTY),
        (["--json", "search", QUERY], "missing", Outcome.ERROR),
        (["--json", "search", QUERY], "crash", Outcome.ERROR),
        (["search", QUERY], "crash", Outcome.ERROR),
        (["--json", "show", "unknown"], "ok", Outcome.EMPTY),
    ],
)
def test_json_exit_paths_still_leave_started_and_completed_but_no_fake_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, args: list[str], driver: str, outcome: Outcome
) -> None:
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    policy.enable_collection(PROFILE)
    got = run(tmp_path, *args, driver=driver)
    assert got.code != 0

    read = events()
    (sequence,) = phases(read).values()
    assert "started" in sequence and "completed" in sequence
    assert "rendered" not in sequence  # an exit before rendering must not fabricate one
    (completed,) = [e for e in read.events if e.phase is Phase.COMPLETED]
    assert completed.outcome == outcome


def test_exception_transparency_when_search_and_journal_append_both_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plain = run(tmp_path / "plain", "--json", "search", QUERY, driver="crash")
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "both" / "data"))
    policy.enable_collection(PROFILE)
    both = run(tmp_path / "both", "--json", "search", QUERY, driver="crash", journal_fails=True)
    assert both == plain
    assert b"service exploded" in both.out + both.err
    assert b"No space" not in both.out + both.err
    assert events().events == []  # the append really failed: nothing was recorded


def test_closed_output_pipe_keeps_exit_behavior_and_started_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    policy.enable_collection(PROFILE)
    plain_root = tmp_path / "plain"  # no policy there: capture disabled

    def piped(root: Path, *args: str) -> tuple[int, int, bytes]:
        """Like ``cmd | head -c 1``: read one byte, close, then collect exit and stderr."""
        producer = subprocess.Popen(
            [sys.executable, str(DRIVER), *args],
            env=_env(root, SSGREP_DRIVER="many"),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert producer.stdout is not None and producer.stderr is not None
        first = producer.stdout.read(1)
        producer.stdout.close()
        stderr = producer.stderr.read()
        return producer.wait(timeout=120), len(first), stderr

    def normalized(stderr: bytes) -> list[bytes]:
        # Tracebacks carry addresses/paths of the same code; compare the error lines.
        return [ln for ln in stderr.splitlines() if ln.startswith((b"BrokenPipeError", b"ERROR"))]

    for args in (("--json", "search", QUERY), ("search", QUERY)):
        without = piped(plain_root, *args)
        with_capture = piped(tmp_path, *args)
        assert with_capture[:2] == without[:2], args
        assert normalized(with_capture[2]) == normalized(without[2]), args
    read = events()
    assert len(phases(read)) == 2
    for sequence in phases(read).values():
        assert "started" in sequence and "completed" in sequence
        assert sequence.count("rendered") <= 1  # never a fabricated or duplicated render
    json_run = [e for e in read.events if e.phase is Phase.COMPLETED]
    assert {e.outcome for e in json_run} == {Outcome.OK}


def test_interruption_after_start_leaves_an_incomplete_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    policy.enable_collection(PROFILE)
    env = _env(tmp_path, SSGREP_DRIVER="slow")
    process = subprocess.Popen(
        [sys.executable, str(DRIVER), "--json", "search", QUERY],
        env=env,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stderr is not None
    assert process.stderr.readline().strip() == b"SLOW-STARTED"
    process.send_signal(signal.SIGKILL)  # no handlers run: a hard kill between phases
    process.wait(timeout=30)

    read = events()
    (sequence,) = phases(read).values()
    assert sequence == ["started"]  # no invented completion or rendering
    assert read.damage == []


def test_sigint_after_start_is_recorded_as_interrupted_not_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    policy.enable_collection(PROFILE)
    process = subprocess.Popen(
        [sys.executable, str(DRIVER), "--json", "search", QUERY],
        env=_env(tmp_path, SSGREP_DRIVER="slow"),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stderr is not None
    assert process.stderr.readline().strip() == b"SLOW-STARTED"
    process.send_signal(signal.SIGINT)
    process.wait(timeout=30)

    read = events()
    completed = [e for e in read.events if e.phase is Phase.COMPLETED]
    assert [e.outcome for e in completed] == [Outcome.INTERRUPTED]
    assert [e for e in read.events if e.phase is Phase.RENDERED] == []


def test_accounting_never_appears_under_notes_or_repository_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    policy.enable_collection(PROFILE)
    run(tmp_path, "--json", "search", QUERY)
    data = tmp_path / "data"
    assert sorted(p.name for p in data.iterdir()) == ["accounting"]
    for path in data.rglob("*"):
        mode = stat.S_IMODE(os.lstat(path).st_mode)
        assert mode in (0o700, 0o600), (path, oct(mode))
    assert not (ROOT / ".ssgrep").exists() or "accounting" not in {
        p.name for p in (ROOT / ".ssgrep").rglob("*")
    }


def test_two_processes_executing_concurrently_record_distinct_reducible_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "data"))
    policy.enable_collection(PROFILE)
    processes = [
        subprocess.Popen(
            [sys.executable, str(DRIVER), "--json", "search", QUERY],
            env=_env(tmp_path, SSGREP_DRIVER="ok"),
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(4)
    ]
    for process in processes:
        process.communicate(timeout=120)
        assert process.returncode == 0

    read = events()
    sequences = phases(read)
    assert len(sequences) == 4 and len({k.split("-")[0] for k in sequences}) == 4
    assert all(s == ["completed", "rendered", "started"] for s in sequences.values())
    # Same query, same key: identical fingerprints, distinct executions (nothing fused them).
    assert len({e.fingerprint for e in read.events}) == 1


def test_cold_import_of_capture_adds_only_stdlib_and_light_accounting_modules() -> None:
    code = (
        "import sys; import ssgrep; before=set(sys.modules);"
        "import ssgrep.accounting.capture.capture;"
        "print(sorted(set(sys.modules)-before))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    added = set(eval(done.stdout))  # noqa: S307 - our own child's sorted list repr
    assert {m for m in added if m.startswith("ssgrep")} == {
        "ssgrep.accounting",
        "ssgrep.accounting.capture",
        "ssgrep.accounting.capture.capture",
        "ssgrep.accounting.capture.capture_policy",
        "ssgrep.accounting.ledger",
        "ssgrep.accounting.ledger.contracts",
    }
    heavy = ("lancedb", "numpy", "torch", "pyarrow", "fastmcp", "rich", "pandas", "scipy")
    assert not [m for m in added if m.split(".")[0] in heavy]
    assert len(added) < 40  # stdlib json/hashlib/dataclasses/contextvars only


def test_data_root_resolution_matches_the_store_package_for_both_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.accounting.capture import capture as capture_module
    from ssgrep.accounting.privacy import storage as datadir
    from ssgrep.store.paths import data_dir

    monkeypatch.setenv("SSGREP_DATA_DIR", str(tmp_path / "override"))
    assert (
        capture_module._configured_root()
        == datadir.data_root()
        == data_dir()
        == tmp_path / "override"
    )
    monkeypatch.delenv("SSGREP_DATA_DIR")
    assert capture_module._configured_root() == datadir.data_root() == data_dir()


def test_enabled_and_disabled_capture_never_import_the_heavy_store_package(
    tmp_path: Path,
) -> None:
    code = (
        "import sys;"
        "from ssgrep.accounting.capture import capture;"
        "from ssgrep.accounting.ledger.contracts import Operation, Outcome;"
        "x = capture.begin(Operation.SEARCH, text='q'); capture.complete(x, Outcome.OK);"
        "print(x is not None, [m for m in ('lancedb','numpy','pyarrow','torch','ssgrep.store',"
        "'ssgrep.indexing') if m in sys.modules])"
    )
    for enabled in (False, True):
        root = tmp_path / ("on" if enabled else "off")
        env = {**os.environ, "PYTHONPATH": str(ROOT), "SSGREP_DATA_DIR": str(root / "data")}
        if enabled:
            seed = (
                "from ssgrep.accounting.privacy import policy; "
                "policy.enable_collection('personal-local')"
            )
            subprocess.run([sys.executable, "-c", seed], env=env, check=True, timeout=60)
        done = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60
        )
        assert done.stdout.strip() == f"{enabled} []", done.stdout + done.stderr


def test_disabled_capture_does_not_load_journal_or_policy_modules(tmp_path: Path) -> None:
    code = (
        "import sys, os;"
        "from ssgrep.accounting.capture import capture;"
        "from ssgrep.accounting.ledger.contracts import Operation;"
        "assert capture.begin(Operation.SEARCH, text='q') is None;"
        "print(sorted(m for m in sys.modules if m.startswith('ssgrep.accounting.')"
        " and m.split('.')[-1] in ('journal','policy','fingerprint','storage')))"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT),
            "SSGREP_DATA_DIR": str(tmp_path / "data"),
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.stdout.strip() == "[]", done.stdout + done.stderr
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    ("mcp", "arguments"),
    [
        (False, ("--json", "search", QUERY)),
        (False, ("--json", "show", "claude:s:ep:1")),
        (True, ("search_sessions", QUERY)),
        (True, ("show_session", "claude:s:ep:1")),
    ],
)
def test_render_metadata_failure_preserves_real_cli_and_mcp_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mcp: bool, arguments: tuple[str, ...]
) -> None:
    live = tmp_path / "render-fault"
    monkeypatch.setenv("SSGREP_DATA_DIR", str(live / "data"))
    policy.enable_collection(PROFILE)
    control = run(live, *arguments, mcp=mcp)
    assert control.code == 0
    accepted = events().events
    assert {event.phase for event in accepted} == {
        Phase.STARTED,
        Phase.COMPLETED,
        Phase.RENDERED,
    }
    original_ids = {event.execution_id for event in accepted}
    failed_metadata = run(live, *arguments, mcp=mcp, render_metadata_fails=True)
    assert failed_metadata == control
    assert (live / "data" / "render-metadata-fault").read_text() == "bytes"
    recent = [event for event in events().events if event.execution_id not in original_ids]
    assert [event.phase for event in recent] == [Phase.STARTED, Phase.COMPLETED]
    assert recent[-1].outcome is Outcome.OK
