"""Read-only accounting startup stays machine-readable with networking denied."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("args", "exit_code", "state"),
    [
        (("savings",), 0, "disabled"),
        (("savings", "status"), 0, "disabled"),
        (("savings", "status", "--sources"), 0, "disabled"),
        (("savings", "--runtime", "invalid-runtime"), 2, "invalid_filter"),
        (("savings", "status", "--profile", "../invalid"), 2, "invalid_profile"),
    ],
)
def test_offline_accounting_startup_emits_one_json_document(
    tmp_path: Path, args: tuple[str, ...], exit_code: int, state: str
) -> None:
    sandbox = shutil.which("sandbox-exec") if sys.platform == "darwin" else None
    if sandbox is None:
        pytest.skip("Requires macOS sandbox-exec to deny native dependency network access")
    home = tmp_path / "home"
    home.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data = tmp_path / "data"
    # Deliberately do not inherit logging or telemetry opt-outs: importing an
    # unrelated network-enabled dependency must not corrupt the JSON contract.
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "XDG_DATA_HOME": str(tmp_path / "xdg-data"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg-config"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
        "SSGREP_DATA_DIR": str(data),
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "SSGREP_CODEX_SESSIONS_DIR": str(home / ".codex" / "sessions"),
        "SSGREP_OMP_SESSIONS_DIR": str(home / ".omp" / "agent" / "sessions"),
        "SSGREP_PI_SESSIONS_DIR": str(home / ".pi" / "agent" / "sessions"),
        "SSGREP_PRIME_AGENT_SESSIONS_DIR": str(home / ".prime" / "agent" / "sessions"),
        "SSGREP_OPENCODE_DB": str(home / ".local" / "share" / "opencode" / "opencode.db"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_COLOR": "1",
    }
    result = subprocess.run(
        [
            sandbox,
            "-p",
            "(version 1) (allow default) (deny network*)",
            str(Path(sys.executable).parent / "ssgrep"),
            "--json",
            *args,
        ],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == exit_code, result.stderr
    assert result.stderr == ""
    document = json.loads(result.stdout)
    if exit_code == 0:
        assert document["ok"] is True
        assert document["data"]["state"] == state
        if args == ("savings", "status", "--sources"):
            inventory = document["data"]["data"]["source_census"]["inventory"]
            assert inventory["totals"]["sources"] == 0
        elif args == ("savings",):
            assert document["data"]["data"]["collection_enabled"] is False
        else:
            assert document["data"]["data"]["capture"] == "off"
    else:
        assert document["ok"] is False
        assert document["condition"] == state
        assert document["message"] and document["command"]
    assert not data.exists()
    assert not tuple(home.iterdir())
    assert not tuple(workspace.iterdir())


ROOT = Path(__file__).resolve().parents[3]
OPTIONAL_PREFIXES = (
    "ssgrep.accounting.collect",
    "ssgrep.accounting.export",
    "ssgrep.accounting.export.delivery",
    "ssgrep.accounting.serve.http",
    "ssgrep.accounting.serve",
    "ssgrep.accounting.ledger.valuation",
    "ssgrep.accounting.readers.classify",
    "ssgrep.accounting.privacy.purge",
    "eval.accounting",
    "ssgrep.services.mcp_server",
    "http.server",
    "socketserver",
)

# Only instrument process exit to record imports; dispatch, serializers, errors and
# os._exit itself remain real. This also covers JSON commands that skip atexit.
CHILD = r"""
import json
import os
import sys
from pathlib import Path

report_path, *arguments = sys.argv[1:]
sys.argv = ["ssgrep", *arguments]


def reject_external_work(event, args):
    if event == "import" and args[0].split(".")[0] in {
        "torch", "transformers", "sentence_transformers", "pylate"
    }:
        raise AssertionError("startup attempted model import: " + args[0])
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise AssertionError("startup attempted network access")

# find_spec availability probes are not imports and remain permitted.
sys.addaudithook(reject_external_work)

def snapshot():
    Path(report_path).write_text(json.dumps(sorted(sys.modules)))

real_exit = os._exit

def recorded_exit(code):
    snapshot()
    real_exit(code)

os._exit = recorded_exit
try:
    from ssgrep.cli import main
    main()
finally:
    snapshot()
"""


def fresh_process(
    tmp_path: Path, *arguments: str
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("SSGREP_", "USECLI_"))
    }
    env.update(
        HOME=str(home),
        SSGREP_DATA_DIR=str(tmp_path / "data"),
        XDG_DATA_HOME=str(tmp_path / "xdg-data"),
        XDG_CONFIG_HOME=str(tmp_path / "xdg-config"),
        CLAUDE_CONFIG_DIR=str(tmp_path / "claude"),
        OPENCODE_CONFIG_DIR=str(tmp_path / "opencode"),
        CODEX_HOME=str(tmp_path / "codex"),
        PI_CODING_AGENT_DIR=str(tmp_path / "pi-agent"),
        PRIME_AGENT_CODING_AGENT_DIR=str(tmp_path / "prime-agent"),
        PI_SESSION_DIR=str(tmp_path / "pi-sessions"),
        PRIME_AGENT_SESSION_DIR=str(tmp_path / "prime-sessions"),
        OMP_SESSIONS_DIR=str(tmp_path / "omp-sessions"),
        OMP_AGENT_DIR=str(tmp_path / "omp-agent"),
        PYTHONPATH=str(ROOT / "src"),
        NO_COLOR="1",
        COLUMNS="120",
        COCOINDEX_DISABLE_USAGE_TRACKING="1",
    )
    report = tmp_path / "import-report.json"
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(report), *arguments],
        cwd=ROOT,
        env=env,
        input="",
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert report.is_file(), done.stderr
    return done, json.loads(report.read_text())


def assert_optional_graph_absent(modules: list[str]) -> None:
    assert not [name for name in modules if name.startswith(OPTIONAL_PREFIXES)]


@pytest.mark.parametrize(
    ("command", "option"),
    [
        ("savings status", "--sources"),
        ("savings collect", "--backfill"),
        ("savings export", "--dry-run"),
        ("savings pause", "--profile"),
        ("savings purge", "--dry-run"),
        ("savings resume", "--profile"),
        ("savings serve", "--port"),
        ("savings setup", "--policy"),
        ("savings status", "--profile"),
        ("savings explain", "--include-content"),
        ("savings history", "--page"),
    ],
)
def test_real_help_dispatch_does_not_load_optional_accounting(
    tmp_path: Path, command: str, option: str
) -> None:
    done, modules = fresh_process(tmp_path, *command.split(), "--help")
    assert done.returncode == 0, done.stderr
    assert "Usage:" in done.stdout
    assert f"ssgrep {command}" in done.stdout
    assert option in done.stdout
    assert_optional_graph_absent(modules)
    assert not (tmp_path / "data").exists()


def test_savings_group_help_dispatch_keeps_real_subcommands_available(tmp_path: Path) -> None:
    done, modules = fresh_process(tmp_path, "savings", "--help")
    assert done.returncode == 0, done.stderr
    assert "savings" in done.stdout and "explain" in done.stdout and "history" in done.stdout
    assert_optional_graph_absent(modules)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("machine", (False, True), ids=("human", "json"))
@pytest.mark.parametrize(
    ("arguments", "code", "condition"),
    [
        (("search", ""), 2, "empty_query"),
        (("show", "invalid-ref"), 4, "missing_index"),
        (("status",), 0, None),
    ],
)
def test_disabled_actual_dispatch_excludes_optional_graph(
    tmp_path: Path, machine: bool, arguments: tuple[str, ...], code: int, condition: str | None
) -> None:
    args = ("--json", *arguments) if machine else arguments
    done, modules = fresh_process(tmp_path, *args)
    assert done.returncode == code, done.stderr
    assert_optional_graph_absent(modules)
    assert not (tmp_path / "data" / "accounting").exists()
    if machine:
        document = json.loads(done.stdout)
        assert document["ok"] is (condition is None)
        if condition is not None:
            assert document["condition"] == condition
        else:
            assert document["data"]["index_exists"] is False
    else:
        expected = {"search": "empty", "show": "ssgrep index", "status": "ssgrep index"}
        assert expected[arguments[0]] in (done.stdout + done.stderr).lower()


@pytest.mark.parametrize(
    ("arguments", "condition"),
    [
        (("savings", "setup"), "confirmation_required"),
        (("savings", "collect", "--backfill"), "backfill_requires_interval"),
        (("savings", "collect", "--reset-frequency-authority"), "reset_frequency_requires_repin"),
        (("savings", "serve", "--port", "-1"), "invalid_port"),
        (("savings", "history", "--limit", "0"), "invalid_limit"),
        (("savings", "history", "--limit", "101"), "invalid_limit"),
    ],
)
def test_lazy_dispatch_preserves_validation_protocol_without_worker_imports(
    tmp_path: Path, arguments: tuple[str, ...], condition: str
) -> None:
    done, modules = fresh_process(tmp_path, "--json", *arguments)
    assert done.returncode == 2, done.stderr
    document = json.loads(done.stdout)
    assert document["ok"] is False
    assert document["condition"] == condition
    assert isinstance(document["message"], str) and document["message"]
    assert isinstance(document["command"], str) and document["command"]
    assert_optional_graph_absent(modules)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("options", [(), ("--reset-frequency-authority", "--repin-frequency")])
def test_disabled_collection_refuses_before_loading_collector_or_renderer(
    tmp_path: Path, options: tuple[str, ...]
) -> None:
    done, modules = fresh_process(tmp_path, "--json", "savings", "collect", *options)
    assert done.returncode == 1, done.stderr
    document = json.loads(done.stdout)
    assert document["ok"] is False and document["condition"] == "collection_disabled"
    assert document["command"] == "ssgrep savings setup"
    assert_optional_graph_absent(modules)
    assert not (tmp_path / "data").exists()


def test_export_date_validation_keeps_preview_and_delivery_dependencies_lazy(
    tmp_path: Path,
) -> None:
    done, modules = fresh_process(tmp_path, "--json", "savings", "export", "--from", "2026-10-01")
    assert done.returncode == 2, done.stderr
    document = json.loads(done.stdout)
    assert document["ok"] is False
    assert not set(modules).intersection(
        {
            "ssgrep.accounting.export.consent",
            "ssgrep.accounting.export.delivery",
            "ssgrep.accounting.serve.http",
            "ssgrep.accounting.serve.serve_loop",
            "ssgrep.cli.export_render",
        }
    )
    assert not (tmp_path / "data").exists()
