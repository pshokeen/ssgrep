"""Real usecli discovery and dispatch must not eagerly load optional accounting work."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
OPTIONAL_PREFIXES = (
    "ssgrep.accounting.collect",
    "ssgrep.accounting.export",
    "ssgrep.accounting.delivery",
    "ssgrep.accounting.http",
    "ssgrep.accounting.serve",
    "ssgrep.accounting.valuation",
    "ssgrep.accounting.census",
    "ssgrep.accounting.purge",
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
        ("metrics census", "--root"),
        ("metrics collect", "--backfill"),
        ("metrics export", "--destination"),
        ("metrics pause", "--profile"),
        ("metrics purge", "--dry-run"),
        ("metrics resume", "--profile"),
        ("metrics serve", "--port"),
        ("metrics setup", "--policy"),
        ("metrics status", "--profile"),
        ("savings explain", "--include-content"),
        ("savings history", "--cursor"),
    ],
)
def test_real_help_dispatch_does_not_load_optional_accounting(
    tmp_path: Path, command: str, option: str
) -> None:
    done, modules = fresh_process(tmp_path, *command.split(), "--help")
    assert done.returncode == 0, done.stderr
    assert "Usage:" in done.stdout
    assert f"ssgrep {command.split()[-1]}" in done.stdout
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
        (("show", "invalid-ref"), 3, "unknown_ref"),
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
        expected = {"search": "empty", "show": "invalid-ref", "status": "ssgrep index"}
        assert expected[arguments[0]] in (done.stdout + done.stderr).lower()


@pytest.mark.parametrize(
    ("arguments", "condition"),
    [
        (("metrics", "setup"), "confirmation_required"),
        (("metrics", "collect", "--backfill"), "backfill_requires_interval"),
        (("metrics", "collect", "--reset-frequency-authority"), "reset_frequency_requires_repin"),
        (("metrics", "serve", "--port", "-1"), "invalid_port"),
        (("savings", "--limit", "0"), "invalid_limit"),
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
    done, modules = fresh_process(tmp_path, "--json", "metrics", "collect", *options)
    assert done.returncode == 1, done.stderr
    document = json.loads(done.stdout)
    assert document["ok"] is False and document["condition"] == "collection_disabled"
    assert document["command"] == "ssgrep metrics setup"
    assert_optional_graph_absent(modules)
    assert not (tmp_path / "data").exists()


def test_export_mode_validation_keeps_preview_and_delivery_dependencies_lazy(
    tmp_path: Path,
) -> None:
    done, modules = fresh_process(tmp_path, "--json", "metrics", "export")
    assert done.returncode == 2, done.stderr
    document = json.loads(done.stdout)
    assert document["ok"] is False and document["condition"] == "destination_required"
    assert document["command"] == "ssgrep metrics status"
    assert not set(modules).intersection(
        {
            "ssgrep.accounting.export_consent_preview",
            "ssgrep.accounting.delivery",
            "ssgrep.accounting.http",
            "ssgrep.accounting.serve_loop",
            "ssgrep.cli.export_render",
        }
    )
    assert not (tmp_path / "data").exists()
