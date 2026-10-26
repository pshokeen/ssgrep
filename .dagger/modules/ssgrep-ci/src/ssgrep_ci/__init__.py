from __future__ import annotations

from dagger import check, function, object_type
from dagger_clients.core import Container, Directory, Workspace, core

_BASE_IMAGE = (
    "python:3.12-slim@sha256:6c6ae3622772575c3ac02ecb2b02a06d05f19e1713a0cbff476ef37bca79e90f"
)
_UV_VERSION = "0.13.5"
_WORKSPACE_EXCLUDES = [
    ".dagger",
    ".git",
    ".venv",
    "dagger.lock",
    "dagger.toml",
    "openspec",
    "**/.aws",
    "**/.cache",
    "**/.claude",
    "**/.codex",
    "**/.git",
    "**/.opencode",
    "**/.pi",
    "**/.ssh",
    "**/.venv",
    "**/__pycache__",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.DS_Store",
    "eval/results",
]
# Slowest tests (~40 s together), run in their own check so they overlap with the rest of the suite.
# The `tests` shard deselects exactly these IDs and `tests-eval` selects exactly these IDs, so every
# test runs once; a renamed ID fails `tests-eval` ("not found") instead of being silently skipped.
# `--deselect` is a node-ID prefix match, so `tests-eval` also proves that each ID matches exactly
# one collected test (a new `..._json` variant would otherwise be deselected and never run).
_EVAL_FILE = "tests/eval/test_run_eval.py"
_HEAVY_EVAL_TESTS = [
    f"{_EVAL_FILE}::{name}"
    for name in (
        "test_gates_derived_and_checked_from_baseline",
        "test_cli_smoke_writes_valid_payload",
        "test_reference_arm_block_populated",
        "test_metric_limit_option_plumbed_through",
        "test_summary_includes_prefetch_keys_after_cli_run",
    )
]
# Directory shards of the pytest suite. Each named shard runs exactly its directories; the `tests`
# shard runs the whole suite with those directories `--ignore`d, so every test runs exactly once and
# any new directory lands in `tests` automatically.
_TEST_SHARDS = {
    "views": ["tests/accounting/views", "tests/accounting/readers"],
    "collect": ["tests/accounting/collect", "tests/accounting/capture"],
}
_HF_HOME = "/root/.cache/huggingface"
_PREFETCH_MODEL = (
    "from huggingface_hub import snapshot_download\n"
    "from ssgrep.indexing.embed import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION\n"
    "snapshot_download(repo_id=DEFAULT_MODEL_ID, revision=DEFAULT_MODEL_REVISION, max_workers=2)\n"
)


@object_type
class SsgrepCi:
    source: Directory
    dependency_files: Directory
    git_metadata: Directory
    module_source: Directory

    @classmethod
    def create(cls, ws: Workspace) -> SsgrepCi:
        return cls(
            source=ws.directory(  # ty: ignore[unknown-argument]
                "/",
                exclude=_WORKSPACE_EXCLUDES,
                gitignore=True,
            ),
            dependency_files=ws.directory(  # ty: ignore[unknown-argument]
                "/",
                include=["pyproject.toml", "uv.lock"],
            ),
            git_metadata=ws.git().directory(),  # ty: ignore[unknown-argument]
            module_source=ws.directory(  # ty: ignore[unknown-argument]
                ".dagger/modules/ssgrep-ci",
                exclude=["**/.venv", "**/__pycache__"],
            ),
        )

    def _base_container(self) -> Container:
        return (
            core()
            .container()
            .from_(_BASE_IMAGE)
            .with_workdir("/repo")
            .with_env_variable("UV_PYTHON", "3.12")
            .with_env_variable("PIP_CACHE_DIR", "/root/.cache/pip")
            .with_env_variable("UV_NO_SYNC", "1")
            .with_env_variable("UV_CONCURRENT_BUILDS", "1")
            .with_env_variable("UV_CONCURRENT_INSTALLS", "1")
            .with_env_variable("UV_LINK_MODE", "copy")
            .with_env_variable("SSGREP_EVAL_WORKERS", "2")
            .with_mounted_cache(
                "/root/.cache/pip",
                core().cache_volume("ssgrep-ci-pip-linux-arm64-py312"),
            )
            .with_mounted_cache(
                "/root/.cache/uv",
                core().cache_volume("ssgrep-ci-uv-linux-arm64-py312"),
            )
            .with_exec(
                [
                    "python",
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    f"uv=={_UV_VERSION}",
                ]
            )
        )

    def _dependencies(self) -> Container:
        return (
            self._base_container()
            .with_exec(["apt-get", "update"])
            .with_exec(
                [
                    "apt-get",
                    "install",
                    "-y",
                    "--no-install-recommends",
                    "build-essential",
                    "git",
                    "lsof",
                    "procps",
                ]
            )
            .with_exec(["rm", "-rf", "/var/lib/apt/lists"])
            .with_directory("/repo", self.dependency_files)
            .with_env_variable("UV_COMPILE_BYTECODE", "1")
            .with_env_variable("UV_CONCURRENT_INSTALLS", "4")
            .with_exec(
                [
                    "uv",
                    "sync",
                    "--locked",
                    "--no-install-package",
                    "fast-plaid",
                    "--no-install-project",
                    "--no-default-groups",
                    "--group",
                    "dev",
                ]
            )
            .without_env_variable("UV_COMPILE_BYTECODE")
            .with_env_variable("UV_CONCURRENT_INSTALLS", "1")
        )

    def _app(self) -> Container:
        return (
            self._dependencies()
            .with_directory("/repo", self.source)
            .with_exec(
                [
                    "uv",
                    "sync",
                    "--locked",
                    "--no-install-package",
                    "fast-plaid",
                    "--no-default-groups",
                    "--group",
                    "dev",
                ]
            )
            # The Linux lane exists to exercise Linux/root semantics; state them explicitly.
            .with_exec(["sh", "-ec", 'test "$(id -u)" = 0; test "$(uname -s)" = Linux'])
        )

    def _module_quality_container(self) -> Container:
        return (
            self._base_container()
            .with_env_variable("UV_PYTHON", "3.14")
            # Keep uv-managed interpreters in the cache volume so 3.14 is not re-downloaded.
            .with_env_variable("UV_PYTHON_INSTALL_DIR", "/root/.cache/uv/python")
            .with_workdir("/module")
            .with_mounted_cache(
                "/root/.cache/uv",
                core().cache_volume("ssgrep-ci-uv-module-linux-arm64-py314"),
            )
            .with_directory("/module", self.module_source)
            .with_exec(["uv", "sync", "--locked", "--group", "dev"])
        )

    def _test_container(self) -> Container:
        """Application container with Git metadata (for the hygiene test) and the pinned
        embedding model prefetched into a cache volume. ssgrep switches the Hub to offline mode
        itself once the model is cached; a process-wide HF_HUB_OFFLINE would break the test that
        proves a cache miss stays online (huggingface_hub reads it at import time)."""
        return (
            self._app()
            .with_env_variable("HF_HOME", _HF_HOME)
            .with_mounted_cache(_HF_HOME, core().cache_volume("ssgrep-ci-huggingface"))
            .with_exec(["uv", "run", "python", "-c", _PREFETCH_MODEL])
            .with_directory("/repo/.git", self.git_metadata)
            .with_exec(["git", "config", "--global", "--add", "safe.directory", "/repo"])
            .with_exec(["git", "rev-parse", "--show-toplevel"])
        )

    async def _pytest(self, args: list[str]) -> None:
        await self._test_container().with_exec(["uv", "run", "pytest", "-v", *args]).sync()

    @function  # ty: ignore[invalid-argument-type]
    @check
    async def module_quality(self) -> None:
        """Lint, format-check, and type-check first-party Dagger module code."""
        container = self._module_quality_container()
        container = container.with_exec(["uv", "run", "ruff", "check", "src/ssgrep_ci"])
        container = container.with_exec(["uv", "run", "ruff", "format", "--check", "src/ssgrep_ci"])
        await container.with_exec(
            [
                "uv",
                "run",
                "ty",
                "check",
                "--no-respect-ignore-files",
                "src/ssgrep_ci/__init__.py",
            ]
        ).sync()

    @function  # ty: ignore[invalid-argument-type]
    @check
    async def lint(self) -> None:
        """Run the repository's non-mutating Ruff lint check."""
        await self._app().with_exec(["uv", "run", "poe", "lint"]).sync()

    @function  # ty: ignore[invalid-argument-type]
    @check
    async def format(self) -> None:
        """Check Ruff formatting without modifying source files."""
        await self._app().with_exec(["uv", "run", "poe", "fmt"]).sync()

    @function  # ty: ignore[invalid-argument-type]
    @check
    async def typecheck(self) -> None:
        """Run the repository's ty type check."""
        await self._app().with_exec(["uv", "run", "poe", "typecheck"]).sync()

    @function  # ty: ignore[invalid-argument-type]
    @check
    async def sizecheck(self) -> None:
        """Run the source file-size gate."""
        await self._app().with_exec(["uv", "run", "poe", "sizecheck"]).sync()

    @function  # ty: ignore[invalid-argument-type]
    @check
    async def tests(self) -> None:
        """Run the non-network pytest suite (hygiene included) except the other test shards."""
        ignore = [f"--ignore={path}" for paths in _TEST_SHARDS.values() for path in paths]
        deselect = [arg for node in _HEAVY_EVAL_TESTS for arg in ("--deselect", node)]
        await self._pytest(["tests/", *ignore, *deselect])

    @function  # ty: ignore[invalid-argument-type]
    @check
    async def tests_views(self) -> None:
        """Run the accounting views and readers tests (a shard of the suite)."""
        await self._pytest(_TEST_SHARDS["views"])

    @function  # ty: ignore[invalid-argument-type]
    @check
    async def tests_collect(self) -> None:
        """Run the accounting collect and capture tests (a shard of the suite)."""
        await self._pytest(_TEST_SHARDS["collect"])

    @function  # ty: ignore[invalid-argument-type]
    @check
    async def tests_eval(self) -> None:
        """Run the heavy eval tests that `tests` deselects, after proving each ID is exact."""
        guard = (
            "import subprocess, sys\n"
            "out = subprocess.run(\n"
            "    ['uv', 'run', 'pytest', '--collect-only', '-q', 'tests/'],\n"
            "    capture_output=True, text=True, check=True,\n"
            ").stdout.split()\n"
            "for node in sys.argv[1:]:\n"
            "    hits = [item for item in out if item.startswith(node)]\n"
            "    if hits != [node]:\n"
            "        sys.exit(f'--deselect {node} would match {hits}')\n"
        )
        container = self._test_container().with_exec(
            ["uv", "run", "python", "-c", guard, *_HEAVY_EVAL_TESTS]
        )
        await container.with_exec(["uv", "run", "pytest", "-v", *_HEAVY_EVAL_TESTS]).sync()

    @function  # ty: ignore[invalid-argument-type]
    @check
    async def package(self) -> None:
        """Build the wheel and verify the CLI identity from a clean runtime-only install."""
        await (
            self._app()
            .with_exec(
                [
                    "sh",
                    "-ec",
                    "uv build --wheel; "
                    'venv="$(mktemp -d)"; testdir="$(mktemp -d)"; '
                    'trap \'rm -rf "$venv" "$testdir"\' EXIT; '
                    'uv venv "$venv"; '
                    # Runtime dependencies only: a CLI importing a dev-only package must fail.
                    'VIRTUAL_ENV="$venv" PATH="$venv/bin:$PATH" '
                    "uv sync --locked --no-install-package fast-plaid --no-default-groups "
                    "--no-install-project --active --project /repo; "
                    'wheel="$(find dist -maxdepth 1 -name "*.whl" -type f -print -quit)"; '
                    'test -n "$wheel"; '
                    'uv pip install --python "$venv/bin/python" --no-deps "$wheel"; '
                    'cd "$testdir"; '
                    '/repo/scripts/check_cli_identity.sh "$venv/bin/ssgrep"',
                ]
            )
            .sync()
        )
