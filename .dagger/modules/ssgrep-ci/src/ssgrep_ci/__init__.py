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
# `tests` deselects exactly these IDs and `tests-eval` selects exactly these IDs, so every test runs
# once; a renamed ID fails `tests-eval` ("not found") instead of being silently skipped.
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
            .with_exec(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    "/repo/.venv/bin/python",
                    "--no-deps",
                    "data-designer==0.9.1",
                ]
            )
        )

    def _module_quality_container(self) -> Container:
        return (
            self._base_container()
            .with_env_variable("UV_PYTHON", "3.14")
            .with_workdir("/module")
            .with_mounted_cache(
                "/root/.cache/uv",
                core().cache_volume("ssgrep-ci-uv-module-linux-arm64-py314"),
            )
            .with_directory("/module", self.module_source)
            .with_exec(["uv", "sync", "--locked", "--group", "dev"])
        )

    def _hygiene_test_container(self) -> Container:
        return (
            self._app()
            .with_directory("/repo/.git", self.git_metadata)
            .with_exec(["git", "config", "--global", "--add", "safe.directory", "/repo"])
            .with_exec(["git", "rev-parse", "--show-toplevel"])
        )

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
        """Run the non-network pytest suite (hygiene included) minus the heavy eval tests.

        The deselected tests run in `tests-eval`, so together the two checks cover the full suite.
        """
        deselect = [arg for node in _HEAVY_EVAL_TESTS for arg in ("--deselect", node)]
        await (
            self._hygiene_test_container().with_exec(["uv", "run", "poe", "test", *deselect]).sync()
        )

    @function  # ty: ignore[invalid-argument-type]
    @check
    async def tests_eval(self) -> None:
        """Run the heavy eval tests that `tests` deselects; a renamed test node ID fails here."""
        container = self._hygiene_test_container()
        await container.with_exec(["uv", "run", "pytest", "-v", *_HEAVY_EVAL_TESTS]).sync()

    @function  # ty: ignore[invalid-argument-type]
    @check
    async def package(self) -> None:
        """Build the wheel and verify the CLI identity from a clean install outside the checkout."""
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
                    'VIRTUAL_ENV="$venv" PATH="$venv/bin:$PATH" '
                    "uv sync --locked --no-install-package fast-plaid "
                    "--no-default-groups --group dev "
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
