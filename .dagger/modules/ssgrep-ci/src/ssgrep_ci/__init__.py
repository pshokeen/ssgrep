from __future__ import annotations

from dagger import check, function, object_type
from dagger_clients.core import Container, Directory, Workspace, core

from .pin import DEPS_IMAGE, DEPS_IMAGE_BASE
from .recipe import INTERPRETER_CHECK, admitted_seed, apt_install

# Multi-platform image index digest: each engine pulls its native architecture.
_BASE_IMAGE = (
    "python:3.13-slim@sha256:2d44738ebf4ce5f63b9d9ea871e9c7cfdc0426f7b0c97cac915fd88ded1d86bb"
)
# fast-plaid publishes no Linux/aarch64 wheel; omit it only there. Every other platform installs
# the complete locked graph.
_SKIP_UNAVAILABLE = '$(test "$(uname -m)" = aarch64 && echo --no-install-package fast-plaid)'
_UV_VERSION = "0.13.5"
# Published dependency image (a cache seed, never a correctness input), see `pin.py`. It is admitted
# only when built from this exact `_BASE_IMAGE`; every step on top of it is idempotent.
_DEPS_SEED = admitted_seed(DEPS_IMAGE, DEPS_IMAGE_BASE, _BASE_IMAGE)
_APT_PACKAGES = ("build-essential", "git", "lsof", "procps")
_EMBED_MODULE = "src/ssgrep/indexing/embed.py"
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
# The model id and pinned revision are plain literals in ssgrep.indexing.embed. Reading them with
# `ast` lets the prefetch run in the dependency stage (the project is not installed there) keyed
# only on that one file, instead of re-running whenever any source file changes.
_PREFETCH_MODEL = (
    "import ast, pathlib\n"
    "from huggingface_hub import snapshot_download\n"
    "names = {'DEFAULT_MODEL_ID', 'DEFAULT_MODEL_REVISION'}\n"
    f"tree = ast.parse(pathlib.Path('/embed/{_EMBED_MODULE}').read_text())\n"
    "values = {\n"
    "    node.targets[0].id: ast.literal_eval(node.value)\n"
    "    for node in tree.body\n"
    "    if isinstance(node, ast.Assign)\n"
    "    and isinstance(node.targets[0], ast.Name)\n"
    "    and node.targets[0].id in names\n"
    "}\n"
    "snapshot_download(\n"
    "    repo_id=values['DEFAULT_MODEL_ID'],\n"
    "    revision=values['DEFAULT_MODEL_REVISION'],\n"
    "    max_workers=2,\n"
    ")\n"
)


def _uv_sync(*args: str) -> list[str]:
    return ["sh", "-ec", " ".join(["uv", "sync", "--locked", _SKIP_UNAVAILABLE, *args])]


@object_type
class SsgrepCi:
    source: Directory
    dependency_files: Directory
    embed_source: Directory
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
            embed_source=ws.directory(  # ty: ignore[unknown-argument]
                "/",
                include=[_EMBED_MODULE],
            ),
            git_metadata=ws.git().directory(),  # ty: ignore[unknown-argument]
            module_source=ws.directory(  # ty: ignore[unknown-argument]
                ".dagger/modules/ssgrep-ci",
                exclude=["**/.venv", "**/__pycache__"],
            ),
        )

    def _base_container(self, image: str = _BASE_IMAGE) -> Container:
        return (
            core()
            .container()
            .from_(image)
            .with_workdir("/repo")
            .with_env_variable("UV_PYTHON", "3.13")
            .with_env_variable("PIP_CACHE_DIR", "/root/.cache/pip")
            .with_env_variable("UV_NO_SYNC", "1")
            .with_env_variable("UV_CONCURRENT_BUILDS", "1")
            .with_env_variable("UV_CONCURRENT_INSTALLS", "1")
            .with_env_variable("UV_LINK_MODE", "copy")
            .with_env_variable("SSGREP_EVAL_WORKERS", "2")
            .with_env_variable("HF_HOME", _HF_HOME)
            .with_mounted_cache(
                "/root/.cache/pip",
                core().cache_volume("ssgrep-ci-pip-py313"),
            )
            .with_mounted_cache(
                "/root/.cache/uv",
                core().cache_volume("ssgrep-ci-uv-py313"),
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

    def _environment(self, image: str) -> Container:
        """The locked dependency environment on top of `image`.

        The one recipe for both the in-engine build (`image` is the base image) and the published
        cache seed (`image` is the pinned deps image). Each step is idempotent: apt installs only
        what `dpkg` reports missing, pip/uv skip satisfied requirements, and the locked sync and
        model prefetch reconcile with the checkout's lockfile and pinned model revision.
        """
        return (
            self._base_container(image)
            .with_exec(apt_install(_APT_PACKAGES))
            .with_directory("/repo", self.dependency_files)
            .with_env_variable("UV_COMPILE_BYTECODE", "1")
            .with_env_variable("UV_CONCURRENT_INSTALLS", "4")
            .with_exec(_uv_sync("--no-install-project", "--no-default-groups", "--group", "dev"))
            .without_env_variable("UV_COMPILE_BYTECODE")
            .with_env_variable("UV_CONCURRENT_INSTALLS", "1")
            .with_exec(["uv", "run", "python", "-c", INTERPRETER_CHECK])
            .with_directory("/embed", self.embed_source)
            .with_exec(["uv", "run", "python", "-c", _PREFETCH_MODEL])
        )

    def _dependencies(self) -> Container:
        return self._environment(_DEPS_SEED or _BASE_IMAGE)

    @function
    def deps_image(
        self, revision: str = "", lock_hash: str = "", source_url: str = ""
    ) -> Container:
        """Build the dependency image published by `deps-image.yml`: always from the base image,
        so every refresh re-resolves the OS packages instead of stacking on the previous image.
        `source_url` labels the repository so GHCR links the package to it. The image records the
        base it was built from; the pin PR copies that value so `admitted_seed` can refuse the
        seed once the checkout's base moves."""
        image = self._environment(_BASE_IMAGE)
        for key, value in (
            ("org.opencontainers.image.source", source_url),
            ("org.opencontainers.image.revision", revision),
            ("dev.ssgrep.ci.lock-hash", lock_hash),
            ("dev.ssgrep.ci.base-image", _BASE_IMAGE),
        ):
            if value:
                image = image.with_label(key, value)
        return image

    def _app(self) -> Container:
        return (
            self._dependencies()
            .with_directory("/repo", self.source)
            .with_exec(_uv_sync("--no-default-groups", "--group", "dev"))
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
                core().cache_volume("ssgrep-ci-uv-module-py314"),
            )
            .with_directory("/module", self.module_source)
            .with_exec(["uv", "sync", "--locked", "--group", "dev"])
        )

    def _test_container(self) -> Container:
        """Application container with Git metadata (for the hygiene test). The pinned embedding
        model is already in the dependency stage; ssgrep switches the Hub to offline mode itself
        once the model is cached, and a process-wide HF_HUB_OFFLINE would break the test that
        proves a cache miss stays online (huggingface_hub reads it at import time)."""
        return (
            self._app()
            .with_directory("/repo/.git", self.git_metadata)
            .with_exec(["git", "config", "--global", "--add", "safe.directory", "/repo"])
            .with_exec(["git", "rev-parse", "--show-toplevel"])
        )

    async def _pytest(self, args: list[str]) -> None:
        await self._test_container().with_exec(["uv", "run", "pytest", "-v", *args]).sync()

    @function  # ty: ignore[invalid-argument-type]
    @check
    async def module_quality(self) -> None:
        """Lint, format-check, type-check and unit-test first-party Dagger module code."""
        container = self._module_quality_container()
        container = container.with_exec(["uv", "run", "ruff", "check", "src/ssgrep_ci", "tests"])
        container = container.with_exec(
            ["uv", "run", "ruff", "format", "--check", "src/ssgrep_ci", "tests"]
        )
        container = container.with_exec(
            [
                "uv",
                "run",
                "ty",
                "check",
                "--no-respect-ignore-files",
                "src/ssgrep_ci/__init__.py",
                "src/ssgrep_ci/recipe.py",
                "src/ssgrep_ci/pin.py",
                "tests/test_recipe.py",
            ]
        )
        await container.with_exec(["uv", "run", "pytest", "-q", "tests"]).sync()

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
                    f"uv sync --locked {_SKIP_UNAVAILABLE} --no-default-groups "
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
