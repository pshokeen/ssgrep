"""Benchmark ingestion helper (Task 13 of the retrieval-eval overhaul).

``build_benchmark_index`` points the five runtime env overrides at a dataset
directory produced by the T7-T11 emitters, runs the REAL indexer once under a
temporary ``SSGREP_DATA_DIR`` (harness pattern), explicitly creates the vector
index, and validates the result.
This is the single place production ingestion is exercised end-to-end on the
synthetic benchmark; T14 (judge) and T16 (baseline) consume the returned
``BenchmarkIndex`` handle.

Dataset layout (the shape T15 freezes under ``eval/dataset/v1/``):

    <dataset_dir>/
      transcripts/
        claude/                 # claude emitter output
                                #   -> SSGREP_TRANSCRIPT_DIRS=native=<dir>
        codex/                  # codex emitter output
                                #   -> SSGREP_CODEX_SESSIONS_DIR=<dir>
        pi/                     # pi emitter output
                                #   -> SSGREP_PI_SESSIONS_DIR=<dir>
        prime-agent/
          sessions/             # prime-agent main sessions
                                #   -> SSGREP_PRIME_AGENT_SESSIONS_DIR=<dir>
          session-artifacts/    # sibling root the adapter computes
                                #   (pi.py:381-382, root.parent / "session-artifacts")
      opencode.db               # opencode emitter output
                                #   -> SSGREP_OPENCODE_DB=<path>

Runtime labels after ingestion (documented in T4 section 9): claude external
roots stamp ``native`` (native.py:26-27), NOT ``claude``; codex/pi/prime-agent/
opencode keep their names. Manifest runtime keys therefore use the
post-ingestion labels (``derive_manifest`` maps ``claude`` rows to ``native``).

The caller is responsible for the fake-embedder boundary in tests (patch
``app_mod.ColBERTEmbedder`` + ``ensure_model_downloaded`` +
``ssgrep.search._query_matrix``); this helper never touches the live data root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from eval import harness
from ssgrep.indexing import indexer
from ssgrep.store import CHUNKS_TABLE, LanceStore
from ssgrep.store.paths import database_dir
from ssgrep.utilities.types import IndexStats

#: Dataset subdirectory names (T15 freezes the same shape under
#: ``eval/dataset/v1/transcripts/<runtime>/``).
TRANSCRIPTS_DIR = "transcripts"
CLAUDE_DIR = "claude"
CODEX_DIR = "codex"
PI_DIR = "pi"
PRIME_AGENT_DIR = "prime-agent"
PRIME_AGENT_SESSIONS_DIR = "sessions"
OPENCODE_DB_NAME = "opencode.db"

#: Runtime labels stamped by the real adapters after ingestion. Claude
#: external roots become ``native`` (native.py:26-27), so the ``claude``
#: census bucket is satisfied by the file format, not the stored label.
RUNTIME_LABELS = ("native", "codex", "pi", "prime-agent", "opencode")

#: Env override names, in the same order as ``RUNTIME_LABELS``.
_ENV_OVERRIDES = (
    "SSGREP_TRANSCRIPT_DIRS",
    "SSGREP_CODEX_SESSIONS_DIR",
    "SSGREP_PI_SESSIONS_DIR",
    "SSGREP_PRIME_AGENT_SESSIONS_DIR",
    "SSGREP_OPENCODE_DB",
)


@dataclass(frozen=True)
class BenchmarkIndex:
    """Handle exposing the built index for T14/T16."""

    index_dir: Path
    db_path: Path
    stats: IndexStats

    def store(self) -> LanceStore:
        """A LanceStore bound to this index's database."""
        with harness._private_data_dir(self.index_dir):
            return LanceStore()


def _dataset_paths(dataset_dir: Path) -> dict[str, Path]:
    """Resolve the five emitter-output paths, raising on any missing one."""
    transcripts = dataset_dir / TRANSCRIPTS_DIR
    paths = {
        "native": transcripts / CLAUDE_DIR,
        "codex": transcripts / CODEX_DIR,
        "pi": transcripts / PI_DIR,
        "prime-agent": transcripts / PRIME_AGENT_DIR / PRIME_AGENT_SESSIONS_DIR,
        "opencode": dataset_dir / OPENCODE_DB_NAME,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"dataset {dataset_dir} is missing emitter output: " + ", ".join(missing)
        )
    return paths


@contextmanager
def _dataset_env(dataset_dir: Path) -> Iterator[None]:
    """Point the five runtime overrides at the dataset's emitted dirs."""
    paths = _dataset_paths(dataset_dir)
    previous = {name: os.environ.get(name) for name in _ENV_OVERRIDES}
    os.environ["SSGREP_TRANSCRIPT_DIRS"] = f"native={paths['native']}"
    os.environ["SSGREP_CODEX_SESSIONS_DIR"] = str(paths["codex"])
    os.environ["SSGREP_PI_SESSIONS_DIR"] = str(paths["pi"])
    os.environ["SSGREP_PRIME_AGENT_SESSIONS_DIR"] = str(paths["prime-agent"])
    os.environ["SSGREP_OPENCODE_DB"] = str(paths["opencode"])
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _episode_count(episodes: object) -> int:
    """Count text-bearing episodes the way the emitters emit them.

    Emitters drop text-less prompts (episodes.py:81-87 folds them), so a
    manifest derived from a parquet with empty prompts would over-count.
    """
    if not isinstance(episodes, list):
        return 0
    count = 0
    for item in episodes:
        if isinstance(item, dict):
            prompt = item.get("prompt")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            prompt = item[0]
        else:
            continue
        if prompt is not None and str(prompt) != "":
            count += 1
    return count


def derive_manifest(parquet_path: str | Path) -> dict[str, dict[str, int]]:
    """Expected per-runtime session/episode counts from ``sessions.parquet``.

    Runtime keys use the post-ingestion labels (``claude`` rows map to
    ``native``). The returned shape is the manifest contract T15 freezes:
    ``{"sessions": {runtime: int}, "episodes": {runtime: int}}``.
    """
    table = pq.read_table(parquet_path)
    sessions: dict[str, int] = {}
    episodes: dict[str, int] = {}
    for row in table.to_pylist():
        runtime = row.get("runtime")
        label = "native" if runtime == "claude" else runtime
        sessions[label] = sessions.get(label, 0) + 1
        episodes[label] = episodes.get(label, 0) + _episode_count(row.get("episodes"))
    return {"sessions": sessions, "episodes": episodes}


def _validate(
    store: LanceStore,
    stats: IndexStats,
    manifest: Mapping[str, Mapping[str, int]] | None,
) -> None:
    """Fail loudly when the built index disagrees with the manifest."""
    if stats.malformed_records != 0:
        raise AssertionError(
            f"index build reported {stats.malformed_records} malformed records; expected 0"
        )
    for label in RUNTIME_LABELS:
        chunk_count = store.count(CHUNKS_TABLE, where=f"runtime = '{label}'")
        if chunk_count == 0:
            raise AssertionError(f"runtime {label!r} has no chunks in the built index")
    if manifest is None:
        return
    for table, expected in manifest.items():
        for label, want in expected.items():
            got = store.count(table, where=f"runtime = '{label}'")
            if got != want:
                raise AssertionError(f"{table}[{label}] count {got} != expected {want}")


def build_benchmark_index(
    dataset_dir: str | Path,
    index_dir: str | Path,
    *,
    manifest: Mapping[str, Mapping[str, int]] | None = None,
) -> BenchmarkIndex:
    """Build a private index from a full emitter-output dataset.

    Points all five runtime env overrides at the dataset's emitted dirs, runs
    a SINGLE real-indexer pass under a temporary ``SSGREP_DATA_DIR``, then
    explicitly creates the vector index (T5-fix pattern). Cocoindex allows
    exactly one open Environment per process, so a second ``indexer.index``
    call would raise ``RuntimeError: environment already open`` under the
    real model; the vector index defers at ``initialize()`` time while the
    chunks table is empty, so ``ensure_vector_index()`` after the pass is
    required (T3/T5 learnings). ``CLAUDE_CONFIG_DIR`` is sandboxed to an
    empty temp dir so the native adapter's live ``~/.claude`` auto-discovery
    cannot leak real sessions into a benchmark index (T16 build pattern).

    Validates the result: every runtime present in the chunks table, zero
    malformed records, and (when ``manifest`` is given) per-runtime
    session/episode counts equal to the manifest's expectations.

    Raises ``FileNotFoundError`` naming the missing directory when any of the
    five emitter outputs is absent. Never touches the live data root.
    """
    dataset = Path(dataset_dir)
    index = Path(index_dir)
    index.mkdir(parents=True, exist_ok=True)
    sandbox_claude = tempfile.mkdtemp(prefix="ssgrep-claude-empty-")
    previous_claude = os.environ.get("CLAUDE_CONFIG_DIR")
    os.environ["CLAUDE_CONFIG_DIR"] = sandbox_claude
    try:
        with _dataset_env(dataset), harness._private_data_dir(index):
            stats = indexer.index(rebuild=True, allow_shrink=True, quiet=True)
            # Single pass: the index pass initializes before rows land, so the
            # vector index only builds once live rows exist (T3 learning); a
            # second index() pass would crash with "environment already open"
            # under the real model (T5 learning).
            LanceStore().ensure_vector_index()
            store = LanceStore()
            _validate(store, stats, manifest)
            db_path = database_dir()
    finally:
        if previous_claude is None:
            os.environ.pop("CLAUDE_CONFIG_DIR", None)
        else:
            os.environ["CLAUDE_CONFIG_DIR"] = previous_claude
    return BenchmarkIndex(index_dir=index, db_path=db_path, stats=stats)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="dataset directory (emitter output)")
    parser.add_argument("--index-dir", required=True, help="private index directory to build into")
    parser.add_argument(
        "--manifest",
        default=None,
        help="manifest.json path (default: <dataset>/manifest.json when present)",
    )
    args = parser.parse_args(argv)

    dataset = Path(args.dataset)
    manifest_path = Path(args.manifest) if args.manifest else dataset / "manifest.json"
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get("counts")

    try:
        handle = build_benchmark_index(dataset, args.index_dir, manifest=manifest)
    except (AssertionError, OSError) as error:
        print(f"ingest failed: {error}", file=sys.stderr)
        return 2
    stats = handle.stats
    print(
        f"built {handle.index_dir}: sessions={stats.session_count} "
        f"episodes={stats.episode_count} chunks={stats.chunk_count} "
        f"malformed={stats.malformed_records}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchmarkIndex",
    "CLAUDE_DIR",
    "CODEX_DIR",
    "OPENCODE_DB_NAME",
    "PI_DIR",
    "PRIME_AGENT_DIR",
    "PRIME_AGENT_SESSIONS_DIR",
    "RUNTIME_LABELS",
    "TRANSCRIPTS_DIR",
    "build_benchmark_index",
    "derive_manifest",
    "main",
]
