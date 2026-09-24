"""BEIR export + artifact freeze (Task 15 of the retrieval-eval overhaul).

Consumes the REAL generated data (``sessions.parquet`` from T6,
``queries.jsonl`` from T12) and the five T7-T11 emitters, then writes the
immutable benchmark artifact::

    <out-dir>/
        manifest.json        # sha256 of EVERY file + provenance counts
        corpus.jsonl         # BEIR: {"_id", "title", "text"} per episode
        queries.jsonl        # T12 records + BEIR _id/text aliases
        qrels.tsv            # merged graded qrels (BEIR header)
        qrels/train.tsv      # train-split qrels (same header)
        qrels/holdout.tsv    # holdout-split qrels
        transcripts/
            claude/          # T7 native JSONL output
            codex/           # T8 codex JSONL output
            pi/              # T9 pi JSONL output
            prime-agent/     # T10 sessions/ + session-artifacts/
        opencode.db          # T11 opencode SQLite output

The transcript layout matches the dataset contract the T13 ingestion helper
(``eval/datasetgen/ingest.py::build_benchmark_index``) and ``eval.run_eval``
consume: claude via ``SSGREP_TRANSCRIPT_DIRS`` (stamped ``native``),
codex/pi/prime-agent via their env overrides, opencode.db at the dataset root.

Canonical id space (a real T6 quirk is handled here): the parquet emits
colliding ``session_id`` values across distinct opencode and prime-agent rows
(verified on the committed T6 parquet: 14 opencode + 2 prime-agent distinct
sessions share raw ids, e.g. ``session_0423``), which would crash the opencode
emitter's PRIMARY KEY and silently overwrite prime-agent files. This module:

#. de-duplicates those ids deterministically before emission (the N-th
   occurrence becomes ``<id>-<N>``) -- the canonical row set;
#. REBUILDS every T12 ``target_episode_ids`` / ``hard_negative_episode_ids``
   reference onto the canonical id space by exact (runtime, prompt, response)
   content match.

T12 authored its references against ``_build_index`` on the RAW rows, which
collapses the colliding ids (``dict`` overwrite -- 827 episodes instead of the
real 867 text-bearing episodes). Without the remap the frozen qrels would
reference episode ids that do not survive ingestion (the deduplicated ids do),
so ``run_eval``'s ``validate_qrel_targets`` preflight would fail. Each
reference is re-anchored to the canonical episode whose content matches
exactly; an absent or ambiguous anchor raises loudly (never silently dropped).

Qrel grades come from the T12 grounding seeds (grade 3 targets, grade 0 hard
negatives) -- the frozen T15 contract. When the T14 judge's graded qrels
(``--judge-qrels``) are supplied, its 1-2 grades for non-seed episodes are
merged in (seeds/hard negatives stay authoritative), so the frozen qrels carry
the full 0-3 graded relevance scale. The manifest records ``created_from`` = the
git HEAD at freeze time, per-runtime session/episode counts
(``derive_manifest``), split sizes, and generation config / profile census
digests.

Freeze contract: the artifact directory is immutable after commit; fixes are a
new version directory (``v2/``). ``verify_manifest`` / ``--verify`` proves
every file matches its sha256 and that no unlisted file exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from eval.datasetgen.emitters import (
    claude as claude_emitter,
    codex as codex_emitter,
    opencode as opencode_emitter,
    pi as pi_emitter,
    prime_agent as prime_agent_emitter,
)
from eval.datasetgen.generate_queries import _build_index
from eval.datasetgen.ingest import (
    CLAUDE_DIR,
    CODEX_DIR,
    OPENCODE_DB_NAME,
    PI_DIR,
    PRIME_AGENT_DIR,
    TRANSCRIPTS_DIR,
    derive_manifest,
)

#: Manifest schema ``version`` value; ``run_eval`` reports it as dataset version.
MANIFEST_SCHEMA = "v2"
#: BEIR qrels column header (the interchange contract).
QRELS_HEADER = "query-id\tcorpus-id\tscore"
#: Grounding-seed grades (T15 frozen contract; T14 refines in a v2).
GRADE_TARGET = 3
GRADE_HARD_NEGATIVE = 0

_MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    """HEAD sha of the repository at freeze time ("" when git is unavailable)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _runtime_rows(rows: list[dict[str, Any]], runtime: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("runtime") == runtime]


def _dedupe_session_ids(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Name every ``session_id`` in a runtime's rows uniquely (deterministic).

    Later occurrences of a duplicated id become ``<original>-<n>`` (n = 1 for
    the second occurrence). Row count and content are preserved.
    """
    seen: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        original = str(row.get("session_id") or "")
        occurrence = seen.get(original, 0)
        seen[original] = occurrence + 1
        if occurrence == 0 or not original:
            out.append(row)
            continue
        renamed = dict(row)
        renamed["session_id"] = f"{original}-{occurrence}"
        out.append(renamed)
    return out


def canonical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The canonical full row set the frozen corpus / id space is built from.

    opencode/prime-agent rows carry colliding ``session_id`` values that would
    crash their emitters (opencode PRIMARY KEY / prime-agent file overwrite),
    so those runtime rows are de-duplicated. claude/codex/pi never collide
    (their emitters rebuild raw ids from a content sort key internally, so the
    parquet ``session_id`` values do not reach the index), so their rows pass
    through unchanged.
    """
    fixed = ("opencode", "prime-agent")
    return [row for row in rows if row.get("runtime") not in fixed] + [
        row for runtime in fixed for row in _dedupe_session_ids(_runtime_rows(rows, runtime))
    ]


def _load_queries(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open()]


def _content_key(episode: Mapping[str, Any]) -> tuple[str, str, str]:
    """The identity of an episode's message content (id-independent)."""
    return (str(episode["runtime"]), str(episode["prompt"]), str(episode["response"]))


def _content_to_canonical_id(
    canonical: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str], str]:
    """Map content (runtime, prompt, response) -> canonical episode id.

    The canonical (de-duplicated) space is a bijection between episode content
    and episode id on the T6 corpus (verified: zero content collisions). A
    collision here means the corpus is ambiguous and raises.
    """
    index = _build_index([dict(row) for row in canonical])
    mapping: dict[tuple[str, str, str], str] = {}
    for episode_id, episode in index.episodes.items():
        key = _content_key(episode)
        previous = mapping.get(key)
        if previous is not None and previous != episode_id:
            raise ValueError(f"canonical content collision for {key!r}")
        mapping[key] = episode_id
    return mapping


def _remap_references(
    query_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
    canonical: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Re-anchor every ``*_episode_ids`` reference onto the canonical space.

    T12 authored episode ids against the raw (collapsed) id space; the frozen
    corpus uses the canonical (de-duplicated) one. Resolution is exact content:
    the canonical episode whose (runtime, prompt, response) equals the content
    of the referenced collapsed episode. An unresolvable or ambiguous anchor
    raises (never silently dropped).
    """
    collapsed = _build_index([dict(row) for row in raw_rows])
    canonical_index = _build_index([dict(row) for row in canonical])
    by_content = _content_to_canonical_id(canonical)

    def resolve(reference: str) -> str:
        episode = collapsed.episodes.get(reference)
        if episode is None:
            raise ValueError(f"unrecognized episode reference {reference!r}")
        key = _content_key(episode)
        target = by_content.get(key)
        if target is None:
            raise ValueError(
                f"cannot re-anchor {reference!r}: no canonical episode with that content"
            )
        return target

    remapped: list[dict[str, Any]] = []
    for query in query_rows:
        record = dict(query)
        for field in ("target_episode_ids", "hard_negative_episode_ids"):
            if field in record:
                record[field] = [resolve(ref) for ref in record[field]]
        # Session ids ride the episode ids: T12 authored them from the collapsed
        # space, so recompute them from the canonical target episodes.
        if "target_episode_ids" in record and "target_session_ids" in record:
            record["target_session_ids"] = sorted(
                {canonical_index.episodes[ep]["session_id"] for ep in record["target_episode_ids"]}
            )
        remapped.append(record)
    return remapped


def _corpus_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """BEIR corpus rows (``id``/``title``/``text``) from the canonical space."""
    index = _build_index(rows)
    corpus: list[dict[str, Any]] = []
    for episode_id in sorted(index.episodes):
        episode = index.episodes[episode_id]
        title = index.sessions[episode["session_id"]]["title"]
        text = f"{episode['prompt']}\n\n{episode['response']}".strip()
        corpus.append({"_id": episode_id, "title": title, "text": text})
    return corpus


def _freeze_queries(queries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Frozen queries.jsonl records: T12 fields plus BEIR ``_id``/``text``."""
    return [{**dict(query), "_id": query["id"], "text": query["query"]} for query in queries]


def _qrels_lines(query_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Rendered qrels TSV (BEIR header first; grade 3/0 respectively)."""
    lines = [QRELS_HEADER]
    for row in query_rows:
        qid = str(row["id"])
        for episode_id in row.get("target_episode_ids") or []:
            lines.append(f"{qid}\t{episode_id}\t{GRADE_TARGET}")
        for episode_id in row.get("hard_negative_episode_ids") or []:
            lines.append(f"{qid}\t{episode_id}\t{GRADE_HARD_NEGATIVE}")
    return lines


def _load_judge_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Parse the judge's BEIR qrels TSV into ``{qid: {episode: grade}}``.

    Accepts the 3-column ``query-id corpus-id grade`` form the judge writes and
    the trec_eval 4-column ``query-id Q0 corpus-id grade`` form; a header line
    is skipped. Mirrors ``run_eval.load_qrels`` locally (no import needed).
    """
    qrels: dict[str, dict[str, int]] = {}
    for line_number, line in enumerate(Path(path).open(encoding="utf-8"), start=1):
        fields = line.split()
        if not fields or fields[0].lower() in {"query-id", "qid"}:
            continue
        if len(fields) == 4 and fields[1] == "Q0":
            query_id, _, episode_id, grade = fields
        elif len(fields) == 3:
            query_id, episode_id, grade = fields
        else:
            raise ValueError(
                f"{path}:{line_number}: expected 'query-id corpus-id grade' or "
                f"'query-id Q0 corpus-id grade' (got {len(fields)} columns)"
            )
        qrels.setdefault(query_id, {})[episode_id] = int(grade)
    return qrels


def _reanchor_judge_grades(
    judge: Mapping[str, Mapping[str, int]],
    canonical: Sequence[Mapping[str, Any]],
    by_content: Mapping[tuple[str, str, str], str],
    raw_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, int]], int, int]:
    """Re-anchor judge episode ids onto the canonical id space by content.

    The judge ran against the draft index whose claude ids use absolute-path
    hashes, while the frozen claude ids use relative-path hashes; resolution is
    exact content (runtime, prompt, response), mirroring ``_remap_references``.
    A judge episode id already in the canonical space verbatim is kept; else it
    is looked up in the collapsed (raw) id space and re-anchored by content.
    Unresolvable rows are counted and skipped (never silently dropped).
    Returns ``(merged, resolved_count, skipped_count)``.
    """
    canonical_index = _build_index([dict(row) for row in canonical])
    collapsed = _build_index([dict(row) for row in raw_rows])
    merged: dict[str, dict[str, int]] = {}
    resolved = 0
    skipped = 0
    for qid in sorted(judge):
        for episode_id in sorted(judge[qid]):
            grade = judge[qid][episode_id]
            if episode_id in canonical_index.episodes:
                merged.setdefault(qid, {})[episode_id] = grade
                resolved += 1
                continue
            episode = collapsed.episodes.get(episode_id)
            target = None
            if episode is not None:
                target = by_content.get(_content_key(episode))
            if target is None:
                skipped += 1
                continue
            merged.setdefault(qid, {})[target] = grade
            resolved += 1
    return merged, resolved, skipped


def _qrels_lines_with_judge(
    query_rows: Sequence[Mapping[str, Any]],
    judge_grades: Mapping[str, Mapping[str, int]],
) -> list[str]:
    """Rendered qrels TSV with judge grades merged (BEIR header first).

    Seed targets stay grade 3 and hard negatives stay grade 0 (authoritative);
    judge grades for any OTHER episode are appended with their judged grade.
    Judge grade-0 rows for non-seed episodes are skipped (unjudged is 0 anyway).
    Deterministic: seed rows in ``_qrels_lines`` order, then judge-added rows
    sorted by (qid, episode_id). Only judge grades for queries in ``query_rows``
    are emitted (so split files carry only their own queries' grades).
    """
    lines = [QRELS_HEADER]
    seen: set[tuple[str, str]] = set()
    qids_in_subset = {str(row["id"]) for row in query_rows}
    for row in query_rows:
        qid = str(row["id"])
        for episode_id in row.get("target_episode_ids") or []:
            lines.append(f"{qid}\t{episode_id}\t{GRADE_TARGET}")
            seen.add((qid, episode_id))
        for episode_id in row.get("hard_negative_episode_ids") or []:
            lines.append(f"{qid}\t{episode_id}\t{GRADE_HARD_NEGATIVE}")
            seen.add((qid, episode_id))
    added: list[tuple[str, str, int]] = []
    for qid in sorted(judge_grades):
        if qid not in qids_in_subset:
            continue
        for episode_id in sorted(judge_grades[qid]):
            grade = judge_grades[qid][episode_id]
            if (qid, episode_id) in seen:
                continue  # seed/hard-negative authoritative; judge cannot override
            if grade <= 0:
                continue  # unjudged is 0 anyway
            added.append((qid, episode_id, grade))
    for qid, episode_id, grade in sorted(added):
        lines.append(f"{qid}\t{episode_id}\t{grade}")
    return lines


def _split_sizes(query_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for row in query_rows:
        split = str(row.get("split") or "train")
        sizes[split] = sizes.get(split, 0) + 1
    return sizes


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    _write_lines(path, [json.dumps(record, ensure_ascii=False) for record in records])


def _dataset_files(dataset_dir: Path) -> list[tuple[str, Path]]:
    """``(relative posix path, absolute path)`` for every regular file.

    The manifest itself is excluded so ``write_manifest`` can hash the tree.
    """
    found: list[tuple[str, Path]] = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.name != _MANIFEST_NAME:
            found.append((path.relative_to(dataset_dir).as_posix(), path))
    return sorted(found, key=lambda item: item[0])


def _write_manifest_files(dataset_dir: Path) -> dict[str, str]:
    """Compute ``{relpath: sha256-hex}`` for every frozen file."""
    return {rel: _sha256(path) for rel, path in _dataset_files(dataset_dir)}


def write_manifest(dataset_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Write ``manifest.json`` with sha256 of every file; returns the payload."""
    payload = {**dict(manifest), "files": _write_manifest_files(dataset_dir)}
    (dataset_dir / _MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def check_manifest(dataset_dir: str | Path) -> list[str]:
    """Integrity problems for a frozen dataset (empty list == intact).

    Verifies the manifest exists and parses, every listed file exists with a
    matching sha256, and no unlisted file lives in the dataset tree.
    """
    directory = Path(dataset_dir)
    manifest_path = directory / _MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"no manifest.json in {directory}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [f"{manifest_path} is not valid JSON ({error})"]
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return [f"{manifest_path} must declare a non-empty 'files' map"]

    on_disk = dict(_dataset_files(directory))
    problems: list[str] = []
    for name in sorted(files):
        if name not in on_disk:
            problems.append(f"missing {name}")
    for name in sorted(on_disk):
        if name not in files:
            problems.append(f"unlisted file {name}")
    if problems:
        return problems
    for name, path in on_disk.items():
        if _sha256(path) != files[name]:
            problems.append(f"sha256 mismatch for {name}")
    return problems


def verify_manifest(dataset_dir: str | Path) -> dict[str, Any]:
    """Verify a frozen dataset; raises ``ValueError`` listing every problem.

    Returns the parsed manifest on success. ``--verify`` and the acceptance
    tamper scenarios rely on the loud contract.
    """
    problems = check_manifest(dataset_dir)
    if problems:
        raise ValueError("; ".join(problems))
    return json.loads((Path(dataset_dir) / _MANIFEST_NAME).read_text(encoding="utf-8"))


def freeze_benchmark(
    *,
    parquet: str | Path,
    queries: str | Path,
    out_dir: str | Path,
    generation_config: str | Path | None = None,
    profile: str | Path | None = None,
    git_sha: str | None = None,
    judge_qrels: str | Path | None = None,
) -> dict[str, Any]:
    """Build the frozen v1 artifact under ``out_dir``; returns the manifest.

    Refuses to write into an existing non-empty directory (version contract).
    Wraps emitter failures in ``ValueError`` with the output path.

    When ``judge_qrels`` (the T14 judge's BEIR qrels TSV) is given, its graded
    rows are merged into the frozen qrels: seed targets stay grade 3, hard
    negatives stay grade 0 (authoritative), and any OTHER judged episode keeps
    its judged grade (1-3). Without it the output is byte-identical to the
    binary seed-only qrels.
    """
    parquet_path = Path(parquet)
    queries_path = Path(queries)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} already exists and is not empty; fixes are a new version dir")
    if not parquet_path.is_file():
        raise FileNotFoundError(f"parquet source not found: {parquet_path}")
    if not queries_path.is_file():
        raise FileNotFoundError(f"queries source not found: {queries_path}")
    out.mkdir(parents=True, exist_ok=True)

    rows = pq.read_table(parquet_path).to_pylist()
    query_rows = _load_queries(queries_path)
    canon = canonical_rows(rows)
    query_rows = _remap_references(query_rows, raw_rows=rows, canonical=canon)

    judge_manifest: dict[str, int] | None = None
    judge_grades: dict[str, dict[str, int]] = {}
    if judge_qrels is not None:
        judge = _load_judge_qrels(judge_qrels)
        by_content = _content_to_canonical_id(canon)
        judge_grades, resolved, skipped = _reanchor_judge_grades(judge, canon, by_content, rows)
        judge_manifest = {"resolved": resolved, "skipped": skipped}

    try:
        transcripts = out / TRANSCRIPTS_DIR
        claude_emitter.emit_sessions(parquet_path, transcripts / CLAUDE_DIR)
        codex_emitter.emit_sessions(parquet_path, transcripts / CODEX_DIR)
        pi_emitter.emit_sessions(parquet_path, transcripts / PI_DIR)
        prime_agent_emitter.emit(
            _dedupe_session_ids(_runtime_rows(rows, "prime-agent")),
            transcripts / PRIME_AGENT_DIR,
        )
        opencode_emitter.emit_sessions(
            _dedupe_session_ids(_runtime_rows(rows, "opencode")),
            out / OPENCODE_DB_NAME,
        )
    except Exception as error:  # noqa: BLE001 -- surfaced with the dataset path
        raise ValueError(f"freeze failed while emitting into {out}: {error}") from error

    corpus = _corpus_rows(canon)
    _write_jsonl(out / "corpus.jsonl", corpus)
    frozen = _freeze_queries(query_rows)
    _write_jsonl(out / "queries.jsonl", frozen)
    if judge_qrels is not None:
        _write_lines(out / "qrels.tsv", _qrels_lines_with_judge(query_rows, judge_grades))
    else:
        _write_lines(out / "qrels.tsv", _qrels_lines(query_rows))

    qrels_dir = out / "qrels"
    qrels_dir.mkdir(parents=True, exist_ok=True)
    if judge_qrels is not None:
        _write_lines(
            qrels_dir / "train.tsv",
            _qrels_lines_with_judge(
                [q for q in query_rows if q.get("split") == "train"], judge_grades
            ),
        )
        _write_lines(
            qrels_dir / "holdout.tsv",
            _qrels_lines_with_judge(
                [q for q in query_rows if q.get("split") == "holdout"], judge_grades
            ),
        )
    else:
        _write_lines(
            qrels_dir / "train.tsv",
            _qrels_lines([q for q in query_rows if q.get("split") == "train"]),
        )
        _write_lines(
            qrels_dir / "holdout.tsv",
            _qrels_lines([q for q in query_rows if q.get("split") == "holdout"]),
        )

    manifest: dict[str, Any] = {
        "version": MANIFEST_SCHEMA,
        "created_from": git_sha or _git_head(),
        "counts": derive_manifest(parquet_path),
        "split_sizes": _split_sizes(query_rows),
        "generation_config_digest": (
            _sha256(Path(generation_config)) if generation_config is not None else None
        ),
        "profile_census_digest": _sha256(Path(profile)) if profile is not None else None,
        "queries_total": len(query_rows),
        "corpus_total": len(corpus),
    }
    if judge_manifest is not None:
        manifest["judge_qrels"] = judge_manifest
    write_manifest(out, manifest)
    problems = check_manifest(out)  # self-check before returning
    if problems:
        raise ValueError(f"freeze produced an inconsistent artifact: {problems}")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="eval/datasetgen/sessions.parquet")
    parser.add_argument("--queries", default="eval/datasetgen/queries.jsonl")
    parser.add_argument("--out", default="eval/dataset/v1")
    parser.add_argument("--generation-config", default="eval/datasetgen/generation_config.json")
    parser.add_argument("--profile", default="eval/datasetgen/profile.json")
    parser.add_argument("--git-sha", default=None, help="override created_from (testing)")
    parser.add_argument(
        "--judge-qrels",
        default=None,
        metavar="PATH",
        help="T14 judge graded qrels (BEIR TSV) merged into the frozen qrels",
    )
    parser.add_argument(
        "--verify",
        nargs="?",
        const="eval/dataset/v2",
        default=None,
        metavar="DIR",
        help="verify a frozen dataset instead of freezing (exit 0 when OK)",
    )
    args = parser.parse_args(argv)

    if args.verify is not None:
        problems = check_manifest(args.verify)
        if problems:
            for problem in problems:
                print(f"freeze: {problem}", file=sys.stderr)
            return 1
        print("OK")
        return 0

    try:
        freeze_benchmark(
            parquet=args.parquet,
            queries=args.queries,
            out_dir=args.out,
            generation_config=(
                Path(args.generation_config) if Path(args.generation_config).is_file() else None
            ),
            profile=Path(args.profile) if Path(args.profile).is_file() else None,
            git_sha=args.git_sha,
            judge_qrels=Path(args.judge_qrels) if args.judge_qrels else None,
        )
    except (ValueError, OSError) as error:
        print(f"freeze failed: {error}", file=sys.stderr)
        return 2
    print(f"froze {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
