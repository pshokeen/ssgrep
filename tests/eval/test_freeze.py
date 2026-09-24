"""Tests for ``eval.datasetgen.freeze`` (Task 15: BEIR export + artifact freeze).

The freeze module runs the five T7-T11 emitters on sessions.parquet and writes
the immutable BEIR-shaped artifact with a sha256 manifest. These tests build a
small multi-runtime fixture parquet + a tasked queries.jsonl, freeze it into a
temp dir, then exercise the acceptance contract: verify round-trip, tamper
detection (non-zero verify + file named), BEIR files load through
ir_measures' read helpers, and the immutability guard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from eval.datasetgen import freeze
from eval.datasetgen.generate_queries import _build_index
from eval.datasetgen.ingest import derive_manifest

#: Fixed rows matching the shared emitter contract (T6 schema subset).
ROWS: list[dict] = [
    {
        "runtime": "claude",
        "session_id": "claude-fix-0",
        "title": "Backoff policy",
        "project": "kepler-apps/uptime-monitor",
        "files_touched": ["src/retry.py"],
        "tool_names": ["Read", "Edit"],
        "episodes": [
            {"prompt": "Where is the backoff policy?", "response": "In retry.py, capped at 60s."},
            {"prompt": "How do I test it?", "response": "Run the three-attempt retry harness."},
        ],
    },
    {
        "runtime": "claude",
        "session_id": "claude-fixed-1",
        "title": "Async migration",
        "project": "kepler-apps/uptime-monitor",
        "files_touched": ["migrate.py"],
        "tool_names": ["Write"],
        "episodes": [
            {
                "prompt": "Why did the async migration fail?",
                "response": "The batch writer raced the flush.",
            },
        ],
    },
    {
        "runtime": "codex",
        "session_id": "codex-fixed-0",
        "title": "Cache invalidation",
        "project": "kepler-apps/uptime-monitor",
        "files_touched": ["cache.py"],
        "tool_names": ["read"],
        "episodes": [
            {"prompt": "Why is the cache stale?", "response": "The TTL resets on every read."},
            {"prompt": "Fix it?", "response": "Bump the version key on writes."},
        ],
    },
    {
        "runtime": "pi",
        "session_id": "pi-fixed-0",
        "title": "Template fix",
        "project": "jasperdev/invoicer",
        "files_touched": ["tpl.py"],
        "tool_names": ["edit"],
        "episodes": [
            {"prompt": "PDFs are blank", "response": "The template resolves the wrong key."},
        ],
    },
    {
        "runtime": "prime-agent",
        "session_id": "session_shared",
        "title": "Pool ceiling",
        "project": "jasperdev/invoicer",
        "episodes": [
            {"prompt": "Raise the pool?", "response": "Bump MAX_POOL."},
        ],
    },
    {
        "runtime": "opencode",
        "session_id": "session_shared",
        "title": "Version key",
        "project": "jasperdev/invoicer",
        "episodes": [
            {"prompt": "Invalidate the cache?", "response": "Bump the version key on writes."},
            {"prompt": "Golden fixtures?", "response": "Commit one fixture per template."},
        ],
    },
]


def _write_parquet(tmp_path: Path) -> Path:
    parquet_path = tmp_path / "sessions.parquet"
    names = {key for row in ROWS for key in row}
    columns = {name: [row.get(name) for row in ROWS] for name in names}
    pq.write_table(pa.Table.from_pydict(columns), parquet_path)
    return parquet_path


def _write_queries(parquet_path: Path, queries_path: Path) -> None:
    """Author a small labelled query set against the real mini id space.

    Uses the derived id space (same mapping as T12) so the frozen qrels refer
    to real corpus _ids. Rows resemble the T12 schema (id/query/class/
    target_episode_ids/hard_negative_episode_ids/split/runtime/...).
    """
    index = _build_index(pq.read_table(parquet_path).to_pylist())
    all_episodes = sorted(index.episodes)
    primary = all_episodes[0]
    secondary = all_episodes[1]
    queries = [
        {
            "id": "exact-0001",
            "query": "retry harness backoff",
            "class": "exact-identifier",
            "anchors": ["retry harness"],
            "anchor_mode": "all",
            "target_episode_ids": [primary],
            "target_session_ids": [index.episodes[primary]["session_id"]],
            "subagent_only": False,
            "split": "train",
            "runtime": index.episodes[primary]["runtime"],
            "hard_negative_episode_ids": [secondary],
            "grounding": [
                {
                    "episode_id": primary,
                    "project": index.episodes[primary]["project"],
                    "runtime": index.episodes[primary]["runtime"],
                    "prompt": index.episodes[primary]["prompt"][:300],
                    "response": index.episodes[primary]["response"][:300],
                }
            ],
            "notes": "",
        },
        {
            "id": "error-0001",
            "query": "why did the flush race",
            "class": "error-string",
            "anchors": ["flush race"],
            "anchor_mode": "all",
            "target_episode_ids": [secondary],
            "target_session_ids": [index.episodes[secondary]["session_id"]],
            "subagent_only": False,
            "split": "holdout",
            "runtime": index.episodes[secondary]["runtime"],
            "hard_negative_episode_ids": [primary],
            "grounding": [
                {
                    "episode_id": secondary,
                    "project": index.episodes[secondary]["project"],
                    "runtime": index.episodes[secondary]["runtime"],
                    "prompt": index.episodes[secondary]["prompt"][:300],
                    "response": index.episodes[secondary]["response"][:300],
                }
            ],
            "notes": "",
        },
        {
            "id": "para-0001",
            "query": "how do we handle rendering",
            "class": "paraphrase",
            "anchors": ["template"],
            "anchor_mode": "all",
            "target_episode_ids": [all_episodes[2]],
            "target_session_ids": [index.episodes[all_episodes[2]]["session_id"]],
            "subagent_only": False,
            "split": "train",
            "runtime": index.episodes[all_episodes[2]]["runtime"],
            "hard_negative_episode_ids": [],
            "grounding": [
                {
                    "episode_id": all_episodes[2],
                    "project": index.episodes[all_episodes[2]]["project"],
                    "runtime": index.episodes[all_episodes[2]]["runtime"],
                    "prompt": index.episodes[all_episodes[2]]["prompt"][:300],
                    "response": index.episodes[all_episodes[2]]["response"][:300],
                }
            ],
            "notes": "",
        },
    ]
    with queries_path.open("w", encoding="utf-8") as fh:
        for row in queries:
            fh.write(json.dumps(row) + "\n")


def _freeze(tmp_path: Path) -> Path:
    parquet_path = _write_parquet(tmp_path)
    queries_path = tmp_path / "queries.jsonl"
    _write_queries(parquet_path, queries_path)
    out_dir = tmp_path / "v1"
    freeze.freeze_benchmark(
        parquet=parquet_path,
        queries=queries_path,
        out_dir=out_dir,
        generation_config=None,
        profile=None,
        git_sha="0000000",
    )
    return out_dir


def test_freeze_roundtrip_verify(tmp_path: Path) -> None:
    """Freeze a small dataset; verify_manifest round-trips with OK."""
    out_dir = _freeze(tmp_path)

    manifest = freeze.verify_manifest(out_dir)
    assert manifest["version"] == freeze.MANIFEST_SCHEMA
    assert manifest["created_from"] == "0000000"
    assert isinstance(manifest["files"], dict)
    assert manifest["files"]  # non-empty
    assert "corpus.jsonl" in manifest["files"]
    assert "qrels/train.tsv" in manifest["files"]
    assert "qrels/holdout.tsv" in manifest["files"]
    assert "transcripts/claude/claude-0000.jsonl" in manifest["files"]
    assert "transcripts/prime-agent/sessions/session_shared.jsonl" in manifest["files"]
    assert "opencode.db" in manifest["files"]

    # Counts reconcile.
    counts = derive_manifest(tmp_path / "sessions.parquet")
    assert manifest["counts"]["sessions"]["native"] == counts["sessions"]["native"] == 2
    assert manifest["counts"]["sessions"]["opencode"] == counts["sessions"]["opencode"] == 1
    # corpus_total equals episodes derived from the parquet
    assert manifest["corpus_total"] == sum(counts["episodes"].values())
    assert manifest["queries_total"] == len(list(open(Path(tmp_path) / "queries.jsonl")))

    # CLI --verify exits 0 and prints "OK"
    from eval.datasetgen.freeze import main

    assert main(["--verify", str(out_dir)]) == 0


def test_tamper_detection_names_mismatched_file(tmp_path: Path, capsys) -> None:
    """Appending whitespace to a copied corpus.jsonl fails verify naming it."""
    out_dir = _freeze(tmp_path)
    tampered = tmp_path / "tampered"
    import shutil

    shutil.copytree(out_dir, tampered)
    with (tampered / "corpus.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(" \n")

    from eval.datasetgen.freeze import main

    rc = main(["--verify", str(tampered)])
    assert rc == 1
    assert "sha256 mismatch for corpus.jsonl" in capsys.readouterr().err


def _trec_qrels(path: Path) -> list[tuple[str, str, int]]:
    """Convert BEIR 3-column (header + query-id\\tcorpus-id\\tscore) to trec 4-col.

    ``ir_measures.read_trec_qrels`` expects the trec_eval ``qid Q0 docid rel``
    shape; the BEIR contract's 3-column form is what ``run_eval.load_qrels``
    and the frozen artifact ship. The conversion is the standard treatment.
    """
    rows: list[tuple[str, str, int]] = []
    for line in path.open():
        fields = line.strip().split()
        if not fields or fields[0] == "query-id":
            continue
        rows.append((fields[0], fields[1], int(fields[2])))
    return rows


def test_beir_files_load_via_ir_measures(tmp_path: Path) -> None:
    """Corpus/queries/qrels load through ir_measures read helpers cleanly."""
    out_dir = _freeze(tmp_path)

    import ir_measures

    corpus = [json.loads(line) for line in (out_dir / "corpus.jsonl").open()]
    assert corpus
    assert sorted(corpus[0]) == ["_id", "text", "title"]
    ids = {row["_id"] for row in corpus}
    assert len(ids) == len(corpus)

    queries = [json.loads(line) for line in (out_dir / "queries.jsonl").open()]
    query_ids = {row["id"] for row in queries}
    assert all("_id" in row and "text" in row for row in queries)

    for path in (
        out_dir / "qrels.tsv",
        out_dir / "qrels" / "train.tsv",
        out_dir / "qrels" / "holdout.tsv",
    ):
        trec_lines = "\n".join(
            f"{qid} Q0 {docid} {grade}" for qid, docid, grade in _trec_qrels(path)
        )
        rels = list(ir_measures.read_trec_qrels(trec_lines))
        assert rels, path
        for rel in rels:
            assert rel.query_id in query_ids
            assert rel.doc_id in ids
            assert rel.relevance in {0, 3}


def test_freeze_refuses_non_empty_out_dir(tmp_path: Path) -> None:
    """Immutability contract: an existing artifact is never overwritten."""
    _freeze(tmp_path)
    with pytest.raises(ValueError, match="not empty"):
        freeze.freeze_benchmark(
            parquet=tmp_path / "sessions.parquet",
            queries=tmp_path / "queries.jsonl",
            out_dir=tmp_path / "v1",
        )


def _write_judge_qrels(path: Path, rows: list[tuple[str, str, int]]) -> None:
    """Write a BEIR-format judge qrels TSV (header + rows)."""
    lines = ["query-id\tcorpus-id\tscore"]
    for qid, episode_id, grade in rows:
        lines.append(f"{qid}\t{episode_id}\t{grade}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _canonical_episodes(parquet_path: Path) -> list[str]:
    """Sorted canonical episode ids (the frozen id space)."""
    rows = pq.read_table(parquet_path).to_pylist()
    canon = freeze.canonical_rows(rows)
    return sorted(_build_index([dict(r) for r in canon]).episodes)


def _seed_canonical_ids(parquet_path: Path) -> set[str]:
    """Canonical ids of the raw all_episodes[0..2] grounding seeds."""
    rows = pq.read_table(parquet_path).to_pylist()
    raw_index = _build_index([dict(r) for r in rows])
    canon = freeze.canonical_rows(rows)
    by_content = freeze._content_to_canonical_id(canon)
    all_eps = sorted(raw_index.episodes)
    return {
        by_content[freeze._content_key(raw_index.episodes[ep])]
        for ep in (all_eps[0], all_eps[1], all_eps[2])
    }


def _freeze_with_judge(tmp_path: Path, judge_qrels: Path) -> Path:
    parquet_path = _write_parquet(tmp_path)
    queries_path = tmp_path / "queries.jsonl"
    _write_queries(parquet_path, queries_path)
    out_dir = tmp_path / "v1"
    freeze.freeze_benchmark(
        parquet=parquet_path,
        queries=queries_path,
        out_dir=out_dir,
        generation_config=None,
        profile=None,
        git_sha="0000000",
        judge_qrels=judge_qrels,
    )
    return out_dir


def test_judge_qrels_grades_merged_into_frozen_qrels(tmp_path: Path) -> None:
    """Judge 1-2 grades for non-seed episodes land in the frozen qrels."""
    parquet_path = _write_parquet(tmp_path)
    queries_path = tmp_path / "queries.jsonl"
    _write_queries(parquet_path, queries_path)
    canon_eps = _canonical_episodes(parquet_path)
    seed_ids = _seed_canonical_ids(parquet_path)
    non_seed = [ep for ep in canon_eps if ep not in seed_ids]
    assert len(non_seed) >= 2
    judge_ep_a, judge_ep_b = non_seed[0], non_seed[1]

    judge_path = tmp_path / "judge_qrels.tsv"
    _write_judge_qrels(
        judge_path,
        [
            ("exact-0001", judge_ep_a, 2),
            ("error-0001", judge_ep_b, 1),
        ],
    )

    out_dir = _freeze_with_judge(tmp_path, judge_path)
    qrels = _trec_qrels(out_dir / "qrels.tsv")
    by_key = {(qid, doc): grade for qid, doc, grade in qrels}

    # Judge grades merged for non-seed episodes.
    assert by_key[("exact-0001", judge_ep_a)] == 2
    assert by_key[("error-0001", judge_ep_b)] == 1

    # Seeds stay grade 3, hard negatives stay grade 0 (authoritative).
    rows = pq.read_table(parquet_path).to_pylist()
    raw_index = _build_index([dict(r) for r in rows])
    canon = freeze.canonical_rows(rows)
    by_content = freeze._content_to_canonical_id(canon)
    all_eps = sorted(raw_index.episodes)
    target_canon = by_content[freeze._content_key(raw_index.episodes[all_eps[0]])]
    hn_canon = by_content[freeze._content_key(raw_index.episodes[all_eps[1]])]
    assert by_key[("exact-0001", target_canon)] == 3
    assert by_key[("exact-0001", hn_canon)] == 0

    # The frozen qrels carry the full 0-3 scale.
    assert {grade for _, _, grade in qrels} == {0, 1, 2, 3}


def test_judge_qrels_resolution_skips_unknown_episodes(tmp_path: Path) -> None:
    """A judge row pointing at an unresolvable episode is skipped and counted."""
    parquet_path = _write_parquet(tmp_path)
    queries_path = tmp_path / "queries.jsonl"
    _write_queries(parquet_path, queries_path)
    judge_path = tmp_path / "judge_qrels.tsv"
    _write_judge_qrels(judge_path, [("exact-0001", "nonexistent-episode", 2)])

    out_dir = _freeze_with_judge(tmp_path, judge_path)
    qrels = _trec_qrels(out_dir / "qrels.tsv")
    assert ("exact-0001", "nonexistent-episode", 2) not in qrels

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["judge_qrels"] == {"resolved": 0, "skipped": 1}


def test_freeze_without_judge_qrels_unchanged(tmp_path: Path) -> None:
    """Without judge_qrels the frozen output is byte-identical across runs."""
    parquet_path = _write_parquet(tmp_path)
    queries_path = tmp_path / "queries.jsonl"
    _write_queries(parquet_path, queries_path)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    freeze.freeze_benchmark(
        parquet=parquet_path,
        queries=queries_path,
        out_dir=out_a,
        generation_config=None,
        profile=None,
        git_sha="0000000",
    )
    freeze.freeze_benchmark(
        parquet=parquet_path,
        queries=queries_path,
        out_dir=out_b,
        generation_config=None,
        profile=None,
        git_sha="0000000",
    )
    for rel in (
        "qrels.tsv",
        "qrels/train.tsv",
        "qrels/holdout.tsv",
        "queries.jsonl",
        "corpus.jsonl",
        "manifest.json",
    ):
        assert (out_a / rel).read_bytes() == (out_b / rel).read_bytes(), rel
    manifest = json.loads((out_a / "manifest.json").read_text(encoding="utf-8"))
    assert "judge_qrels" not in manifest
