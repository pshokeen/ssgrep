"""Builds eval/queries.jsonl — a labelled retrieval query set for the ssgrep harness.

METHODOLOGY (read this before trusting the numbers in eval/README.md).

The failure mode this file exists to avoid: picking targets by running the
current search and labelling whatever it returns. That measures agreement of
the retriever with itself and produces excellent numbers that mean nothing.

Instead, every target here is located two steps removed from ranking:

1. The corpus is segmented into episodes using ssgrep's own PARSER
   (discovery.discover_sessions, records.read_records, episodes.segment_episodes).
   This is deterministic structural work — turning bytes into (session, episode)
   units — not the thing being evaluated. The thing being evaluated is the
   SCORED retriever: chunker + BM25 (FTS5) + vector cosine + RRF + main-session
   boost, in search.py. Segmentation and scoring are different code paths, and
   only the second is under test.

2. Each query's target episode(s) are located by plain substring containment
   over full episode text (prompt_text + response_text concatenated) against
   ANCHOR strings drawn independently from the project's own design and
   handoff notes — prose written to describe what happened in this project,
   not queries against it.
   Containment matching does not rank, does not chunk, does not embed, and
   does not touch FTS5 or model2vec. It cannot agree with the retriever by
   construction: it never runs the retriever.

3. QUERY TEXT is then composed by hand, in different words than the anchor
   wherever the class calls for it (paraphrase, multi-hop). The exact-identifier
   and error-string classes deliberately use verbatim tokens/strings, because
   that is what those classes test — that literal-token queries actually hit
   the episode containing that literal token. Paraphrase and multi-hop queries
   are phrased as a person would plausibly ask them, never copied from the
   anchor, and never checked against search() before being locked in below.

A real corpus is messy: many facts (e.g. "the BufferError bug", "the prune
WHERE 1=1 mutation") get restated across several episodes, because later
sessions routinely recap earlier findings. So a query can legitimately have
more than one correct target. target_episode_ids is therefore a SET, not a
single id:
recall@10 counts a hit if ANY member appears in the top 10, and MRR uses the
BEST (lowest) rank among them. This is the standard treatment for multi-
relevant-document evaluation and is documented, not hidden.

Anchors that match more than ANCHOR_MATCH_RATIO * corpus_size episodes are
rejected by an assertion below — a query whose "correct answer" spans a
large fraction of the corpus is testing retrieval of a topic, not a specific
episode, which is out of scope for this harness. The cap is proportional,
not a fixed count: a fixed cap goes stale the moment the corpus grows past
it, so it is expressed as a ratio and re-derived from today's corpus size
every time build() runs (see eval/README.md's methodology section, point 5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ssgrep import discovery, episodes, records

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = Path(__file__).resolve().parent / "queries.jsonl"

# An anchor matching more than this fraction of the corpus stops being a
# specific target and starts being a topic. Reject rather than silently
# keep it. Proportional, not absolute, so the cap scales with corpus growth
# automatically instead of needing manual re-tuning as the corpus grows.
ANCHOR_MATCH_RATIO = 0.01


@dataclass(frozen=True)
class QuerySpec:
    id: str
    query: str
    query_class: str  # exact-identifier | error-string | paraphrase | multi-hop
    anchors: tuple[str, ...]
    anchor_mode: str = "all"  # "all": every anchor must appear; "any": at least one
    notes: str = ""
    subagent_only_expected: bool = field(default=False)


# ---------------------------------------------------------------------------
# The labelled query set. Anchors are verbatim strings independently sourced
# from the project's design and handoff notes — read those documents'
# claims, then located in
# the raw corpus by containment, never by search().
# ---------------------------------------------------------------------------
QUERY_SPECS: list[QuerySpec] = [
    # ---- exact-identifier: verbatim code tokens a developer would grep for ----
    QuerySpec(
        "exact-01",
        "reciprocal_rank_fusion RRF_K",
        "exact-identifier",
        ("reciprocal_rank_fusion",),
    ),
    QuerySpec(
        "exact-02",
        "apply_main_session_boost",
        "exact-identifier",
        ("apply_main_session_boost",),
    ),
    QuerySpec(
        "exact-03",
        "_should_exclude_path memory tool-results workflows",
        "exact-identifier",
        ("_should_exclude_path",),
    ),
    QuerySpec(
        "exact-04",
        "get_vector_path generation store",
        "exact-identifier",
        ("get_vector_path",),
    ),
    QuerySpec(
        "exact-05",
        'db_path.parent / "vectors.f32" hardcoded',
        "exact-identifier",
        ('db_path.parent / "vectors.f32"',),
    ),
    QuerySpec(
        "exact-06",
        "potion-retrieval-32M MTEB Retrieval 35.06",
        "exact-identifier",
        ("MTEB Retrieval 35.06",),
        subagent_only_expected=True,
    ),
    QuerySpec(
        "exact-07",
        "chunks_fts table delete ordering",
        "exact-identifier",
        ("chunks_fts",),
    ),
    QuerySpec(
        "exact-08",
        "WorktreeCreate WorktreeRemove hooks",
        "exact-identifier",
        ("WorktreeCreate",),
    ),
    QuerySpec(
        "exact-09",
        "commit 6149eb9 usecli JSON output",
        "exact-identifier",
        ("6149eb9",),
    ),
    # ---- error-string: verbatim messages / literal output a user would paste ----
    QuerySpec(
        "error-01",
        "BufferError: cannot close exported pointers exist",
        "error-string",
        ("cannot close exported pointers exist",),
    ),
    QuerySpec(
        "error-02",
        "WHERE s.source_status = 'absent'",
        "error-string",
        ("WHERE s.source_status = 'absent'",),
    ),
    QuerySpec(
        "error-03",
        "prune mutated to WHERE 1=1",
        "error-string",
        ("WHERE 1=1",),
    ),
    QuerySpec(
        "error-04",
        "OSError: [Errno 45] Operation not supported",
        "error-string",
        ("[Errno 45] Operation not supported",),
        subagent_only_expected=True,
        notes="flock on a filesystem without lock support aborts the index commit",
    ),
    QuerySpec(
        "error-05",
        "ValueError: refusing to commit generation 1: not mutually consistent",
        "error-string",
        ("not mutually consistent",),
        subagent_only_expected=True,
        notes="generation-commit consistency guard refusal",
    ),
    QuerySpec(
        "error-06",
        "RuntimeError: Ran out of disk space while writing the index",
        "error-string",
        ("Ran out of disk space while writing the index",),
        subagent_only_expected=True,
    ),
    QuerySpec(
        "error-07",
        "OSError: [Errno 22] Invalid argument",
        "error-string",
        ("[Errno 22] Invalid argument",),
        subagent_only_expected=True,
        notes="directory fsync EINVAL during generation commit",
    ),
    QuerySpec(
        "error-08",
        "hand-rolled read 12,062 bytes on append",
        "error-string",
        ("12,062 bytes",),
    ),
    QuerySpec(
        "error-09",
        "CocoIndex read 67,264,584 bytes",
        "error-string",
        ("67,264,584",),
    ),
    # ---- paraphrase: natural-language asks, phrased differently than the anchor ----
    QuerySpec(
        "para-01",
        "why did we decide not to use CocoIndex's file reader for incremental updates",
        "paraphrase",
        ("67,264,584",),
        notes="anchor is the benchmark number; query never states it",
    ),
    QuerySpec(
        "para-02",
        "why did excluding subagent transcripts from indexing make ssgrep refuse "
        "to rebuild the index",
        "paraphrase",
        ("shrink guard", "--no-subagents"),
        subagent_only_expected=True,
    ),
    QuerySpec(
        "para-03",
        "why did a stale system binary invalidate a round of manual verification",
        "paraphrase",
        ("opt/homebrew/bin/ssgrep", "shadowed"),
        subagent_only_expected=True,
    ),
    QuerySpec(
        "para-04",
        "why did pruned session text keep showing up and distorting search results",
        "paraphrase",
        ("chunks_fts",),
    ),
    QuerySpec(
        "para-05",
        "why did the deletion fix only actually work on a freshly created index, "
        "and not on one that had already grown",
        "paraphrase",
        ("prune", "get_vector_path"),
    ),
    QuerySpec(
        "para-06",
        "what was structurally wrong with the SessionEnd hook we installed for Claude Code",
        "paraphrase",
        ("matcher group",),
        subagent_only_expected=True,
    ),
    QuerySpec(
        "para-07",
        "how many project working directories were undercounted by only reading the "
        "first record of a session file",
        "paraphrase",
        ("12 of 16",),
    ),
    QuerySpec(
        "para-08",
        "what did fixing session discovery turn up that path-based folder encoding "
        "had been missing",
        "paraphrase",
        ("213 files", "5,662 records"),
    ),
    QuerySpec(
        "para-09",
        "how many documents per second can the embedding model process",
        "paraphrase",
        ("132k docs/sec",),
    ),
    # ---- multi-hop: requires connecting two facts stated together in one episode ----
    QuerySpec(
        "multi-01",
        "which SQL change turned prune into a hard delete of a user's live archive, "
        "and why did the test suite stay green through it",
        "multi-hop",
        ("WHERE 1=1", "hard-delete"),
    ),
    QuerySpec(
        "multi-02",
        "which model tier is only allowed to gate decisions and never permitted to "
        "write code itself",
        "multi-hop",
        ("opus", "adjudicator"),
    ),
    QuerySpec(
        "multi-03",
        "why does a deletion fix that passes every test still leave a mature, "
        "already-grown index unsearchable",
        "multi-hop",
        ("prune", "generation", "db_path.parent"),
    ),
    QuerySpec(
        "multi-04",
        "how does a scope typed at the command line match a working directory "
        "recorded with different capitalization or a tilde",
        "multi-hop",
        ("case-fold", "expanduser"),
        subagent_only_expected=True,
    ),
    QuerySpec(
        "multi-05",
        "which locking and durability syscalls can abort an index commit on " "unusual filesystems",
        "multi-hop",
        ("flock", "fsync"),
        subagent_only_expected=True,
    ),
    QuerySpec(
        "multi-06",
        "connect the buffer export crash to which specific append in a process it fired on",
        "multi-hop",
        ("fresh empty", "BufferError"),
    ),
    QuerySpec(
        "multi-07",
        "which two production-shape defects did the phase-4 gate find that mutation "
        "testing inside the dev tree could not have caught",
        "multi-hop",
        ("matcher group",),
        subagent_only_expected=True,
    ),
    QuerySpec(
        "multi-08",
        "how does the guard against a shrinking rebuild treat sessions whose "
        "transcript files were deleted",
        "multi-hop",
        ("RebuildWouldShrinkError", "tombstone"),
    ),
    QuerySpec(
        "multi-09",
        "which CLI framework upgrade made structured JSON error output reliable "
        "for the setup command",
        "multi-hop",
        ("usecli 0.1.78", "init --json"),
    ),
    QuerySpec(
        "exact-10",
        "repair_current_session_tail bounded byte cap",
        "exact-identifier",
        ("repair_current_session_tail",),
    ),
    QuerySpec(
        "exact-11",
        "cosine_top_k brute-force vector leg",
        "exact-identifier",
        ("cosine_top_k",),
    ),
    QuerySpec(
        "exact-12",
        "textsafe ESC byte escaping",
        "exact-identifier",
        ("textsafe",),
    ),
    QuerySpec(
        "error-10",
        "sqlite3.OperationalError: fts5 syntax error near",
        "error-string",
        ("sqlite3.OperationalError",),
    ),
    QuerySpec(
        "error-11",
        "index.db reports no such table: chunks",
        "error-string",
        ("no such table",),
    ),
    QuerySpec(
        "error-12",
        "BufferError: memoryview export",
        "error-string",
        ("BufferError",),
    ),
    QuerySpec(
        "para-10",
        "how does an interrupted indexing run avoid replacing a good index with a "
        "half-written one",
        "paraphrase",
        ("manifest swap",),
    ),
    QuerySpec(
        "para-11",
        "which parts of a transcript count as prose worth indexing rather than " "machine noise",
        "paraphrase",
        ("signal.py",),
    ),
    QuerySpec(
        "para-12",
        "why did deleting rows with an always-true filter survive the test suite",
        "paraphrase",
        ("WHERE 1=1",),
    ),
    QuerySpec(
        "multi-10",
        "which record subtype forces an episode split during segmentation",
        "multi-hop",
        ("compact_boundary", "segment_episodes"),
    ),
    QuerySpec(
        "multi-11",
        "why does each retrieval leg fetch far more candidates than the ten " "results shown",
        "multi-hop",
        ("LEG_POOL_SIZE",),
    ),
    QuerySpec(
        "multi-12",
        "how are vector rows kept aligned with chunk rows after a crash between " "the two writes",
        "multi-hop",
        ("validate_alignment",),
    ),
]


def _collect_episodes() -> list[tuple]:
    """(SessionFile, Episode) pairs for every episode in this project's corpus."""
    sessions = discovery.discover_sessions(PROJECT_DIR)
    out = []
    for sf in sessions:
        recs = list(records.read_records(sf.path))
        eps = episodes.segment_episodes(recs, sf.session_id)
        for e in eps:
            out.append((sf, e))
    return out


def labelled_session_ids() -> set[str]:
    """The session ids the committed queries.jsonl was built against.

    Read from the artifact rather than recomputed, so this describes the
    corpus the labels actually came from and not whatever is on disk today.
    """
    if not OUTPUT_PATH.exists():
        return set()
    ids: set[str] = set()
    for line in OUTPUT_PATH.read_text().splitlines():
        if line.strip():
            ids.update(json.loads(line)["target_session_ids"])
    return ids


def corpus_shortfall() -> str | None:
    """Why build()'s invariants cannot be checked today, or None if they can.

    The precondition build() actually needs is not "a corpus exists" but
    "THIS corpus exists": every anchor is a literal string drawn from a
    specific transcript, so the invariants are only meaningful when the
    sessions those anchors live in are discoverable at PROJECT_DIR.

    Checking that the ``~/.claude/projects`` directory merely EXISTS was a
    proxy for that, and the proxy broke the moment the repo moved. Claude
    Code derives a project's transcript directory from its path, creates a
    NEW one on a move, and never migrates the old transcripts -- so
    discovery at the new path returns a large, healthy, and completely
    DIFFERENT corpus (measured: 2095 episodes, of which 38 of the 39
    labelled sessions were absent). The old guard saw the directory, let the
    test run, and every anchor failed to match. That is a corpus that is not
    there, reported as a methodology violation.

    This checks the real thing, and reports a shortfall only when the
    labelled sessions cannot be found. Anything else -- an anchor that
    stopped matching, one that grew past the proportional cap, a
    subagent-only anchor that leaked into a main session -- still fails
    build() as it should, because those are properties of the methodology
    rather than of the corpus's availability.
    """
    labelled = labelled_session_ids()
    if not labelled:
        return f"{OUTPUT_PATH} holds no labelled sessions to check against"
    discoverable = {sf.session_id for sf in discovery.discover_sessions(PROJECT_DIR)}
    missing = labelled - discoverable
    if missing:
        return (
            f"{len(missing)} of {len(labelled)} labelled sessions are not discoverable "
            f"under {PROJECT_DIR}; the corpus these anchors were drawn from is not "
            "present on this machine"
        )
    return None


def _matches(spec: QuerySpec, pairs: list[tuple]) -> list[tuple]:
    hits = []
    for sf, e in pairs:
        text = e.prompt_text + e.response_text
        if spec.anchor_mode == "all":
            ok = all(a in text for a in spec.anchors)
        else:
            ok = any(a in text for a in spec.anchors)
        if ok:
            hits.append((sf, e))
    return hits


def build() -> list[dict]:
    pairs = _collect_episodes()
    assert pairs, (
        f"No episodes found scoped to {PROJECT_DIR}. This script must run against "
        "the real ~/.claude/projects corpus for this project; it is not meant to "
        "run in CI or against a fixture."
    )

    corpus_size = len(pairs)
    anchor_match_cap = max(1, int(corpus_size * ANCHOR_MATCH_RATIO))

    rows = []
    for spec in QUERY_SPECS:
        hits = _matches(spec, pairs)
        assert hits, f"{spec.id}: anchor(s) {spec.anchors} matched zero episodes"
        assert len(hits) <= anchor_match_cap, (
            f"{spec.id}: anchor(s) {spec.anchors} matched {len(hits)} of {corpus_size} "
            f"episodes (cap {anchor_match_cap}, {ANCHOR_MATCH_RATIO:.1%} of corpus) — "
            f"this is a topic, not a specific target"
        )
        all_main = all(sf.is_main for sf, _e in hits)
        any_sub = any(not sf.is_main for sf, _e in hits)
        if spec.subagent_only_expected:
            assert not all_main, (
                f"{spec.id}: expected a subagent-only target but a main-session "
                f"episode also matched — anchor is not subagent-exclusive"
            )
        rows.append(
            {
                "id": spec.id,
                "query": spec.query,
                "class": spec.query_class,
                "anchors": list(spec.anchors),
                "anchor_mode": spec.anchor_mode,
                "target_episode_ids": sorted({e.episode_id for _sf, e in hits}),
                "target_session_ids": sorted({sf.session_id for sf, _e in hits}),
                "subagent_only": all_main is False
                and any_sub
                and all(not sf.is_main for sf, _e in hits),
                "match_count": len(hits),
                "notes": spec.notes,
            }
        )
    return rows


def main() -> None:
    rows = build()
    with open(OUTPUT_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    classes = {}
    subagent_only_count = 0
    for row in rows:
        classes[row["class"]] = classes.get(row["class"], 0) + 1
        if row["subagent_only"]:
            subagent_only_count += 1
    print(f"Wrote {len(rows)} queries to {OUTPUT_PATH}")
    print(f"By class: {classes}")
    print(f"Subagent-only targets: {subagent_only_count}")


if __name__ == "__main__":
    main()
