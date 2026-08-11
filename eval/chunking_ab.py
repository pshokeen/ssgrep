"""Settles open question 2: does per-turn indexing beat uniform
chunking for short turns?

The reference design (CocoIndex's entire-session-search) keeps each turn as
one row, drops anything under 20 characters, and only size-chunks turns that
run long. Our D5 chunker instead flattens an episode's prompt and response
into two strings and slides a uniform 1200-char/200-overlap window over each,
regardless of the turn boundaries inside them.

METHODOLOGY

`episodes.Episode` only carries the flattened `prompt_text` / `response_text`
strings -- by the time chunker.py sees an episode, individual turn
boundaries are already gone (assistant turns are joined with a single
"\\n" by `episodes.segment_episodes`). To compare per-turn chunking against
production chunking on the SAME episodes the query set was labelled
against, this module re-walks the same raw records with a **mirror** of
`episodes.segment_episodes`'s own boundary logic (compaction boundary, user
record, EOF flush) that preserves each turn instead of flattening it.

Trusting a hand-mirrored copy of someone else's control flow is exactly the
kind of thing that silently drifts, so `_verify_alignment` asserts, for
every episode, that re-joining the mirrored turns reproduces byte-identical
`prompt_text` / `response_text` against the real `episodes.segment_episodes`
output before any index gets built. If that assertion ever fails, the
mirror has drifted from episodes.py and the experiment is invalid -- it is
checked, not assumed. Episode numbering also has to line up 1:1, since
queries.jsonl's target_episode_ids were labelled against the real
segmentation; `_verify_alignment` checks that too.

Per-turn chunking rule (matching the reference design): a turn under
MIN_TURN_CHARS is dropped entirely (it was noise, not signal, at 5.4% prose
density); a turn at or under CHUNK_TARGET_SIZE becomes exactly one chunk
row; a longer turn is size-chunked with the same sliding window chunker.py
uses, so the two strategies differ only in whether turn boundaries are
respected, not in the windowing algorithm itself.

Wired into indexing by monkeypatching `ssgrep.chunker.chunk_episode` for the
duration of one private-index build -- the same technique model_ab.py uses
for the embedding model -- and reverted in a `finally`. Never edits
chunker.py.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

from eval import harness, provenance
from ssgrep import chunker, discovery, episodes, records
from ssgrep.types import Chunk, ContentType

PROJECT_DIR = Path(__file__).resolve().parent.parent

MIN_TURN_CHARS = 20  # reference design's floor for dropping a turn outright
CHUNK_TARGET_SIZE = chunker.CHUNK_TARGET_SIZE
CHUNK_OVERLAP = chunker.CHUNK_OVERLAP

LABELS = {
    "uniform": "uniform (current production chunking, D5)",
    "per_turn": "per_turn (reference-design rule)",
}


def _extract_text(record: dict) -> str:
    message = record.get("message", {})
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    text = ""
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
    return text


def _segment_turns(recs: list[dict], session_id: str) -> dict[str, list[tuple[str, ContentType]]]:
    """Mirror of episodes.segment_episodes' boundary logic, preserving each
    turn (one user prompt, each individual assistant message) instead of
    flattening. Same flush conditions, same episode numbering formula
    (episodes.build_episode_id), so that turns_by_episode[i] lines up with
    the real segment_episodes(recs, session_id)[i]. Verified, not assumed --
    see _verify_alignment.
    """
    turns_by_episode: dict[str, list[tuple[str, ContentType]]] = {}
    current_prompt_text: str | None = None
    current_turns: list[tuple[str, ContentType]] = []
    episode_index = 0

    def flush() -> None:
        nonlocal current_turns, episode_index, current_prompt_text
        if current_prompt_text or current_turns:
            episode_id = episodes.build_episode_id(session_id, episode_index)
            turns_by_episode[episode_id] = list(current_turns)
            episode_index += 1
        current_turns = []
        current_prompt_text = None

    for record in recs:
        rec_type = record.get("type", "")
        subtype = record.get("subtype", "")

        if rec_type == "system" and subtype == "compact_boundary":
            flush()
            continue

        if rec_type == "user":
            flush()
            text = _extract_text(record)
            current_prompt_text = text
            if text:
                current_turns = [(text, ContentType.PROMPT)]
            else:
                current_turns = []
            continue

        if rec_type == "assistant":
            text = _extract_text(record)
            if text:
                current_turns.append((text, ContentType.RESPONSE))

    flush()
    return turns_by_episode


def _verify_alignment(project_dir: Path = PROJECT_DIR) -> int:
    """Assert the mirrored per-turn segmentation reproduces the real
    episodes.segment_episodes output byte-for-byte, episode-for-episode,
    across every session in scope. Returns the episode count checked.
    Raises AssertionError (not a soft warning) on any drift -- a silent
    mismatch here would invalidate every number this module produces.
    """
    sessions = discovery.discover_sessions(project_dir)
    checked = 0
    for sf in sessions:
        recs = list(records.read_records(sf.path))
        real_episodes = episodes.segment_episodes(recs, sf.session_id)
        turns_by_episode = _segment_turns(recs, sf.session_id)
        assert len(turns_by_episode) == len(real_episodes), (
            f"{sf.path}: mirrored segmentation produced {len(turns_by_episode)} "
            f"episodes, real segment_episodes produced {len(real_episodes)}"
        )
        for ep in real_episodes:
            turns = turns_by_episode.get(ep.episode_id)
            assert turns is not None, f"{ep.episode_id}: missing from mirrored segmentation"
            prompt_turns = [t for t, ct in turns if ct == ContentType.PROMPT]
            response_turns = [t for t, ct in turns if ct == ContentType.RESPONSE]
            rebuilt_prompt = prompt_turns[0] if prompt_turns else ""
            rebuilt_response = "\n".join(response_turns)
            assert rebuilt_prompt == ep.prompt_text, (
                f"{ep.episode_id}: prompt_text mismatch\n"
                f"  real:     {ep.prompt_text!r}\n"
                f"  mirrored: {rebuilt_prompt!r}"
            )
            assert rebuilt_response == ep.response_text, (
                f"{ep.episode_id}: response_text mismatch\n"
                f"  real:     {ep.response_text!r}\n"
                f"  mirrored: {rebuilt_response!r}"
            )
            checked += 1
    return checked


def _turns_index(project_dir: Path = PROJECT_DIR) -> dict[str, list[tuple[str, ContentType]]]:
    sessions = discovery.discover_sessions(project_dir)
    out: dict[str, list[tuple[str, ContentType]]] = {}
    for sf in sessions:
        recs = list(records.read_records(sf.path))
        out.update(_segment_turns(recs, sf.session_id))
    return out


def _windowed_chunks(
    text: str,
    episode_id: str,
    session_id: str,
    content_type: ContentType,
    start_index: int,
) -> list[Chunk]:
    """Copy of chunker.chunk_text's sliding-window algorithm, parameterized
    with a starting chunk index so multiple long turns in the same episode
    and content type don't collide on chunk_id. Same target size / overlap
    constants as production; only the index offset differs.
    """
    chunks = []
    start = 0
    index = start_index
    while start < len(text):
        end = min(start + CHUNK_TARGET_SIZE, len(text))
        if end < len(text):
            last_newline = text.rfind("\n\n", start, end)
            if last_newline > start + CHUNK_TARGET_SIZE // 2:
                end = last_newline + 2
        piece = text[start:end].strip()
        if piece:
            chunk_id = f"{episode_id}:{content_type.value}:turn:{index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    episode_id=episode_id,
                    session_id=session_id,
                    text=piece,
                    content_type=content_type,
                )
            )
            index += 1
        start = end - CHUNK_OVERLAP if end < len(text) else end
    return chunks


def make_per_turn_chunk_episode(turns_by_episode: dict[str, list[tuple[str, ContentType]]]):
    """Build a chunker.chunk_episode-compatible function closed over a
    precomputed turns index, for monkeypatching into ssgrep.chunker during
    one private-index build. Falls back to the real chunker.chunk_episode
    (captured before patching) for any episode not found in the index --
    should not happen for episodes built from the same corpus scope, but a
    silent gap is worse than an explicit fallback.
    """
    original_chunk_episode = chunker.chunk_episode

    def _per_turn_chunk_episode(episode) -> list[Chunk]:  # noqa: ANN001
        turns = turns_by_episode.get(episode.episode_id)
        if turns is None:
            return original_chunk_episode(episode)

        out: list[Chunk] = []
        counters: dict[ContentType, int] = {ContentType.PROMPT: 0, ContentType.RESPONSE: 0}
        for text, content_type in turns:
            stripped = text.strip()
            if len(stripped) < MIN_TURN_CHARS:
                continue
            if len(stripped) <= CHUNK_TARGET_SIZE:
                idx = counters[content_type]
                chunk_id = f"{episode.episode_id}:{content_type.value}:turn:{idx}"
                out.append(
                    Chunk(
                        chunk_id=chunk_id,
                        episode_id=episode.episode_id,
                        session_id=episode.session_id,
                        text=stripped,
                        content_type=content_type,
                    )
                )
                counters[content_type] = idx + 1
            else:
                windowed = _windowed_chunks(
                    stripped,
                    episode.episode_id,
                    episode.session_id,
                    content_type,
                    counters[content_type],
                )
                out.extend(windowed)
                counters[content_type] += len(windowed)
        return out

    return _per_turn_chunk_episode


def run_chunking_ab(
    *, main_session_boost: float, base_dir: Path | None = None, task: str = ""
) -> dict:
    base_dir = base_dir or Path(tempfile.mkdtemp(prefix="ssgrep-eval-chunking-ab-"))
    run_date = date.today().isoformat()

    checked = _verify_alignment()
    turns_by_episode = _turns_index()

    original_chunk_episode = chunker.chunk_episode
    out: dict = {"episodes_verified": checked}
    try:
        # --- uniform (production) chunking ---
        index_dir = base_dir / "uniform"
        results, summary, stats = harness.run(
            index_dir, main_session_boost=main_session_boost, rebuild=True
        )
        db_path, _vec_path = harness._index_paths_for(index_dir)
        out["uniform"] = {
            "episode_count": stats.episode_count,
            "chunk_count": stats.chunk_count,
            "summary": summary,
            "provenance": provenance.build_provenance(
                date=run_date,
                task=task,
                label=LABELS["uniform"],
                stats=stats,
                db_path=db_path,
            ),
        }

        # --- per-turn chunking ---
        chunker.chunk_episode = make_per_turn_chunk_episode(turns_by_episode)
        index_dir = base_dir / "per_turn"
        results, summary, stats = harness.run(
            index_dir, main_session_boost=main_session_boost, rebuild=True
        )
        db_path, _vec_path = harness._index_paths_for(index_dir)
        out["per_turn"] = {
            "episode_count": stats.episode_count,
            "chunk_count": stats.chunk_count,
            "summary": summary,
            "provenance": provenance.build_provenance(
                date=run_date,
                task=task,
                label=LABELS["per_turn"],
                stats=stats,
                db_path=db_path,
            ),
        }
    finally:
        chunker.chunk_episode = original_chunk_episode

    for label in ("uniform", "per_turn"):
        print(f"=== {label} (chunk_count={out[label]['chunk_count']}) ===")
        for key, agg in out[label]["summary"].items():
            if agg["n"] == 0:
                continue
            print(
                f"  {key:20s} n={agg['n']:3d}  "
                f"recall@10={agg['recall_at_10']:.3f}  mrr={agg['mrr']:.3f}"
            )

    return out


def main() -> None:
    import argparse

    from ssgrep import search as search_module

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-session-boost", type=float, default=search_module.MAIN_SESSION_BOOST
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--task",
        type=str,
        default="Re-run chunking A/B (uniform vs per-turn) after the chunk-dedup fix, "
        "at the shipped default boost (p6-abrerun)",
    )
    args = parser.parse_args()

    result = run_chunking_ab(main_session_boost=args.main_session_boost, task=args.task)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
