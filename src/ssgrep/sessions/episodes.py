"""Episode segmentation from classified records."""

from __future__ import annotations

from ssgrep.utilities.types import Episode


def build_episode_id(session_id: str, index: int) -> str:
    return f"{session_id}:ep:{index}"


def segment_episode_groups(
    records: list[dict], session_id: str
) -> list[tuple[Episode, list[dict]]]:
    """Segment records into episodes, paired with the exact records that
    produced each one.

    This is the one and only boundary-detection pass. It answers both
    "what are the episodes" and "which raw records belong to episode i" from
    the same loop, at the same decision points, so the two questions cannot
    disagree the way two independently maintained implementations of
    "where episodes start and end" could.

    That is not a hypothetical risk: indexer.py used to re-derive groups
    with a second, structurally different copy of this logic (opening a new
    group at every `user` record, unconditionally) solely to recover each
    episode's raw records for metadata/signal-text extraction. That copy
    disagreed with this function's episode count whenever a `user` record's
    extracted text was empty -- e.g. a turn that is entirely a tool_result
    echo, with no human-authored text block. This function's flush
    condition (current_prompt or current_responses, both text-based) treats
    "nothing extracted yet" as "this turn didn't happen" and silently folds
    it into whichever episode follows; the old second copy split on the
    `user` record's mere presence regardless of content, so it produced one
    more group there than this function produced episodes. Every such fold
    shifted the two counts apart by one, and the indexer treated any count
    mismatch as license to fall back to chunking every episode in the batch
    from the *entire* batch of new_records -- silently multiplying the
    index by the episode count (measured 733x on this repo's own main
    session). Folding a textless turn forward is harmless by itself (an
    episode with no prompt and no response text chunks to nothing either
    way -- see chunker.chunk_text's guard); recomputing that same fold
    decision twice, in two places free to reach different answers, was not.
    There is now exactly one place it is decided, and the two questions are
    answered together so they cannot drift apart again.

    The returned groups omit only the boundary markers themselves
    (`system`/`compact_boundary` records): every other record -- including a
    `user` record whose extracted text is empty -- stays in whichever group
    is open when it arrives, even one that a fold later merges forward. No
    record is ever discarded; a fold changes which episode a run of records
    ultimately lands in, never whether it lands in one.
    """
    episode_list: list[Episode] = []
    groups: list[list[dict]] = []
    current_prompt: str | None = None
    current_responses: list[str] = []
    current_records: list[dict] = []
    episode_index = 0

    def flush() -> None:
        nonlocal episode_index, current_prompt, current_responses, current_records
        episode_list.append(
            _build_episode(session_id, episode_index, current_prompt, current_responses)
        )
        groups.append(current_records)
        episode_index += 1
        current_prompt = None
        current_responses = []
        current_records = []

    for record in records:
        rec_type = record.get("type", "")
        subtype = record.get("subtype", "")

        if rec_type == "system" and subtype == "compact_boundary":
            if current_prompt or current_responses:
                flush()
            continue  # the boundary marker itself never joins a group

        if rec_type == "user":
            if current_prompt or current_responses:
                flush()
            current_prompt = _extract_text(record)
            current_responses = []
            current_records.append(record)
            continue

        current_records.append(record)
        if rec_type == "assistant":
            text = _extract_text(record)
            if text:
                current_responses.append(text)

    if current_prompt or current_responses:
        flush()

    return list(zip(episode_list, groups, strict=True))


def _extract_text(record: dict) -> str:
    """Light inline text extraction for one record's message content.

    Content can be a plain string (user prompts) or a list of blocks, of
    which only "text"-typed blocks contribute. This is intentionally the
    crude extraction episodes.py has always done -- indexer.py separately
    re-derives cleaner text per episode via signal.classify_signal() (markup
    stripping, away_summary surfacing, tool-result/thinking exclusion); see
    indexer.py's module docstring.
    """
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


def _build_episode(
    session_id: str, index: int, prompt: str | None, responses: list[str]
) -> Episode:
    return Episode(
        episode_id=build_episode_id(session_id, index),
        session_id=session_id,
        prompt_text=prompt or "",
        response_text="\n".join(responses),
        title=f"Episode {index}",
    )
