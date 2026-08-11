"""Tests for episode segmentation."""

import json
from pathlib import Path

from ssgrep.episodes import segment_episode_groups, segment_episodes

FIXTURES = Path(__file__).parent / "fixtures"


def load_records(filename):
    records = []
    with open(FIXTURES / filename) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def test_basic_segmentation():
    records = load_records("main-session.jsonl")
    episodes = segment_episodes(records, "test-session")
    assert len(episodes) > 0


def test_compaction_not_new_session():
    """Verify that compact_boundary records close episodes but don't create new sessions."""
    records = load_records("compaction-session.jsonl")
    episodes = segment_episodes(records, "compaction-test")

    # All episodes should belong to the same session
    session_ids = {ep.session_id for ep in episodes}
    assert len(session_ids) == 1, "All episodes must belong to same session"

    user_records = [r for r in records if r.get("type") == "user"]

    # Each user record starts an episode, and each compact_boundary can close an episode.
    # The fixture has 4 user records and 3 compact boundaries, so we should have
    # multiple episodes created. If compact_boundary was not handled, we'd only have
    # as many episodes as user records.
    assert len(episodes) > len(user_records), (
        f"compact_boundary should create episode boundaries; "
        f"expected > {len(user_records)} episodes, got {len(episodes)}"
    )


def test_consecutive_users():
    records = [
        {"type": "user", "message": {"content": [{"type": "text", "text": "Q1"}]}},
        {"type": "user", "message": {"content": [{"type": "text", "text": "Q2"}]}},
    ]
    episodes = segment_episodes(records, "test")
    assert len(episodes) == 2


def test_assistant_at_start():
    records = [
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Hello"}]}},
        {"type": "user", "message": {"content": [{"type": "text", "text": "Question"}]}},
    ]
    episodes = segment_episodes(records, "test")
    assert len(episodes) >= 1


# ---------------------------------------------------------------------------
# segment_episode_groups: the single segmentation pass behind both
# segment_episodes() and indexer.py's per-episode record grouping. See
# eliminated indexer._split_into_episode_groups / _build_episodes for what
# a second, disagreeing implementation of this used to cost.
# ---------------------------------------------------------------------------


def _tool_result_user(uid: str) -> dict:
    """A `user`-type record carrying only a tool_result echo: no "text"
    block, so its extracted text is empty. This is the real-world shape
    that triggers the empty-content merge: assistant tool calls are echoed
    back as `user` records with no human-authored prompt text.
    """
    return {
        "type": "user",
        "uuid": uid,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": uid, "content": "ok"}],
        },
    }


def _text_user(uid: str, text: str) -> dict:
    return {"type": "user", "uuid": uid, "message": {"content": [{"type": "text", "text": text}]}}


def _text_assistant(uid: str, text: str) -> dict:
    return {
        "type": "assistant",
        "uuid": uid,
        "message": {"content": [{"type": "text", "text": text}]},
    }


def _compact_boundary() -> dict:
    """Minimal system/compact_boundary marker record. segment_episode_groups
    only inspects `type`/`subtype` for this record, mirroring the other
    minimal builders above.
    """
    return {"type": "system", "subtype": "compact_boundary"}


def test_segment_episode_groups_matches_segment_episodes_exactly():
    """segment_episodes() must be exactly the episode half of
    segment_episode_groups() -- byte for byte, not just same length. Every
    other caller (episodes.py's own tests below, eval/'s mirrored
    verification) relies on segment_episodes()'s output being unchanged by
    the existence of the grouped variant.
    """
    records = load_records("compaction-session.jsonl")
    grouped = segment_episode_groups(records, "s")
    episodes_only = segment_episodes(records, "s")
    assert [ep for ep, _group in grouped] == episodes_only


def test_groups_always_align_one_to_one_with_episodes():
    """len(groups) == len(episodes) must hold by construction for any
    input -- there is no code path that can produce one without the other.
    """
    for fixture in ["main-session.jsonl", "compaction-session.jsonl"]:
        records = load_records(fixture)
        grouped = segment_episode_groups(records, "s")
        episodes_only = segment_episodes(records, "s")
        assert len(grouped) == len(episodes_only), fixture


def test_empty_content_turn_folds_without_losing_its_records():
    """The empty-content merge quirk, reproduced directly: two consecutive
    tool-result-only `user` turns (empty extracted text) followed by a real
    assistant response. The old indexer.py-side grouping function split a
    new group at every `user` record regardless of content and produced 2
    groups here; segment_episodes's flush condition folds a textless turn
    into whatever follows and produces only 1 episode. That one-episode/
    two-group disagreement is exactly what used to trip the whole-batch
    fallback. Assert here that: exactly one episode results, and its group
    holds all three raw records -- the fold changes which episode the two
    tool-result turns land in, but never discards them.
    """
    records = [
        _tool_result_user("u0"),
        _tool_result_user("u1"),
        _text_assistant("a0", "the real response text"),
    ]
    grouped = segment_episode_groups(records, "s")

    assert len(grouped) == 1
    ep, group = grouped[0]
    assert ep.prompt_text == ""
    assert ep.response_text == "the real response text"
    assert group == records, "all three raw records must survive the fold, none discarded"


def test_group_records_never_bleed_into_a_different_episode():
    """Each episode's group must contain only records belonging to that
    episode -- never another episode's. Builds three distinguishable
    episodes (including one preceded by the empty-content merge quirk) and
    checks each group's uuids against the others'.
    """
    records = [
        _text_user("q0", "first question MARKER_A"),
        _text_assistant("r0", "first answer MARKER_A"),
        _tool_result_user("tr1"),
        _tool_result_user("tr2"),
        _text_assistant("r1", "second answer MARKER_B, after two empty turns"),
        _text_user("q2", "third question MARKER_C"),
        _text_assistant("r2", "third answer MARKER_C"),
    ]
    grouped = segment_episode_groups(records, "s")
    assert len(grouped) == 3

    all_uuids = [r["uuid"] for r in records]
    seen: set[str] = set()
    for _ep, group in grouped:
        group_uuids = {r["uuid"] for r in group}
        assert not (group_uuids & seen), "a record uuid appeared in more than one group"
        seen |= group_uuids
    assert seen == set(all_uuids), "every record must land in exactly one group"

    # Content-level check mirroring the ticket's ask: no episode's group
    # carries another episode's distinguishing marker.
    markers = {0: "MARKER_A", 1: "MARKER_B", 2: "MARKER_C"}
    for i, (_ep, group) in enumerate(grouped):
        texts = json.dumps(group)
        for j, marker in markers.items():
            if i == j:
                continue
            assert marker not in texts, f"episode {i}'s group contains episode {j}'s marker"


# ---------------------------------------------------------------------------
# compact_boundary flush condition: `if current_prompt or current_responses:
# flush()`. Its structural twin one line below (the `user`-record flush
# condition) is covered by test_compaction_not_new_session; this site was
# not covered by anything. Both tests below construct a "half-open" episode
# -- one that has accumulated only a prompt with no response, or only a
# response with no prompt, per segment_episode_groups's own docstring -- and
# check that the boundary closes it immediately, rather than leaving it open
# to silently merge with whatever accumulates next.
# ---------------------------------------------------------------------------


def test_compact_boundary_flushes_prompt_only_episode():
    """A prompt with no response yet must close AT the compact_boundary, not
    stay open and merge with the assistant text that arrives after it.

    If the boundary's flush condition were weakened from `or` to `and`
    (current_prompt truthy, current_responses still empty -> False), the
    open prompt survives the boundary untouched. It then gets swept into
    the next flush trigger regardless -- the very next `user` record's own
    (unweakened) `or` check still fires -- but by then the post-boundary
    assistant reply has already accumulated into the SAME still-open
    episode, merging content that should have been split at the boundary.
    """
    records = [
        _text_user("q0", "PROMPT_HALF_OPEN_MARKER"),
        _compact_boundary(),
        _text_assistant("a0", "RESPONSE_AFTER_BOUNDARY_MARKER"),
        _text_user("q1", "second question"),
        _text_assistant("a1", "second response"),
    ]
    grouped = segment_episode_groups(records, "s")

    assert len(grouped) == 3, (
        "the boundary must close the prompt-only episode by itself, before "
        "the post-boundary response ever has a chance to accumulate into "
        "it -- a failure to close here collapses this to 2 episodes"
    )
    ep0, group0 = grouped[0]
    ep1, _group1 = grouped[1]
    ep2, _group2 = grouped[2]

    assert ep0.prompt_text == "PROMPT_HALF_OPEN_MARKER"
    assert ep0.response_text == "", "episode 0 must close with no response yet"
    assert group0 == [records[0]], "episode 0's group must hold only the prompt record"

    assert ep1.prompt_text == ""
    assert ep1.response_text == "RESPONSE_AFTER_BOUNDARY_MARKER"
    assert "RESPONSE_AFTER_BOUNDARY_MARKER" not in ep0.response_text, (
        "the post-boundary response must land in its own episode, not merge "
        "backward into the prompt-only episode the boundary should have closed"
    )

    assert ep2.prompt_text == "second question"
    assert ep2.response_text == "second response"


def test_compact_boundary_flushes_response_only_episode():
    """Mirror of the above for the other half-open shape the function's own
    docstring names: response text with no prompt (e.g. a stray assistant
    record before any `user` turn). The boundary must close it before a
    further assistant record -- arriving after the boundary, before the
    next `user` record -- accumulates into the SAME still-open response list.
    """
    records = [
        _text_assistant("a0", "PRE_BOUNDARY_MARKER"),
        _compact_boundary(),
        _text_assistant("a1", "POST_BOUNDARY_MARKER"),
        _text_user("q1", "next question"),
    ]
    grouped = segment_episode_groups(records, "s")

    assert len(grouped) == 3, (
        "the boundary must close the response-only episode before the next "
        "assistant record's text joins the same open response list -- a "
        "failure to close here collapses this to 2 episodes"
    )
    ep0, group0 = grouped[0]
    ep1, group1 = grouped[1]
    ep2, _group2 = grouped[2]

    assert ep0.prompt_text == ""
    assert ep0.response_text == "PRE_BOUNDARY_MARKER"
    assert group0 == [records[0]], "episode 0's group must hold only the pre-boundary response"
    assert "POST_BOUNDARY_MARKER" not in ep0.response_text, (
        "the post-boundary response must not merge backward into the "
        "episode the boundary should have already closed"
    )

    assert ep1.prompt_text == ""
    assert ep1.response_text == "POST_BOUNDARY_MARKER"
    assert group1 == [records[2]]

    assert ep2.prompt_text == "next question"
    assert ep2.response_text == ""
