"""Unit tests for the single-pass episode segmenter."""

from __future__ import annotations

from ssgrep.sessions import episodes


def user(text=None, *, uid="u"):
    if text is None:
        content = [{"type": "tool_result", "content": "echo"}]
    else:
        content = [{"type": "text", "text": text}]
    return {"type": "user", "uuid": uid, "message": {"content": content}}


def assistant(text=None, *, uid="a"):
    content = [] if text is None else [{"type": "text", "text": text}]
    return {"type": "assistant", "uuid": uid, "message": {"content": content}}


def boundary():
    return {"type": "system", "subtype": "compact_boundary"}


def test_build_episode_id():
    assert episodes.build_episode_id("session", 12) == "session:ep:12"


def test_segment_groups_flushes_on_users_boundaries_and_end():
    records = [
        boundary(),  # empty boundary creates no episode and never joins a group
        {"type": "queue-operation", "uuid": "pre"},
        assistant("orphan response", uid="orphan"),
        boundary(),  # response-only episode
        user("first question", uid="q1"),
        assistant("answer one", uid="a1"),
        assistant(None, uid="empty-assistant"),
        user("second question", uid="q2"),  # flushes first prompt episode
        assistant("part one", uid="a2"),
        assistant("part two", uid="a3"),
    ]
    grouped = episodes.segment_episode_groups(records, "session")
    assert len(grouped) == 3

    first, first_records = grouped[0]
    assert first.episode_id == "session:ep:0"
    assert first.prompt_text == ""
    assert first.response_text == "orphan response"
    assert [r["uuid"] for r in first_records] == ["pre", "orphan"]

    second, second_records = grouped[1]
    assert second.prompt_text == "first question"
    assert second.response_text == "answer one"
    assert [r["uuid"] for r in second_records] == ["q1", "a1", "empty-assistant"]

    third, third_records = grouped[2]
    assert third.title == "Episode 2"
    assert third.response_text == "part one\npart two"
    assert [r["uuid"] for r in third_records] == ["q2", "a2", "a3"]


def test_textless_user_turns_fold_without_losing_records():
    records = [user(None, uid="tool-1"), user(None, uid="tool-2"), assistant("result")]
    grouped = episodes.segment_episode_groups(records, "s")
    assert len(grouped) == 1
    episode, raw = grouped[0]
    assert episode.prompt_text == ""
    assert episode.response_text == "result"
    assert raw == records


def test_prompt_only_episode_is_closed_by_boundary():
    records = [user("prompt"), boundary(), assistant("after")]
    grouped = episodes.segment_episode_groups(records, "s")
    assert [(ep.prompt_text, ep.response_text) for ep, _ in grouped] == [
        ("prompt", ""),
        ("", "after"),
    ]
    assert grouped[0][1] == [records[0]]
    assert grouped[1][1] == [records[2]]


def test_no_text_means_no_episode_but_records_fold_forward():
    records = [{"type": "queue-operation"}, user(None), boundary()]
    assert episodes.segment_episode_groups(records, "s") == []


def test_extract_text_supports_string_and_selected_list_blocks():
    assert episodes._extract_text({"message": {"content": "plain prompt"}}) == "plain prompt"
    assert (
        episodes._extract_text(
            {
                "message": {
                    "content": [
                        "not a block",
                        {"type": "thinking", "thinking": "hidden"},
                        {"type": "text", "text": "visible "},
                        {"type": "text", "text": "text"},
                    ]
                }
            }
        )
        == "visible text"
    )
    assert episodes._extract_text({}) == ""
    assert episodes._extract_text({"message": {"content": ("tuple",)}}) == ""


def test_build_episode_applies_empty_prompt_and_joins_responses():
    episode = episodes._build_episode("s", 4, None, ["one", "two"])
    assert episode.episode_id == "s:ep:4"
    assert episode.session_id == "s"
    assert episode.prompt_text == ""
    assert episode.response_text == "one\ntwo"
    assert episode.title == "Episode 4"
