"""Unit tests for pipeline row models and builders."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from cocoindex.connectors.lancedb import LanceType
from pydantic import BaseModel

from ssgrep.pipeline.rows import (
    PROXY_LANCE,
    SEARCH_PROJECT_TOKENS,
    TIMESTAMP_SPEC,
    ChunkRow,
    EpisodeRow,
    SessionRow,
    _contextual_search_text,
    _project_tail,
    _truncate_to_tokens,
    build_chunk_row,
    build_episode_row,
    build_session_row,
    chunk_payloads,
    directory_size,
    project_path,
)
from ssgrep.utilities.types import Chunk, ContentType


def test_rows_are_pydantic_models(sample_episode, sample_session) -> None:
    vector = np.zeros(256, dtype=np.float32)
    chunk = ChunkRow(
        chunk_id="session-1:ep:0:response:x",
        episode_id="session-1:ep:0",
        session_id="session-1",
        project="/work/app",
        source_path="/work/session.jsonl",
        vector=vector,
        text="hello",
        search_text="Title: Hi\nContent: hello",
        timestamp=datetime(2025, 1, 2, 3, 4, tzinfo=UTC),
    )
    assert chunk.timestamp == datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    assert isinstance(chunk, BaseModel)
    episode = EpisodeRow(episode_id="e", session_id="s", project="/p", title="t")
    assert episode.episode_id == "e" and episode.source_status == "available"
    session = SessionRow(session_id="s", path="/p/a.jsonl")
    assert session.absent_since is None and session.runtime == "claude"


def test_project_path_prefers_episode_cwd(sample_episode, sample_session) -> None:
    relative = "relative-project"
    episode = replace(sample_episode, cwd=relative)
    assert project_path(episode, sample_session) == str(Path(relative).absolute())


def test_project_path_uses_session_candidates_and_can_be_empty(sample_session) -> None:
    empty = replace(sample_session, project_paths=("",))
    assert project_path(None, empty) == ""
    assert project_path(None, sample_session) == str(
        Path(sample_session.project_paths[0]).absolute()
    )


def test_build_session_row_defaults() -> None:
    row = build_session_row(session_id="s-1", path="native:///x", runtime="pi")
    assert row.session_id == "s-1" and row.path == "native:///x" and row.runtime == "pi"
    assert row.source_status == "available"


def test_build_episode_row_denormalizes_metadata(sample_episode, sample_session) -> None:
    row = build_episode_row(sample_episode, sample_session)
    assert row.episode_id == sample_episode.episode_id
    assert row.project == sample_episode.project
    assert row.timestamp == sample_episode.timestamp
    assert row.files_touched == "\n".join(sample_episode.files_touched)
    assert row.tool_names == "\n".join(sample_episode.tool_names)
    assert row.is_subagent == sample_episode.is_subagent
    assert row.source_path == str(sample_session.path.absolute())
    assert row.runtime == "claude"
    assert row.title == "Testing"


def test_build_chunk_row_denormalizes_and_casts_vector(sample_episode, sample_session) -> None:
    part, search_text = chunk_payloads(sample_episode, sample_session)[0]
    vector = np.arange(256, dtype=np.float32)
    row = build_chunk_row(part, search_text, vector, sample_episode, sample_session)
    assert row.chunk_id == part.chunk_id
    assert row.episode_id == sample_episode.episode_id
    assert row.search_text == search_text
    assert row.text == part.text
    assert row.content_type == part.content_type.value
    assert row.source_project == sample_episode.source_project
    np.testing.assert_array_equal(row.vector, vector)


def test_build_chunk_row_fills_unit_norm_mean_proxy(sample_episode, sample_session) -> None:
    """proxy_vector = L2-normalized mean of the chunk's token vectors."""
    part, search_text = chunk_payloads(sample_episode, sample_session)[0]
    rng = np.random.default_rng(5)
    vector = rng.normal(size=(7, 96)).astype(np.float32)
    row = build_chunk_row(part, search_text, vector, sample_episode, sample_session)

    mean = vector.mean(axis=0)
    assert row.proxy_vector is not None
    np.testing.assert_allclose(row.proxy_vector, mean / np.linalg.norm(mean), rtol=1e-6)
    assert abs(np.linalg.norm(row.proxy_vector) - 1.0) < 1e-6


def test_proxy_lance_spec_mirrors_declared_v6_column() -> None:
    """The mirror's arrow type must equal ChunkModel's declared f32 column."""
    from ssgrep.store.schema import ChunkModel

    assert str(PROXY_LANCE.pa_type) == "fixed_size_list<item: float>[96]"
    declared = ChunkModel.to_arrow_schema().field("proxy_vector").type
    assert str(declared) == str(PROXY_LANCE.pa_type)
    vector = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert PROXY_LANCE.encoder is not None
    assert PROXY_LANCE.encoder(vector) == [1.0, 2.0, 3.0, 4.0]
    assert PROXY_LANCE.encoder(None) is None


def test_chunk_payloads_add_title_context(sample_episode, sample_session) -> None:
    from ssgrep.indexing.chunker import _tokenizer

    parts = chunk_payloads(sample_episode, sample_session)
    assert parts
    project_tail = _truncate_to_tokens(
        _tokenizer(), _project_tail(sample_episode.project), SEARCH_PROJECT_TOKENS
    )
    for part, search_text in parts:
        assert isinstance(part, object) and part.content_type in (
            ContentType.PROMPT,
            ContentType.RESPONSE,
        )
        assert search_text.startswith("Project: ")
        assert search_text.startswith(f"Project: {project_tail}\nTitle: ")
        assert search_text.endswith(part.text)


def test_chunk_payloads_search_text_fits_model_window(sample_episode, sample_session) -> None:
    """The embedded search_text never exceeds the model's 299-token window."""
    from ssgrep.indexing.chunker import _tokenizer
    from ssgrep.pipeline.rows import SEARCH_MAX_TOKENS

    tokenizer = _tokenizer()
    # The code hard-caps search_text at SEARCH_MAX_TOKENS (290), which sits
    # under the model's 299-token window; assert against that real cap rather
    # than the tokenizer's own model_max_length (which is not the window).
    limit = SEARCH_MAX_TOKENS
    parts = chunk_payloads(sample_episode, sample_session)
    assert parts
    for _part, search_text in parts:
        assert len(tokenizer.encode(search_text)) <= limit


def test_chunk_payloads_pass_episode_project(sample_episode, sample_session) -> None:
    """The episode's project flows into the embedded text as a two-part tail."""
    episode = replace(sample_episode, project="org/repo")
    parts = chunk_payloads(episode, sample_session)
    assert parts
    for _part, search_text in parts:
        assert search_text.startswith("Project: org/repo\nTitle: ")
    assert _project_tail(episode.project) == "org/repo"


def test_chunk_payloads_no_project_no_prefix_line(sample_episode, sample_session) -> None:
    """Without a project the search text keeps the old ``Title:``-only format."""
    episode = replace(sample_episode, project=None, cwd="")
    session = replace(sample_session, project_paths=())
    parts = chunk_payloads(episode, session)
    assert parts
    for _part, search_text in parts:
        assert search_text.startswith("Title: ")
        assert not search_text.startswith("Project: ")


class _FakeTokenizer:
    """One token per character, with a controllable model window."""

    def __init__(self, limit: int) -> None:
        self.model_max_length = limit

    def encode(self, text: str, add_special_tokens: bool = True) -> list[str]:
        return list(text)

    def decode(self, ids: list[str], skip_special_tokens: bool = True) -> str:
        return "".join(ids)


class _OscillatingTokenizer(_FakeTokenizer):
    """One token per char plus one extra token, so decoding a truncated prefix
    re-encodes to one over the budget every time.  The old re-clamping loop
    could not terminate on this; the window-search implementation must."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[str]:
        return list(text) + (["\x00"] if text else [])


def _fake_tokenizer(monkeypatch, limit: int) -> None:
    from ssgrep.indexing import chunker as chunker_mod

    monkeypatch.setattr(chunker_mod, "_tokenizer", lambda: _FakeTokenizer(limit))


def test_truncate_to_tokens_zero_or_negative_budget_returns_empty() -> None:
    """A zero/negative token budget yields an empty string, never an error."""
    tokenizer = _FakeTokenizer(limit=100)
    assert _truncate_to_tokens(tokenizer, "some text", 0) == ""
    assert _truncate_to_tokens(tokenizer, "some text", -5) == ""


def test_truncate_to_tokens_terminates_when_reencode_oscillates() -> None:
    """A truncated prefix that re-encodes over budget must not hang the loop."""
    tokenizer = _OscillatingTokenizer(limit=3)
    # The decoded prefix re-encodes to one token over the budget forever; the
    # fix steps the window down and returns the largest prefix that fits.
    assert _truncate_to_tokens(tokenizer, "abcdef", 3) == "ab"
    assert _truncate_to_tokens(tokenizer, "ab", 3) == "ab"


def test_truncate_to_tokens_returns_empty_when_even_one_token_overflows() -> None:
    """Budget 1 with a prefix that re-encodes to two tokens exhausts the loop."""
    tokenizer = _OscillatingTokenizer(limit=3)
    assert _truncate_to_tokens(tokenizer, "ab", 1) == ""


def test_contextual_search_text_empty_title_keeps_content(sample_episode) -> None:
    """A blank title yields ``Title: \nContent: <chunk>`` with no title text."""
    part = Chunk(chunk_id="c", text="hello", content_type=ContentType.PROMPT)
    blank = replace(sample_episode, title="   ")
    assert _contextual_search_text(part, blank, project="") == "Title: \nContent: hello"


def test_contextual_search_text_with_project_prefix_keeps_title_and_content(sample_episode) -> None:
    """A non-empty project yields a ``Project:`` prefix ahead of title/content."""
    part = Chunk(chunk_id="c", text="hello", content_type=ContentType.PROMPT)
    result = _contextual_search_text(part, sample_episode, project="org/repo")
    assert result.startswith("Project: org/repo\nTitle: Testing\nContent: hello")
    assert "Title: " in result
    assert result.endswith("hello")


def test_contextual_search_text_hard_caps_title_to_fixed_budget(sample_episode) -> None:
    """A title longer than SEARCH_TITLE_TOKENS is truncated to that budget."""
    from ssgrep.indexing.chunker import _tokenizer

    tokenizer = _tokenizer()
    content = "hello"
    part = Chunk(chunk_id="c", text=content, content_type=ContentType.PROMPT)
    ep = replace(sample_episode, title="T" * 500)
    result = _contextual_search_text(part, ep, project="")
    title_part = result.split("Title: ", 1)[1].split("\nContent: ")[0]
    assert len(tokenizer.encode(title_part, add_special_tokens=False)) <= 40
    assert result.endswith(content)


def test_contextual_search_text_never_exceeds_hard_cap(sample_episode, monkeypatch) -> None:
    """The combined search_text is deterministically capped at SEARCH_MAX_TOKENS."""
    from ssgrep.indexing.chunker import _tokenizer
    from ssgrep.pipeline.rows import SEARCH_MAX_TOKENS

    tokenizer = _tokenizer()
    # A huge chunk plus a huge title plus a deep project path: the result
    # must never exceed the cap, and the Project line keeps only the
    # last-two-components tail of the path.
    part = Chunk(chunk_id="c", text="w " * 3000, content_type=ContentType.PROMPT)
    ep = replace(sample_episode, title="T" * 500)
    project = "/a/very/long/absolutly/long/path/of/a/project/with/very/deep/nesting"
    result = _contextual_search_text(part, ep, project=project)
    assert len(tokenizer.encode(result, add_special_tokens=False)) <= SEARCH_MAX_TOKENS
    project_line = result.split("\n", 1)[0]
    assert project_line.startswith("Project: ")
    assert "deep/nesting" in project_line
    assert "/a/very" not in result and "absolutly" not in result
    # The project tail alone encodes within its reserved token budget.
    project_tail = project_line[len("Project: ") :]
    assert len(tokenizer.encode(project_tail, add_special_tokens=False)) <= SEARCH_PROJECT_TOKENS


def test_contextual_search_text_content_truncated_when_over_budget(sample_episode) -> None:
    """The chunk content is truncated to the remaining budget under the cap."""
    from ssgrep.indexing.chunker import _tokenizer

    tokenizer = _tokenizer()
    part = Chunk(chunk_id="c", text="w " * 3000, content_type=ContentType.PROMPT)
    ep = replace(sample_episode, title="")
    result = _contextual_search_text(part, ep)
    assert result.startswith("Title: \nContent: ")
    assert len(tokenizer.encode(result, add_special_tokens=False)) <= 290


def test_timestamp_spec_is_real_timestamp_with_epoch_encoder() -> None:
    assert isinstance(TIMESTAMP_SPEC, LanceType)
    assert str(TIMESTAMP_SPEC.pa_type) == "timestamp[us]"
    encoder = TIMESTAMP_SPEC.encoder
    assert encoder is not None
    aware = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert encoder(aware) == int(aware.timestamp() * 1_000_000)
    naive = datetime(2025, 1, 2, 3, 4, 5)
    assert encoder(naive) == int(naive.replace(tzinfo=UTC).timestamp() * 1_000_000)


def test_directory_size_sums_regular_files(tmp_path) -> None:
    (tmp_path / "a").write_bytes(b"12345")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b").write_bytes(b"678")
    assert directory_size(tmp_path) == 8
