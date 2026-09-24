"""Unit tests for :mod:`ssgrep.indexing.chunker`."""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from types import SimpleNamespace

from ssgrep.indexing import chunker
from ssgrep.utilities.types import ContentType


def test_chunk_budget_fits_model_window() -> None:
    """Chunk plus prefix overlap stays within the model's 299-token window."""
    # The embedding model's window is 299 model tokens (pylate wraps the
    # transformer at max_seq_length=299, not the tokenizer's 512). OverlapRefinery
    # prefix-merges up to CHUNK_TOKEN_OVERLAP tokens onto each chunk; the full
    # chunk (as the model encodes it, with special tokens) must stay within the
    # window. CHUNK_TOKEN_BUDGET + OVERLAP (260) sits just under 299.
    assert chunker.CHUNK_TOKEN_OVERLAP == 25
    tokenizer = chunker._tokenizer()
    budget = chunker.CHUNK_TOKEN_BUDGET + chunker.CHUNK_TOKEN_OVERLAP
    assert budget < 299
    window = 299
    text = " ".join(f"word {i} " + "sentence with several tokens here. " for i in range(200))
    chunks = chunker.chunk_text(text, "s", ContentType.PROMPT)
    assert chunks
    for chunk in chunks:
        assert len(tokenizer.encode(chunk.text)) <= window


def test_chunker_import_does_not_import_torch() -> None:
    """Importing the chunker must not pull in torch or pylate."""
    code = (
        "import sys; "
        "import ssgrep.indexing.chunker; "
        "assert 'torch' not in sys.modules; "
        "assert 'pylate' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
    )


def test_chunk_text_tiny_input_yields_one_chunk() -> None:
    chunks = chunker.chunk_text("hello", "s", ContentType.PROMPT)
    assert len(chunks) >= 1
    assert chunks[0].text == "hello"


def test_chunk_id_is_stable_and_namespaced() -> None:
    digest = hashlib.sha256(b"7:hello").hexdigest()[:16]
    assert chunker._chunk_id("session", ContentType.PROMPT, 7, "hello") == (
        f"session:prompt:{digest}"
    )


def test_chunk_text_rejects_empty_input() -> None:
    assert chunker.chunk_text("", "s", ContentType.PROMPT) == []
    assert chunker.chunk_text("  \n", "s", ContentType.RESPONSE) == []


def test_chunk_text_strips_and_skips_pieces(monkeypatch) -> None:
    raw_pieces = object()
    fake_chunker = SimpleNamespace(chunk=lambda text: raw_pieces)
    pieces = [
        SimpleNamespace(text="  first chunk  ", start_index=3),
        SimpleNamespace(text=" \t", start_index=9),
        SimpleNamespace(text="second", start_index=20),
    ]
    overlap_calls = []

    def overlap(value):
        overlap_calls.append(value)
        return pieces

    monkeypatch.setattr(chunker, "_chunker", fake_chunker)
    monkeypatch.setattr(chunker, "_overlap", overlap)

    result = chunker.chunk_text("source", "sid", ContentType.RESPONSE)

    assert overlap_calls == [raw_pieces]
    assert [part.text for part in result] == ["first chunk", "second"]
    assert all(part.content_type is ContentType.RESPONSE for part in result)
    assert result[0].chunk_id == chunker._chunk_id("sid", ContentType.RESPONSE, 3, "first chunk")


def test_chunk_episode_keeps_prompt_and_response_separate(sample_episode, monkeypatch) -> None:
    calls = []

    def fake_chunk_text(text, episode_id, content_type):
        calls.append((text, episode_id, content_type))
        return [content_type.value]

    monkeypatch.setattr(chunker, "chunk_text", fake_chunk_text)

    assert chunker.chunk_episode(sample_episode) == ["prompt", "response"]
    assert calls == [
        (sample_episode.prompt_text, sample_episode.episode_id, ContentType.PROMPT),
        (sample_episode.response_text, sample_episode.episode_id, ContentType.RESPONSE),
    ]


def test_overlap_env_unset_uses_default(monkeypatch) -> None:
    monkeypatch.delenv("SSGREP_CHUNK_OVERLAP", raising=False)
    assert chunker._resolved_overlap() == chunker.CHUNK_TOKEN_OVERLAP


def test_overlap_env_unparsable_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv("SSGREP_CHUNK_OVERLAP", "not-a-number")
    assert chunker._resolved_overlap() == chunker.CHUNK_TOKEN_OVERLAP


def test_overlap_env_clamped_to_half_budget_with_warning(monkeypatch, caplog) -> None:
    monkeypatch.setenv("SSGREP_CHUNK_OVERLAP", "200")
    with caplog.at_level(logging.WARNING, logger="ssgrep.indexing.chunker"):
        resolved = chunker._resolved_overlap()
    assert resolved == chunker.CHUNK_TOKEN_BUDGET // 2
    assert resolved == 117
    assert "117" in caplog.text


def test_overlap_env_boundary_half_budget_accepted(monkeypatch, caplog) -> None:
    monkeypatch.setenv("SSGREP_CHUNK_OVERLAP", str(chunker.CHUNK_TOKEN_BUDGET // 2))
    with caplog.at_level(logging.WARNING, logger="ssgrep.indexing.chunker"):
        resolved = chunker._resolved_overlap()
    assert resolved == 117
    assert caplog.text == ""


def test_overlap_env_zero_allowed(monkeypatch, caplog) -> None:
    monkeypatch.setenv("SSGREP_CHUNK_OVERLAP", "0")
    with caplog.at_level(logging.WARNING, logger="ssgrep.indexing.chunker"):
        resolved = chunker._resolved_overlap()
    assert resolved == 0
    assert caplog.text == ""


def test_overlap_env_negative_clamped_to_zero(monkeypatch) -> None:
    monkeypatch.setenv("SSGREP_CHUNK_OVERLAP", "-5")
    assert chunker._resolved_overlap() == 0


def test_get_overlap_passes_resolved_context_size(monkeypatch) -> None:
    monkeypatch.setenv("SSGREP_CHUNK_OVERLAP", "25")
    monkeypatch.setattr(chunker, "_overlap", None)
    monkeypatch.setattr(chunker, "_tokenizer", lambda: object())
    recorded: dict[str, object] = {}

    class FakeRefinery:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr(chunker, "OverlapRefinery", FakeRefinery)
    chunker._get_overlap()
    assert recorded["context_size"] == 25
    monkeypatch.setattr(chunker, "_overlap", None)


def test_chunk_id_scheme_unchanged_by_overlap_env(monkeypatch) -> None:
    """The id derivation (sha256 of start:text) must not depend on the overlap knob."""
    monkeypatch.setenv("SSGREP_CHUNK_OVERLAP", "12")
    digest = hashlib.sha256(b"7:hello").hexdigest()[:16]
    assert chunker._chunk_id("session", ContentType.PROMPT, 7, "hello") == (
        f"session:prompt:{digest}"
    )
