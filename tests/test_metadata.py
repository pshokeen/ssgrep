"""Tests for metadata harvesting."""

import json
from pathlib import Path

from ssgrep.metadata import derive_files_touched, harvest_metadata, load_agent_meta

FIXTURES = Path(__file__).parent / "fixtures"


def test_title_precedence_custom_wins():
    records = [
        {"type": "last-prompt", "last-prompt": "Last Prompt"},
        {"type": "ai-title", "ai-title": "AI Title"},
        {"type": "custom-title", "custom-title": "Custom Title"},
    ]
    meta = harvest_metadata(records, "test")
    assert meta.title == "Custom Title"


def test_title_precedence_ai_wins():
    records = [
        {"type": "last-prompt", "last-prompt": "Last Prompt"},
        {"type": "ai-title", "ai-title": "AI Title"},
    ]
    meta = harvest_metadata(records, "test")
    assert meta.title == "AI Title"


def test_files_from_read_edit_write():
    """Name promises Read+Edit+Write coverage: body must actually construct
    all three, each with a distinct file_path, not just Read (+ a Bash call
    to prove exclusion). Previously this test's body gave zero coverage to
    Edit and Write despite its name -- dropping either from
    derive_files_touched's filter tuple passed the whole suite clean.
    """
    records = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "src/main.py"}},
                    {"type": "tool_use", "name": "Edit", "input": {"file_path": "src/other.py"}},
                    {"type": "tool_use", "name": "Write", "input": {"file_path": "src/new.py"}},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                ]
            },
        }
    ]
    files = derive_files_touched(records)
    assert "src/main.py" in files
    assert "src/other.py" in files
    assert "src/new.py" in files
    assert len(files) == 3


def test_ismeta_records_excluded_from_derive_files_touched():
    """isMeta records must not contribute to files_touched.

    Real isMeta:true records observed in the live corpus are always
    type=user with no tool_use content (see fixtures/manifest.md's
    ismeta-record.jsonl note) -- a tool_use block never actually appears on
    one today. But derive_files_touched's isMeta guard is unconditional on
    record shape (it doesn't check `type` at all), so this proves the guard
    itself holds rather than relying on today's producer never emitting the
    combination -- the same defensive-contract reasoning as the block-type
    filter in episodes._extract_text.
    """
    records = [
        {
            "type": "assistant",
            "isMeta": True,
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {"file_path": "/should/not/appear/poison.py"},
                    },
                ]
            },
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Write", "input": {"file_path": "src/real.py"}},
                ]
            },
        },
    ]
    files = derive_files_touched(records)
    assert files == ("src/real.py",)


def test_ismeta_records_excluded_from_harvest_metadata():
    """isMeta records must contribute nothing to harvest_metadata's title,
    git_branch, or cwd -- all guarded by one `continue` at the top of its
    per-record loop, so a fixture with isMeta lines carrying deliberately
    wrong values for all three, processed ahead of one real record,
    proves the guard holds for all three fields at once (not just that the
    overall result happens to be falsy).

    Uses ismeta-record.jsonl, the one fixture in the repo with isMeta: true
    (every other fixture is isMeta: false throughout -- see manifest.md).
    """
    records = []
    with open(FIXTURES / "ismeta-record.jsonl") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    real_record = {
        "type": "user",
        "message": {"content": "Real prompt text from a non-meta record"},
        "gitBranch": "real-branch",
        "cwd": "/real/project/cwd",
    }
    meta = harvest_metadata(records + [real_record], "test")

    assert meta.title == "Real prompt text from a non-meta record"
    assert meta.git_branch == "real-branch"
    assert meta.cwd == "/real/project/cwd"
    assert "ISMETA_POISON" not in meta.title
    assert meta.git_branch != "should-not-surface-ismeta-branch"
    assert meta.cwd != "/should/not/surface/ismeta-cwd"


def test_load_agent_meta():
    meta = load_agent_meta(FIXTURES / "agent-aexample-agent-deadbeef01.meta.json")
    assert meta is not None
    assert meta.agent_type is not None


def test_missing_agent_meta():
    meta = load_agent_meta(FIXTURES / "nonexistent.meta.json")
    assert meta is None


def test_agent_meta_description_cleaned():
    """Agent meta description goes through the same cleaning pipeline as titles."""
    import tempfile
    from pathlib import Path

    # Create a temporary .meta.json with markup and newlines in description
    # Using a description that matches the real corpus issue: incomplete wrapper tag
    meta_data = {
        "agentType": "test-agent",
        "name": "Test Agent",
        "model": "test-model",
        "description": (
            r"<system-reminder>" + "\n"
            "Background indexing is still running.\n"
            "Retry once it settles"
        ),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        meta_path = Path(tmpdir) / "test-agent.meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta_data, f)

        meta = load_agent_meta(meta_path)
        assert meta is not None
        # Description should have markup and newlines stripped/collapsed
        assert "<" not in meta.description
        assert "\n" not in meta.description
        assert meta.description == "Background indexing is still running. Retry once it settles"


def test_markup_stripped_from_slash_command_prompt():
    """A title derived from a slash-command prompt contains no markup tags."""
    markup_content = (
        "<command-name>/model</command-name>\n"
        "            <command-message>model</command-message>"
    )
    records = [
        {
            "type": "user",
            "message": {
                "content": markup_content,
            },
        }
    ]
    meta = harvest_metadata(records, "test")
    assert "<command-name>" not in meta.title
    assert "<command-message>" not in meta.title
    assert meta.title.strip()  # Non-empty
    assert "\n" not in meta.title  # Single line


def test_markup_collapsed_whitespace():
    """Embedded newlines and whitespace runs are collapsed."""
    records = [
        {
            "type": "ai-title",
            "ai-title": "Fix   bug\n\nwith   spaces",
        }
    ]
    meta = harvest_metadata(records, "test")
    assert meta.title == "Fix bug with spaces"
    assert "\n" not in meta.title


def test_empty_title_after_stripping_falls_through():
    """A title that is only markup becomes empty and falls through to next source."""
    records = [
        {
            "type": "ai-title",
            "ai-title": "<command-name>/model</command-name>",
        },
        {
            "type": "last-prompt",
            "last-prompt": "Actual prompt text",
        },
    ]
    meta = harvest_metadata(records, "test")
    # ai-title is only markup, so it becomes empty and falls through to last-prompt
    assert meta.title == "Actual prompt text"


def test_markup_stripped_local_command_stdout():
    """<local-command-stdout> wrapper tags are stripped and empty titles fall through."""
    records = [
        {
            "type": "ai-title",
            "ai-title": "<local-command-stdout>Set model to Opus</local-command-stdout>",
        },
        {
            "type": "last-prompt",
            "last-prompt": "Real prompt",
        },
    ]
    meta = harvest_metadata(records, "test")
    # ai-title is only markup, so falls through to last-prompt
    assert meta.title == "Real prompt"
    assert "<local-command-stdout>" not in meta.title


def test_markup_stripped_system_reminder():
    """<system-reminder> wrapper tags are stripped."""
    records = [
        {
            "type": "custom-title",
            "custom-title": "<system-reminder>reminder</system-reminder>",
        },
        {
            "type": "ai-title",
            "ai-title": "AI generated title",
        },
    ]
    meta = harvest_metadata(records, "test")
    # custom-title is only markup, so falls through to ai-title
    assert meta.title == "AI generated title"
    assert "<system-reminder>" not in meta.title


def test_unknown_wrapper_tags_handled():
    """Unknown future wrapper tags are automatically handled by generic stripping."""
    records = [
        {
            "type": "ai-title",
            "ai-title": "<future-unknown-tag>content</future-unknown-tag>",
        },
        {
            "type": "last-prompt",
            "last-prompt": "Fallback text",
        },
    ]
    meta = harvest_metadata(records, "test")
    # Unknown tag is stripped, empty title falls through
    assert meta.title == "Fallback text"
    assert "<future-unknown-tag>" not in meta.title


def test_ansi_escapes_removed():
    """ANSI escape sequences are removed from titles."""
    records = [
        {
            "type": "ai-title",
            "ai-title": "Set model to \x1b[1mOpus 5\x1b[22m context",
        }
    ]
    meta = harvest_metadata(records, "test")
    # ANSI escapes are removed
    assert "\x1b" not in meta.title
    assert "[1m" not in meta.title
    assert "[22m" not in meta.title
    assert meta.title == "Set model to Opus 5 context"


def test_multiline_title_collapsed_to_single_line():
    """Titles with embedded newlines are collapsed to single line."""
    records = [
        {
            "type": "ai-title",
            "ai-title": "First line\nSecond line\nThird line",
        }
    ]
    meta = harvest_metadata(records, "test")
    # No newlines in final title
    assert "\n" not in meta.title
    assert "\r" not in meta.title
    assert meta.title == "First line Second line Third line"


def test_episode_index_fallback_ordinal():
    """When no title can be derived, uses episode index as ordinal fallback."""
    # Empty records - no title sources available
    records = []
    meta = harvest_metadata(records, "test", episode_index=5)
    # Falls back to episode ordinal
    assert meta.title == "Episode 5"


def test_episode_index_precedence_still_respected():
    """Episode index is only used as fallback, not when title exists."""
    records = [
        {
            "type": "user",
            "message": {
                "content": "Real prompt text",
            },
        }
    ]
    meta = harvest_metadata(records, "test", episode_index=5)
    # Should use the real prompt, not the ordinal
    assert meta.title == "Real prompt text"
    assert "Episode 5" not in meta.title


def test_all_fixtures_get_titles():
    """All fixtures must get non-empty, single-line titles without markup."""
    for fixture_file in FIXTURES.glob("*.jsonl"):
        records = []
        with open(fixture_file) as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        if records:
            meta = harvest_metadata(records, fixture_file.stem)
            assert meta.title, f"No title for {fixture_file.name}"
            assert meta.title.strip(), f"Empty title for {fixture_file.name}"
            assert "\n" not in meta.title, f"Newline in title for {fixture_file.name}"
            assert "<" not in meta.title, f"Markup tag in title for {fixture_file.name}"
            assert "\x1b" not in meta.title, f"ANSI escape in title for {fixture_file.name}"


def test_normalize_title_adversarial_repeated_tags_is_fast():
    """A pathological repeated-tag title must not cost seconds of backtracking.

    normalize_title caps its input at 2000 chars before the tag regexes run;
    without the cap, '<a>' * 40000 measured ~8s. The bound here is generous
    (0.5s) to stay robust on loaded CI machines while still catching a
    reintroduced quadratic blowup by orders of magnitude.
    """
    import time

    from ssgrep.metadata import normalize_title

    adversarial = "<a>" * 40000
    start = time.perf_counter()
    result = normalize_title(adversarial)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"normalize_title took {elapsed:.2f}s on adversarial input"
    assert isinstance(result, str)
