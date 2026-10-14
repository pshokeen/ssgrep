"""Visible output, token privacy, and shell attribution at public interpretation boundaries."""

from __future__ import annotations

import json

import pytest

from ssgrep.accounting.contracts import Exposure, Outcome
from ssgrep.accounting.evidence import extract, validates
from ssgrep.accounting.evidence_tokens import MAX_SCAN_CHARS, Token, scan_text
from ssgrep.accounting.exposure import exposure_of, interpret, shell_shape
from ssgrep.accounting.frequency import milli_log2
from ssgrep.accounting.timeline import call_text, ts_ms


def test_extracted_identifiers_keep_structure_but_not_private_or_generic_tokens() -> None:
    visible = (
        "BUILD_CONFIG AB-123 alpha.beta.gamma module.some_value "
        "/123/456 docs/2026-10-12/log.txt /generic a/.. "
        "/home/alice_user/config.py alice_user --help BUILD-2026-10-12 "
        "a_b " + "long_" + "x" * 170
    )
    scan = scan_text(visible)
    assert scan.kept == {
        Token("identifier", "BUILD_CONFIG"),
        Token("identifier", "AB-123"),
        Token("identifier", "alpha.beta.gamma"),
        Token("identifier", "module.some_value"),
    }
    assert scan.excluded[Token("identifier", "alice_user")] == "username"
    assert scan.excluded[Token("identifier", "BUILD-2026-10-12")] == "date"
    assert scan.excluded[Token("flag", "--help")] == "generic"
    assert scan.excluded[Token("path", "/home/alice_user/config.py")] == "home_path"


def test_scan_bound_does_not_create_candidates_from_unseen_suffix() -> None:
    text = "visible_fact " + " " * MAX_SCAN_CHARS + " hidden_fact"
    assert scan_text(text).kept == {Token("identifier", "visible_fact")}
    assert scan_text("").kept == set()


def test_cli_success_envelope_extracts_show_files_not_hidden_envelope_fields() -> None:
    output = json.dumps(
        {
            "ok": True,
            "metadata": {"secret": "hidden_fact"},
            "data": {
                "episode_id": "claude:source:ep:0",
                "response_text": "useful_fact",
                "files_touched": ["src/repair.py", 42],
            },
        }
    )
    result = interpret({"output": output})
    assert result.outcome is Outcome.OK and result.explicit_success
    extracted = extract(result)
    assert {item.token for item in extracted.exposed} == {
        Token("identifier", "useful_fact"),
        Token("path", "src/repair.py"),
    }
    assert {item.refs for item in extracted.exposed} == {("claude:source:ep:0",)}


@pytest.mark.parametrize(
    "body, expected",
    [
        ("{broken json", Outcome.UNKNOWN),
        ('{"ok":false,"message":"incomplete index"}', Outcome.ERROR),
        ('{"error":"Episode not found: gone:ep:0"}', Outcome.EMPTY),
    ],
)
def test_result_semantics_override_transport_success(body: str, expected: Outcome) -> None:
    result = interpret({"output": body, "is_error": False})
    assert result.outcome is expected
    assert result.chars == len(body)
    assert exposure_of("pure", result) is Exposure.OBSERVED
    assert not result.explicit_success


def test_verification_requires_verifier_and_nonerror_result() -> None:
    success = interpret({"output": "8 passed", "exit_code": 0})
    failure = interpret({"output": "8 passed", "is_error": True})
    assert validates("pytest tests/", success)
    assert not validates("cat tests/report", success)
    assert not validates("pytest tests/", failure)


def test_shell_escaped_quotes_and_structural_foreign_work_preserve_attribution() -> None:
    assert shell_shape(r'ssgrep search "say \"hello\"" | head -n 2').kind == "filtered"
    shape = shell_shape("if true; then echo unrelated; else ssgrep search repair; fi")
    assert shape.kind == "mixed"
    assert exposure_of(shape.kind, interpret({"output": "visible_fact"})) is Exposure.PARTIAL


def test_call_text_bounds_nested_arguments_and_accepts_native_string_arguments() -> None:
    nested: object = "hidden_fact"
    for _ in range(10):
        nested = [nested]
    assert call_text({"arguments": {"visible": "known_fact", "nested": nested}}) == "known_fact"
    assert call_text({"arguments": "ssgrep search repair"}) == "ssgrep search repair"
    assert ts_ms("2026-10-12T10:00:00") is None
    assert ts_ms("2026-10-12T10:00:00Z") == ts_ms("2026-10-12T12:00:00+02:00")


@pytest.mark.parametrize("numerator, denominator", [(0, 0), (2, 3), (-1, 1)])
def test_invalid_frequency_ratios_cannot_fabricate_rarity(numerator: int, denominator: int) -> None:
    with pytest.raises(ValueError):
        milli_log2(numerator, denominator)
