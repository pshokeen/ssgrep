"""Tests for the public CLI exit-code contract."""

from ssgrep.cli import exit_codes


def test_exit_code_contract() -> None:
    assert exit_codes.INTERNAL_FAILURE == 1
    assert exit_codes.USAGE_ERROR == 2
    assert exit_codes.NO_MATCHING_DATA == 3
    assert exit_codes.MISSING_INDEX == 4
    assert (
        len(
            {
                exit_codes.INTERNAL_FAILURE,
                exit_codes.USAGE_ERROR,
                exit_codes.NO_MATCHING_DATA,
                exit_codes.MISSING_INDEX,
            }
        )
        == 4
    )
