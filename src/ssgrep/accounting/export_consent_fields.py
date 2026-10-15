"""Explicit per-target field-version approval, separate from the policy permission.

Only confirmed enable calls write this owner-only metadata. Missing, old or unsafe
approval cannot silently authorize newly added fields; preview/check never writes.
"""

import os

from ssgrep.accounting import export_files, storage
from ssgrep.accounting.export_model import FIELDS_VERSION
from ssgrep.accounting.ledger_contracts import Signal, valid_profile_id
from ssgrep.accounting.plain_codec import loads

_MAX_BYTES = 4096


def _name(signal: Signal) -> str:
    return f"consent-{signal.value}.json"


def _expected(profile: str, destination: str, signal: Signal) -> dict:
    if not valid_profile_id(profile) or not valid_profile_id(destination):
        raise ValueError("invalid_consent_target")
    return {
        "schema": 1,
        "profile": profile,
        "destination": destination,
        "signal": signal.value,
        "fields_version": FIELDS_VERSION[signal.value],
    }


def _matches(document: object, expected: dict) -> bool:
    return (
        isinstance(document, dict) and type(document.get("schema")) is int and document == expected
    )


def approval_reason(profile: str, destination: str, signal: Signal) -> str | None:
    expected = _expected(profile, destination, signal)
    try:
        directory = storage.open_profile_dir(profile, "export", destination, create=False)
    except FileNotFoundError:
        return "consent_reapproval_required"
    except OSError:
        return "consent_state_unavailable"
    try:
        raw = storage.read_file(directory, _name(signal), _MAX_BYTES)
        document = loads(raw) if raw is not None else None
    except (OSError, ValueError, TypeError):
        return "consent_state_unavailable"
    finally:
        os.close(directory)
    return None if _matches(document, expected) else "consent_reapproval_required"


def approve(profile: str, destination: str, signal: Signal) -> bool:
    """Called ONLY after explicit current-field preview confirmation; idempotent."""
    document = _expected(profile, destination, signal)
    with export_files.locked(
        profile, ("export", destination), f"consent-{signal.value}"
    ) as directory:
        if directory is None:
            raise ValueError("consent_state_unavailable")
        raw = storage.read_file(directory, _name(signal), _MAX_BYTES)
        previous = loads(raw) if raw is not None else None
        if _matches(previous, document):
            return False
        export_files.write_json(directory, _name(signal), document)
    return True
