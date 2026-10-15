"""Consent JSON describes persisted classes without opening collector state."""

from dataclasses import replace

from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.cli.metrics_json import lifecycle_document, retained_data


def test_default_disclosure_has_no_plaintext_consent_and_explicit_replay_limits():
    policy = ProfilePolicy()
    data = retained_data(policy)
    assert data["schema"] == 1
    assert data["evidence_format"] == "compact-v1"
    assert data["plaintext_cache"] == {
        "enabled": False,
        "retention_days": 0,
        "max_bytes": 0,
        "fields": ["prompts", "queries", "tool_arguments", "tool_outputs"],
    }
    replay = data["replay"]
    assert isinstance(replay, dict)
    assert replay["missing_or_changed_original"] == "unavailable_not_zero"
    assert replay["accepted_totals"] == "preserved"
    classes = data["classes"]
    assert isinstance(classes, dict)
    assert set(classes) >= {
        "journal",
        "compact_evidence",
        "rarity_metadata",
        "receipts",
        "accepted_totals",
        "queues",
        "temporary_files",
        "fingerprint_keys",
    }


def test_lifecycle_json_discloses_separate_cache_limits():
    policy = replace(
        ProfilePolicy(),
        collection_enabled=True,
        evidence_cache_enabled=True,
        evidence_cache_retention_days=2,
        evidence_cache_max_bytes=8192,
    )
    result = lifecycle_document("enabled", policy, policy.profile_id, "metrics collect", "Collect.")
    data = result["data"]
    assert isinstance(data, dict)
    disclosed_policy = data["policy"]
    assert isinstance(disclosed_policy, dict)
    assert disclosed_policy["evidence_cache_enabled"] is True
    retained = data["retained_data"]
    assert isinstance(retained, dict)
    cache = retained["plaintext_cache"]
    assert isinstance(cache, dict)
    assert cache["retention_days"] == 2
    assert cache["max_bytes"] == 8192
