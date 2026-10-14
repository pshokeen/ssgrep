"""Policy validation, persistence, and pause behavior (real files under tmp_path)."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.accounting import policy
from ssgrep.accounting.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.policy_rules import PolicyError


def listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


@pytest.fixture
def data(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def work(tmp_path: Path) -> Path:
    for name in ("team", "team-other", "outside"):
        (tmp_path / "work" / name).mkdir(parents=True)
    return tmp_path / "work"


def managed(profile: str, *roots: Path, **extra: object) -> dict[str, object]:
    return {"profile_id": profile, "roots": [str(r) for r in roots], **extra}


def test_defaults_are_off_and_personal_local(data: Path) -> None:
    loaded = policy.load_profile_policy()
    assert loaded.profile_id == "personal-local"
    assert (loaded.collection_enabled, loaded.paused, loaded.enabled_exports) == (False, False, ())
    assert not data.exists(), "a read must not create storage"


def test_prefix_sibling_is_not_inside_root(work: Path) -> None:
    approved = policy.preview_enable("acme", managed("acme", work / "team"))
    assert policy.path_in_scope(approved, str(work / "team" / "src"))
    assert not policy.path_in_scope(approved, str(work / "team-other"))
    assert not policy.path_in_scope(approved, str(work / "team-other" / "x"))


def test_symlink_escape_is_outside_scope_and_symlinked_root_resolves(work: Path) -> None:
    (work / "team" / "link").symlink_to(work / "outside")
    approved = policy.preview_enable("acme", managed("acme", work / "team"))
    assert not policy.path_in_scope(approved, str(work / "team" / "link"))
    alias = work / "alias"
    alias.symlink_to(work / "team")
    via_alias = policy.preview_enable("acme", managed("acme", alias))
    assert via_alias.roots == (str(os.path.realpath(work / "team")),)


def test_organization_needs_roots_and_personal_default_covers_everything(work: Path) -> None:
    with pytest.raises(PolicyError) as caught:
        policy.preview_enable("acme", managed("acme"))
    assert caught.value.condition == "invalid_policy"
    personal = policy.load_profile_policy()
    assert policy.path_in_scope(personal, "/anywhere/at/all")
    assert not policy.path_in_scope(ProfilePolicy(profile_id="acme"), str(work))


@pytest.mark.parametrize(
    "bad", ["../etc", "a/b", "", "UPPER", "-lead", "x" * 33, "a b", "..", "a\0b", "~root"]
)
def test_traversal_and_malformed_profile_ids_rejected_before_any_write(
    data: Path, bad: str
) -> None:
    for call in (
        lambda: policy.enable_collection(bad),
        lambda: policy.pause(bad),
        lambda: policy.resume(bad),
        lambda: policy.policy_status(bad),
    ):
        with pytest.raises(PolicyError) as caught:
            call()
        assert caught.value.condition == "invalid_profile"
    assert not data.exists()


def test_overlapping_org_roots_rejected_unless_assigned(data: Path, work: Path) -> None:
    policy.enable_collection("acme", managed("acme", work))
    before = (data / "accounting" / "policy.json").read_bytes()
    with pytest.raises(PolicyError) as caught:
        policy.enable_collection("globex", managed("globex", work / "team"))
    assert caught.value.condition == "overlapping_roots"
    assert (data / "accounting" / "policy.json").read_bytes() == before
    assert policy.load_profile_policy("globex").collection_enabled is False
    resolved = managed("globex", work / "team", assignments={str(work / "team"): "globex"})
    assert policy.enable_collection("globex", resolved).collection_enabled
    assert policy.load_profile_policy("globex").roots == (os.path.realpath(work / "team"),)


def test_overlap_through_symlink_is_detected(data: Path, work: Path) -> None:
    (work / "alias").symlink_to(work / "team")
    policy.enable_collection("acme", managed("acme", work / "team"))
    with pytest.raises(PolicyError) as caught:
        policy.enable_collection("globex", managed("globex", work / "alias"))
    assert caught.value.condition == "overlapping_roots"


def test_disjoint_prefix_siblings_do_not_overlap(work: Path) -> None:
    policy.enable_collection("acme", managed("acme", work / "team"))
    assert policy.enable_collection("globex", managed("globex", work / "team-other")).roots


def test_ninth_profile_is_rejected_and_nothing_is_written(data: Path, work: Path) -> None:
    policy.save_profile(ProfilePolicy())
    for index in range(1, 8):
        root = work / f"ws{index}"
        root.mkdir()
        policy.save_profile(ProfilePolicy(profile_id=f"p{index}", roots=(str(root),)))
    before = (data / "accounting" / "policy.json").read_bytes()
    with pytest.raises(PolicyError) as caught:
        policy.pause("p9")
    assert caught.value.condition == "too_many_profiles"
    with pytest.raises(PolicyError):
        policy.save_profile(ProfilePolicy(profile_id="p9", roots=(str(work),)))
    assert (data / "accounting" / "policy.json").read_bytes() == before


def test_retention_widening_rejected_narrowing_allowed(data: Path, work: Path) -> None:
    with pytest.raises(PolicyError) as caught:
        policy.save_profile(ProfilePolicy(evidence_retention_days=31))
    assert caught.value.condition == "retention_widened"
    assert not data.exists()
    assert (
        policy.save_profile(ProfilePolicy(evidence_retention_days=7)).evidence_retention_days == 7
    )
    ceiling = managed("acme", work / "team", evidence_retention_days=10, receipt_retention_days=100)
    policy.enable_collection("acme", ceiling)
    with pytest.raises(PolicyError) as caught:
        policy.save_profile(replace(policy.load_profile_policy("acme"), receipt_retention_days=101))
    assert caught.value.condition == "retention_widened"
    stored = policy.load_profile_policy("acme")
    assert (stored.evidence_retention_days, stored.receipt_retention_days) == (10, 100)


def test_local_roots_cannot_exceed_managed_roots(work: Path) -> None:
    policy.enable_collection("acme", managed("acme", work / "team"))
    widened = replace(policy.load_profile_policy("acme"), roots=(str(work / "team-other"),))
    with pytest.raises(PolicyError) as caught:
        policy.save_profile(widened)
    assert caught.value.condition == "scope_expansion"


def test_destination_cannot_exceed_managed_allowlist(work: Path) -> None:
    allowed = {
        "destination_id": "otel",
        "endpoint": "https://collector.example.test/v1",
        "signals": ["metrics"],
    }
    policy.enable_collection("acme", managed("acme", work / "team", destinations=[allowed]))
    current = policy.load_profile_policy("acme")
    rogue = DestinationPolicy("evil", "https://elsewhere.example.test", (Signal.METRICS,))
    with pytest.raises(PolicyError) as caught:
        policy.save_profile(replace(current, destinations=(*current.destinations, rogue)))
    assert caught.value.condition == "scope_expansion"
    logs = replace(
        current.destinations[0], signals=(Signal.METRICS, Signal.LOGS), logs_permitted=True
    )
    with pytest.raises(PolicyError):
        policy.save_profile(replace(current, destinations=(logs,)))


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://collector.example.test",
        "https://user:pw@host.test",
        "ftp://host",
        "https://h/p?token=1",
    ],
)
def test_unsafe_endpoints_rejected(endpoint: str) -> None:
    dest = DestinationPolicy("otel", endpoint)
    with pytest.raises(PolicyError):
        policy.save_profile(ProfilePolicy(destinations=(dest,)))


def test_loopback_http_endpoint_allowed_and_credentials_by_reference_only() -> None:
    dest = DestinationPolicy("dev", "http://127.0.0.1:4318", credential_ref="env:OTLP_TOKEN")
    saved = policy.save_profile(ProfilePolicy(destinations=(dest,)))
    assert saved.destinations[0].credential_ref == "env:OTLP_TOKEN"
    bad = DestinationPolicy("dev", "http://127.0.0.1:4318", credential_ref="hunter2")
    with pytest.raises(PolicyError):
        policy.save_profile(ProfilePolicy(destinations=(bad,)))


def test_invalid_managed_update_keeps_last_valid_and_pauses_sending(data: Path, work: Path) -> None:
    policy.enable_collection("acme", managed("acme", work / "team"))
    good = policy.load_profile_policy("acme")
    ok, error = policy.apply_managed_update("acme", {"profile_id": "acme", "roots": "nonsense"})
    assert (ok, error is not None) == (False, True)
    after = policy.load_profile_policy("acme")
    assert after.roots == good.roots and after.collection_enabled
    with pytest.raises(PolicyError) as caught:
        policy.require_not_paused("acme", "export")
    assert (caught.value.condition, caught.value.exit_code) == ("export_paused", 1)
    policy.require_not_paused("acme", "collection")
    status = policy.policy_status("acme")
    assert status.sending_paused_reason == "invalid_policy"
    ok, error = policy.apply_managed_update(
        "acme", managed("acme", work / "team", policy_version=2)
    )
    assert ok and error is None
    assert policy.policy_status("acme").sending_paused_reason is None
    assert policy.load_profile_policy("acme").policy_version == 2


def test_managed_update_narrowing_clamps_existing_state(work: Path) -> None:
    dest = {"destination_id": "otel", "endpoint": "https://c.example.test", "signals": ["metrics"]}
    policy.enable_collection("acme", managed("acme", work / "team", destinations=[dest]))
    current = policy.load_profile_policy("acme")
    policy.save_profile(replace(current, enabled_exports=(("otel", "metrics"),)))
    ok, _ = policy.apply_managed_update("acme", managed("acme", work / "team", policy_version=3))
    assert ok
    assert policy.load_profile_policy("acme").enabled_exports == ()


def test_corrupt_or_foreign_policy_never_enables_capture(data: Path) -> None:
    (data / "accounting").mkdir(parents=True)
    (data / "accounting").chmod(0o700)
    target = data / "accounting" / "policy.json"
    target.write_text("{not json")
    target.chmod(0o600)
    assert policy.load_profile_policy().collection_enabled is False
    with pytest.raises(PolicyError) as caught:
        policy.read_policy_file()
    assert caught.value.condition == "policy_corrupt"
    with pytest.raises(PolicyError) as caught:
        policy.require_not_paused("personal-local", "export")
    assert caught.value.condition == "export_paused"
    target.write_text(json.dumps({"schema": 99}))
    assert policy.policy_status().error is not None
    target.write_text("{}")
    target.chmod(0o666)
    assert policy.policy_status().error is not None


def test_symlinked_accounting_directory_is_refused(data: Path, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    data.mkdir()
    (data / "accounting").symlink_to(elsewhere)
    with pytest.raises(PolicyError):
        policy.enable_collection("personal-local")
    assert list(elsewhere.iterdir()) == []


def test_written_policy_is_owner_only_and_atomic(data: Path) -> None:
    policy.enable_collection("personal-local")
    target = data / "accounting" / "policy.json"
    assert target.stat().st_mode & 0o777 == 0o600
    assert (data / "accounting").stat().st_mode & 0o777 == 0o700
    assert [p.name for p in (data / "accounting").iterdir() if p.name != ".lock"] == ["policy.json"]
    assert (
        json.loads(target.read_text())["profiles"]["personal-local"]["collection_enabled"] is True
    )


def test_pause_resume_cycle_records_gap_and_never_reenables_exports(data: Path) -> None:
    dest = DestinationPolicy("dev", "http://127.0.0.1:4318")
    base = policy.save_profile(ProfilePolicy(destinations=(dest,), collection_enabled=True))
    policy.save_profile(replace(base, enabled_exports=(("dev", "metrics"),)))
    t1 = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
    paused = policy.pause("personal-local", t1)
    assert (paused.paused, paused.paused_at) == (True, "2026-10-12T09:00:00Z")
    again = policy.pause("personal-local", datetime(2026, 10, 12, 9, 5, tzinfo=UTC))
    assert again.paused_at == "2026-10-12T09:00:00Z"
    for kind, condition in (("collection", "collection_paused"), ("export", "export_paused")):
        with pytest.raises(PolicyError) as caught:
            policy.require_not_paused("personal-local", kind)  # type: ignore[arg-type]
        assert (caught.value.condition, caught.value.exit_code) == (condition, 1)
        assert "resume" in caught.value.command
    # disabling a signal while paused, then resuming, must not bring it back
    paused_policy = policy.load_profile_policy()
    policy.save_profile(replace(paused_policy, enabled_exports=()))
    resumed = policy.resume("personal-local", datetime(2026, 10, 12, 9, 30, tzinfo=UTC))
    assert resumed.paused is False and resumed.enabled_exports == ()
    policy.require_not_paused("personal-local", "export")
    status = policy.policy_status()
    assert status.gaps == (("2026-10-12T09:00:00Z", "2026-10-12T09:30:00Z"),)


def test_noop_calls_do_not_write(data: Path) -> None:
    policy.resume("personal-local")
    assert not data.exists()
    policy.enable_collection("personal-local")
    stamp = (data / "accounting" / "policy.json").stat().st_mtime_ns
    policy.resume("personal-local")
    assert (data / "accounting" / "policy.json").stat().st_mtime_ns == stamp


def test_enable_keeps_export_disabled(work: Path) -> None:
    dest = {"destination_id": "otel", "endpoint": "https://c.example.test", "signals": ["metrics"]}
    enabled = policy.enable_collection("acme", managed("acme", work / "team", destinations=[dest]))
    assert enabled.collection_enabled and enabled.enabled_exports == ()
    assert enabled.destinations[0].destination_id == "otel"
