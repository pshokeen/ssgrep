"""Policy validation, persistence, and pause behavior (real files under tmp_path)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ssgrep.accounting.ledger.ledger_contracts import (
    MAX_DESTINATIONS,
    DestinationPolicy,
    ProfilePolicy,
    Signal,
)
from ssgrep.accounting.privacy import policy
from ssgrep.accounting.privacy.policy import (
    MAX_ROOTS,
    PolicyError,
    PolicyFile,
    check_endpoint,
    check_proxy,
    historical_workspace_owner,
    path_in_roots,
    validate_profile,
)
from tests.accounting.export.http_builders import tree_state

DEST = DestinationPolicy("otel", "https://collector.example.test", (Signal.METRICS,))


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


def test_oversized_policy_file_is_policy_corrupt_without_echoing_the_path(data: Path) -> None:
    (data / "accounting").mkdir(parents=True)
    (data / "accounting").chmod(0o700)
    target = data / "accounting" / "policy.json"
    target.write_bytes(b"{" + b" " * (policy.MAX_POLICY_BYTES + 1) + b"}")
    target.chmod(0o600)
    with pytest.raises(PolicyError) as caught:
        policy.read_policy_file()
    assert caught.value.condition == "policy_corrupt"
    assert caught.value.exit_code == 1 and "sending is paused" in str(caught.value)
    assert str(data) not in str(caught.value)  # the read error text carries the private root
    assert isinstance(caught.value.__cause__, ValueError)
    assert policy.load_profile_policy().collection_enabled is False


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


def test_nonobject_profile_revokes_collection_authority_without_repair(tmp_path: Path):
    policy.enable_collection("personal-local")
    path = policy.policy_path()
    original = path.read_bytes()
    raw = json.loads(original)
    raw["profiles"]["personal-local"] = []
    path.write_text(json.dumps(raw))
    before = tree_state(tmp_path / "data")
    status = policy.policy_status()
    assert status.error is not None and status.error.condition == "policy_corrupt"
    assert not status.profile.collection_enabled
    assert tree_state(tmp_path / "data") == before
    path.write_bytes(original)
    assert policy.load_profile_policy().collection_enabled


@pytest.mark.parametrize(
    "raw",
    [
        [],
        {"extra": True},
        {"profile_id": "other"},
        {"evidence_retention_days": 3651},
        {"assignments": []},
    ],
)
def test_invalid_managed_input_cannot_replace_existing_policy(tmp_path: Path, raw):
    policy.enable_collection("personal-local")
    before = tree_state(tmp_path / "data")
    with pytest.raises(PolicyError) as caught:
        policy.enable_collection("personal-local", raw)
    assert caught.value.condition == "invalid_policy"
    assert policy.load_profile_policy().collection_enabled
    assert tree_state(tmp_path / "data") == before


@pytest.mark.parametrize("url", [None, "", "x" * 2049])
def test_endpoint_requires_bounded_url(url):
    with pytest.raises(PolicyError) as caught:
        check_endpoint(url)
    assert caught.value.condition == "invalid_policy"
    assert check_endpoint("https://collector.example.test") == "https://collector.example.test"


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "x" * 2049,
        "ftp://proxy.test",
        "https:///",
        "http://secret@proxy.test",
        "http://proxy.test/path",
        "http://proxy.test/?token=secret",
        "http://proxy.test/#fragment",
    ],
)
def test_proxy_refuses_credential_and_unbounded_routing(url):
    with pytest.raises(PolicyError) as caught:
        check_proxy(url, endpoint=DEST.endpoint)
    assert caught.value.condition == "invalid_policy"
    assert (
        check_proxy("http://proxy.test:3128/", endpoint=DEST.endpoint) == "http://proxy.test:3128/"
    )


@pytest.mark.parametrize(
    ("endpoint", "proxy"),
    [
        ("https://collector.example.test", "https://proxy.test:8443"),
        ("http://127.0.0.1:4318", "http://proxy.test:3128"),
    ],
)
def test_setup_rejects_unsupported_proxy_route_before_creating_policy(
    data: Path, endpoint: str, proxy: str
) -> None:
    raw = {
        "destinations": [
            {"destination_id": "otel", "endpoint": endpoint, "proxy": proxy, "signals": ["metrics"]}
        ]
    }
    with pytest.raises(PolicyError) as caught:
        policy.enable_collection("personal-local", raw)
    assert caught.value.condition == "invalid_policy"
    assert not data.exists()
    assert not policy.load_profile_policy().collection_enabled


def test_setup_accepts_http_proxy_for_https_without_export_consent(data: Path) -> None:
    approved = policy.enable_collection(
        "personal-local",
        {
            "destinations": [
                {
                    "destination_id": "otel",
                    "endpoint": "https://collector.example.test",
                    "proxy": "http://proxy.test:3128",
                    "signals": ["metrics"],
                }
            ]
        },
    )
    assert approved.collection_enabled and approved.enabled_exports == ()
    (destination,) = policy.load_profile_policy().destinations
    assert destination.endpoint == "https://collector.example.test"
    assert destination.proxy == "http://proxy.test:3128"
    assert (data / "accounting" / "policy.json").is_file()


@pytest.mark.parametrize(
    "change",
    [
        {"roots": tuple(f"/root/{i}" for i in range(MAX_ROOTS + 1))},
        {
            "destinations": tuple(
                replace(DEST, destination_id=f"d{i}") for i in range(MAX_DESTINATIONS + 1)
            )
        },
        {"destinations": (DEST, DEST)},
        {"destinations": (replace(DEST, destination_id="../escape"),)},
        {"destinations": (replace(DEST, signals=()),)},
        {"destinations": (replace(DEST, signals=("traces",)),)},
        {"destinations": (replace(DEST, signals=(Signal.LOGS,)),)},
        {"evidence_retention_days": True},
        {"receipt_retention_days": 0},
        {"enabled_exports": (("missing", "metrics"),)},
    ],
)
def test_invalid_profile_candidate_cannot_gain_authority(change):
    baseline = ProfilePolicy(destinations=(DEST,))
    file = PolicyFile(profiles={"personal-local": baseline})
    with pytest.raises(PolicyError) as caught:
        validate_profile(replace(baseline, **change), file)
    assert caught.value.condition == "invalid_policy"
    assert file.profiles["personal-local"].enabled_exports == ()
    accepted = validate_profile(replace(baseline, enabled_exports=(("otel", "metrics"),)), file)
    assert accepted.enabled_exports == (("otel", "metrics"),)


def test_path_in_roots_resolves_aliases_without_accepting_escape(tmp_path: Path) -> None:
    approved = tmp_path / "approved"
    outside = tmp_path / "outside"
    approved.mkdir()
    outside.mkdir()
    (approved / "child").mkdir()
    (tmp_path / "inside-alias").symlink_to(approved / "child")
    (tmp_path / "outside-alias").symlink_to(outside)

    roots = (str(approved),)
    assert path_in_roots(roots, str(tmp_path / "inside-alias"))
    assert not path_in_roots(roots, str(tmp_path / "outside-alias"))
    assert not path_in_roots(roots, str(tmp_path / "approved-sibling"))


def test_historical_workspace_owner_rejects_noncanonical_paths() -> None:
    profile = ProfilePolicy(profile_id="acme", roots=("/work/team",))
    ownership = PolicyFile(profiles={"acme": profile})

    assert historical_workspace_owner(ownership, "/work/team/project") == "acme"
    for invalid in ("relative/project", "/work/team/../secret", "/work/team/\0secret"):
        assert historical_workspace_owner(ownership, invalid) is None


@pytest.mark.parametrize(
    "change",
    [
        {"evidence_cache_enabled": 1},
        {"evidence_cache_retention_days": True},
        {"evidence_cache_max_bytes": False},
        {"evidence_cache_retention_days": 1},
        {"evidence_cache_max_bytes": 1},
    ],
)
def test_invalid_cache_consent_or_disabled_bounds_are_rejected(change) -> None:
    candidate = replace(ProfilePolicy(), **change)
    file = PolicyFile(profiles={"personal-local": ProfilePolicy()})

    with pytest.raises(PolicyError) as caught:
        validate_profile(candidate, file)

    assert caught.value.condition == "invalid_policy"
    assert file.profiles["personal-local"].evidence_cache_enabled is False
    assert file.profiles["personal-local"].evidence_cache_max_bytes == 0


@pytest.mark.parametrize(
    "change",
    [
        {"evidence_cache_retention_days": 4},
        {"evidence_cache_max_bytes": 2049},
    ],
)
def test_evidence_cache_cannot_exceed_managed_limits(change) -> None:
    ceiling = ProfilePolicy(
        profile_id="acme",
        roots=("/managed/acme",),
        managed=True,
        evidence_cache_enabled=True,
        evidence_cache_retention_days=3,
        evidence_cache_max_bytes=2048,
    )
    current = replace(
        ceiling,
        evidence_cache_enabled=True,
        evidence_cache_retention_days=2,
        evidence_cache_max_bytes=1024,
    )
    candidate = replace(current, **change)
    file = PolicyFile(profiles={"acme": current}, ceilings={"acme": ceiling})

    with pytest.raises(PolicyError) as caught:
        validate_profile(candidate, file)

    assert caught.value.condition == "retention_widened"
    assert file.profiles["acme"] == current
    assert validate_profile(current, file).evidence_cache_max_bytes == 1024


def test_managed_profile_without_cache_consent_cannot_enable_cache() -> None:
    ceiling = ProfilePolicy(profile_id="acme", roots=("/managed/acme",), managed=True)
    candidate = replace(
        ceiling,
        evidence_cache_enabled=True,
        evidence_cache_retention_days=1,
        evidence_cache_max_bytes=1,
    )
    file = PolicyFile(profiles={"acme": ceiling}, ceilings={"acme": ceiling})

    with pytest.raises(PolicyError) as caught:
        validate_profile(candidate, file)

    assert caught.value.condition == "retention_widened"
    assert file.profiles["acme"].evidence_cache_enabled is False


def test_evidence_cache_positive_control_respects_managed_ceiling() -> None:
    ceiling = ProfilePolicy(
        profile_id="acme",
        roots=("/managed/acme",),
        managed=True,
        evidence_cache_enabled=True,
        evidence_cache_retention_days=3,
        evidence_cache_max_bytes=2048,
    )
    candidate = replace(
        ceiling,
        evidence_cache_retention_days=2,
        evidence_cache_max_bytes=1024,
    )
    file = PolicyFile(profiles={"acme": candidate}, ceilings={"acme": ceiling})

    saved = validate_profile(candidate, file)

    assert saved.evidence_cache_enabled
    assert (saved.evidence_cache_retention_days, saved.evidence_cache_max_bytes) == (2, 1024)


def test_private_write_failure_keeps_previous_document_and_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.accounting.privacy import storage

    base = tmp_path / "private"
    storage.write_private(base, "accounting", "policy.json", data=b"valid-policy")
    target = base / "accounting" / "policy.json"
    replace_file = storage.os.replace

    def fail_replace(*args, **kwargs):
        raise OSError("simulated atomic replacement failure")

    with monkeypatch.context() as fault:
        fault.setattr(storage.os, "replace", fail_replace)
        with pytest.raises(OSError, match="replacement failure"):
            storage.write_private(base, "accounting", "policy.json", data=b"uncommitted")

    assert target.read_bytes() == b"valid-policy"
    assert sorted(path.name for path in target.parent.iterdir()) == ["policy.json"]
    assert storage.os.replace is replace_file


def test_root_snapshot_rejects_replacement_and_unavailable_original_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.accounting.privacy import storage

    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("SSGREP_DATA_DIR", str(root))
    snapshot = storage.snapshot(root)
    assert snapshot is not None
    root.rename(tmp_path / "old-data")
    with pytest.raises(storage.RootChanged):
        storage.open_root(snapshot)

    root.mkdir()
    try:
        with pytest.raises(storage.RootChanged):
            storage.open_root(snapshot)
        assert list(root.iterdir()) == []
    finally:
        root.rmdir()


def test_private_reader_rechecks_size_after_concurrent_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ssgrep.accounting.privacy import storage

    base = tmp_path / "private"
    storage.write_private(base, "accounting", "policy.json", data=b"safe")
    target = base / "accounting" / "policy.json"
    real_fdopen = storage.os.fdopen

    def append_between_stat_and_read(descriptor, mode, *, closefd=False):
        handle = real_fdopen(descriptor, mode, closefd=closefd)
        with target.open("ab") as writer:
            writer.write(b"-too-large")
        return handle

    with monkeypatch.context() as race:
        race.setattr(storage.os, "fdopen", append_between_stat_and_read)
        with pytest.raises(ValueError, match="exceeds"):
            storage.read_private(base, "accounting", "policy.json", max_bytes=4)

    assert target.read_bytes() == b"safe-too-large"


def test_private_reader_rejects_path_traversal_without_creating_storage(tmp_path: Path) -> None:
    from ssgrep.accounting.privacy import storage

    base = tmp_path / "absent"
    with pytest.raises(storage.UnsafePathError):
        storage.read_private(base, "..", "policy.json", max_bytes=1024)
    assert not base.exists()


def test_private_reader_rejects_group_writable_directory_without_repairing_it(
    tmp_path: Path,
) -> None:
    from ssgrep.accounting.privacy import storage

    base = tmp_path / "private"
    storage.write_private(base, "accounting", "policy.json", data=b"valid")
    directory = base / "accounting"
    directory.chmod(0o770)

    with pytest.raises(storage.UnsafePathError):
        storage.read_private(base, "accounting", "policy.json", max_bytes=1024)

    assert directory.stat().st_mode & 0o777 == 0o770
    directory.chmod(0o700)
    assert storage.read_private(base, "accounting", "policy.json", max_bytes=1024) == b"valid"


def test_private_directory_check_refuses_regular_file_without_modifying_it(tmp_path: Path) -> None:
    from ssgrep.accounting.privacy import storage

    directory = tmp_path / "directory"
    directory.mkdir()
    directory.chmod(0o700)
    regular_file = tmp_path / "not-a-directory"
    regular_file.write_bytes(b"keep this data")

    assert storage._check_dir(directory, private=True)
    with pytest.raises(storage.UnsafePathError):
        storage._check_dir(regular_file, private=True)
    assert regular_file.read_bytes() == b"keep this data"


def test_private_codec_decodes_frozen_set_values_and_rejects_nonlist_data() -> None:
    from ssgrep.accounting.privacy import storage

    assert storage.from_plain(frozenset[str], ["workspace-a", "workspace-b"]) == frozenset(
        {"workspace-a", "workspace-b"}
    )
    with pytest.raises(ValueError):
        storage.from_plain(frozenset[str], {"workspace-a": True})


PROBE = r"""
import json
import os
import signal
import socket
import sys
from pathlib import Path
from ssgrep.accounting.privacy import storage as plainfile

base = Path(sys.argv[1])
kind = sys.argv[2]
folder = base / "accounting"
folder.mkdir(mode=0o700)
target = folder / "policy.json"
expected = b'{"collection_enabled": true}'
listener = None
os.chdir(folder)
if kind == "directory":
    target.mkdir(mode=0o700)
elif kind == "socket":
    listener = socket.socket(socket.AF_UNIX)
    listener.bind("policy.json")
elif kind == "symlink":
    (folder / "outside").write_bytes(b"must not be consumed")
    target.symlink_to(folder / "outside")
elif kind == "fifo":
    os.mkfifo(target, 0o600)
else:
    target.write_bytes(expected)
    target.chmod(0o600)

if kind == "fifo_race":
    original_open = plainfile.os.open
    replaced = False
    parent_info = folder.stat()
    parent_identity = (parent_info.st_dev, parent_info.st_ino)
    def swap(path, flags, *args, **kwargs):
        global replaced
        parent_fd = kwargs.get("dir_fd")
        relative_leaf = os.fspath(path) == "policy.json" and parent_fd is not None
        if relative_leaf:
            info = os.fstat(parent_fd)
            relative_leaf = (info.st_dev, info.st_ino) == parent_identity
        if (Path(path) == target or relative_leaf) and not replaced:
            replaced = True
            target.unlink()
            os.mkfifo(target, 0o600)
        return original_open(path, flags, *args, **kwargs)
    plainfile.os.open = swap

signal.signal(signal.SIGALRM, lambda *_: sys.exit(93))
signal.alarm(2)
try:
    try:
        value = plainfile.read_private(base, "accounting", "policy.json", max_bytes=1024)
    except plainfile.UnsafePathError:
        assert kind != "regular"
        rejected = True
    else:
        assert kind == "regular" and value == expected
        rejected = False
finally:
    signal.alarm(0)
    if listener is not None:
        listener.close()
if kind == "fifo_race":
    assert replaced, "the leaf replacement race must have occurred"

if target.is_dir():
    target.rmdir()
else:
    target.unlink()
target.write_bytes(expected)
target.chmod(0o600)
assert plainfile.read_private(base, "accounting", "policy.json", max_bytes=1024) == expected
print(json.dumps({"rejected": rejected, "regular_afterward": expected.decode()}))
"""


@pytest.mark.parametrize("kind", ["fifo", "fifo_race", "socket", "directory", "symlink", "regular"])
def test_private_reader_preserves_availability_and_valid_data(tmp_path: Path, kind: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE, str(tmp_path), kind],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, (kind, completed.returncode, completed.stderr)
    result = json.loads(completed.stdout)
    assert result["rejected"] is (kind != "regular")
    assert json.loads(result["regular_afterward"]) == {"collection_enabled": True}


def test_cached_policy_permission_revocation_stops_capture_then_recovers() -> None:
    from ssgrep.accounting.capture import journal, journal_reader
    from ssgrep.accounting.ledger.contracts import Phase
    from ssgrep.accounting.privacy import policy, storage as datadir
    from ssgrep.services import mcp_server

    journal.close_all()
    policy.enable_collection("personal-local")
    response = mcp_server.search_sessions("isolated-policy-permission")

    def completed_ids() -> set[str]:
        return {
            event.execution_id
            for event in journal_reader.read_journal("personal-local", quarantine=False).events
            if event.phase is Phase.COMPLETED
        }

    before = completed_ids()
    assert len(before) == 1
    target = datadir.data_root() / "accounting" / "policy.json"
    original_mtime = target.stat().st_mtime_ns
    target.chmod(0o644)
    assert target.stat().st_mtime_ns == original_mtime
    try:
        assert mcp_server.search_sessions("isolated-policy-permission") == response
        assert completed_ids() == before
    finally:
        target.chmod(0o600)
    assert mcp_server.search_sessions("isolated-policy-permission") == response
    assert len(completed_ids() - before) == 1
