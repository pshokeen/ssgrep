"""Damaged policy recovery and ownership precedence at the persisted boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ssgrep.accounting.ledger.ledger_contracts import DestinationPolicy, ProfilePolicy, Signal
from ssgrep.accounting.ledger.valuation_inputs import AtomStatus
from ssgrep.accounting.privacy import policy
from ssgrep.accounting.privacy.policy import PolicyError, PolicyFile, workspace_owner
from ssgrep.accounting.views.filter import Scoper
from tests.accounting.export.http_builders import tree_state


@pytest.mark.parametrize(
    ("section", "value"),
    [
        ("schema", True),
        ("profiles", []),
        ("assignments", []),
        ("sending_paused", []),
        ("gaps", []),
        ("gaps", {"personal-local": None}),
        ("gaps", {"personal-local": [None]}),
        ("gaps", {"personal-local": [[]]}),
        ("gaps", {"personal-local": [["start"]]}),
        ("gaps", {"personal-local": [["start", None, "extra"]]}),
        ("gaps", {"personal-local": [[False, None]]}),
        ("gaps", {"personal-local": [["start", 1]]}),
    ],
)
def test_malformed_policy_fails_closed_without_repair(
    tmp_path: Path, section: str, value: Any
) -> None:
    policy.enable_collection("personal-local")
    path = policy.policy_path()
    document = json.loads(path.read_bytes())
    document[section] = value
    path.write_text(json.dumps(document))
    before = tree_state(tmp_path / "data")
    status = policy.policy_status()
    assert status.error is not None and status.error.condition == "policy_corrupt"
    assert status.sending_paused_reason == "policy_corrupt"
    assert not status.profile.collection_enabled and status.gaps == ()
    assert not policy.load_profile_policy().collection_enabled
    with pytest.raises(PolicyError) as caught:
        policy.require_not_paused("personal-local", "export")
    assert caught.value.condition == "export_paused"
    assert tree_state(tmp_path / "data") == before


@pytest.mark.parametrize("section", ["profiles", "ceilings"])
def test_persisted_profile_key_cannot_impersonate_another_profile(
    tmp_path: Path, section: str
) -> None:
    policy.enable_collection("acme", {"roots": [str(tmp_path / "work")]})
    path = policy.policy_path()
    document = json.loads(path.read_bytes())
    document[section]["acme"]["profile_id"] = "personal-local"
    path.write_text(json.dumps(document))
    before = tree_state(tmp_path / "data")
    status = policy.policy_status("acme")
    assert status.error is not None and status.error.condition == "policy_corrupt"
    loaded = policy.load_profile_policy("acme")
    assert loaded.profile_id == "acme" and not loaded.collection_enabled
    assert not policy.path_in_scope(loaded, str(tmp_path / "work"))
    assert tree_state(tmp_path / "data") == before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("roots", [False]),
        ("policy_version", True),
        ("collection_enabled", 1),
        ("enabled_exports", [["otel"]]),
        ("destinations", [None]),
        (
            "destinations",
            [{"destination_id": "otel", "endpoint": "https://c.test", "signals": ["traces"]}],
        ),
    ],
)
def test_corrupt_profile_fields_do_not_retain_enabled_authority(field: str, value: Any) -> None:
    policy.enable_collection("personal-local")
    path = policy.policy_path()
    document = json.loads(path.read_bytes())
    document["profiles"]["personal-local"][field] = value
    path.write_text(json.dumps(document))
    original = path.read_bytes()
    status = policy.policy_status()
    assert status.error is not None and status.error.condition == "policy_corrupt"
    assert not status.profile.collection_enabled
    assert path.read_bytes() == original


def test_longest_assignment_wins_but_invalid_assignee_never_falls_back(tmp_path: Path) -> None:
    root = str(tmp_path.resolve() / "work")
    nested = root + "/team"
    profiles = {
        "personal-local": ProfilePolicy(collection_enabled=True),
        "acme": ProfilePolicy(profile_id="acme", roots=(root,)),
        "team": ProfilePolicy(profile_id="team", roots=(nested,)),
    }
    file = PolicyFile(profiles=profiles, assignments={root: "acme", nested: "team"})
    assert workspace_owner(file, nested + "/src") == "team"
    assert workspace_owner(file, root + "/other") == "acme"
    assert workspace_owner(file, root + "-other") == "personal-local"
    for owner in ("missing", "team"):
        invalid = replace(file, assignments={root: owner})
        assert workspace_owner(invalid, root + "/other") is None
    tied = replace(
        file,
        profiles={**profiles, "other": replace(profiles["acme"], profile_id="other")},
        assignments={},
    )
    assert workspace_owner(tied, root + "/other") is None
    assert workspace_owner(tied, nested + "/src") == "team"
    restricted = PolicyFile(profiles={"personal-local": ProfilePolicy(roots=(nested,))})
    assert workspace_owner(restricted, root + "/other") is None
    assert workspace_owner(restricted, nested) == "personal-local"


def test_managed_update_narrows_signals_proxy_roots_and_retention_without_reenabling(
    tmp_path: Path,
) -> None:
    root = str(tmp_path / "work")
    endpoint = "https://collector.example.test"
    initial = {
        "roots": [root],
        "destinations": [
            {
                "destination_id": "otel",
                "endpoint": endpoint,
                "signals": ["metrics", "logs"],
                "logs_permitted": True,
            },
            {"destination_id": "old", "endpoint": endpoint, "signals": ["metrics"]},
        ],
    }
    policy.enable_collection("acme", initial)
    current = policy.load_profile_policy("acme")
    policy.save_profile(
        replace(
            current,
            paused=True,
            enabled_exports=(("otel", "metrics"), ("otel", "logs"), ("old", "metrics")),
        )
    )
    narrowed = {
        "roots": [root + "/team"],
        "policy_version": 2,
        "evidence_retention_days": 5,
        "receipt_retention_days": 20,
        "destinations": [
            {
                "destination_id": "otel",
                "endpoint": endpoint,
                "signals": ["metrics"],
                "proxy": "http://proxy.example.test",
            }
        ],
    }
    assert policy.apply_managed_update("acme", narrowed) == (True, None)
    result = policy.load_profile_policy("acme")
    assert result.roots == (str(Path(root + "/team").resolve()),)
    assert result.paused and result.collection_enabled
    assert result.enabled_exports == (("otel", "metrics"),)
    assert result.destinations == (
        DestinationPolicy("otel", endpoint, (Signal.METRICS,), proxy="http://proxy.example.test"),
    )
    assert (result.evidence_retention_days, result.receipt_retention_days) == (5, 20)
    assert result.policy_version == 2


def test_source_scope_never_borrows_personal_authority_and_corruption_revokes_it(
    tmp_path: Path,
) -> None:
    root = str((tmp_path / "org").resolve())
    policy.enable_collection("personal-local")
    policy.enable_collection("acme", {"roots": [root]})
    organization = Scoper(policy.load_profile_policy("acme"))
    personal = Scoper(policy.load_profile_policy())
    assert organization.in_scope(root + "/repo")
    assert organization.source_status(root + "/repo") == (AtomStatus.OK, "")
    assert organization.source_status(None) == (AtomStatus.OUT_OF_SCOPE, "source_scope_unknown")
    assert organization.source_status(root + "-other") == (
        AtomStatus.OUT_OF_SCOPE,
        "source_out_of_scope",
    )
    assert personal.in_scope(root + "-other") and personal.in_scope(None)
    assert not personal.in_scope(root + "/repo")
    active = policy.load_profile_policy("acme")
    personal_policy = policy.load_profile_policy()
    policy.policy_path().write_text("{corrupt")
    for profile in (active, personal_policy):
        denied = Scoper(profile)
        assert not denied.in_scope(root + "/repo")
        assert not denied.in_scope(root + "-other")
        assert not denied.in_scope(None)
