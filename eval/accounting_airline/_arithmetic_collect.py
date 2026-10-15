"""Private arithmetic-fixture execution of the real collector and CLI publication path.

Only this isolated validation process injects a reviewed candidate-frequency view.
Production collection still consumes its actual index statistics. No source content,
policy authority, thresholds, native identities or source digests are synthesized.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

ARITHMETIC_METHOD = "airline-lab-arithmetic-v1"


def fixture_methodology(view):
    """A genuinely separate ledger namespace, retaining production units/thresholds."""
    from ssgrep.accounting.reduce import methodology_for

    production = methodology_for(view)
    return replace(
        production, label=f"{ARITHMETIC_METHOD}:observed_identifier_whitelist_v1:{production.label}"
    )


def fixture_frequency_resolver(resolve, view):
    """Use the existing injected-index seam; keep erasure and revision handling intact."""

    def controlled(profile, **kwargs):
        return resolve(profile, **kwargs, index_view=lambda: view)

    return controlled


def validate_inputs(request):
    """Require a marked lab-owned derived root and exact original-source authorities."""
    from .lab_product import ProductError
    from .roots import ROOT_PREFIX, LabRoots

    source = Path(request["source_data"]).resolve(strict=True)
    destination = Path(os.environ["SSGREP_DATA_DIR"]).resolve(strict=True)
    owner = next(
        (
            path
            for path in source.parents
            if path.name.startswith(ROOT_PREFIX)
            and (path / "verifier" / ".lab-owner.json").is_file()
        ),
        None,
    )
    if owner is None:
        raise ProductError("controlled arithmetic requires a marked lab-owned source")
    roots = LabRoots(owner, owner.name[len(ROOT_PREFIX) :])
    roots.verify_ownership()
    if (
        source == destination
        or not source.is_relative_to(owner)
        or not destination.is_relative_to(roots.artifacts)
    ):
        raise ProductError("controlled arithmetic requires a separate private artifact root")
    index, alias = source / "lancedb", destination / "lancedb"
    if index.exists() and (not alias.is_symlink() or alias.resolve(strict=True) != index):
        raise ProductError("controlled arithmetic lost its original locator-index authority")
    approved = []
    for raw in request.get("source_roots", ()):
        path = Path(raw).resolve(strict=True)
        info = path.stat()
        if (
            not path.is_dir()
            or not path.is_relative_to(owner)
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o022
        ):
            raise ProductError("controlled arithmetic source root is not an exact owned lab root")
        approved.append(path)
    return tuple(approved)


def main() -> None:
    from ssgrep.accounting import collector, collector_cycle, policy, reduce_io
    from ssgrep.accounting.collector_reduce import Reducer
    from ssgrep.accounting.collector_types import CollectOptions, CollectResult
    from ssgrep.accounting.frequency import build_view
    from ssgrep.accounting.ledger_contracts import ProfilePolicy
    from ssgrep.accounting.native_source import NativeSourceAdapter, NativeSourceReference
    from ssgrep.accounting.policy_rules import PolicyFile
    from ssgrep.accounting.scope_events import Scoper
    from ssgrep.cli import main as cli_main

    from .lab_product import PROFILE, ProductError, reviewed_view_documents

    request = json.loads(sys.stdin.read())
    source_roots = validate_inputs(request)
    profile = policy.load_profile_policy(PROFILE)
    if profile.enabled_exports or profile.destinations:
        raise ProductError("controlled arithmetic must remain export-disabled")
    view = build_view(
        reviewed_view_documents(
            request["tokens"],
            common_tokens=request.get("common_tokens", ()),
        )
    )
    method = fixture_methodology(view)
    if (
        request["methodology"] != {"label": method.label, "hash": method.hash}
        or request["fixture_inputs"]["frequency_view_id"] != view.version_id
    ):
        raise ProductError("controlled arithmetic inputs differ from their pre-collection freeze")
    real_collect, real_adapter = collector.collect_once, reduce_io.native_source_adapter
    collector_cycle.resolve_frequency = fixture_frequency_resolver(
        collector_cycle.resolve_frequency, view
    )

    def adapter(
        profile_policy: ProfilePolicy,
        env: Mapping[str, str] | None = None,
        *,
        now_ms: int,
        references: Mapping[str, NativeSourceReference] | None = None,
        repository: Any = None,
        ownership: PolicyFile | None = None,
        scoper: Scoper | None = None,
    ) -> NativeSourceAdapter:
        original = real_adapter(
            profile_policy,
            env,
            now_ms=now_ms,
            references=references,
            repository=repository,
            ownership=ownership,
            scoper=scoper,
        )
        return replace(
            original, approved_roots=tuple(dict.fromkeys((*original.approved_roots, *source_roots)))
        )

    def controlled_collect(
        profile: str,
        options: CollectOptions,
        *,
        now: datetime | None = None,
        clock: Callable[[], float] = time.monotonic,
        should_stop: Callable[[], bool] = lambda: False,
        reducer: Reducer | None = None,
        env: Mapping[str, str] | None = None,
        on_step: Callable[[str], None] | None = None,
    ) -> CollectResult:
        from ssgrep.accounting.reduce import reduce_profile

        return real_collect(
            profile,
            options,
            now=now,
            clock=clock,
            should_stop=should_stop,
            reducer=Reducer(reduce_profile, fixture_methodology),
            env=env,
            on_step=on_step,
        )

    # Execute the actual CLI validator, collect_once, journal/partition/checkpoint
    # machinery and JSON renderer. Only frequency and method identity are fixture inputs.
    sys.argv = [request["program"], "metrics", "collect", *sys.argv[1:]]
    with (
        patch.object(reduce_io, "native_source_adapter", adapter),
        patch.object(collector, "collect_once", controlled_collect),
    ):
        cli_main()


if __name__ == "__main__":
    main()
