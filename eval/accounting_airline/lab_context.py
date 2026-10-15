"""Shared state of one stand-in pipeline run plus checkpoint/stage plumbing.

``Run`` owns everything a stage needs (roots, observer, report, database, actor kits, product
environments). ``Stage`` builds one checkpoint: it times the stage, accumulates domain and
accounting assertions, writes the private per-checkpoint artifact and prints the redacted
progress line. Nothing here decides correctness.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import accounting_oracle as acct
from .checkpoints import Assertion, Checkpoint, input_prefix, write_private_json
from .database import BaselineInfo, LabInstance
from .lab_product import Collected, Product, collect, controlled_arithmetic, copy_accounting
from .lab_session import ActorKit
from .manifest import RunManifest
from .observer import Frozen, NativeSession, Observer
from .oracle_inputs import IndexStats, accept_lookup_inputs, freeze_lookups, read_index_inputs
from .oracle_time import read_timing_events
from .report import Report, progress_line
from .roots import LabRoots
from .sandbox import SandboxProfile

Emit = Callable[[str], None]
PRODUCTION = acct.PRODUCTION
ARITHMETIC = acct.ARITHMETIC


@dataclass
class Run:
    roots: LabRoots
    observer: Observer
    report: Report
    manifest: RunManifest
    instance: LabInstance
    base: BaselineInfo
    cache: Path | None
    cache_blocked_reason: str | None
    secrets: list[str]
    emit: Emit
    product: Product
    kit_a: ActorKit
    kit_b: ActorKit
    a_db: str
    b_db: str
    a_sid: str
    b_sid: str
    t0: float = field(default_factory=time.monotonic)
    profile: SandboxProfile | None = None
    sandbox_ok: bool = False
    notes: dict[str, Any] = field(default_factory=dict)
    arith_count: int = 0
    native: dict[str, Path] = field(default_factory=dict)  # label -> native session file
    collected: dict[str, Collected] = field(default_factory=dict)  # stage -> real binaries' view
    failed: bool = False

    # ------------------------------------------------------------------ evidence
    def frozen(self) -> Frozen:
        return self.observer.freeze()

    def sessions_of(self, frozen: Frozen, actor: str) -> dict[str, NativeSession]:
        return {k: v[2] for k, v in frozen.sessions.items() if v[0] == actor}

    def production_profile(self, frozen: Frozen) -> acct.ProfileSpec:
        """Use first-collected raw inputs; an uncollected lookup remains explicitly pending."""
        inputs = freeze_lookups(
            self.notes.get("oracle_lookup_stats", {}),
            frozen.calls,
            IndexStats(unavailable_reason="lookup_not_collected"),
        )
        return acct.production_profile(
            inputs, read_timing_events(self.roots.ssgrep_data, frozen.calls)
        )

    def arithmetic_profile(self, frozen: Frozen) -> acct.ProfileSpec:
        reviewed = acct.review_candidates(frozen)
        return acct.arithmetic_profile(
            [(r.kind, r.token) for r in reviewed],
            read_timing_events(self.roots.ssgrep_data, frozen.calls),
        )

    # ------------------------------------------------------------------- product
    def collect(self, stage: str, *flags: str) -> Collected:
        """Collect after a native turn; post-cutoff arrivals retain unproved acceptance."""
        observed = self.frozen()
        before = read_index_inputs(self.roots.ssgrep_data, self.roots.project)
        result = collect(self.product, *flags)
        after = read_index_inputs(self.roots.ssgrep_data, self.roots.project)
        observed_after = self.frozen()
        self.notes["oracle_lookup_stats"] = accept_lookup_inputs(
            self.notes.get("oracle_lookup_stats", {}),
            observed.calls,
            before,
            after,
            calls_after=observed_after.calls,
        )
        self.attach_receipt(result, self.product)
        self.collected[stage] = result
        return result

    def native_dirs(self) -> dict[str, Path]:
        dirs = {"a": self.kit_a.sessions_dir}
        if any(self.kit_b.sessions_dir.rglob("*.jsonl")):
            dirs["b"] = self.kit_b.sessions_dir
        return dirs

    def collect_arithmetic(self, frozen: Frozen) -> tuple[Collected, list[str]]:
        """Real collector publication under isolated, explicit arithmetic-fixture inputs."""
        self.arith_count += 1
        dest = self.roots.artifacts / f"arith-{self.arith_count}"
        copy_accounting(self.roots.ssgrep_data, dest, journal_only=True)
        product = Product(
            {**self.product.env, "SSGREP_DATA_DIR": str(dest)},
            self.product.cwd,
            self.product.binary,
        )
        tokens = [t for _, t in self.arithmetic_profile(frozen).reviewed or ()]
        common = [
            token
            for kind, token in acct.candidate_vocabulary(frozen)
            if kind == "identifier" and token not in tokens
        ]
        result = controlled_arithmetic(
            product,
            tokens,
            source_data=self.roots.ssgrep_data,
            source_roots=tuple(self.native_dirs().values()),
            common_tokens=common,
        )
        self.attach_receipt(result, product)
        return result, tokens

    @staticmethod
    def attach_receipt(collected: Collected, product: Product) -> None:
        identities = {}
        for path in sorted(
            (product.data_dir / "accounting" / "personal-local" / "receipts").glob("*.json")
        ):
            if not re.fullmatch(r"[0-9a-f]{24}", path.stem) or path.stat().st_size > 1000000:
                continue
            receipt = json.loads(path.read_bytes())
            unit = receipt.get("unit")
            if unit in ("context_tokens:chars_div4_v1", "work_seconds:capped_gap_v1"):
                identities.setdefault(unit, path.stem)
            if len(identities) == 2:
                break
        collected.savings_document["lab_receipts"] = [
            product.data("savings", "explain", identity) for identity in identities.values()
        ]

    def verify_product(self) -> Product:
        """The verifier/control environment (same data dir, lab-owned discovery roots)."""
        return self.product

    # ------------------------------------------------------------------ printing
    def say(self, text: str) -> None:
        from .report import redact

        self.emit(redact(text, self.secrets))

    def elapsed_s(self) -> int:
        return int(time.monotonic() - self.t0)


def utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Stage:
    """Builds one checkpoint; ``finish`` records it and prints its progress line."""

    def __init__(self, run: Run, cid: str, name: str, methodology: str | None = None) -> None:
        self.run, self.cid, self.name = run, cid, name
        self.domain: list[Assertion] = []
        self.accounting: list[Assertion] = []
        self.input_paths: list[Path] = []
        self.task_ids: list[str] = []
        self.runtime_ids: dict[str, str] = {}
        self.expected_delta: dict[str, Any] = {}
        self.actual_delta: dict[str, Any] = {}
        self.coverage_missing: list[str] = []
        self.methods: list[str] = []
        self.snapshot_before: str | None = None
        self.snapshot_after: str | None = None
        self.methodology = methodology
        self.started = utc_stamp()
        self._t = time.monotonic()

    def add_domain(self, *items: Assertion) -> None:
        self.domain.extend(items)

    def add_accounting(self, *items: Assertion) -> None:
        self.accounting.extend(items)

    def finish(self, next_action: str) -> Checkpoint:
        run = self.run
        frozen = run.frozen()
        by_actor = Counter(c.actor for c in frozen.calls)
        prefixes: dict[str, str] = {}
        for path in self.input_paths:
            if path.is_file():
                prefixes[path.name] = input_prefix(path)
        for seg in sorted(
            (run.roots.ssgrep_data / "accounting" / "personal-local" / "journal").glob("seg-*")
        ):
            prefixes[f"journal/{seg.name}"] = input_prefix(seg)
        artifact = run.roots.artifacts / "checkpoints" / f"{self.cid}.json"
        cp = Checkpoint(
            cid=self.cid,
            name=self.name,
            domain=self.domain,
            accounting=self.accounting,
            input_prefixes=prefixes,
            events_by_actor=dict(by_actor),
            runtime_ids=self.runtime_ids,
            task_ids=self.task_ids,
            snapshot_before=self.snapshot_before,
            snapshot_after=self.snapshot_after,
            expected_delta=self.expected_delta,
            actual_delta=self.actual_delta,
            comparison_method="; ".join(dict.fromkeys(self.methods)),
            coverage_missing=self.coverage_missing,
            artifact_path=str(artifact),
            next_action=next_action,
            started_utc=self.started,
            elapsed_ms=int((time.monotonic() - self._t) * 1000),
            methodology=self.methodology,
        )
        run.report.checkpoints.add(cp)
        artifact.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_private_json(artifact, cp.as_dict())
        run.say(
            progress_line(
                cp,
                elapsed_s=run.elapsed_s(),
                artifact=cp.name and f"checkpoints/{self.cid}.json",
                next_action=next_action,
                redactions=run.secrets,
            )
        )
        return cp


def dig(doc: Any, key: str) -> Any:
    """First value stored under ``key`` anywhere in a nested JSON document."""
    if isinstance(doc, Mapping):
        if key in doc:
            return doc[key]
        for value in doc.values():
            found = dig(value, key)
            if found is not None:
                return found
    elif isinstance(doc, list):
        for value in doc:
            found = dig(value, key)
            if found is not None:
                return found
    return None
