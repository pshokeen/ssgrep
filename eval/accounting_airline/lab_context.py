"""Shared state of one stand-in pipeline run plus checkpoint/stage plumbing.

``Run`` owns everything a stage needs (roots, observer, report, database, actor kits, product
environments). ``Stage`` builds one checkpoint: it times the stage, accumulates domain and
accounting assertions, writes the private per-checkpoint artifact and prints the redacted
progress line. Nothing here decides correctness.
"""

from __future__ import annotations

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
from .lab_product import Collected, Product, collect, copy_accounting, pin_arithmetic_view
from .lab_session import ActorKit
from .manifest import RunManifest
from .observer import Frozen, NativeSession, Observer
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

    def corpus_docs(self, frozen: Frozen) -> int:
        """Independent count of the documents a retained corpus would hold (turns + results)."""
        return sum(
            1
            for _actor, _task, native in frozen.sessions.values()
            for r in native.records
            if r.kind in ("user_turn", "tool_result")
        )

    def production_profile(self, frozen: Frozen) -> acct.ProfileSpec:
        return acct.production_profile(corpus_docs=self.corpus_docs(frozen))

    def arithmetic_profile(self, frozen: Frozen) -> acct.ProfileSpec:
        reviewed = acct.review_candidates(frozen)
        return acct.arithmetic_profile([(r.kind, r.token) for r in reviewed])

    # ------------------------------------------------------------------- product
    def collect(self, stage: str, *flags: str) -> Collected:
        from .lab_session import refresh_view

        refresh_view(self.roots, self.native_dirs())
        result = collect(self.product, *flags)
        self.collected[stage] = result
        return result

    def native_dirs(self) -> dict[str, Path]:
        dirs = {"a": self.kit_a.sessions_dir}
        if any(self.kit_b.sessions_dir.rglob("*.jsonl")):
            dirs["b"] = self.kit_b.sessions_dir
        return dirs

    def collect_arithmetic(self, frozen: Frozen) -> tuple[Collected, list[str]]:
        """Journal-only isolated copy with the test view pinned, collected there (never summed)."""
        self.arith_count += 1
        dest = self.roots.artifacts / f"arith-{self.arith_count}"
        copy_accounting(self.roots.ssgrep_data, dest, journal_only=True)
        product = Product(
            {**self.product.env, "SSGREP_DATA_DIR": str(dest)},
            self.product.cwd,
            self.product.binary,
        )
        tokens = [t for _, t in self.arithmetic_profile(frozen).reviewed or ()]
        pin_arithmetic_view(product, tokens)
        return collect(product), tokens

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
