"""Independent accounting oracle for the airline lab (validation-only).

Recomputes exposures, exclusions, source atoms, credits, debits and nets from FROZEN observer
evidence alone, by the rules in ``validation-lab.md`` section 5 / ``design.md`` D4-D7. It imports
no ``ssgrep`` code, so it can never be the production reducer renamed. Two profiles exist and
are never summed: ``production-policy`` and ``airline-lab-arithmetic-v1``. All results are ints.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import oracle_rules as rules
from .observer import Frozen
from .oracle_rules import SCENARIOS, import_violations
from .oracle_scan import LOOKUPS, UNOBSERVED, Invocation, Scan, scan_frozen

__all__ = [
    "AggregationRefused", "ProfileSpec", "Estimate", "Verdict", "TaskDebit", "Reviewed",
    "OracleResult", "production_profile", "arithmetic_profile", "evaluate",
    "review_candidates", "refuse_aggregate", "compare_with_snapshot", "import_violations",
]  # fmt: skip

PRODUCTION = "production-policy"
ARITHMETIC = "airline-lab-arithmetic-v1"
UNIT = "context_tokens"
METHOD = "chars_div4_v1"
FREQUENCY_MIN_DOCS = 200
REUSE = ("none_observed", "inspected", "action_reference", "contradicted")


class AggregationRefused(RuntimeError):
    """Production-policy and arithmetic-fixture results must never be summed."""


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    reviewed: tuple[tuple[str, str], ...] | None
    threshold_milli: int | None
    corpus_docs: int | None


def production_profile(corpus_docs: int | None) -> ProfileSpec:
    return ProfileSpec(PRODUCTION, None, None, corpus_docs)


def arithmetic_profile(reviewed: Sequence[tuple[str, str]]) -> ProfileSpec:
    return ProfileSpec(ARITHMETIC, tuple((str(k), str(t)) for k, t in reviewed), 0, None)


@dataclass(frozen=True)
class Estimate:
    scenario: str
    credit: int | None
    debit: int | None
    net: int | None
    known_net: int | None
    missing: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario, "credit": self.credit, "debit": self.debit,
            "net": self.net, "known_net": self.known_net, "missing": list(self.missing),
        }  # fmt: skip


@dataclass(frozen=True)
class Verdict:
    task: str
    token: tuple[str, str]
    exposure_call: str
    eligible: bool
    reason: str | None
    reuse: str
    source_session: str | None
    witness_record: str | None
    atoms: Mapping[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task, "token": list(self.token), "exposure_call": self.exposure_call,
            "eligible": self.eligible, "reason": self.reason, "reuse": self.reuse,
            "source_session": self.source_session, "witness_record": self.witness_record,
            "atoms": {k: list(v) for k, v in self.atoms.items()},
        }  # fmt: skip


@dataclass(frozen=True)
class TaskDebit:
    task: str
    call: str
    chars: int | None
    partial: bool
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"task": self.task, "call": self.call, "chars": self.chars,
                "partial": self.partial, "reason": self.reason}  # fmt: skip


@dataclass(frozen=True)
class Reviewed:
    token: str
    kind: str
    task: str
    reason: str
    exposure_call: str
    action_call: str | None


@dataclass(frozen=True)
class OracleResult:
    profile: str
    methodology_label: str
    estimates: Mapping[str, Estimate]
    tasks: tuple[str, ...]
    debits: tuple[TaskDebit, ...]
    verdicts: tuple[Verdict, ...]
    counts: Mapping[str, int]
    unknowns: tuple[str, ...]
    non_aggregation: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile, "methodology_label": self.methodology_label,
            "estimates": {k: v.as_dict() for k, v in self.estimates.items()},
            "tasks": list(self.tasks), "debits": [d.as_dict() for d in self.debits],
            "verdicts": [v.as_dict() for v in self.verdicts], "counts": dict(self.counts),
            "unknowns": list(self.unknowns), "non_aggregation": self.non_aggregation,
        }  # fmt: skip


# ----------------------------------------------------------------------------- arithmetic


def union_chars(per_task: Iterable[Iterable[str]], chars: Mapping[str, int]) -> int:
    """Characters of the per-task UNION of atom ids, summed over tasks (no division here)."""
    return sum(sum(chars[a] for a in set(atoms)) for atoms in per_task)


def _estimates(
    scan: Scan, verdicts: Sequence[Verdict], debits: Sequence[TaskDebit], profile: ProfileSpec
) -> tuple[dict[str, Estimate], list[str]]:
    debit_missing = sorted({f"debit:{d.reason}" for d in debits if d.chars is None})
    debit = sum(d.chars for d in debits if d.chars is not None) // 4
    if profile.name == PRODUCTION and (profile.corpus_docs or 0) >= FREQUENCY_MIN_DOCS:
        reasons = ["credit:production_rarity_not_evaluated", *debit_missing]
        out = {s: Estimate(s, None, debit, None, None, tuple(reasons)) for s in SCENARIOS}
        return out, ["production_rarity_not_evaluated"]
    credit_missing = sorted({"credit:source_atom_unavailable"} if any(
        v.reason == "source_atom_unavailable" for v in verdicts) else [])  # fmt: skip
    missing = tuple(credit_missing + debit_missing)
    out = {}
    for scenario in SCENARIOS:
        by_task: dict[str, list[str]] = {t: [] for t in scan.tasks()}
        for v in verdicts:
            by_task.setdefault(v.task, []).extend(v.atoms[scenario])
        credit = union_chars(by_task.values(), scan.atom_chars) // 4
        net = None if missing else credit - debit
        out[scenario] = Estimate(scenario, credit, debit, net, credit - debit, missing)
    return out, []


# ------------------------------------------------------------------------------ verdicts


def _tokens_of(inv: Invocation, profile: ProfileSpec, prefix: re.Pattern[str] | None) -> list:
    """(kind, token) candidates visible in the invocation's cards."""
    texts = rules.card_texts(inv.parsed) if inv.parsed else []
    if profile.reviewed is not None:
        return [(k, t) for k, t in profile.reviewed if any(rules.has_token(x, t) for x in texts)]
    pattern = prefix or re.compile(rules.DEFAULT_IDENTIFIER)
    seen: dict[tuple[str, str], None] = {}
    for text in texts:
        for token in rules.extract_tokens(text, pattern):
            seen.setdefault(("identifier", token))
    return list(seen)


def _source(scan: Scan, inv: Invocation, token: str) -> tuple[str | None, str | None, int | None]:
    """(reason, source session key, witness index) over the refs the token is visible under."""
    assert inv.parsed is not None and inv.result is not None
    stamp = scan.records[inv.session][inv.result].ts_ms
    first: tuple[str | None, str | None, int | None] | None = None
    for ref in rules.refs_for_token(inv.parsed, token):
        found = rules.REF_RE.fullmatch(ref or "")
        name = found.group(1) if found else None
        key = name if name in scan.records else scan.alias.get(name or "")
        if not key:
            outcome: tuple[str | None, str | None, int | None] = ("source_unresolved", None, None)
        elif key == inv.session:
            outcome = ("source_self_or_later", key, None)
        else:
            src = rules.resolve_source(token, scan.records[key], stamp)
            outcome = (src.reason, key, src.witness)
        if outcome[0] is None:
            return outcome
        first = first or outcome
    return first or ("source_unresolved", None, None)


def _judge(scan: Scan, inv: Invocation, kind: str, token: str, profile: ProfileSpec) -> Verdict:
    assert inv.result is not None
    none: dict[str, tuple[str, ...]] = {s: () for s in SCENARIOS}
    refs = rules.refs_for_token(inv.parsed, token) if inv.parsed else []
    reuse, _ = scan.reuse(inv, token, refs)

    def verdict(
        reason: str | None,
        src: str | None = None,
        wit: str | None = None,
        atoms: Mapping[str, tuple[str, ...]] = none,
    ) -> Verdict:
        eligible = reason is None
        return Verdict(
            inv.task, (kind, token), inv.call_id, eligible, reason, reuse, src, wit, atoms
        )

    if profile.name == PRODUCTION:
        pinned = (profile.corpus_docs or 0) >= FREQUENCY_MIN_DOCS
        return verdict("production_rarity_not_evaluated" if pinned else "rarity_unknown")
    if (why := scan.novelty(inv, token)) is not None:
        return verdict(why)
    reason, key, witness = _source(scan, inv, token)
    if reason is not None or key is None or witness is None:
        return verdict(reason or "source_unresolved", key)
    records = scan.records[key]
    wit_id = records[witness].record_id
    if reuse != "action_reference":
        return verdict(None, key, wit_id)  # eligible, but nothing earned without an action
    chain = rules.source_atoms(key, records, witness)
    if chain is None:
        return verdict("source_atom_unavailable", key, wit_id)
    immediate, discovery, episode = chain  # cumulative: each scenario keeps the lower one's atoms
    atoms = {
        "conservative": immediate,
        "central": tuple(sorted({*immediate, *discovery})),
        "optimistic": tuple(sorted({*immediate, *discovery, *episode})),
    }
    return verdict(None, key, wit_id, atoms)


def _exposing(scan: Scan) -> list[Invocation]:
    return [i for i in scan.invocations if i.has_cards and i.chars is not None]


def _debits(scan: Scan) -> list[TaskDebit]:
    out = [
        TaskDebit(i.task, i.call_id, i.chars, i.partial, i.reason)
        for i in scan.invocations
        if i.is_lookup
    ]
    for call in scan.unlinked:
        if call.argv[1] in LOOKUPS:
            out.append(
                TaskDebit(call.task or "unattributed", call.call_id, None, False, UNOBSERVED)
            )
    return out


def _counts(scan: Scan, verdicts: Sequence[Verdict]) -> dict[str, int]:
    counts = dict(scan.counts)
    lookups = [i for i in scan.invocations if i.is_lookup]
    counts["invocations"] = len(scan.invocations)
    counts["unlinked_executions"] = len(scan.unlinked)
    counts["exposures"] = sum(1 for i in lookups if i.chars is not None)
    counts["eligible_candidates"] = sum(1 for v in verdicts if v.eligible)
    executions = [i for i in lookups if i.is_execution]
    counts["empty"] = sum(1 for i in executions if i.outcome == "empty")
    counts["error"] = sum(1 for i in executions if i.outcome == "error")
    counts["truncated_exposures"] = sum(1 for i in lookups if i.truncated)
    return counts


def evaluate(frozen: Frozen, profile: ProfileSpec) -> OracleResult:
    scan = scan_frozen(frozen)
    verdicts = [
        _judge(scan, inv, kind, token, profile)
        for inv in _exposing(scan)
        for kind, token in _tokens_of(inv, profile, None)
    ]
    debits = _debits(scan)
    estimates, unknowns = _estimates(scan, verdicts, debits, profile)
    if profile.name == PRODUCTION and (profile.corpus_docs or 0) < FREQUENCY_MIN_DOCS:
        unknowns.append(f"frequency_insufficient_corpus:{profile.corpus_docs}/{FREQUENCY_MIN_DOCS}")
    unknowns += [f"{d.reason}:{d.call}" for d in debits if d.chars is None]
    unknowns += [f"source_atom_unavailable:{v.task}:{v.token[1]}" for v in verdicts
                 if v.reason == "source_atom_unavailable"]  # fmt: skip
    return OracleResult(
        profile.name, f"{profile.name}:{UNIT}:{METHOD}", estimates, tuple(scan.tasks()),
        tuple(debits), tuple(verdicts), _counts(scan, verdicts), tuple(dict.fromkeys(unknowns)),
        f"{profile.name} results are never summed with any other profile",
    )  # fmt: skip


def review_candidates(frozen: Frozen, *, prefix: str | None = None) -> tuple[Reviewed, ...]:
    """Tokens with visible solver exposure, novelty and a later successful action reference."""
    scan = scan_frozen(frozen)
    pattern = re.compile(prefix) if prefix else None
    spec = ProfileSpec(ARITHMETIC, None, 0, None)
    out: dict[tuple[str, str], Reviewed] = {}
    for inv in _exposing(scan):
        for kind, token in _tokens_of(inv, spec, pattern):
            refs = rules.refs_for_token(inv.parsed, token) if inv.parsed else []
            if scan.novelty(inv, token) is not None:
                continue
            reuse, action = scan.reuse(inv, token, refs)
            if reuse == "action_reference":
                out.setdefault(
                    (inv.task, token),
                    Reviewed(token, kind, inv.task, "exposed_novel_acted", inv.call_id, action),
                )
    return tuple(out.values())


# ------------------------------------------------------------------- aggregation, snapshot


def refuse_aggregate(results: Iterable[OracleResult]) -> None:
    profiles = sorted({r.profile for r in results})
    if len(profiles) > 1:
        raise AggregationRefused(f"refusing to aggregate distinct profiles: {profiles}")


def _find(doc: Any, key: str) -> Mapping[str, Any] | None:
    if isinstance(doc, Mapping):
        if isinstance(doc.get(key), Mapping):
            return doc[key]
        for value in doc.values():
            if (found := _find(value, key)) is not None:
                return found
    return None


def _same(label: str, expected: int | None, actual: Any, out: list[str]) -> None:
    if not isinstance(actual, Mapping) or "value" not in actual:
        out.append(f"{label}: missing in snapshot")
        return
    value = actual["value"]
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        out.append(f"{label}: non-integer value {value!r}")
    elif value != expected:
        out.append(f"{label}: oracle {expected!r} != snapshot {value!r}")


def compare_with_snapshot(result: OracleResult, savings_json: Mapping[str, Any]) -> list[str]:
    """Mismatch descriptions against the real ``ssgrep savings --json`` document."""
    out: list[str] = []
    tokens = (_find(savings_json, "estimates") or {}).get(UNIT)
    for scenario, est in result.estimates.items():
        row = tokens.get(scenario) if isinstance(tokens, Mapping) else None
        if not isinstance(row, Mapping):
            out.append(f"{UNIT}.{scenario}: missing in snapshot")
            continue
        for field in ("credit", "debit", "net"):
            _same(f"{UNIT}.{scenario}.{field}", getattr(est, field), row.get(field), out)
    measured = _find(savings_json, "measured")
    for name, key in (("searches", "journaled_search"), ("shows", "journaled_show")):
        if measured is None or name not in measured:
            out.append(f"measured.{name}: missing in snapshot")
        else:
            _same(f"measured.{name}", result.counts[key], measured[name], out)
    return out
