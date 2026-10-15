"""Independent accounting oracle for the airline lab (validation-only).

Recomputes exposures, source unions, known subtotals and sound bounds from frozen native,
observer and raw read-only index/journal inputs. No production accounting reducer, frequency
or valuation output is used as an expectation. Production v2 and explicitly reviewed fixture
arithmetic remain separate methods; neither proves causal savings or positive ROI.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from . import oracle_rules as rules
from .observer import Frozen
from .oracle_estimates import Estimate, bounded_estimate, compare_estimates, observed_timing
from .oracle_inputs import IndexStats, rarity_bands, rarity_upper_df
from .oracle_receipts import compare_receipt
from .oracle_rules import SCENARIOS
from .oracle_scan import LOOKUPS, UNOBSERVED, Invocation, Scan, scan_frozen
from .oracle_time import time_estimates

__all__ = [
    "AggregationRefused", "ProfileSpec", "Estimate", "Verdict", "TaskDebit", "Reviewed",
    "OracleResult", "production_profile", "arithmetic_profile", "evaluate",
    "review_candidates", "candidate_vocabulary", "refuse_aggregate", "compare_with_snapshot",
]  # fmt: skip

PRODUCTION = "production-policy"
ARITHMETIC = "airline-lab-arithmetic-v1"
UNIT = "context_tokens"
METHOD = "chars_div4_v1"
REUSE = ("none_observed", "inspected", "action_reference", "contradicted")


class AggregationRefused(RuntimeError):
    """Production-policy and arithmetic-fixture results must never be summed."""


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    reviewed: tuple[tuple[str, str], ...] | None
    threshold_milli: int | None
    lookup_stats: Mapping[str, IndexStats] | None
    timing_events: tuple[Mapping[str, Any], ...] = ()
    compare_time: bool = False


def production_profile(
    lookup_stats: Mapping[str, IndexStats],
    timing_events: Sequence[Mapping[str, Any]] | None = None,
) -> ProfileSpec:
    return ProfileSpec(
        PRODUCTION,
        None,
        None,
        MappingProxyType(dict(lookup_stats)),
        tuple(MappingProxyType(dict(e)) for e in timing_events or ()),
        timing_events is not None,
    )


def arithmetic_profile(
    reviewed: Sequence[tuple[str, str]],
    timing_events: Sequence[Mapping[str, Any]] | None = None,
) -> ProfileSpec:
    return ProfileSpec(
        ARITHMETIC,
        tuple((str(k), str(t)) for k, t in reviewed),
        0,
        None,
        tuple(MappingProxyType(dict(e)) for e in timing_events or ()),
        timing_events is not None,
    )


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
    proven_bands: tuple[bool, ...] = (True, True, True)
    provenance: tuple[tuple[str, str], ...] = ()
    action_outcome_proven: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task, "token": list(self.token), "exposure_call": self.exposure_call,
            "eligible": self.eligible, "reason": self.reason, "reuse": self.reuse,
            "source_session": self.source_session, "witness_record": self.witness_record,
            "proven_bands": list(self.proven_bands), "provenance": dict(self.provenance),
            "action_outcome_proven": self.action_outcome_proven,
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
    time_estimates: Mapping[str, Estimate]
    timing: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile, "methodology_label": self.methodology_label,
            "estimates": {k: v.as_dict() for k, v in self.estimates.items()},
            "tasks": list(self.tasks), "debits": [d.as_dict() for d in self.debits],
            "verdicts": [v.as_dict() for v in self.verdicts], "counts": dict(self.counts),
            "unknowns": list(self.unknowns), "non_aggregation": self.non_aggregation,
            "time_estimates": {k: v.as_dict() for k, v in self.time_estimates.items()},
            "timing": dict(self.timing),
        }  # fmt: skip


# ----------------------------------------------------------------------------- arithmetic


def union_chars(per_task: Iterable[Iterable[str]], chars: Mapping[str, int]) -> int:
    """Characters of the per-task UNION of atom ids, summed over tasks (no division here)."""
    return sum(sum(chars[a] for a in set(atoms)) for atoms in per_task)


def _estimates(
    scan: Scan, verdicts: Sequence[Verdict], debits: Sequence[TaskDebit], profile: ProfileSpec
) -> tuple[dict[str, Estimate], list[str]]:
    debit_missing = sorted({f"debit:{d.reason}" for d in debits if d.chars is None})
    debit_raw = sum(d.chars for d in debits if d.chars is not None)
    out, unknowns = {}, []
    for band, scenario in enumerate(SCENARIOS):
        by_task: dict[str, list[str]] = {t: [] for t in scan.tasks()}
        missing = list(debit_missing)
        for v in verdicts:
            if v.reuse == "action_reference" and v.reason in (
                "source_atom_unavailable",
                "source_unresolved",
                "source_time_unknown",
                "source_evidence_conflict",
                "prior_context_conflict",
            ):
                missing.append(f"credit:{v.reason}")
            if (
                (v.eligible or v.reason == "rarity_unknown")
                and v.reuse == "action_reference"
                and not v.proven_bands[band]
            ):
                missing.append("credit:rarity_bound_unproved")
            if v.proven_bands[band]:
                by_task.setdefault(v.task, []).extend(v.atoms[scenario])
        credit_raw = union_chars(by_task.values(), scan.atom_chars)
        missing = tuple(sorted(set(missing)))
        out[scenario] = bounded_estimate(
            scenario,
            credit_raw,
            any(m.startswith("credit:") for m in missing),
            debit_raw,
            debit_raw,
            None if any(m.startswith("debit:") for m in missing) else debit_raw,
            4,
            missing,
        )
        unknowns.extend(missing)
    return out, sorted(set(unknowns))


# ------------------------------------------------------------------------------ verdicts


def _tokens_of(inv: Invocation, profile: ProfileSpec, prefix: re.Pattern[str] | None) -> list:
    """(kind, token) candidates visible in the invocation's cards."""
    texts = rules.card_texts(inv.parsed) if inv.parsed else []
    if profile.reviewed is not None:
        return [(k, t) for k, t in profile.reviewed if any(rules.has_token(x, t) for x in texts)]
    if profile.name == PRODUCTION:
        return list(
            dict.fromkeys(candidate for text in texts for candidate in rules.candidate_tokens(text))
        )
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
        elif any(
            (key, record.record_id) in scan.conflicted_records
            for record in scan.records[key]
            if record.kind in rules.CONTENT_KINDS
        ):
            outcome = ("source_evidence_conflict", key, None)
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
    reuse, _action, action_ok = scan.reuse(inv, token, refs)
    evidence = [
        profile.lookup_stats.get(d.call_id)
        for d in inv.dispatches
        if profile.lookup_stats is not None
    ]
    stats = evidence[0] if evidence and all(s == evidence[0] for s in evidence) else None
    bands = rarity_bands(stats, kind, token) if profile.name == PRODUCTION else (True,) * 3
    provenance = tuple(sorted(stats.provenance().items())) if stats else ()
    if evidence and any(s != evidence[0] for s in evidence[1:]):
        provenance = (("unavailable_reason", "conflicting_lookup_stats"),)
    elif profile.name == PRODUCTION and stats is None:
        provenance = (("unavailable_reason", "lookup_stats_unavailable"),)
    if stats:
        upper = rarity_upper_df(stats, kind, token)
        provenance += (
            ("rarity_df_upper_bound", str(upper) if upper is not None else ""),
            (
                "rarity_proven_bits",
                ",".join(
                    str(bits) for bits, proven in zip((8, 6, 4), bands, strict=True) if proven
                ),
            ),
        )

    def verdict(
        reason: str | None,
        src: str | None = None,
        wit: str | None = None,
        atoms: Mapping[str, tuple[str, ...]] = none,
    ) -> Verdict:
        eligible = reason is None
        return Verdict(
            inv.task,
            (kind, token),
            inv.call_id,
            eligible,
            reason,
            reuse,
            src,
            wit,
            atoms,
            bands,
            provenance,
            action_ok,
        )

    if (why := scan.novelty(inv, token)) is not None:
        return verdict(why)
    if profile.name == PRODUCTION and not any(bands):
        return verdict("rarity_unknown")
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
        "conservative": immediate if action_ok else (),
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
    work_estimates, timing = time_estimates(scan, verdicts, profile.timing_events)
    timing["observed_dispatch_ms"] = observed_timing(scan)[0]
    timing["compare_snapshot"] = profile.compare_time
    unknowns += [
        f"frequency:{reason}:{v.exposure_call}"
        for v in verdicts
        for name, reason in v.provenance
        if name == "unavailable_reason" and reason
    ]
    unknowns += [f"{d.reason}:{d.call}" for d in debits if d.chars is None]
    unknowns += [f"source_atom_unavailable:{v.task}:{v.token[1]}" for v in verdicts
                 if v.reason == "source_atom_unavailable"]  # fmt: skip
    return OracleResult(
        profile.name, f"{profile.name}:{UNIT}:{METHOD}", estimates, tuple(scan.tasks()),
        tuple(debits), tuple(verdicts), _counts(scan, verdicts), tuple(dict.fromkeys(unknowns)),
        f"{profile.name} results are never summed with any other profile",
        work_estimates, timing,
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
            reuse, action, _action_ok = scan.reuse(inv, token, refs)
            if reuse == "action_reference":
                out.setdefault(
                    (inv.task, token),
                    Reviewed(token, kind, inv.task, "exposed_novel_acted", inv.call_id, action),
                )
    return tuple(out.values())


def candidate_vocabulary(frozen: Frozen) -> tuple[tuple[str, str], ...]:
    """Independent vocabulary from accepted, actually visible lookup cards only.

    This supplies explicit closed-world fixture inputs, never production frequency.
    Hidden source text, result metadata and unobserved output cannot populate it.
    """
    scan = scan_frozen(frozen)
    spec = production_profile({})
    return tuple(
        dict.fromkeys(
            token for invocation in _exposing(scan) for token in _tokens_of(invocation, spec, None)
        )
    )


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
    out = compare_estimates(result, savings_json, _find, _same)
    if result.timing.get("compare_snapshot"):
        out.extend(
            compare_estimates(
                result,
                savings_json,
                _find,
                _same,
                estimates=result.time_estimates,
                unit="work_seconds",
            )
        )
    measured = _find(savings_json, "measured")
    for name, key in (("searches", "journaled_search"), ("shows", "journaled_show")):
        if measured is None or name not in measured:
            out.append(f"measured.{name}: missing in snapshot")
        else:
            _same(f"measured.{name}", result.counts[key], measured[name], out)
    for receipt in savings_json.get("lab_receipts", ()):
        out.extend(compare_receipt(result, receipt))
    return out
