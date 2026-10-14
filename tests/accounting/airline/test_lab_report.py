"""Report gates, metric book, valuation separation, privacy, redaction and scans."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest

from eval.accounting_airline.checkpoints import (
    BLOCKED,
    FAIL,
    NOT_RUN,
    PASS,
    UNKNOWN,
    Assertion,
    Checkpoint,
)
from eval.accounting_airline.report import (
    ARITHMETIC_PROFILE,
    BASES,
    FAMILIES,
    GATE_GROUPS,
    PRODUCTION_PROFILE,
    REDACTED,
    GateEntry,
    Metric,
    MetricBook,
    Report,
    assert_no_high_cardinality,
    canary_scan,
    gate_table,
    progress_line,
    redact,
)

CANARY = "CANARY-7f3a91c2-secret"
IDS = {
    "run": "run-20261101-abc123",
    "session": "sess-a1b2c3d4e5",
    "task": "task-L3-0042",
    "source": "src-omp-9f8e7d6c",
    "receipt": "rcpt-55aa66bb",
    "query": "q-deadbeef01",
}


def _metric(**kw) -> Metric:
    base: dict[str, Any] = {
        "family": "fixture_domain",
        "name": "row_count",
        "value": 82,
        "unit": "rows",
        "basis": "observed",
        "source": "database",
        "samples": 1,
        "actor_scope": "verifier",
        "window": "C0",
        "methodology": "count",
    }
    base.update(kw)
    return Metric(**base)


def _standin() -> Report:
    return Report("run-1", "standin", "scripted-standin", False, "scripted-standin")


def _live() -> Report:
    return Report("run-1", "live", "omp", True, "omp")


def _keys(obj, out=None):
    out = set() if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k).lower())
            _keys(v, out)
    elif isinstance(obj, list | tuple):
        for v in obj:
            _keys(v, out)
    return out


# ---------------------------------------------------------------- Metric


def test_metric_rejects_float_value_and_float_samples():
    with pytest.raises(TypeError, match="float"):
        _metric(value=1.5)
    with pytest.raises(TypeError, match="float"):
        _metric(samples=2.0)
    assert _metric(value=3).value == 3
    assert _metric(value="USD", unit="currency").value == "USD"


def test_metric_rejects_bool_value():
    with pytest.raises(TypeError):
        _metric(value=True)


def test_metric_none_value_needs_unavailable_and_reason():
    with pytest.raises(ValueError, match="unavailable"):
        _metric(value=None, basis="observed", reason="missing")
    with pytest.raises(ValueError, match="reason"):
        _metric(value=None, basis="unavailable", samples=None)
    with pytest.raises(ValueError, match="reason"):
        _metric(value=None, basis="unavailable", samples=None, reason="  ")
    ok = _metric(value=None, basis="unavailable", samples=None, reason="runtime reports no usage")
    assert ok.value is None and ok.reason == "runtime reports no usage"


def test_metric_unavailable_cannot_carry_a_value():
    with pytest.raises(ValueError, match="no value"):
        _metric(value=5, basis="unavailable", reason="r")


@pytest.mark.parametrize("basis", ["observed", "reported", "estimated"])
def test_metric_basis_requires_samples(basis):
    with pytest.raises(ValueError, match="sample"):
        _metric(basis=basis, samples=None)
    assert _metric(basis=basis, samples=0).samples == 0


def test_metric_rejects_unknown_family_basis_and_blank_fields():
    with pytest.raises(ValueError, match="family"):
        _metric(family="vibes")
    with pytest.raises(ValueError, match="basis"):
        _metric(basis="guessed")
    for field in ("unit", "source", "actor_scope", "window", "methodology", "name"):
        with pytest.raises(ValueError):
            _metric(**{field: ""})
    with pytest.raises(TypeError):
        _metric(samples=-1)


def test_metric_book_refuses_duplicate_but_allows_distinct_scope_window_methodology():
    book = MetricBook()
    book.add(_metric())
    with pytest.raises(ValueError, match="duplicate"):
        book.add(_metric(value=83))
    book.add(_metric(actor_scope="A"))
    book.add(_metric(window="C1"))
    book.add(_metric(methodology=ARITHMETIC_PROFILE))
    book.add(_metric(name="other"))
    assert len(book) == 5


def test_metric_book_family_and_counts():
    book = MetricBook()
    book.add(_metric())
    book.add(_metric(family="retrieval", name="searches", value=2, basis="reported"))
    book.add(
        _metric(
            family="retrieval",
            name="rank",
            value=None,
            basis="unavailable",
            samples=None,
            reason="not exposed",
        )
    )
    assert [m.name for m in book.family("retrieval")] == ["searches", "rank"]
    with pytest.raises(ValueError):
        book.family("nope")
    counts = book.counts()
    assert list(counts) == list(FAMILIES)
    assert counts["retrieval"] == {"observed": 0, "reported": 1, "estimated": 0, "unavailable": 1}
    assert counts["fixture_domain"]["observed"] == 1
    assert counts["valuation"] == dict.fromkeys(BASES, 0)


def test_require_complete_reports_missing_declared_names():
    book = MetricBook()
    book.add(_metric(name="row_count"))
    book.add(
        _metric(name="digest", value=None, basis="unavailable", samples=None, reason="not computed")
    )
    declared = {"fixture_domain": ["row_count", "digest", "fk_violations"], "retrieval": ["search"]}
    assert book.require_complete(declared) == ["fixture_domain/fk_violations", "retrieval/search"]
    # unavailable-with-reason counts as present; adding the rest empties the list
    book.add(_metric(name="fk_violations", value=0))
    book.add(_metric(family="retrieval", name="search", value=1, basis="reported"))
    assert book.require_complete(declared) == []
    with pytest.raises(ValueError):
        book.require_complete({"made_up": ["x"]})


# ---------------------------------------------------------------- gates


def test_gate_status_precedence_and_empty_is_not_run():
    r = _live()
    assert r.gate_status("l1") == NOT_RUN
    r.add_gate("l1", GateEntry("a", PASS, ""))
    assert r.gate_status("l1") == PASS
    r.add_gate("l1", GateEntry("b", UNKNOWN, ""))
    assert r.gate_status("l1") == UNKNOWN
    r.add_gate("l1", GateEntry("c", NOT_RUN, ""))
    assert r.gate_status("l1") == NOT_RUN
    r.add_gate("l1", GateEntry("d", BLOCKED, ""))
    assert r.gate_status("l1") == BLOCKED
    r.add_gate("l1", GateEntry("e", FAIL, ""))
    assert r.gate_status("l1") == FAIL


def test_gate_rejects_unknown_group_and_status():
    r = _live()
    with pytest.raises(ValueError):
        r.add_gate("l9", GateEntry("a", PASS, ""))
    with pytest.raises(ValueError):
        r.gate_status("l9")
    with pytest.raises(ValueError):
        GateEntry("a", "GREEN", "")


def test_gate_groups_are_independent_and_nothing_hides_blocked_reuse():
    r = _live()
    r.add_gate("l3", GateEntry("plan", PASS, "joint optimum matches"))
    r.add_gate("reuse_evidence", GateEntry("receipt", BLOCKED, "no traceable reuse"))
    r.add_gate("accounting_checkpoints", GateEntry("C1..C10", FAIL, "C6 accounting"))
    assert r.gate_status("l3") == PASS
    assert r.gate_status("reuse_evidence") == BLOCKED
    assert r.gate_status("accounting_checkpoints") == FAIL
    lines = r.summary_lines()
    assert "l3: PASS" in lines
    assert "reuse_evidence: BLOCKED" in lines
    assert "accounting_checkpoints: FAIL" in lines
    assert "live_session_gate: NOT_RUN" in lines
    # one line per group, each independent
    group_lines = [ln for ln in lines if ": " in ln and ln.split(": ")[0] in GATE_GROUPS]
    assert [ln.split(": ")[0] for ln in group_lines] == list(GATE_GROUPS)
    data = r.as_dict()
    assert data["gates"]["l3"]["status"] == PASS
    assert data["gates"]["reuse_evidence"]["status"] == BLOCKED


def test_no_aggregate_success_key_anywhere():
    r = _live()
    r.add_gate("l1", GateEntry("x", PASS, ""))
    r.metrics.add(_metric())
    r.valuation(PRODUCTION_PROFILE, {"profile": PRODUCTION_PROFILE, "eligible": 0})
    for blob in (r.as_dict(), r.public_summary()):
        found = _keys(blob) & {"overall", "all_passed", "success", "ok"}
        assert found == set()
    # twin: the walker would notice such a key if one existed
    assert _keys({"gates": [{"Overall": 1}]}) & {"overall"} == {"overall"}
    assert not any(ln.lower().startswith(("overall", "success", "ok")) for ln in r.summary_lines())


def test_valuation_payload_with_aggregate_keys_is_refused():
    r = _live()
    with pytest.raises(ValueError, match="aggregate"):
        r.valuation(PRODUCTION_PROFILE, {"profile": PRODUCTION_PROFILE, "detail": {"success": 1}})
    with pytest.raises(TypeError, match="float"):
        r.valuation(PRODUCTION_PROFILE, {"profile": PRODUCTION_PROFILE, "net": [1, 2.5]})
    # refused payloads are not recorded; a clean one still goes in
    r.valuation(PRODUCTION_PROFILE, {"profile": PRODUCTION_PROFILE, "net": 0})


def test_gate_table_lists_every_group_with_independent_status():
    r = _live()
    r.add_gate("l3", GateEntry("plan", PASS, ""))
    r.add_gate("reuse_evidence", GateEntry("receipt", BLOCKED, "no traceable reuse"))
    rows = gate_table(r).splitlines()
    assert len(rows) == 2 + len(GATE_GROUPS)
    by_group = {row.split("|")[0].strip(): row for row in rows[2:]}
    assert list(by_group) == list(GATE_GROUPS)
    assert "PASS" in by_group["l3"]
    assert (
        "BLOCKED" in by_group["reuse_evidence"]
        and "no traceable reuse" in by_group["reuse_evidence"]
    )
    assert "NOT_RUN" in by_group["cleanup"]


# ---------------------------------------------------------------- stand-in labeling


def test_standin_report_is_labeled_and_cannot_count_toward_live_gate():
    r = _standin()
    assert r.counts_toward_live_gate is False
    assert r.gate_status("live_session_gate") == NOT_RUN
    entry = r.gates["live_session_gate"][0]
    assert entry.detail == "scripted stand-in never counts toward the live gate"
    data = r.as_dict()
    assert data["runtime"] == "scripted-standin"
    assert data["scripted_standin"] is True
    assert data["runtime_label"] == "scripted-standin"
    assert data["counts_toward_live_gate"] is False
    assert r.public_summary()["scripted_standin"] is True
    assert r.public_summary()["gates"]["live_session_gate"] == NOT_RUN
    assert r.summary_lines()[0] == (
        "mode=standin runtime=scripted-standin counts_toward_live_gate=False"
    )
    # passing deterministic gates cannot touch the live gate
    r.add_gate("l1", GateEntry("x", PASS, ""))
    assert r.gate_status("live_session_gate") == NOT_RUN


def test_standin_can_never_pass_the_live_gate_and_cannot_claim_to_count():
    r = _standin()
    with pytest.raises(ValueError, match="live gate"):
        r.add_gate("live_session_gate", GateEntry("x", PASS, ""))
    assert r.gate_status("live_session_gate") == NOT_RUN
    with pytest.raises(ValueError, match="stand-in"):
        Report("r", "standin", "scripted-standin", True, "scripted-standin")


def test_live_report_has_no_preset_live_gate_and_can_pass_it():
    r = _live()
    assert r.gate_status("live_session_gate") == NOT_RUN  # empty, no reason entry
    assert r.gates["live_session_gate"] == []
    d = r.as_dict()
    assert d["scripted_standin"] is False and d["runtime_label"] == "omp"
    assert d["counts_toward_live_gate"] is True
    r.add_gate("live_session_gate", GateEntry("two sessions", PASS, "A and B ran"))
    assert r.gate_status("live_session_gate") == PASS


def test_non_counting_live_style_run_cannot_pass_live_gate():
    r = Report("r", "deterministic", "omp", False, "omp")
    with pytest.raises(ValueError, match="live gate"):
        r.add_gate("live_session_gate", GateEntry("x", PASS, ""))
    r.add_gate("live_session_gate", GateEntry("x", BLOCKED, "no authorization"))
    assert r.gate_status("live_session_gate") == BLOCKED


# ---------------------------------------------------------------- valuation


def test_valuation_separate_keys_and_no_combined_total():
    r = _live()
    r.valuation(PRODUCTION_PROFILE, {"profile": PRODUCTION_PROFILE, "net_tokens": -10})
    r.valuation(ARITHMETIC_PROFILE, {"profile": ARITHMETIC_PROFILE, "net_tokens": 5})
    data = r.as_dict()
    assert data["production"]["net_tokens"] == -10
    assert data["arithmetic_fixture"]["net_tokens"] == 5
    for combined in ("valuation", "total", "net_tokens", "combined"):
        assert combined not in data
    assert data["production"]["profile"] != data["arithmetic_fixture"]["profile"]
    assert r.public_summary()["valuation_present"] == {
        "production": True,
        "arithmetic_fixture": True,
    }


def test_valuation_absent_profile_is_none_not_zero():
    r = _live()
    r.valuation(PRODUCTION_PROFILE, {"profile": PRODUCTION_PROFILE, "net_tokens": 0})
    data = r.as_dict()
    assert data["production"] == {"profile": PRODUCTION_PROFILE, "net_tokens": 0}
    assert data["arithmetic_fixture"] is None
    assert r.public_summary()["valuation_present"] == {
        "production": True,
        "arithmetic_fixture": False,
    }


def test_valuation_refuses_duplicate_unknown_and_mislabeled_profile():
    r = _live()
    r.valuation(PRODUCTION_PROFILE, {"profile": PRODUCTION_PROFILE, "n": 1})
    with pytest.raises(ValueError, match="already recorded"):
        r.valuation(PRODUCTION_PROFILE, {"profile": PRODUCTION_PROFILE, "n": 2})
    assert r.as_dict()["production"]["n"] == 1
    with pytest.raises(ValueError, match="unknown valuation profile"):
        r.valuation("calibrated-roi", {"profile": "calibrated-roi"})
    with pytest.raises(ValueError, match="not a known profile"):
        r.valuation(ARITHMETIC_PROFILE, {"profile": "something-else"})
    with pytest.raises(ValueError, match="not a known profile"):
        r.valuation(ARITHMETIC_PROFILE, {"n": 1})
    # an arithmetic payload cannot be filed under the production profile, or vice versa
    with pytest.raises(ValueError, match="cannot be filed"):
        r.valuation(ARITHMETIC_PROFILE, {"profile": PRODUCTION_PROFILE})
    r.valuation(ARITHMETIC_PROFILE, {"profile": ARITHMETIC_PROFILE, "n": 3})
    assert r.as_dict()["arithmetic_fixture"]["n"] == 3


# ---------------------------------------------------------------- private / public


def _report_with_ids() -> Report:
    r = _live()
    r.run_id = IDS["run"]
    cp = Checkpoint(
        cid="C5",
        name="each B search/show",
        domain=[Assertion("source", PASS, IDS["source"], IDS["source"], "show")],
        accounting=[Assertion("receipt", PASS, IDS["receipt"], IDS["receipt"], "journal")],
        runtime_ids={"A": IDS["session"]},
        task_ids=[IDS["task"]],
        events_by_actor={"A": 3},
    )
    r.checkpoints.add(cp)
    r.add_gate(
        "restart_retrieval",
        GateEntry(f"query {IDS['query']}", PASS, f"opened {IDS['source']}", artifact="/lab/a.json"),
    )
    r.metrics.add(_metric(actor_scope=IDS["session"]))
    r.valuation(PRODUCTION_PROFILE, {"profile": PRODUCTION_PROFILE, "receipt": IDS["receipt"]})
    return r


def test_private_report_is_0600_and_contains_ids(tmp_path: Path):
    r = _report_with_ids()
    path = tmp_path / "report.json"
    path.write_text("stale")
    path.chmod(0o644)
    r.write_private(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    text = path.read_text()
    assert json.loads(text)["run_id"] == IDS["run"]
    # positive twin of the public check: every id is present in the private detail
    for name, ident in IDS.items():
        assert ident in text, name


def test_public_summary_contains_no_ids():
    r = _report_with_ids()
    summary = r.public_summary()
    rendered = json.dumps(summary)
    for name, ident in IDS.items():
        assert ident not in rendered, name
    assert "run_id" not in summary
    assert summary["gates"]["restart_retrieval"] == PASS
    assert summary["checkpoints"]["C5"] == {"domain": PASS, "accounting": PASS}
    assert summary["checkpoints"]["C6"] == {"domain": NOT_RUN, "accounting": NOT_RUN}
    assert summary["metric_counts"]["fixture_domain"]["observed"] == 1
    # the private dict carries them, so the absence above is meaningful
    private = json.dumps(r.as_dict())
    assert all(ident in private for ident in IDS.values())
    # summary lines are ids free too
    assert not any(ident in "\n".join(r.summary_lines()) for ident in IDS.values())
    assert not any(ident in gate_table(r) for ident in IDS.values() if ident != IDS["query"])


# ---------------------------------------------------------------- populated report


def _full_metrics() -> list[Metric]:
    out = []
    for i, fam in enumerate(FAMILIES):
        out.append(
            Metric(
                family=fam,
                name=f"m_{fam}",
                value=i + 1,
                unit="count",
                basis=BASES[i % 3],
                source=f"src-{fam}",
                samples=i + 1,
                actor_scope="A+B",
                window="run",
                methodology="production-policy",
            )
        )
        out.append(
            Metric(
                family=fam,
                name=f"missing_{fam}",
                value=None,
                unit="count",
                basis="unavailable",
                source="runtime",
                samples=None,
                actor_scope="A+B",
                window="run",
                methodology="production-policy",
                reason="runtime does not report it",
            )
        )
    return out


def test_fully_populated_report_every_metric_is_fully_described():
    r = _live()
    for m in _full_metrics():
        r.metrics.add(m)
    declared = {fam: [f"m_{fam}", f"missing_{fam}"] for fam in FAMILIES}
    assert r.metrics.require_complete(declared) == []
    data = r.as_dict()["metrics"]
    assert list(data) == list(FAMILIES)
    for fam in FAMILIES:
        assert len(data[fam]) == 2
        for item in data[fam]:
            for key in ("unit", "basis", "source", "actor_scope", "window", "methodology"):
                assert item[key], (fam, key)
            assert item["basis"] in BASES
            if item["basis"] == "unavailable":
                assert item["value"] is None and item["reason"]
            else:
                assert item["value"] is not None and item["samples"] is not None
                assert item["reason"] is None
            assert not isinstance(item["value"], float)
    counts = r.public_summary()["metric_counts"]
    assert sum(sum(v.values()) for v in counts.values()) == 2 * len(FAMILIES)
    assert all(c["unavailable"] == 1 for c in counts.values())


def test_ten_families_and_eighteen_gate_groups():
    assert len(FAMILIES) == 10 and len(set(FAMILIES)) == 10
    assert len(GATE_GROUPS) == 18 and len(set(GATE_GROUPS)) == 18
    assert BASES == ("observed", "reported", "estimated", "unavailable")


# ---------------------------------------------------------------- progress / redaction


def _cp_with_canary() -> Checkpoint:
    return Checkpoint(
        cid="C6",
        name="B L3 validation",
        domain=[
            Assertion(
                "plan",
                PASS,
                "B1->F5",
                "B1->F5",
                "oracle",
                detail=f"transcript said: the answer is hidden {CANARY}",
            )
        ],
        accounting=[Assertion("credit", FAIL, 0, 1, "oracle", detail=f"raw sql {CANARY}")],
    )


def test_progress_line_format_and_fields():
    cp = _cp_with_canary()
    line = progress_line(
        cp,
        elapsed_s=42,
        artifact="/lab/artifacts/c6.json",
        next_action="start L4",
        ts="2026-11-01T07:00:00Z",
    )
    assert line == (
        "[2026-11-01T07:00:00Z] C6 B L3 validation domain=PASS accounting=FAIL elapsed=42s "
        "artifact=/lab/artifacts/c6.json next=start L4"
    )
    assert "\n" not in line


def test_progress_line_default_timestamp_and_single_line():
    cp = Checkpoint(cid="C1", name="A L1 completes")
    line = progress_line(cp, elapsed_s=0, artifact=None, next_action="wait\nfor exit")
    assert line.startswith("[") and "\n" not in line
    assert "artifact=-" in line and "next=wait for exit" in line
    assert "domain=UNKNOWN accounting=UNKNOWN" in line


def test_progress_line_redacts_supplied_canary_and_never_prints_assertion_text():
    cp = _cp_with_canary()
    leaky = progress_line(
        cp, elapsed_s=1, artifact=f"/lab/{CANARY}/c6.json", next_action=f"use {CANARY}"
    )
    assert CANARY in leaky  # positive twin: without redactions the planted value does leak
    safe = progress_line(
        cp,
        elapsed_s=1,
        artifact=f"/lab/{CANARY}/c6.json",
        next_action=f"use {CANARY}",
        redactions=[CANARY],
    )
    assert CANARY not in safe
    assert REDACTED in safe
    # assertion details/transcript text never enter the line, even without redaction
    plain = progress_line(cp, elapsed_s=1, artifact=None, next_action="go")
    assert CANARY not in plain and "transcript" not in plain and "raw sql" not in plain


@pytest.mark.parametrize(
    "secret",
    [
        "sk-abcdef1234567890XYZ",
        "Bearer eyJhbGciOi.payload-part_1.sig",
        "PGPASSWORD=hunter2",
        "password=hunter2",
        "password: hunter2",
        "token " + "a1" * 20,
        "api_key=" + "0f" * 20,
    ],
)
def test_redact_generic_secret_shapes(secret):
    out = redact(f"before {secret} after")
    assert REDACTED in out
    assert out.startswith("before ")
    for fragment in ("sk-abcdef", "eyJhbGciOi", "hunter2", "a1a1a1a1", "0f0f0f0f"):
        assert fragment not in out


def test_redact_leaves_ordinary_text_and_short_hex_alone():
    text = "C3 checkpoint ok, key 0f0f short, token abc, sk-1 pass word"
    assert redact(text) == text
    digest = "a" * 64
    assert redact(f"sha256 {digest}") == f"sha256 {digest}"  # not preceded by token/key


def test_redact_longest_secret_first_and_ignores_empty():
    out = redact("abc abcdef", ["abc", "abcdef", ""])
    assert out == f"{REDACTED} {REDACTED}"
    assert redact("plain", [""]) == "plain"


def test_progress_line_redacts_generic_shapes_in_fields():
    cp = Checkpoint(cid="C9", name="scrape/export")
    line = progress_line(
        cp,
        elapsed_s=3,
        artifact="/a/b.json",
        next_action="retry PGPASSWORD=s3cr3t with Bearer abc.def",
    )
    assert "s3cr3t" not in line and "abc.def" not in line
    assert line.count(REDACTED) == 2


def test_progress_line_rejects_float_elapsed():
    cp = Checkpoint(cid="C1", name="x")
    with pytest.raises(TypeError):
        progress_line(cp, elapsed_s=1.5, artifact=None, next_action="n")  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------- scans


def test_canary_scan_finds_planted_canary_and_clean_text_is_empty():
    texts = {
        "report.json": f'{{"detail": "{CANARY}"}}',
        "progress.log": "C1 A L1 completes domain=PASS",
        "scrape.txt": f"x {CANARY} y",
    }
    found = canary_scan(texts, [CANARY])
    assert found == ["report.json: canary found", "scrape.txt: canary found"]
    assert all(CANARY not in f for f in found)
    assert canary_scan({k: v for k, v in texts.items() if k == "progress.log"}, [CANARY]) == []
    assert canary_scan({}, [CANARY]) == []
    with pytest.raises(ValueError):
        canary_scan(texts, [""])


def test_report_public_summary_and_table_pass_canary_scan():
    r = _report_with_ids()
    r.add_gate("export_privacy", GateEntry("labels", PASS, "no canary"))
    texts = {
        "public": json.dumps(r.public_summary()),
        "gate_table": gate_table(r),
        "summary": "\n".join(r.summary_lines()),
        "progress": progress_line(
            _cp_with_canary(), elapsed_s=1, artifact=None, next_action="n", redactions=[CANARY]
        ),
    }
    assert canary_scan(texts, [CANARY]) == []
    texts["private-with-leak"] = json.dumps({"detail": CANARY})
    assert canary_scan(texts, [CANARY]) == ["private-with-leak: canary found"]


OPENMETRICS_CLEAN = (
    "# TYPE ssgrep_lookups counter\n"
    "# HELP ssgrep_lookups lookups by kind\n"
    'ssgrep_lookups_total{kind="search",runtime="omp"} 3\n'
    "# EOF\n"
)


def test_assert_no_high_cardinality_passes_clean_body_and_catches_label_ids():
    ids = list(IDS.values())
    assert assert_no_high_cardinality(OPENMETRICS_CLEAN, ids) == []
    dirty = OPENMETRICS_CLEAN.replace('kind="search"', f'kind="search",session="{IDS["session"]}"')
    violations = assert_no_high_cardinality(dirty, ids)
    assert len(violations) == 1
    assert "line 3" in violations[0] and "label" in violations[0]
    assert IDS["session"] not in violations[0]  # the violation does not echo the id


def test_assert_no_high_cardinality_catches_metadata_and_sample_positions():
    body = (
        f"# HELP m about {IDS['task']}\n"
        f"m_total 1 # {IDS['source']}\n"
        f'm_total{{a="{IDS["receipt"]}",b="{IDS["query"]}"}} 2\n'
        f"{IDS['run']}_total 1\n"
    )
    violations = assert_no_high_cardinality(body, IDS.values())
    kinds = [v.rsplit(" in ", 1)[1] for v in violations]
    assert kinds == ["metadata", "sample", "label", "label", "sample"]
    with pytest.raises(ValueError):
        assert_no_high_cardinality(body, [""])
