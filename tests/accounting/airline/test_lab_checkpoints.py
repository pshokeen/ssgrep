"""Checkpoint status logic, assertion labels, the 11-row table, and prefix digests."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from eval.accounting_airline.checkpoints import (
    BLOCKED,
    DEFINITIONS,
    FAIL,
    NOT_RUN,
    PASS,
    STATUSES,
    UNKNOWN,
    Assertion,
    Checkpoint,
    CheckpointLog,
    combine,
    input_prefix,
)


def _a(status: str, name: str = "a", **kw) -> Assertion:
    return Assertion(name, status, kw.pop("expected", 1), kw.pop("actual", 1), "oracle", **kw)


def _cp(cid: str = "C1", domain=(), accounting=(), **kw) -> Checkpoint:
    return Checkpoint(
        cid=cid, name=DEFINITIONS[cid].name, domain=list(domain), accounting=list(accounting), **kw
    )


# ---------------------------------------------------------------- combine


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([PASS, PASS], PASS),
        ([PASS, UNKNOWN], UNKNOWN),
        ([PASS, UNKNOWN, BLOCKED], BLOCKED),
        ([PASS, UNKNOWN, BLOCKED, FAIL], FAIL),
        ([BLOCKED, FAIL], FAIL),
        ([UNKNOWN, BLOCKED], BLOCKED),
        ([FAIL], FAIL),
        ([PASS], PASS),
    ],
)
def test_combine_precedence(statuses, expected):
    assert combine(statuses) == expected
    assert combine(reversed(statuses)) == expected


def test_combine_empty_is_unknown_never_pass():
    assert combine([]) == UNKNOWN
    assert combine(iter(())) == UNKNOWN


def test_combine_rejects_unknown_status_strings():
    with pytest.raises(ValueError, match="unknown checkpoint status"):
        combine([PASS, "MAYBE"])
    with pytest.raises(ValueError):
        combine([NOT_RUN])
    assert STATUSES == (PASS, FAIL, BLOCKED, UNKNOWN)


def test_assertion_rejects_bad_status():
    with pytest.raises(ValueError):
        Assertion("x", "GREEN", 1, 1, "m")
    assert Assertion("x", PASS, 1, 1, "m").status == PASS


# ---------------------------------------------------------------- checkpoint statuses


def test_empty_checkpoint_is_unknown_and_populated_twin_is_pass():
    empty = _cp()
    assert (empty.domain_status, empty.accounting_status, empty.status) == (
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
    )
    full = _cp(domain=[_a(PASS)], accounting=[_a(PASS, "b")])
    assert (full.domain_status, full.accounting_status, full.status) == (PASS, PASS, PASS)


def test_one_side_empty_keeps_checkpoint_unknown():
    cp = _cp(domain=[_a(PASS)])
    assert cp.domain_status == PASS
    assert cp.accounting_status == UNKNOWN
    assert cp.status == UNKNOWN


def test_domain_fail_and_accounting_pass_stay_separately_visible():
    cp = _cp(domain=[_a(FAIL, "rows")], accounting=[_a(PASS, "events")])
    assert cp.domain_status == FAIL
    assert cp.accounting_status == PASS
    assert cp.status == FAIL
    data = cp.as_dict()
    assert data["domain_status"] == FAIL and data["accounting_status"] == PASS
    # the reverse split also stays split
    rev = _cp(domain=[_a(PASS)], accounting=[_a(BLOCKED, "journal")])
    assert (rev.domain_status, rev.accounting_status, rev.status) == (PASS, BLOCKED, BLOCKED)


def test_assertion_counts_cover_both_lists():
    cp = _cp(
        domain=[_a(PASS, "d1"), _a(FAIL, "d2")],
        accounting=[_a(BLOCKED, "a1"), _a(UNKNOWN, "a2"), _a(PASS, "a3")],
    )
    assert cp.counts() == {PASS: 2, FAIL: 1, BLOCKED: 1, UNKNOWN: 1}


def test_checkpoint_rejects_unknown_id_and_float_elapsed():
    with pytest.raises(ValueError, match="unknown checkpoint id"):
        Checkpoint(cid="C11", name="x")
    with pytest.raises(TypeError):
        Checkpoint(cid="C1", name="x", elapsed_ms=1.5)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError):
        Checkpoint(cid="C1", name="x", events_by_actor={"A": 1.0})  # ty: ignore[invalid-argument-type]
    assert Checkpoint(cid="C1", name="x", elapsed_ms=1500, events_by_actor={"A": 3}).elapsed_ms


# ---------------------------------------------------------------- intentional negatives


def test_intentional_negative_detected_is_pass_and_labeled():
    a = Assertion.negative(
        "fanout-detected",
        detected=True,
        expected="B1!=180000",
        actual="B1=180000 flagged",
        method="mutation",
    )
    assert a.status == PASS
    assert a.intentional_negative is True
    assert "INTENTIONAL NEGATIVE" in a.line()
    assert "expected failure detected" in a.line()
    cp = _cp(domain=[a], accounting=[_a(PASS)])
    assert cp.domain_status == PASS
    assert any("INTENTIONAL NEGATIVE" in line for line in cp.assertion_lines())
    assert cp.as_dict()["domain"][0]["intentional_negative"] is True


def test_intentional_negative_not_detected_is_fail():
    a = Assertion.negative(
        "fanout-detected", detected=False, expected="flagged", actual="not flagged", method="m"
    )
    assert a.status == FAIL
    assert "NOT detected" in a.label
    cp = _cp(domain=[a], accounting=[_a(PASS)])
    assert cp.domain_status == FAIL
    assert cp.status == FAIL


def test_ordinary_assertion_has_no_negative_label():
    a = _a(PASS)
    assert a.label == ""
    assert "INTENTIONAL" not in a.line()


# ---------------------------------------------------------------- table


def test_table_always_has_eleven_rows_with_not_run_for_missing():
    log = CheckpointLog()
    lines = log.table().splitlines()
    assert lines[0].split("|")[0].strip() == "CP"
    header = [c.strip() for c in lines[0].split("|")]
    assert header == [
        "CP",
        "name",
        "domain",
        "accounting",
        "assertions p/f/b/u",
        "elapsed",
        "next",
    ]
    rows = lines[2:]
    assert len(rows) == 11
    assert [r.split("|")[0].strip() for r in rows] == [f"C{i}" for i in range(11)]
    assert all(NOT_RUN in r for r in rows)

    log.add(
        _cp(
            "C10",
            domain=[_a(PASS)],
            accounting=[_a(FAIL, "x")],
            elapsed_ms=2345,
            next_action="stop owned resources",
        )
    )
    log.add(_cp("C2", domain=[_a(PASS)], accounting=[_a(PASS, "y")], elapsed_ms=10))
    rows = log.table().splitlines()[2:]
    assert len(rows) == 11
    # numeric ordering: C2 before C10 and C10 last
    assert [r.split("|")[0].strip() for r in rows] == [f"C{i}" for i in range(11)]
    c10 = rows[10]
    assert "FAIL" in c10 and "1/1/0/0" in c10 and "2.345s" in c10 and "stop owned resources" in c10
    c2 = rows[2]
    assert "PASS" in c2 and "0.010s" in c2 and NOT_RUN not in c2
    assert NOT_RUN in rows[3]


def test_table_is_fixed_width():
    log = CheckpointLog()
    log.add(_cp("C1", domain=[_a(PASS)], accounting=[_a(PASS, "b")], next_action="go on"))
    lines = log.table().splitlines()
    ruler = lines[1]
    seps = [i for i, ch in enumerate(ruler) if ch == "+"]
    for line in (lines[0], *lines[2:]):
        assert [i for i, ch in enumerate(line) if ch == "|"] == seps


def test_log_rejects_duplicate_and_unknown_checkpoint():
    log = CheckpointLog()
    log.add(_cp("C1"))
    assert log.get("C1").cid == "C1"
    with pytest.raises(ValueError, match="already recorded"):
        log.add(_cp("C1"))
    with pytest.raises(KeyError):
        log.get("C2")


def test_as_dict_lists_all_eleven_with_definitions_and_not_run():
    log = CheckpointLog()
    log.add(_cp("C0", domain=[_a(PASS)], accounting=[_a(PASS, "b")]))
    data = log.as_dict()
    assert list(data) == [f"C{i}" for i in range(11)]
    assert data["C0"]["status"] == PASS
    assert data["C1"]["status"] == NOT_RUN
    for cid, item in data.items():
        assert item["definition"]["domain"] == DEFINITIONS[cid].domain
        assert item["definition"]["accounting"] == DEFINITIONS[cid].accounting
    json.dumps(data)


def test_write_is_0600_json_and_tightens_existing_file(tmp_path: Path):
    log = CheckpointLog()
    log.add(_cp("C3", domain=[_a(PASS)], accounting=[_a(FAIL, "z")]))
    path = tmp_path / "cp.json"
    path.write_text("old")
    path.chmod(0o644)
    log.write(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    loaded = json.loads(path.read_text())
    assert loaded["C3"]["domain_status"] == PASS
    assert loaded["C3"]["accounting_status"] == FAIL
    fresh = tmp_path / "new.json"
    log.write(fresh)
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o600


# ---------------------------------------------------------------- definitions


def test_definitions_cover_c0_to_c10_with_section5_text():
    assert list(DEFINITIONS) == [f"C{i}" for i in range(11)]
    assert DEFINITIONS["C0"].name == "seed/setup"
    assert "12 table counts, 82 rows" in DEFINITIONS["C0"].domain
    assert "export off" in DEFINITIONS["C0"].accounting
    assert "35/45-minute" in DEFINITIONS["C2"].domain
    assert "repeat reads cannot add credits" in DEFINITIONS["C5"].accounting
    assert "privacy-safe report" in DEFINITIONS["C10"].accounting
    for d in DEFINITIONS.values():
        assert d.domain and d.accounting


def test_definition_text_matches_the_spec_table():
    spec = Path("openspec/changes/add-savings-ledger/validation-lab.md").read_text()
    for cid, d in DEFINITIONS.items():
        row = next(line for line in spec.splitlines() if line.startswith(f"| {cid}:"))
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        assert cells[0].startswith(f"{cid}: ")
        assert d.domain == cells[1]
        assert d.accounting == cells[2]


# ---------------------------------------------------------------- input_prefix


def test_input_prefix_defaults_to_newline_complete_prefix(tmp_path: Path):
    f = tmp_path / "s.jsonl"
    f.write_bytes(b'{"a":1}\n{"b":2}\n{"par')
    complete = b'{"a":1}\n{"b":2}\n'
    assert input_prefix(f) == f"{hashlib.sha256(complete).hexdigest()}@{len(complete)}"
    # appending a partial tail does not change the digest; finishing the line does
    before = input_prefix(f)
    with f.open("ab") as fh:
        fh.write(b"tial")
    assert input_prefix(f) == before
    with f.open("ab") as fh:
        fh.write(b'"}\n')
    assert input_prefix(f) != before


def test_input_prefix_explicit_bytes_and_bounds(tmp_path: Path):
    f = tmp_path / "s.jsonl"
    f.write_bytes(b"one\ntwo\nthree")
    first = b"one\n"
    assert input_prefix(f, 4) == f"{hashlib.sha256(first).hexdigest()}@4"
    assert input_prefix(f, 0) == f"{hashlib.sha256(b'').hexdigest()}@0"
    with pytest.raises(ValueError, match="shorter"):
        input_prefix(f, 999)
    with pytest.raises(ValueError):
        input_prefix(f, -1)


def test_input_prefix_without_any_newline_is_empty_prefix(tmp_path: Path):
    f = tmp_path / "s.jsonl"
    f.write_bytes(b"no newline yet")
    assert input_prefix(f) == f"{hashlib.sha256(b'').hexdigest()}@0"


def test_input_prefix_spans_chunk_boundaries(tmp_path: Path):
    f = tmp_path / "big.jsonl"
    body = (b"x" * 99 + b"\n") * 20_000  # ~2 MB: more than one read chunk
    f.write_bytes(body + b"tail")
    assert input_prefix(f) == f"{hashlib.sha256(body).hexdigest()}@{len(body)}"
