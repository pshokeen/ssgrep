"""8.5/8.6: ``savings explain`` and ``savings history`` over REAL persisted receipts/revisions.

The dataset is produced by the real collector at three clock points: day 12 (two consumers
reuse ONE fact source: overlap), day 22 (late evidence for the sealed day 12: a superseding
revision), day 57 (retention: the old roots' evidence expires). Every assertion reads what the
real binary and the facade show from the files the collector wrote.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from ssgrep.accounting import service
from tests.accounting import collector_scene as scene, verify_scenes as vs
from tests.accounting.reduce_builders import source_session
from tests.accounting.test_reduce_evidence import FACT, VIEW, consumer
from tests.accounting.test_ux_support import Binary, fingerprint

DAY12 = scene.at(hour=20)
DAY22 = DAY12 + timedelta(days=10)
DAY57 = DAY12 + timedelta(days=45)


def renamed(session: str, records: list[dict]) -> list[dict]:
    return [{**r, "sessionId": session} if "sessionId" in r else dict(r) for r in records]


def put(root: Path, name: str, records: list[dict]) -> None:
    vs.put(vs.projects(root), name, renamed(name, records), scene.at(hour=10))


def start(root: Path) -> None:
    vs.enable(pin=VIEW)
    vs.start_capture(day=11)
    put(root, "src1", source_session("src1", FACT, when_day=11))
    put(root, "first", consumer())
    put(root, "second", consumer())  # a second consumer of the SAME source fact


def receipts_dir(root: Path) -> Path:
    return root / "data" / "accounting" / "personal-local" / "receipts"


def receipt_ids(root: Path) -> list[str]:
    return sorted(p.stem for p in receipts_dir(root).glob("*.json"))


@pytest.fixture
def overlap(tmp_path: Path) -> Path:
    start(tmp_path)
    assert scene.collect(DAY12).state == "published"
    return tmp_path


def test_two_consumers_of_one_source_are_each_explained_and_credited_per_task(
    overlap: Path,
) -> None:
    run = Binary(overlap)
    ids = receipt_ids(overlap)
    assert len(ids) == 2
    views = [service.read_receipt("personal-local", i) for i in ids]
    assert all(v is not None and v.revision == 1 for v in views)
    credits = {v.root: dict(v.credit_raw)["central"] for v in views if v is not None}
    assert set(credits) == {"claude:first", "claude:second"}
    # overlap is removed WITHIN a task; two separate tasks that reused the same source are each
    # credited (design D6), so the all-task credit is the sum of the two task credits
    first = run.data("savings", "--task", "claude:first#turn:u1")["data"]["estimates"]
    second = run.data("savings", "--task", "claude:second#turn:u1")["data"]["estimates"]
    both = run.data("savings")["data"]["estimates"]
    cells = [x["context_tokens"]["central"]["credit"]["value"] for x in (first, second, both)]
    assert cells[0] == cells[1] and cells[0] > 0
    assert cells[2] == cells[0] + cells[1]
    for receipt_id in ids:
        text = " ".join(run("savings", "explain", receipt_id).stdout.split())
        assert "Overlap removed within this task" in text
        assert "association, not proof of cause" in text


def test_explain_shows_costs_exclusions_provenance_and_hides_content_by_default(
    overlap: Path,
) -> None:
    run = Binary(overlap)
    receipt_id = receipt_ids(overlap)[0]
    plain = run("savings", "explain", receipt_id, columns=80)
    assert plain.returncode == 0
    for heading in (
        "CREDIT BY SCENARIO",
        "COST OF THE LOOKUP",
        "NOT CREDITED",
        "EVIDENCE AVAILABILITY",
        "PROVENANCE",
    ):
        assert heading in plain.stdout
    assert "Private content is hidden" in plain.stdout
    stored = json.loads((receipts_dir(overlap) / f"{receipt_id}.json").read_text())
    assert "content" not in stored, "the collector never persists private content"
    assert stored["schema"] == 1 and set(stored["identities"]) == {
        "frequency",
        "tokenizer",
        "pricing",
    }
    data = run.data("savings", "explain", receipt_id)["data"]
    assert data["content"] is None and data["schema"] == 1 and data["unit"]
    assert data["credit_raw"]["central"] >= data["credit_raw"]["conservative"]
    assert data["recomputation_available"] is True
    assert set(data["identities"]) == {"frequency", "pricing", "tokenizer"}


def test_include_content_is_explicit_local_and_never_invented(overlap: Path) -> None:
    run = Binary(overlap)
    receipt_id = receipt_ids(overlap)[0]
    path = receipts_dir(overlap) / f"{receipt_id}.json"
    shown = run("savings", "explain", receipt_id, "--include-content")
    assert (
        "Nothing to show" in " ".join(shown.stdout.split()) and "no private content" in shown.stdout
    )
    document = json.loads(path.read_text())
    document["content"] = "LOCAL-ONLY-CANARY-7731"
    path.write_text(json.dumps(document))
    path.chmod(0o600)
    assert "LOCAL-ONLY-CANARY-7731" not in run("savings", "explain", receipt_id).stdout
    assert "LOCAL-ONLY-CANARY-7731" not in run("savings", "explain", receipt_id, "--json").stdout
    assert (
        "LOCAL-ONLY-CANARY-7731"
        in run("savings", "explain", receipt_id, "--include-content").stdout
    )
    with_json = run.data("savings", "explain", receipt_id, "--include-content")["data"]
    assert with_json["content"] == "LOCAL-ONLY-CANARY-7731"
    for name in ("snapshots", "history"):  # content never leaks into anything exportable
        for file in (overlap / "data" / "accounting" / "personal-local" / name).rglob("*"):
            if file.is_file():
                assert b"LOCAL-ONLY-CANARY-7731" not in file.read_bytes()


def test_unknown_ids_exit_3_in_text_and_json_and_never_look_like_a_zero_receipt(
    overlap: Path,
) -> None:
    run = Binary(overlap)
    for bad in ("0" * 24, "unknown-id", "..%2f..%2fetc", "a" * 65):
        done = run("savings", "explain", bad)
        assert done.returncode == 3 and done.stdout == "" and "No receipt" in done.stderr, bad
        code, body = run.json("savings", "explain", bad)
        assert code == 3 and body["condition"] == "receipt_not_found" and body["ok"] is False


def test_late_evidence_for_a_sealed_day_makes_a_superseding_revision_history_never_double_counts(
    overlap: Path,
) -> None:
    run = Binary(overlap)
    before = run.data("savings", "history")["data"]
    assert [(e["period"], e["revision"], e["current"]) for e in before["entries"]] == [
        ("2026-10-12", 1, True)
    ]
    put(overlap, "late", consumer())  # a third consumer of that day, arriving after the seal
    assert scene.collect(DAY22).state == "published"
    history = run.data("savings", "history")["data"]
    entries = history["entries"]
    revisions = [(e["revision"], e["current"], e["supersedes"], e["kind"]) for e in entries]
    assert revisions == [(1, False, None, "original"), (2, True, 1, "adjustment")]
    original, replacement = entries[0]["net"]["value"], entries[1]["net"]["value"]
    assert history["current_total"]["value"] == replacement, "only the applicable revision counts"
    assert history["current_total"]["value"] != original + replacement
    text = " ".join(run("savings", "history", columns=40).stdout.split())
    assert "superseded" in text and "adjustment" in text and "Applicable revisions only" in text
    ids = receipt_ids(overlap)
    latest = [service.read_receipt("personal-local", i) for i in ids]
    assert all(v is not None for v in latest)


def test_history_pages_are_bound_to_the_data_and_reject_a_moved_cursor(overlap: Path) -> None:
    run = Binary(overlap)
    put(overlap, "late", consumer())
    assert scene.collect(DAY22).state == "published"
    first = run.data("savings", "history", "--limit", "1")["data"]
    cursor = first["next_cursor"]
    assert cursor and len(first["entries"]) == 1
    second = run.data("savings", "history", "--limit", "1", "--cursor", cursor)["data"]
    assert [e["revision"] for e in first["entries"] + second["entries"]] == [1, 2]
    assert second["next_cursor"] is None
    done = run("savings", "history", "--limit", "2", "--cursor", cursor, "--json")
    assert done.returncode == 2 and json.loads(done.stdout)["condition"] == "invalid_cursor"
    put(overlap, "later", consumer())
    assert scene.collect(DAY22 + timedelta(hours=1)).state == "published"
    done = run("savings", "history", "--limit", "1", "--cursor", cursor, "--json")
    assert done.returncode == 2, "the data changed: the old cursor must die, not skip rows"


def test_expired_evidence_keeps_totals_marks_not_recomputable_and_says_so_everywhere(
    overlap: Path,
) -> None:
    run = Binary(overlap)
    totals = run.data("savings", "history")["data"]["current_total"]["value"]
    assert scene.collect(DAY57).state == "published"
    ids = receipt_ids(overlap)
    expired = [service.read_receipt("personal-local", i) for i in ids]
    assert all(v is not None and not v.evidence_available for v in expired)
    explained = run("savings", "explain", ids[0])
    text = " ".join(explained.stdout.split())
    assert explained.returncode == 0
    assert "EXPIRED" in text and "Detailed evidence has expired" in text
    assert (
        "cannot be recomputed from source" in text and "Full recomputation is unavailable" in text
    )
    data = run.data("savings", "explain", ids[0])
    assert data["state"] == "evidence_expired" and data["data"]["recomputation_available"] is False
    assert any("evidence_expired" in w for w in data["warnings"])
    history = run.data("savings", "history")["data"]
    assert history["current_total"]["value"] == totals, "retained closed totals did not shrink"
    assert all(e["recomputable"] is False for e in history["entries"])
    content = run("savings", "explain", ids[0], "--include-content")
    assert "evidence has expired, so no content can be shown" in " ".join(content.stdout.split())
    summary = run.data("savings")
    assert any(c["code"] == "evidence_expired" for c in summary["data"]["conditions"])


def test_a_newer_schema_history_is_unavailable_not_rewritten(overlap: Path) -> None:
    run = Binary(overlap)
    path = overlap / "data" / "accounting" / "personal-local" / "history" / "periods.json"
    rows = json.loads(path.read_text())
    rows[0]["schema"] = 7
    path.write_text(json.dumps(rows))
    path.chmod(0o600)
    before = path.read_bytes()
    done = run("savings", "history", "--json")
    body = json.loads(done.stdout)["data"]
    assert done.returncode == 0 and body["state"] == "empty"
    assert body["data"]["reason"] == "history_unsupported_schema"
    assert body["data"]["current_total"]["value"] is None
    assert body["next_actions"][0]["command"] == "ssgrep metrics status"
    assert path.read_bytes() == before


def test_damaged_receipt_is_a_specific_exit_1_not_a_not_found(overlap: Path) -> None:
    run = Binary(overlap)
    receipt_id = receipt_ids(overlap)[0]
    path = receipts_dir(overlap) / f"{receipt_id}.json"
    path.write_text("{not json")
    path.chmod(0o600)
    code, body = run.json("savings", "explain", receipt_id)
    assert code == 1 and body["condition"] == "receipt_unreadable"
    assert body["command"] == "ssgrep metrics status"
    before = fingerprint(overlap / "data")
    run("savings", "explain", receipt_id)
    assert fingerprint(overlap / "data") == before
