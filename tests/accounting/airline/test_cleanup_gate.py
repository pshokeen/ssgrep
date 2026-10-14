"""Cleanup gate: judged on what the lab itself removed, not on shared-daemon churn.

A fake docker layer replaces the single spawn point (``database._docker_raw``), so these
tests are deterministic and need no daemon. The lab's real choke point (``_docker``) still
runs on top of it: removal targets are identified BEFORE the call and recorded in the ledger.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from eval.accounting_airline import database, domain_lane
from eval.accounting_airline.gates import FAIL, PASS

RUN = "abc123def456"
OWN_ID = "a" * 64
FOREIGN_ID = "f" * 64
RYUK_ID = "7" * 64


class FakeDocker:
    """A tiny in-memory docker CLI: ps / inspect / rm / stop / kill."""

    def __init__(self) -> None:
        self.containers: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, ...]] = []

    def add(self, cid: str, name: str, labels: dict[str, str] | None = None) -> str:
        self.containers[cid] = {"Id": cid, "Name": "/" + name, "Config": {"Labels": labels or {}}}
        return cid

    def add_own(self, run_id: str = RUN, cid: str = OWN_ID) -> str:
        labels = {database.RUN_LABEL: run_id, database.KIND_LABEL: database.KIND_VALUE}
        return self.add(cid, database.NAME_PREFIX + run_id, labels)

    def _find(self, ref: str) -> dict[str, Any] | None:
        for cid, info in self.containers.items():
            if cid == ref or cid.startswith(ref) or info["Name"] == "/" + ref.lstrip("/"):
                return info
        return None

    def __call__(
        self, *args: str, timeout: float = 60, input_text: str | None = None, env: Any = None
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        words = list(args)
        if words[:1] == ["container"]:
            words = words[1:]
        verb, rest = words[0], words[1:]

        def done(code: int, out: str = "", err: str = "") -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args, code, out, err)

        if verb == "ps":
            wanted = [rest[i + 1] for i, token in enumerate(rest) if token == "--filter"]
            ids = []
            for cid, info in self.containers.items():
                labels = info["Config"]["Labels"]
                if all(self._label_match(labels, w) for w in wanted):
                    ids.append(cid)
            return done(0, "\n".join(ids))
        if verb == "inspect":
            info = self._find(rest[-1])
            return done(0, json.dumps([info])) if info else done(1, "", "No such container")
        if verb in database.REMOVAL_VERBS:
            targets, skip = [], False
            for token in rest:
                if skip:
                    skip = False
                elif token in ("-t", "--time", "-s", "--signal"):
                    skip = True
                elif not token.startswith("-"):
                    targets.append(token)
            for ref in targets:
                info = self._find(ref)
                if info is None:
                    return done(1, "", f"No such container: {ref}")
                if verb != "kill" and verb != "stop":
                    del self.containers[info["Id"]]
            return done(0, "\n".join(targets))
        return done(1, "", f"fake docker: unsupported {verb}")

    @staticmethod
    def _label_match(labels: dict[str, str], clause: str) -> bool:
        spec = clause.removeprefix("label=")
        key, _, value = spec.partition("=")
        return key in labels and (not value or labels[key] == value)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    daemon = FakeDocker()
    monkeypatch.setattr(database, "_docker_raw", daemon)
    return daemon


def lab_instance() -> database.LabInstance:
    return database.LabInstance(RUN, OWN_ID, 1, {})


def run_gate(lab: database.LabInstance | None, mark: int, foreign_before: set[str]):
    return domain_lane.gate_cleanup(RUN, lab, database.removals_since(mark), foreign_before)


def test_lab_removing_a_foreign_container_fails_even_if_it_was_foreign_by_name_only(
    fake: FakeDocker,
) -> None:
    fake.add_own()
    fake.add(FOREIGN_ID, "someone-elses-db", {"com.example.project": "other"})
    mark = database.removal_mark()
    before = domain_lane.other_containers()
    assert before == {FOREIGN_ID}
    database._docker("rm", "-f", "-v", FOREIGN_ID)  # the lab (wrongly) removes a foreign id
    assert lab_instance().teardown() is True  # and still removes its own container
    gate = run_gate(lab_instance_removed(), mark, before)
    assert gate.status == FAIL
    assert f"touched a container it does not own: {FOREIGN_ID}" in gate.detail
    assert "someone-elses-db" in gate.detail
    removals = gate.evidence["lab_removals"]
    assert [r["container_id"] for r in removals] == [FOREIGN_ID, OWN_ID]
    assert FOREIGN_ID not in gate.evidence["foreign_disappeared_informational"]  # it is the lab's


def lab_instance_removed() -> database.LabInstance:
    lab = lab_instance()
    lab.removed = True
    return lab


@pytest.mark.parametrize("label_value", ["other-run", None])
def test_lab_removing_a_lookalike_with_the_wrong_run_label_fails(
    fake: FakeDocker, label_value: str | None
) -> None:
    fake.add_own()
    labels = {database.KIND_LABEL: database.KIND_VALUE}
    if label_value:
        labels[database.RUN_LABEL] = label_value
    fake.add(FOREIGN_ID, database.NAME_PREFIX + RUN + "-lookalike", labels)
    mark = database.removal_mark()
    database._docker("rm", "-f", FOREIGN_ID)
    gate = run_gate(lab_instance_removed(), mark, set())
    assert gate.status == FAIL and "does not own" in gate.detail
    assert FOREIGN_ID in gate.detail


def test_external_churn_alone_passes_and_is_reported_informationally(fake: FakeDocker) -> None:
    fake.add_own()
    fake.add(RYUK_ID, "testcontainers-ryuk-1234", {"org.testcontainers": "true"})
    mark = database.removal_mark()
    before = domain_lane.other_containers()
    assert before == {RYUK_ID}
    assert lab_instance().teardown() is True  # the lab removes only its own container
    # another project's reaper removes ryuk and a new unrelated container appears
    del fake.containers[RYUK_ID]
    fake.add("9" * 64, "testcontainers-postgres-new", {"org.testcontainers": "true"})
    gate = run_gate(lab_instance_removed(), mark, before)
    assert gate.status == PASS, gate.detail
    assert gate.evidence["foreign_disappeared_informational"] == [RYUK_ID]
    assert gate.evidence["foreign_appeared_informational"] == ["9" * 64]
    assert [r["container_id"] for r in gate.evidence["lab_removals"]] == [OWN_ID]
    assert all(r["run_label"] == RUN and r["succeeded"] for r in gate.evidence["lab_removals"])


def test_owned_container_left_behind_fails(fake: FakeDocker) -> None:
    fake.add_own()
    mark = database.removal_mark()
    lab = lab_instance()  # never torn down
    gate = run_gate(lab, mark, set())
    assert gate.status == FAIL
    assert "owned container was not removed" in gate.detail
    assert f"run-labelled containers remain: ['{OWN_ID}']" in gate.detail
    assert gate.evidence["lab_removals"] == []


def test_run_labelled_leftover_fails_even_when_the_lab_thinks_it_removed_its_container(
    fake: FakeDocker,
) -> None:
    fake.add_own()
    mark = database.removal_mark()
    gate = run_gate(lab_instance_removed(), mark, set())
    assert gate.status == FAIL and "run-labelled containers remain" in gate.detail
    assert "owned container was not removed" not in gate.detail


def test_clean_run_with_no_foreign_activity_passes(fake: FakeDocker) -> None:
    fake.add_own()
    mark = database.removal_mark()
    lab = lab_instance()
    lab.teardown()
    gate = run_gate(lab, mark, set())
    assert gate.status == PASS and gate.detail == ""
    assert gate.evidence["foreign_disappeared_informational"] == []
    assert fake.containers == {}


def test_ledger_identifies_targets_before_the_call_and_covers_every_verb(fake: FakeDocker) -> None:
    fake.add_own()
    fake.add(FOREIGN_ID, "x", {})
    mark = database.removal_mark()
    database._docker("container", "kill", "--signal", "KILL", FOREIGN_ID)
    database._docker("stop", "-t", "1", OWN_ID)
    database._docker("rm", "-f", "-v", OWN_ID)
    database._docker("rm", "-f", "ghost")  # not inspectable: recorded as unidentified, failed
    records = database.removals_since(mark)
    assert [(r.verb, r.container_id, r.succeeded) for r in records] == [
        ("kill", FOREIGN_ID, True),
        ("stop", OWN_ID, True),
        ("rm", OWN_ID, True),
        ("rm", None, False),
    ]
    assert records[2].owned_by(RUN) and not records[0].owned_by(RUN)
    assert not records[3].owned_by(RUN) and not records[3].identified
    # a failed attempt on something the lab could not even identify is not a violation
    gate = domain_lane.gate_cleanup(RUN, lab_instance_removed(), records[3:], set())
    assert gate.status == PASS
    assert domain_lane.gate_cleanup(RUN, lab_instance_removed(), records[:1], set()).status == FAIL


def test_non_removal_docker_calls_are_not_recorded(fake: FakeDocker) -> None:
    fake.add_own()
    mark = database.removal_mark()
    database._docker("ps", "-aq")
    database._docker("inspect", OWN_ID)
    database._docker("exec", OWN_ID, "true")
    assert database.removals_since(mark) == []


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (("rm", "-f", "-v", "abc"), ("rm", ["abc"])),
        (("container", "rm", "abc", "def"), ("rm", ["abc", "def"])),
        (("stop", "-t", "5", "abc"), ("stop", ["abc"])),
        (("kill", "--signal", "KILL", "abc"), ("kill", ["abc"])),
        (("ps", "-aq"), (None, [])),
        (("run", "-d", "rm"), (None, [])),
    ],
)
def test_removal_target_parsing(
    args: tuple[str, ...], expected: tuple[str | None, list[str]]
) -> None:
    assert database._removal_targets(args) == expected


def test_teardown_still_refuses_foreign_container_and_records_nothing(fake: FakeDocker) -> None:
    fake.add(OWN_ID, "someone-elses-container", {database.RUN_LABEL: "other"})
    mark = database.removal_mark()
    with pytest.raises(database.OwnershipError):
        lab_instance().teardown()
    assert database.removals_since(mark) == []  # refused before any rm was issued
    assert OWN_ID in fake.containers
