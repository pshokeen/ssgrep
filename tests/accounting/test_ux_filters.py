"""10.1: intersecting filters, groupings and stable cursors over a REAL collected ledger.

The dataset is written by the real collector (native transcripts, journal events, pinned
frequency view); the selections are read back through the facade and the real binary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ssgrep.accounting import service
from ssgrep.accounting.policy_rules import PolicyError
from ssgrep.accounting.scope import build_scope
from tests.accounting import collector_scene as scene
from tests.accounting.test_ux_support import NOW, Binary, collected_scene, fingerprint

TASKS = ("claude:cons0#turn:u0", "claude:cons1#turn:u1", "claude:cons2#turn:u2")
PER_TASK = (-51, -45, -45)  # conservative / central / optimistic net tokens of every pair


def nets(view) -> list[int | None]:
    row = next(r for r in view.scenarios if r.unit_key == "context_tokens")
    return [q.value for _, q in row.net]


def read(**kwargs):
    limit = kwargs.pop("limit", 20)
    cursor = kwargs.pop("cursor", None)
    return service.read_summary(scope=build_scope(**kwargs), now=NOW, limit=limit, cursor=cursor)


@pytest.fixture
def scene_root(tmp_path: Path) -> Path:
    collected_scene(tmp_path)
    return tmp_path


def test_unfiltered_total_is_the_sum_of_its_three_tasks(scene_root: Path) -> None:
    total = nets(read())
    # characters are summed BEFORE the single division by 4, so the total may differ from the
    # sum of per-task rounded rows by < 1 token per task, never more
    for got, per_task in zip(total, PER_TASK, strict=True):
        assert got is not None and abs(got - 3 * per_task) <= 3
    snapshot_total = [
        q.value
        for q in (
            service.parse_quantity(
                scene.window(scene.current(), "7d")["estimates"]["context_tokens"][s]["net"],
                "tokens",
                "x",
            )
            for s in ("conservative", "central", "optimistic")
        )
    ]
    assert total == snapshot_total, "unfiltered view agrees with the collector's own window"
    by_task = read(by="task")
    rows = {g.key: g for g in by_task.groups}
    assert set(TASKS) <= set(rows)
    for key in TASKS:
        row = next(r for r in rows[key].scenarios if r.unit_key == "context_tokens")
        assert [q.value for _, q in row.net] == list(PER_TASK)
        assert rows[key].receipts == 1


def test_task_and_session_filters_select_exactly_their_records(scene_root: Path) -> None:
    one = read(task="claude:cons1#turn:u1")
    assert nets(one) == list(PER_TASK)
    assert dict(one.measured)["searches"].value == 1
    by_root = read(task="claude:cons1")  # a root id selects every task of that conversation
    assert nets(by_root) == list(PER_TASK)
    session = read(session="cons2")
    assert nets(session) == list(PER_TASK)
    assert dict(session.measured)["searches"].value == 1
    assert session.coverage["candidate_receipts"] == 1
    none = read(session="no-such-session")
    assert dict(none.measured)["searches"].value == 0
    assert nets(none) == [None, None, None], "an empty selection is unknown, never a zero total"


def test_filters_intersect_never_union(scene_root: Path) -> None:
    both = read(runtime="claude", task="claude:cons0#turn:u0", session="cons0")
    assert nets(both) == list(PER_TASK)
    clash = read(runtime="claude", task="claude:cons0#turn:u0", session="cons1")
    assert dict(clash.measured)["searches"].value == 0
    assert nets(clash) == [None, None, None]
    wrong_runtime = read(runtime="codex", task="claude:cons0#turn:u0")
    assert dict(wrong_runtime.measured)["searches"].value == 0
    assert nets(wrong_runtime) == [None, None, None]


def test_interval_is_utc_half_open_for_measured_counts_and_whole_days_for_estimates(
    scene_root: Path,
) -> None:
    inside = read(since="2026-10-12", until="2026-10-13")
    assert dict(inside.measured)["searches"].value == 3
    assert nets(inside) == nets(read()), "the whole day 12 holds every estimate"
    before = read(since="2026-10-11", until="2026-10-12")  # end exclusive: day 12 excluded
    assert dict(before.measured)["searches"].value == 1, "only the day-11 capture-start search"
    assert nets(before) == [None, None, None], "no estimate period lies in that day"
    exact = read(since="2026-10-12T09:00:00Z", until="2026-10-12T09:00:11Z")
    assert dict(exact.measured)["searches"].value == 0  # event is at 09:00:11, end is exclusive
    covering = read(since="2026-10-12T09:00:11Z", until="2026-10-12T09:00:12Z")
    assert dict(covering.measured)["searches"].value == 1


def test_by_day_runtime_and_project_group_the_same_totals(scene_root: Path) -> None:
    for by in ("day", "runtime", "project"):
        view = read(by=by, limit=100)
        counted = sum(
            dict(g.measured)["searches"].value or 0 for g in view.groups if g.key != "task_unknown"
        )
        assert counted == dict(view.measured)["searches"].value, by
    days = read(by="day", limit=100)
    assert [g.key for g in days.groups] == ["2026-10-11", "2026-10-12"] or "2026-10-12" in [
        g.key for g in days.groups
    ]
    runtime = read(by="runtime", limit=100)
    assert "claude" in {g.key for g in runtime.groups}


def test_project_filter_uses_recorded_working_directories_and_says_when_unknown(
    scene_root: Path,
) -> None:
    other = read(project="/definitely/not/here")
    assert dict(other.measured)["searches"].value == 0
    assert nets(other) == [None, None, None]
    inside = read(project="/work/proj")
    assert dict(inside.measured)["searches"].value == 4  # 3 consumers + the start marker
    prefix = read(project="/work")
    assert dict(prefix.measured)["searches"].value == 4
    demo = read(project="/work/demo")  # where the consumer conversations actually ran
    assert nets(demo) == nets(read()), "estimates follow the recorded working directory"
    assert any("project_unknown" in w for w in inside.warnings) or nets(inside) == [None] * 3
    sibling = read(project="/wor")  # a path prefix is not a directory prefix
    assert dict(sibling.measured)["searches"].value == 0


def test_pages_neither_skip_nor_duplicate_and_every_cursor_is_bound(scene_root: Path) -> None:
    full = read(by="task", limit=100)
    keys = [g.key for g in full.groups]
    assert len(keys) == len(set(keys)) >= 3
    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        page = read(by="task", limit=1, cursor=cursor)
        seen += [g.key for g in page.groups]
        pages += 1
        assert page.group_total == len(keys)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert seen == keys and pages == len(keys)
    first = read(by="task", limit=1)
    assert first.next_cursor is not None
    for bad in (
        read.__name__,  # not even shaped like a cursor
        first.next_cursor.replace("g1.", "g1.9"),  # edited offset
        first.next_cursor[:-1] + ("0" if first.next_cursor[-1] != "0" else "1"),  # edited tag
    ):
        with pytest.raises(PolicyError) as caught:
            read(by="task", limit=1, cursor=bad)
        assert caught.value.condition == "invalid_cursor" and caught.value.exit_code == 2
    for other in (
        {"by": "task", "limit": 2},  # other page size
        {"by": "runtime", "limit": 1},  # other grouping
        {"by": "task", "limit": 1, "runtime": "claude"},  # other selection
        {"by": "task", "limit": 1, "since": "2026-10-12"},
    ):
        with pytest.raises(PolicyError) as caught:
            read(cursor=first.next_cursor, **other)
        assert caught.value.condition == "invalid_cursor"
    with pytest.raises(PolicyError):
        read(cursor=first.next_cursor)  # no grouping: this cursor belongs to no page


def test_a_cursor_dies_when_the_snapshot_changes_instead_of_skipping_rows(
    scene_root: Path,
) -> None:
    first = read(by="task", limit=1)
    cursor = first.next_cursor
    assert cursor is not None
    from tests.accounting import verify_scenes as vs

    vs.pair(vs.projects(scene_root), 3, scene.at(hour=9))  # new evidence -> new generation
    assert scene.collect(scene.at(hour=21)).state == "published"
    with pytest.raises(PolicyError) as caught:
        service.read_summary(
            scope=build_scope(by="task"), now=scene.at(hour=21), limit=1, cursor=cursor
        )
    assert caught.value.condition == "invalid_cursor" and caught.value.exit_code == 2


def test_children_enter_only_with_include_children_and_only_for_a_session(
    scene_root: Path,
) -> None:
    with pytest.raises(PolicyError) as caught:
        build_scope(include_children=True)
    assert caught.value.exit_code == 2 and "--session" in str(caught.value)
    plain = read(session="cons0")
    with_children = read(session="cons0", include_children=True)
    assert nets(plain) == nets(with_children) == list(PER_TASK), "cons0 has no children"
    assert with_children.scope.include_children is True and plain.scope.include_children is False


def test_limit_above_100_is_rejected_by_the_real_binary_before_any_side_effect(
    scene_root: Path,
) -> None:
    run = Binary(scene_root)
    before = fingerprint(scene_root / "data")
    code, body = run.json("savings", "--limit", "101")
    assert code == 2 and body["ok"] is False and body["condition"] == "invalid_limit"
    code, body = run.json("savings", "--by", "task", "--limit", "1", "--cursor", "g1.1.deadbeef")
    assert code == 2 and body["condition"] == "invalid_cursor"
    assert fingerprint(scene_root / "data") == before


def test_real_binary_pages_through_json_and_text_with_a_pasteable_continuation(
    scene_root: Path,
) -> None:
    run = Binary(scene_root)
    first = run.data("savings", "--by", "task", "--limit", "2", "--runtime", "claude")
    groups = first["data"]["groups"]
    assert len(groups["rows"]) == 2 and groups["next_cursor"]
    second = run.data(
        "savings", "--by", "task", "--limit", "2", "--runtime", "claude", "--cursor",
        groups["next_cursor"],
    )  # fmt: skip
    keys = [r["key"] for r in groups["rows"] + second["data"]["groups"]["rows"]]
    assert len(keys) == len(set(keys))
    text = run("savings", "--by", "task", "--limit", "2", "--runtime", "claude", columns=60)
    assert text.returncode == 0
    more = text.stdout.split("More:")[1]
    command = " ".join(more.replace("\\\n", " ").split()).split("Next:")[0].split(" PRIVACY")[0]
    assert "--cursor" in command and "--runtime claude" in command and "--by task" in command
