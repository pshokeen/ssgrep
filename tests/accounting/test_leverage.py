"""Note leverage, payback coverage, and overhead purpose tagging."""

from __future__ import annotations

from ssgrep.accounting.contracts import Basis, Quantity, Scenario, TaskId
from ssgrep.accounting.leverage import (
    NoteInvestment,
    OverheadEvent,
    Purpose,
    note_leverage,
    overhead_views,
    purpose_of,
)
from ssgrep.accounting.units import CONTEXT_TOKENS
from tests.accounting.valuation_builders import CHARS, METHOD, atom, candidate

NOTE_ORIGIN = "note-event-1"


def cost(value):
    if value is None:
        return Quantity.unavailable(CONTEXT_TOKENS, "creation_cost_unobserved")
    return Quantity(value, CONTEXT_TOKENS, Basis.ESTIMATED, "chars_div4_v1")


def consumer(cid, task, chars=400, start=0, origin=NOTE_ORIGIN):
    return candidate(
        cid, task=task, immediate=(atom(f"{cid}-a", chars, start=start, origin=origin),)
    )


def test_overlapping_consumers_of_a_note_are_deduplicated_per_task_and_summed_across_tasks():
    t1, t2 = TaskId("r1", "t1", "main"), TaskId("r2", "t1", "main")
    consumers = [
        consumer("c1", t1, 400, 0),
        consumer("c2", t1, 400, 200),  # overlaps c1 inside one task: union [0,600)
        consumer("c3", t2, 400, 0),  # a different consumer task: credited again
    ]
    result = note_leverage(
        "n1",
        frozenset({NOTE_ORIGIN}),
        consumers,
        [NoteInvestment("n1", "creation", cost(100))],
        METHOD,
        CHARS,
    )
    cons = next(s for s in result.scenarios if s.scenario is Scenario.CONSERVATIVE)
    assert cons.credit.value == (600 + 400) // 4
    assert cons.consumers == 2
    assert result.known_cost == 100 and result.payback_complete
    assert result.payback_permille(Scenario.CONSERVATIVE) == 250 * 1000 // 100


def test_atoms_from_other_sources_do_not_count_toward_this_notes_leverage():
    task = TaskId("r1", "t1", "main")
    consumers = [consumer("c1", task, 400, 0), consumer("c2", task, 800, 0, origin="elsewhere")]
    result = note_leverage(
        "n1",
        frozenset({NOTE_ORIGIN}),
        consumers,
        [NoteInvestment("n1", "creation", cost(10))],
        METHOD,
        CHARS,
    )
    assert result.scenarios[0].credit.value == 100


def test_partly_unavailable_creation_cost_shows_known_investment_and_denies_payback():
    task = TaskId("r1", "t1", "main")
    investments = [
        NoteInvestment("n1", "creation", cost(100)),
        NoteInvestment("n1", "index", cost(None)),
    ]
    result = note_leverage(
        "n1", frozenset({NOTE_ORIGIN}), [consumer("c1", task)], investments, METHOD, CHARS
    )
    assert result.known_cost == 100
    assert (result.known_components, result.cost_components) == (1, 2)
    assert not result.payback_complete
    assert result.missing_reasons == ("creation_cost_unobserved",)
    assert all(result.payback_permille(s.scenario) is None for s in result.scenarios)


def test_no_observed_creation_cost_has_unknown_known_cost_not_zero():
    task = TaskId("r1", "t1", "main")
    result = note_leverage(
        "n1", frozenset({NOTE_ORIGIN}), [consumer("c1", task)], [], METHOD, CHARS
    )
    assert result.known_cost is None and result.cost_components == 0
    assert not result.payback_complete
    assert result.missing_reasons == ("note_creation_cost_unobserved",)


def test_unavailable_source_atom_blocks_payback_even_with_complete_cost():
    from ssgrep.accounting.valuation_inputs import AtomStatus

    task = TaskId("r1", "t1", "main")
    gone = candidate(
        "c1",
        task=task,
        immediate=(atom("gone", 400, origin=NOTE_ORIGIN, status=AtomStatus.UNAVAILABLE),),
    )
    result = note_leverage(
        "n1",
        frozenset({NOTE_ORIGIN}),
        [gone],
        [NoteInvestment("n1", "creation", cost(50))],
        METHOD,
        CHARS,
    )
    assert result.payback_complete and result.scenarios[0].credit_missing == 1
    assert result.payback_permille(Scenario.CONSERVATIVE) is None


# ---- overhead ---------------------------------------------------------------------


def test_protocol_purpose_requires_explicit_tag_or_sequence_evidence():
    tagged = OverheadEvent("e1", "protocol", cost(30), explicit_tag="protocol")
    sequenced = OverheadEvent("e2", "protocol", cost(30), sequence_evidence=True)
    assert purpose_of(tagged) is Purpose.PROTOCOL
    assert purpose_of(sequenced) is Purpose.PROTOCOL
    assert purpose_of(OverheadEvent("e3", "protocol", cost(30))) is Purpose.UNKNOWN
    # an explicit tag on a non-protocol kind does not manufacture protocol purpose
    assert (
        purpose_of(OverheadEvent("e4", "index", cost(5), explicit_tag="protocol"))
        is Purpose.UNKNOWN
    )


def test_overhead_views_group_by_kind_and_purpose_and_keep_missing_visible():
    events = [
        OverheadEvent("e1", "protocol", cost(30), explicit_tag="protocol"),
        OverheadEvent("e2", "protocol", cost(12)),  # purpose unknown stays a cost
        OverheadEvent("e3", "protocol", cost(None)),
        OverheadEvent("e4", "index", cost(200)),
        OverheadEvent("e5", "collector", cost(7)),
    ]
    views = {(v.kind, v.purpose): v for v in overhead_views(events)}
    assert views[("protocol", Purpose.PROTOCOL)].known == 30
    unknown = views[("protocol", Purpose.UNKNOWN)]
    assert (unknown.known, unknown.events, unknown.missing) == (12, 2, 1)
    assert views[("index", Purpose.UNKNOWN)].known == 200
    assert views[("collector", Purpose.UNKNOWN)].known == 7
    # input order does not matter
    assert overhead_views(list(reversed(events))) == overhead_views(events)
