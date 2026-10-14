"""Lab-authored scripted turns for the stand-in agent (L1/L2 for A, L3/L4 for B).

Each script is the mechanical work a competent solver would do, expressed as real shell
commands. Dynamic steps read the REAL outcome of earlier steps (parsed tool output), exactly as
a solver reads its own transcript. The solver-authored SQL comes from ``queries`` (the lab's
own SQL formulations); the oracle never reads this text, it grades the actual rows and states.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from . import l4_reference as l4, prompts, queries
from .observer import parse_psql_table
from .standin_agent import Bash, Ctx, DocRead, Outcome, Say, Step

POLICY_ID = re.compile(r"\bMCT-[A-Z]{3}-\d{3}\b")
WALL = "\n\nWall time:"
EXIT_TRAILER = re.compile(r"\n\[exit code: -?\d+\]$")


def sql(role: str, database: str, statement: str, tag: str | None = None) -> str:
    """One shell command running ``statement`` through the observed restricted SQL tool."""
    marker = f"-- step:{tag}\n" if tag else ""
    return (
        f"sql --role {role} --database {database} <<'LABSQL'\n{marker}{statement.strip()}\nLABSQL"
    )


def tagged(ctx: Ctx, tag: str) -> Outcome | None:
    return _by_command(ctx, f"-- step:{tag}\n")


def body(outcome: Outcome) -> str:
    """The tool's own output without the harness' trailer lines."""
    text = outcome.text.split(WALL)[0]
    return EXIT_TRAILER.sub("", text)


def table(outcome: Outcome | None) -> list[dict[str, str]]:
    rows = parse_psql_table(body(outcome)) if outcome is not None else None
    return rows or []


def _by_command(ctx: Ctx, needle: str) -> Outcome | None:
    for outcome in reversed(ctx.outcomes):
        if needle in outcome.command:
            return outcome
    return None


def _fmt(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> str:
    return "; ".join(" ".join(str(r[c]) for c in columns) for r in rows)


# ------------------------------------------------------------------------------------ A


CASH_COLS = ["booking_id", "currency", "settled_minor"]
SEAT_COLS = ["flight_id", "occupied", "live_holds", "raw_free", "sellable_free"]
GAP_COLS = ["booking_id", "from_flight", "to_flight", "airport", "gap_minutes", "rule_id"]
CONN_COLS = ["booking_id", "from_flight", "to_flight", "gap_minutes", "rule_id", "valid"]
COUPON_COLS = ["coupon_id", "flight_id", "leg_no", "status", "version"]
PERSON_COLS = ["booking_id", "traveller_id"]
RULE_COLS = ["rule_id", "airport_code", "minimum_minutes"]
RULES_SQL = "SELECT rule_id, airport_code, minimum_minutes FROM connection_rules ORDER BY 1"


def _l1(db: str) -> list[Step]:
    def summary(ctx: Ctx) -> str:
        good = table(tagged(ctx, "cash"))
        inflated = {r["booking_id"]: r["settled_minor"] for r in table(tagged(ctx, "cash_naive"))}
        totals: dict[str, int] = {}
        for r in good:
            totals[r["currency"]] = totals.get(r["currency"], 0) + int(r["settled_minor"])
        return (
            "L1 findings (verified by executing SQL). Settled cash per booking in minor "
            f"units, payments pre-aggregated before any join: {_fmt(good, CASH_COLS)}. "
            f"Totals by currency (never merged): {json.dumps(totals, sort_keys=True)}. "
            "Joining payments through tickets, coupons and marketing_flights fans the rows out "
            f"and inflates the sums (naive join gave {json.dumps(inflated, sort_keys=True)}). "
            "Face value is not settled cash; failed payments are excluded and the cancelled "
            "booking nets to zero after its refund. Sellable economy seats per physical "
            "flight (flight occupied live_holds raw_free sellable_free): "
            f"{_fmt(table(tagged(ctx, 'seats')), SEAT_COLS)}. "
            "Holds count only while status is active and expires_at is strictly after the "
            "application clock; cancelled flights are not sellable."
        )

    wrong = "SELECT booking_id, sum(amount) FROM payments GROUP BY 1"
    return [
        # a first attempt against a table that does not exist: a real SQL error, kept in history
        Bash(sql("reader", db, wrong, "wrong_table")),
        Bash(sql("reader", db, queries.CASH_JOIN_FANOUT, "cash_naive")),
        Bash(sql("reader", db, queries.CASH_PREAGGREGATED, "cash")),
        Bash(sql("reader", db, queries.seats(), "seats")),
        Say(summary),
    ]


def _l2(db: str) -> list[Step]:
    def summary(ctx: Ctx) -> str:
        affected = table(tagged(ctx, "affected"))
        conns = table(tagged(ctx, "connections"))
        gaps = [c for c in conns if c["valid"] in ("f", "false")]
        local = [c for c in table(tagged(ctx, "clocks")) if c["flight_id"] in ("F1", "F8")]
        return (
            "L2 findings (verified by executing SQL). Confirmed travellers needing "
            f"reaccommodation (booking traveller): {_fmt(affected, PERSON_COLS)}. "
            "They are distinct travellers even though their display names match; the cancelled "
            "booking is excluded. Their cancelled leg and a short connection are the cause: "
            f"connections below the airport minimum: {_fmt(gaps, GAP_COLS)}. "
            "Minimum-connection policies (rule airport minutes): "
            f"{_fmt(table(tagged(ctx, 'rules_all')), RULE_COLS)}. A connection of "
            "exactly the minimum is valid (comparison is >=). All connections: "
            f"{_fmt(conns, CONN_COLS)}. "
            "The repeated local departure clock is not a duplicate flight: "
            f"{_fmt(local, ['flight_id', 'local_clock', 'utc_offset_hours'])} are one actual hour "
            "apart on distinct UTC instants."
        )

    return [
        Bash(sql("reader", db, queries.affected(), "affected")),
        Bash(sql("reader", db, queries.connections(), "connections")),
        Bash(sql("reader", db, RULES_SQL, "rules_all")),
        Bash(sql("reader", db, queries.local_clock_listing(), "clocks")),
        Bash(sql("reader", db, queries.flight_instants(), "instants")),
        Say(summary),
    ]


# ------------------------------------------------------------------------------------ B

SEARCH_QUERY = "minimum connection policy rules per airport and how connections are validated"


def _search_outcome(ctx: Ctx) -> Outcome | None:
    return _by_command(ctx, "ssgrep search")


def _refs_of(outcome: Outcome | None) -> list[str]:
    if outcome is None:
        return []
    text = body(outcome)
    start = text.find("{")
    refs: list[str] = []
    if start >= 0:
        try:
            doc = json.JSONDecoder().raw_decode(text[start:])[0]
        except ValueError:
            doc = None
        data = doc.get("data", doc) if isinstance(doc, dict) else {}
        for card in (data.get("results") or []) if isinstance(data, dict) else []:
            if isinstance(card, dict) and isinstance(card.get("ref"), str):
                refs.append(card["ref"])
    if not refs:
        refs = re.findall(r"omp:[\w-]+:ep:\d+", text)
    return list(dict.fromkeys(refs))


def _show_command(index: int):  # noqa: ANN202
    def make(ctx: Ctx) -> str | None:
        refs = _refs_of(_search_outcome(ctx))
        return f"ssgrep show {refs[index]} --json" if index < len(refs) else None

    return make


def _policy_ids(ctx: Ctx) -> list[str]:
    seen: list[str] = []
    for outcome in ctx.outcomes:
        if "ssgrep" in outcome.command:
            seen += POLICY_ID.findall(outcome.text)
    return sorted(set(seen))


def _validate_rules(db: str):  # noqa: ANN202
    def make(ctx: Ctx) -> str:
        ids = _policy_ids(ctx)
        if ids:  # validate the retrieved rules against CURRENT database state
            listing = ", ".join(f"'{i}'" for i in ids)
            statement = (
                "SELECT rule_id, airport_code, minimum_minutes FROM connection_rules "
                f"WHERE rule_id IN ({listing}) ORDER BY 1"
            )
        else:  # nothing usable was retrieved: read the rule rows directly
            statement = (
                "SELECT rule_id, airport_code, minimum_minutes FROM connection_rules ORDER BY 1"
            )
        return sql("reader", db, statement, "rules")

    return make


def _plan(ctx: Ctx) -> dict[str, tuple[str, ...]]:
    affected = table(tagged(ctx, "affected"))
    routes = [
        {**r, "legs": tuple(x for x in r["legs"].strip("{}").split(",") if x)}
        for r in table(tagged(ctx, "routes"))
    ]
    seat_rows = table(tagged(ctx, "seats"))
    return queries.greedy_priority_plan(
        [{**r, "service_priority": int(r["service_priority"])} for r in affected], routes, seat_rows
    )


def _validate_policy_for_plan(db: str):  # noqa: ANN202
    def make(ctx: Ctx) -> str | None:
        multi = [legs for legs in _plan(ctx).values() if len(legs) > 1]
        return sql("reader", db, queries.policy_ids_used(multi[0]), "policy") if multi else None

    return make


def _l3(db: str) -> list[Step]:
    def summary(ctx: Ctx) -> str:
        plan = _plan(ctx)
        ctx.memory["plan"] = {b: list(v) for b, v in sorted(plan.items())}
        routes = [{**r, "legs": r["legs"]} for r in table(tagged(ctx, "routes"))]
        used = table(tagged(ctx, "policy"))
        rules = table(tagged(ctx, "rules"))
        return (
            "L3 plan (verified by executing SQL). Feasible single-traveller routes arriving "
            f"by 18:00Z: {_fmt(routes, ['legs', 'arrival_utc'])}. Joint allocation by "
            "priority with shared capacity: "
            f"{json.dumps({b: list(v) for b, v in sorted(plan.items())})}. "
            f"Connection policy rows validated in the current database: "
            f"{_fmt(rules, ['rule_id', 'airport_code', 'minimum_minutes'])}. "
            f"Policy ids applied to the multi-leg itinerary: {_fmt(used, ['rule_id'])}."
        )

    return [
        Bash(f'ssgrep search "{SEARCH_QUERY}" --json'),
        Bash(_show_command(0)),
        Bash(_show_command(1)),
        Bash(_validate_rules(db)),
        Bash(sql("reader", db, queries.affected(), "affected")),
        Bash(sql("reader", db, queries.seats(), "seats")),
        Bash(sql("reader", db, queries.candidate_routes(), "routes")),
        Bash(_validate_policy_for_plan(db)),
        Say(summary),
    ]


def _state_sql() -> tuple[str, str]:
    return (
        "SELECT ticket_id, booking_id FROM tickets ORDER BY 1",
        "SELECT coupon_id, ticket_id, flight_id, leg_no, status, version FROM coupons ORDER BY 1",
    )


def _request_command(db: str, booking: str, legs: tuple[str, ...], ctx: Ctx) -> str:
    tickets = table(tagged(ctx, "tickets"))
    coupons = [
        {**c, "leg_no": int(c["leg_no"]), "version": int(c["version"])}
        for c in table(tagged(ctx, "coupons"))
    ]
    state = {"tickets": tickets, "coupons": coupons}
    request = l4.request_for(state, f"l4:lab:{booking}", booking, legs)
    script = f"BEGIN ISOLATION LEVEL READ COMMITTED;\n{request.call()}\nCOMMIT;"
    return sql("writer", db, script)


def _apply(db: str, booking: str):  # noqa: ANN202
    def make(ctx: Ctx) -> str | None:
        legs = _plan_from_summary(ctx).get(booking)
        if not legs:
            return None
        command = _request_command(db, booking, legs, ctx)
        ctx.memory[f"request:{booking}"] = command  # the exact payload, for a faithful retry
        return command

    return make


def _replay(booking: str):  # noqa: ANN202
    """Resend the IDENTICAL request (same key and digest) as after a lost response."""

    def make(ctx: Ctx) -> str | None:
        return ctx.memory.get(f"request:{booking}")

    return make


def _plan_from_summary(ctx: Ctx) -> dict[str, tuple[str, ...]]:
    raw = ctx.params.get("plan")
    return {b: tuple(v) for b, v in raw.items()} if isinstance(raw, dict) else {}


FINAL_SQL = "SELECT coupon_id, flight_id, leg_no, status, version FROM coupons ORDER BY 1"


def _l4(db: str) -> list[Step]:
    t_sql, c_sql = _state_sql()

    def summary(ctx: Ctx) -> str:
        responses = [body(o)[-400:] for o in ctx.outcomes if "rebook_itinerary" in o.command]
        final = table(tagged(ctx, "final"))
        return (
            "L4 applied the plan atomically, one READ COMMITTED transaction per booking with "
            f"an immutable request key. Operation responses: {responses}. Final coupon rows "
            f"(coupon flight leg status version): {_fmt(final, COUPON_COLS)}."
        )

    return [
        Bash(sql("reader", db, t_sql, "tickets")),
        Bash(sql("reader", db, c_sql, "coupons")),
        Bash(_apply(db, "B1")),
        Bash(_apply(db, "B2")),
        Bash(_replay("B1")),
        Bash(
            sql(
                "reader",
                db,
                "SELECT coupon_id, flight_id, leg_no, status, version FROM coupons ORDER BY 1",
                "final",
            )
        ),
        Say(summary),
    ]


def _l4_params_guard(params: Mapping[str, Any]) -> None:
    if not isinstance(params.get("plan"), dict) or not params["plan"]:
        raise ValueError("L4 needs params['plan'] (booking -> legs) derived from B's own L3 result")


# ---------------------------------------------------------------------------- registry


def build(name: str, params: Mapping[str, Any]) -> tuple[str, list[Step]]:
    """(user prompt, steps) of a named scripted turn."""
    db = str(params["database"])
    first = bool(params.get("first_turn", True))
    prefix = prompts.BRIEF + "\n\n" if first else ""
    if name == "L1":
        return prefix + prompts.L1, _l1(db)
    if name == "L2":
        return prefix + prompts.L2, _l2(db)
    if name == "L3":
        return prefix + prompts.L3, _l3(db)
    if name == "L4":
        _l4_params_guard(params)
        return prefix + prompts.L4, _l4(db)
    if name == "DOCS":  # control: a documentation read that merely MENTIONS ssgrep search
        return prefix + "Read the retrieval tool documentation.", [
            DocRead("xd://mcp__recall_search_sessions", "Use `ssgrep search <query>` to search."),
            Say("Read the documentation; no lookup executed."),
        ]
    raise ValueError(f"unknown scripted turn {name!r}")
