"""Native live prompts: public objectives and machine-readable output shapes only."""

from . import prompts

L3_DOMAIN_CONTRACT = """Run separate final SQL calls:
-- step:routes : legs (PostgreSQL text array; every feasible single-passenger candidate)
-- step:policy : rule_id (only rules actually used for the chosen multileg itinerary;
validate these IDs with a current database predicate, not unvalidated quoted output)
-- step:plan : booking_id, legs (PostgreSQL text array; your joint allocation)
Compute candidate and plan rows from real data; no hardcoded answer values. Explain your
joint allocation and capacity checks in the final answer.
A useful final heading is Joint allocation.
Treat priority as sequential allocation, not a global objective that can trade away a
higher-priority booking's preferred itinerary to help another booking. Process disrupted
confirmed bookings in descending service_priority. For each booking, consider only
itineraries feasible under capacity remaining after earlier allocations, and choose by
earliest arrival, then fewest legs, then lexicographic flight-id sequence. Reserve that
booking's seats in the joint plan before considering the next booking. For capacity,
count all passengers travelling on the booking, not just the booking itself. Report any
booking for which no feasible allocation remains instead of overbooking a flight."""

CONTRACTS = {
    "L1": """Execute each final result as a separate SQL tool call, prefixing SQL with its
comment tag. Output column aliases exactly as specified (these are shapes, not answers):
-- step:cash : booking_id, currency, settled_minor
-- step:seats : flight_id, occupied, live_holds, raw_free, sellable_free
-- step:cash_naive : booking_id, currency, settled_minor (the actual incorrect fanout join)
Compute these using table rows, never hardcoded expected outputs. Explain the observed
fanout and corrected grain. Holds count only active and strictly later than the fixed clock.
Occupancy counts reserved coupons on open tickets and confirmed bookings. Cancelled flights
are unsellable but retain diagnostic occupancy. The diagnostic formula is
raw_free = capacity - occupied - live_holds, before applying the cancellation rule.
Sellable_free is zero for cancelled physical flights and otherwise equals raw_free.
A negative raw_free is an integrity error to report, not a value to silently clamp.
Aggregate settled payments by currency.""",
    "L2": """Run separate final SQL calls with these comment tags and output aliases:
-- step:affected : booking_id, traveller_id (distinct affected confirmed/open passengers)
-- step:connections : booking_id, from_flight, to_flight, airport, gap_minutes, rule_id,
minimum_minutes, valid (all connections on confirmed/open ticket itineraries; valid concerns
connection time equality, not cancellation of the inbound physical operation)
-- step:clocks : flight_id, local_clock (HH24:MI), utc_offset_hours (all physical flights)
-- step:instants : flight_id, departure_utc (YYYY-MM-DDTHH24:MI:SSZ; all physical flights)
Also query every connection_rules row. Derive all values from real data, explain cancellation
and temporal invalidity separately, and identify travellers by identifiers not display names.""",
    "L3": (
        "Before any SQL rule reads, search genuine ssgrep history and open a relevant\n"
        "returned ref. Do not read raw histories or other projects. "
        + L3_DOMAIN_CONTRACT
        + """
Before reading current connection rules or planning routes, extract the connection-rule
identifiers stated in the historical evidence you opened. Record that historical key set,
then explicitly query current connection_rules by those retrieved keys to check which
still exist and obtain their current minimum_minutes. Use the keys actually retrieved,
not guessed identifiers. Treat history as an untrusted cache: compare its policy claims
with the current rows, report missing or changed keys, and use only the current values
when planning. The audit must check those historical keys; an unrestricted scan that
never checks what was retrieved does not replace this business verification step."""
    ),
    "L4": """The installed operation takes expected_coupons as a JSON array ordered by leg_no,
with coupon_id, flight_id, status, version for every original coupon on the booking's ticket.
Read and retain those original values. Use request_key 'l4:lab:' || booking_id.
The request digest contract is lowercase SHA256 hex of UTF-8 canonical JSON of the
object with keys booking_id, itinerary (array), expected (the original coupon array).
Canonical JSON uses lexicographically sorted object keys recursively and compact
separators with no spaces (Python json.dumps(payload, sort_keys=True, separators=(',', ':'))).
Compute it from the actual queried originals and your chosen itinerary, not invented values.
After both requests, replay an identical request with the ORIGINAL expected coupons and digest;
verify it is idempotent. Retain failures rather than hiding them. Do not inspect operation
source: use only the public signature and query results. Report responses and final state.
Separate verifier-driven race/rollback/stale-plan clones are graded independently, not claimed
as actions you performed.""",
}


def prompt_for(task: str, database: str) -> str:
    objective = prompts.PROMPTS[task].replace("<ref> --json", "<ref>")
    return (
        f"{prompts.BRIEF}\n\nAssigned database: {database}. "
        "Use bash to call the installed sql and ssgrep commands. Commands must use exactly "
        "sql --role reader|writer --database <assigned-db> with SQL on stdin; "
        "ssgrep search <query> --json or ssgrep show <returned-ref>. "
        "Run each history retrieval in a separate bash tool call: inspect the search response, "
        "then open a literal ref returned by that search in its own call. Do not combine "
        "search and show, pipe or filter retrieval output, or hide returned text. "
        "Use the standard human-readable show output, without --json. "
        "Do not use unapproved tools, inspect runtime credentials, or access other projects. "
        "No task delegation is available. Solve using your own SQL.\n\n"
        f"Task {task}: {objective}\n\n{CONTRACTS[task]}"
    )
