"""Task 18.3: one small synthetic acceptance corpus across all six runtimes.

Every native record carries a stable ID and a clock frozen from the caller's
``setup_at``; nothing here reads the product journal, runs the estimator, calls a
provider or touches an index. Expectations are declared by hand next to the
records that justify them (gate IDs V01/V03/V04/V05/V06/V07/V08 from
review-followup.md) so the parent can compare reducer/estimator output against
independent numbers, never the other way round.

Consumers compare native call locators and explicitly declared observations.
Locators are not internal deduplication keys; the product may migrate its
identity representation without changing these independently generated inputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from tests.accounting.readers.readers_fixtures import (
    SyntheticHome,
    claude_prompt,
    claude_tool_result,
    claude_tool_use,
    codex_call,
    codex_meta,
    codex_output,
    codex_turn,
    jsonl,
    opencode_add_message,
    opencode_add_part,
    opencode_assistant_data,
    opencode_connect,
    opencode_tool_part,
    pi_header,
    pi_tool_call,
    pi_tool_result,
    pi_user,
    write_records,
)

MCP_SEARCH = "mcp__acme_recall__search_sessions"
OLD_FACT = "ZX-RETENTION-7731"  # witnessed 180 days before setup (V05)
LATER_FACT = "ZX-LATER-9942"  # first seen after corpus generation (V06)
GAP_SECONDS = (299, 28801)  # adjacent source gaps (V08); cap is 300 s per gap
GAP_CAP_SECONDS = 300
OVERLAP_TOKENS = {"input": 30, "output": 9, "totalTokens": 39}  # V04 debit stays 39


def stamp(base: datetime, **delta: float) -> str:
    return (base + timedelta(**delta)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def at(record: dict[str, Any], when: str) -> dict[str, Any]:
    return {**record, "timestamp": when}


def native_claude_records(session: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic valid native UUIDs, independent of product identity code."""
    return [
        {
            **record,
            "uuid": str(
                UUID(
                    hex=hashlib.sha256(f"review-corpus-v2:{session}:{index}".encode()).hexdigest()[
                        :32
                    ],
                    version=4,
                )
            ),
        }
        for index, record in enumerate(records)
    ]


def results_json(ref: str, excerpt: str) -> str:
    card = {"ref": ref, "title": "earlier session", "excerpt": excerpt, "files_touched": []}
    return json.dumps({"results": [card], "total_matches": 1, "omitted_count": 0})


@dataclass(frozen=True)
class Lookup:
    """One native lookup call and the outcome declared for it before any product run."""

    label: str
    gate: str
    runtime: str
    session_id: str
    call_id: str
    surface: str  # cli | mcp
    operation: str  # search | show
    argument: str  # query text or show ref
    ts: str
    counted: bool  # becomes its own execution/debit
    eligibility: str  # supported | rejected | unknown
    reason: str
    debit_tokens: int | None = None
    credit_seconds_max: int | None = None
    source_ref: str | None = None

    @property
    def locator(self) -> tuple[str, str, str]:
        return (self.runtime, self.session_id, self.call_id)


@dataclass
class OverlapCase:
    """omp file read live past an EOF baseline, then backfilled from offset 0 (V04)."""

    path: Path
    baseline_bytes: bytes
    live_bytes: bytes
    lookup: Lookup

    @property
    def baseline_offset(self) -> int:
        return len(self.baseline_bytes)

    def append_live(self) -> int:
        with self.path.open("ab") as handle:
            handle.write(self.live_bytes)
        return self.baseline_offset


@dataclass
class Corpus:
    home: SyntheticHome
    setup_at: datetime
    lookups: list[Lookup] = field(default_factory=list)
    overlap: OverlapCase | None = None
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    copies: dict[Path, Path] = field(default_factory=dict)  # resumed copy -> original (V03)

    def gate(self, name: str) -> list[Lookup]:
        return [item for item in self.lookups if item.gate == name]

    def expected_counts(self) -> dict[str, int]:
        counted = [item for item in self.lookups if item.counted]
        return {
            "unique_controls": len(self.gate("V01")),
            "executions": len(counted),
            "copied_files_adding_zero": len(self.copies),
            "not_imported_old_lookups": sum(not i.counted for i in self.gate("V05")),
            "known_debit_tokens": sum(i.debit_tokens or 0 for i in counted),
            "unknown_eligibility": sum(i.eligibility == "unknown" for i in counted),
            "known_rejected": sum(i.eligibility == "rejected" for i in counted),
        }


def _layout(root: Path) -> SyntheticHome:
    home = SyntheticHome(root=root)
    home.paths = {
        "claude": root / ".claude" / "projects",
        "codex": root / ".codex" / "sessions",
        "omp": root / ".omp" / "agent" / "sessions",
        "pi": root / ".pi" / "agent" / "sessions",
        "prime": root / ".prime" / "agent" / "sessions",
        "artifacts": root / ".prime" / "agent" / "session-artifacts",
        "opencode": root / ".local" / "share" / "opencode" / "opencode.db",
    }
    home.env = {
        "HOME": str(root),
        "CLAUDE_CONFIG_DIR": str(root / ".claude"),
        "CODEX_HOME": str(root / ".codex"),
        "OMP_SESSIONS_DIR": str(home.paths["omp"]),
        "PI_SESSION_DIR": str(home.paths["pi"]),
        "PRIME_AGENT_SESSION_DIR": str(home.paths["prime"]),
        "SSGREP_OPENCODE_DB": str(home.paths["opencode"]),
    }
    git = home.code / "alpha" / ".git"
    git.mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    return home


def _claude(corpus: Corpus) -> None:
    home, base = corpus.home, corpus.setup_at
    alpha, projects = home.repo("alpha"), home.paths["claude"]
    old_ref = "claude:src-old:ep:0"
    t = {k: stamp(base, minutes=m) for k, m in (("p", 60), ("c1", 61), ("r1", 62), ("c2", 70))}
    main = native_claude_records(
        "c-main",
        [
            claude_prompt("c-main", "find the retention token", uuid="u-c-main", ts=t["p"]),
            claude_tool_use(
                "c-main",
                "cl-1",
                MCP_SEARCH,
                {"query": "corpus alpha one", "limit": 5},
                message_id="m-cl-1",
                ts=t["c1"],
                cwd=alpha,
            ),
            claude_tool_result("c-main", "cl-1", results_json(old_ref, OLD_FACT), ts=t["r1"]),
            claude_tool_use(
                "c-main",
                "cl-2",
                "Bash",
                {"command": 'ssgrep search "corpus alpha two"'},
                message_id="m-cl-2",
                ts=t["c2"],
                cwd=alpha,
            ),
            claude_tool_result(
                "c-main", "cl-2", "(no results)", ts=stamp(base, minutes=70, seconds=2)
            ),
        ],
    )
    original = write_records(projects / "-work" / "c-main.jsonl", main)
    resumed = write_records(
        projects / "-work-resumed" / "c-resumed.jsonl",
        [{**record, "sessionId": "c-resumed"} for record in main],
    )
    corpus.copies[resumed] = original
    corpus.lookups += [
        Lookup(
            "L1",
            "V01",
            "claude",
            "c-main",
            "cl-1",
            "mcp",
            "search",
            "corpus alpha one",
            t["c1"],
            True,
            "supported",
            "old_source_authorized_and_indexed",
            credit_seconds_max=2,
            source_ref=old_ref,
        ),
        Lookup(
            "L2",
            "V01",
            "claude",
            "c-main",
            "cl-2",
            "cli",
            "search",
            "corpus alpha two",
            t["c2"],
            True,
            "rejected",
            "empty_result_known_no_credit",
        ),
    ]
    new_ts = stamp(base, minutes=80)
    write_records(
        projects / "-work" / "c-new.jsonl",
        native_claude_records(
            "c-new",
            [
                claude_prompt(
                    "c-new", "find the token", uuid="u-c-new", ts=stamp(base, minutes=79)
                ),
                claude_tool_use(
                    "c-new",
                    "cl-9",
                    MCP_SEARCH,
                    {"query": "corpus alpha one", "limit": 5},
                    message_id="m-cl-9",
                    ts=new_ts,
                    cwd=alpha,
                ),
                claude_tool_result(
                    "c-new",
                    "cl-9",
                    results_json(old_ref, f"{OLD_FACT} {LATER_FACT}"),
                    ts=stamp(base, minutes=81),
                ),
            ],
        ),
    )
    corpus.lookups.append(
        Lookup(
            "N1",
            "V03",
            "claude",
            "c-new",
            "cl-9",
            "mcp",
            "search",
            "corpus alpha one",
            new_ts,
            True,
            "unknown",
            "later_identifier_rarity_not_in_generation",
            source_ref=old_ref,
        )
    )
    identical_ts = stamp(base, minutes=90)
    write_records(
        projects / "-work" / "c-identical.jsonl",
        native_claude_records(
            "c-identical",
            [
                claude_prompt(
                    "c-identical", "another independent lookup", ts=stamp(base, minutes=89)
                ),
                claude_tool_use(
                    "c-identical",
                    "cl-identical",
                    MCP_SEARCH,
                    {"query": "corpus alpha one", "limit": 5},
                    message_id="m-cl-identical",
                    ts=identical_ts,
                    cwd=alpha,
                ),
                claude_tool_result(
                    "c-identical",
                    "cl-identical",
                    results_json(old_ref, OLD_FACT),
                    ts=stamp(base, minutes=91),
                ),
            ],
        ),
    )
    corpus.lookups.append(
        Lookup(
            "N2",
            "V03",
            "claude",
            "c-identical",
            "cl-identical",
            "mcp",
            "search",
            "corpus alpha one",
            identical_ts,
            True,
            "supported",
            "new_native_id_identical_output",
            source_ref=old_ref,
        )
    )
    old = base - timedelta(days=180)
    write_records(
        projects / "-work" / "src-old.jsonl",
        native_claude_records(
            "src-old",
            [
                claude_prompt(
                    "src-old", "why does retention fail", uuid="u-src-old", ts=stamp(old)
                ),
                claude_tool_use(
                    "src-old",
                    "w-old",
                    "Bash",
                    {"command": "cat retention.log"},
                    message_id="m-w-old",
                    ts=stamp(old, seconds=1),
                    cwd=alpha,
                ),
                claude_tool_result(
                    "src-old", "w-old", f"{OLD_FACT} expired", ts=stamp(old, seconds=2)
                ),
                claude_tool_use(
                    "src-old",
                    "cl-old",
                    "Bash",
                    {"command": 'ssgrep search "ancient"'},
                    message_id="m-cl-old",
                    ts=stamp(old, minutes=5),
                    cwd=alpha,
                ),
            ],
        ),
    )
    corpus.sources["src-old"] = {
        "ref": old_ref,
        "fact": OLD_FACT,
        "age_days": 180,
        "witness": ("claude", "src-old", "result:w-old"),
        "age_bars_valuation": False,
        "gaps_seconds": (1, 1),  # prompt -> call -> witnessed result; the old lookup is outside
        "capped_seconds": 2,
    }
    corpus.lookups.append(
        Lookup(
            "O1",
            "V05",
            "claude",
            "src-old",
            "cl-old",
            "cli",
            "search",
            "ancient",
            stamp(old, minutes=5),
            False,
            "rejected",
            "predates_setup_not_imported",
        )
    )


def _codex(corpus: Corpus) -> None:
    base, alpha = corpus.setup_at, corpus.home.repo("alpha")
    when, ref = stamp(base, minutes=90), "omp:o-src-gap:ep:0"
    write_records(
        corpus.home.paths["codex"] / "2026" / "rollout-x-main.jsonl",
        [
            at(codex_meta("x-main", alpha), stamp(base, minutes=89)),
            at(codex_turn("x-turn-1", alpha), stamp(base, minutes=89)),
            at(codex_call("cx-1", "exec_command", {"cmd": f"ssgrep show {ref}"}), when),
            at(codex_output("cx-1", "gap source body"), stamp(base, minutes=90, seconds=1)),
        ],
    )
    corpus.lookups.append(
        Lookup(
            "L3",
            "V01",
            "codex",
            "x-main",
            "cx-1",
            "cli",
            "show",
            ref,
            when,
            True,
            "supported",
            "gap_credit_capped_per_adjacent_gap",
            credit_seconds_max=sum(min(g, GAP_CAP_SECONDS) for g in GAP_SECONDS),
            source_ref=ref,
        )
    )


def _pi_family(corpus: Corpus) -> None:
    home, base, alpha = corpus.home, corpus.setup_at, corpus.home.repo("alpha")
    omp, pi = home.paths["omp"] / "-work", home.paths["pi"] / "-work"
    prime = home.paths["prime"] / "-work"
    t_om = stamp(base, minutes=100)
    usage = {"input": 8, "output": 4, "totalTokens": 12}
    write_records(
        omp / "o-main.jsonl",
        [
            at(pi_header("o-main", cwd=alpha), stamp(base, minutes=99)),
            at(
                pi_tool_call(
                    "e1",
                    "om-1",
                    "bash",
                    {"command": 'ssgrep search "corpus alpha five"'},
                    usage=usage,
                ),
                t_om,
            ),
            at(
                pi_tool_result("e2", "om-1", "ssgrep: index unavailable", is_error=True),
                stamp(base, minutes=100, seconds=1),
            ),
        ],
    )
    t_pi, t_pr = stamp(base, minutes=110), stamp(base, minutes=120)
    write_records(
        pi / "p-main.jsonl",
        [
            at(pi_header("p-main", cwd=alpha), stamp(base, minutes=109)),
            at(
                pi_tool_call(
                    "e1",
                    "pi-1",
                    "write",
                    {
                        "path": f"xd://{MCP_SEARCH}",
                        "content": json.dumps({"query": "corpus alpha six"}),
                    },
                ),
                t_pi,
            ),
            at(
                pi_tool_result("e2", "pi-1", json.dumps({"results": [], "total_matches": 0})),
                stamp(base, minutes=110, seconds=1),
            ),
        ],
    )
    missing = "prime-agent:never-written:ep:0"
    write_records(
        prime / "m-main.jsonl",
        [
            at(pi_header("m-main", cwd=alpha), stamp(base, minutes=119)),
            at(
                pi_tool_call(
                    "e1", "pr-1", "bash", {"command": 'ssgrep search "corpus alpha seven"'}
                ),
                t_pr,
            ),
            at(
                pi_tool_result("e2", "pr-1", results_json(missing, "unresolvable excerpt")),
                stamp(base, minutes=120, seconds=1),
            ),
        ],
    )
    corpus.lookups += [
        Lookup(
            "L4",
            "V01",
            "omp",
            "o-main",
            "om-1",
            "cli",
            "search",
            "corpus alpha five",
            t_om,
            True,
            "rejected",
            "error_result_known_no_credit",
            debit_tokens=12,
        ),
        Lookup(
            "L5",
            "V01",
            "pi",
            "p-main",
            "pi-1",
            "mcp",
            "search",
            "corpus alpha six",
            t_pi,
            True,
            "rejected",
            "empty_result_known_no_credit",
        ),
        Lookup(
            "L6",
            "V01",
            "prime-agent",
            "m-main",
            "pr-1",
            "cli",
            "search",
            "corpus alpha seven",
            t_pr,
            True,
            "unknown",
            "source_unavailable_not_zero",
            source_ref=missing,
        ),
    ]
    # V08: a source whose adjacent records sit 299 s and 28801 s apart.
    g0 = base - timedelta(days=3)
    g1, g2 = g0 + timedelta(seconds=GAP_SECONDS[0]), g0 + timedelta(seconds=sum(GAP_SECONDS))
    write_records(
        omp / "o-src-gap.jsonl",
        [
            at(pi_header("o-src-gap", cwd=alpha), stamp(g0)),
            at(pi_user("e1", "inspect the gap"), stamp(g0)),
            at(pi_tool_call("e2", "g-1", "read", {"path": "notes.md"}), stamp(g1)),
            at(pi_tool_result("e3", "g-1", "gap source body"), stamp(g2)),
        ],
    )
    corpus.sources["o-src-gap"] = {
        "ref": "omp:o-src-gap:ep:0",
        "gaps_seconds": GAP_SECONDS,
        "uncapped_seconds": sum(GAP_SECONDS),
        "capped_seconds": sum(min(g, GAP_CAP_SECONDS) for g in GAP_SECONDS),
        "witness": ("omp", "o-src-gap", "result:g-1"),
    }
    # V04: EOF baseline, live append, then overlapping backfill from offset 0.
    t_live = stamp(base, minutes=130)
    baseline = [
        at(pi_header("o-live", cwd=alpha), stamp(base, minutes=129)),
        at(pi_user("e1", "earlier turn before baseline"), stamp(base, minutes=129)),
    ]
    live = [
        at(
            pi_tool_call(
                "e2",
                "om-live-1",
                "bash",
                {"command": 'ssgrep search "corpus live"'},
                usage=OVERLAP_TOKENS,
            ),
            t_live,
        ),
        at(pi_tool_result("e3", "om-live-1", "(no results)"), stamp(base, minutes=130, seconds=1)),
    ]
    lookup = Lookup(
        "B1",
        "V04",
        "omp",
        "o-live",
        "om-live-1",
        "cli",
        "search",
        "corpus live",
        t_live,
        True,
        "rejected",
        "live_and_backfill_debit_once",
        debit_tokens=OVERLAP_TOKENS["totalTokens"],
    )
    path = write_records(omp / "o-live.jsonl", baseline)
    corpus.overlap = OverlapCase(path, jsonl(baseline), jsonl(live), lookup)
    corpus.lookups.append(lookup)


def _opencode(corpus: Corpus) -> None:
    base, alpha = corpus.setup_at, corpus.home.repo("alpha")
    ms = int((base + timedelta(minutes=140)).timestamp() * 1000)
    connection = opencode_connect(corpus.home.paths["opencode"])
    connection.execute(
        "INSERT INTO session VALUES (?,?,?,?,?,?,?)",
        ("ses-main", "proj", None, alpha, "corpus", ms, ms),
    )
    opencode_add_message(connection, "msg-main", "ses-main", opencode_assistant_data(), at=ms)
    old_ref = corpus.sources["src-old"]["ref"]
    part = opencode_tool_part(
        "oc-1",
        "bash",
        {"command": 'ssgrep search "corpus alpha four"'},
        output=results_json(old_ref, OLD_FACT),
        start=ms,
        end=ms + 900,
    )
    opencode_add_part(connection, "prt-main", "msg-main", "ses-main", part, at=ms)
    connection.close()
    corpus.lookups.append(
        Lookup(
            "L7",
            "V01",
            "opencode",
            "ses-main",
            "oc-1",
            "cli",
            "search",
            "corpus alpha four",
            stamp(base, minutes=140),
            True,
            "supported",
            "old_source_authorized_and_indexed",
            credit_seconds_max=2,
            source_ref=old_ref,
        )
    )


def build_corpus(root: Path, setup_at: datetime) -> Corpus:
    """Populate ``root`` with the acceptance corpus; ``setup_at`` must be tz-aware UTC."""
    if setup_at.tzinfo is None or setup_at.utcoffset() != timedelta(0):
        raise ValueError("setup_at must be an aware UTC datetime")
    corpus = Corpus(home=_layout(root), setup_at=setup_at.replace(microsecond=0))
    _claude(corpus)
    _codex(corpus)
    _pi_family(corpus)
    _opencode(corpus)
    return corpus


FROZEN_SETUP = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
