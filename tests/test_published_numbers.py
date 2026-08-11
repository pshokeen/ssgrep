"""Guards README.md's published headline retrieval numbers against the
committed eval artifact they cite, and the shipped MAIN_SESSION_BOOST against
the boost that artifact was actually measured at.

This project has hit exactly this failure mode twice: a number published in a
doc drifted out of sync with the eval/results/*.json artifact that was
supposed to back it, and nothing caught it automatically -- that absence is
why README.md and eval/README.md's tables were able to drift out of sync with
the artifacts they cite without anything going red. It is also the shape of
the defect that prompted this module: MAIN_SESSION_BOOST was tuned to 0.0 by
the eval harness and never shipped, so the code and the numbers describing
the code silently disagreed for an unknown stretch of time. This module is
the guard against a third occurrence.

Deliberately narrow, not a general markdown-number parser (which would be
brittle and would itself rot): it targets exactly the one canonical
"| **Overall** | **87.5%** | **0.5871** |" row README.md publishes, and the
shipped MAIN_SESSION_BOOST constant. Nothing else in either file is parsed.
(This row read 77.8% / 0.558 before the 2026-08-06 label regeneration.)

Tolerance, not equality: recall@10 and MRR are measured against a live,
continuously-growing corpus (~/.claude/projects, still being written to by
concurrent agents at measurement time -- see the full methodology in
eval/results/noise_floor_2026-07-27.json). An exact-match assertion would
fail on every legitimate re-measurement and get deleted within a week, which
is worse than no test at all. The tolerances below bound CORPUS SENSITIVITY,
not measurement noise: a dedicated study ran 152 same-config re-measurements
across 11 corpus states and found 60 independent full rebuilds at an
UNCHANGED corpus bit-identical on every metric, zero variance -- this
pipeline has no inherent randomness anywhere in indexing, embedding, BM25,
or ranking. Every tolerance below exists because the corpus itself can
genuinely differ between two measurements, not because repeated runs at a
fixed corpus would disagree (they don't, ever, per that proof). See
eval/results/noise_floor_2026-07-27.json's `provenance.corpus_
determinism_proof` and `floors` blocks:
  - recall@10: a step function over n labelled queries -- the smallest
    possible non-zero move is one query, 1/n (currently 1/48 =~ 0.0208).
  - MRR: continuous; ~0.012 is the measured cross-experiment corpus drift
    (same code, same config, minutes apart -- the only thing that can
    differ, and does, is the live corpus -- see noise_floor_2026-07-27.json's
    `floors.mrr.cross_experiment_*` fields for exactly how this was
    measured).

MAIN_SESSION_BOOST is checked for exact equality, not tolerance: it is a
discrete shipped constant, not a corpus-sensitive measurement, so any
disagreement between what search.py ships and what the cited artifact was
built at is a real defect, not something a corpus difference could ever
cause.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ssgrep import search

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"

# Hardcoded to the specific, dated artifact README.md's headline numbers are
# actually drawn from -- not a glob. If retrieval
# quality is re-measured under a new dated filename, updating README.md's
# numbers and repointing this constant at the new artifact are the same
# change; a test that silently followed a glob to whichever file matched
# would not catch a case where only one of the two was updated.
ARTIFACT_PATH = REPO_ROOT / "eval" / "results" / "baseline_n66_2026-08-07b.json"

# Anchored specifically to the "Overall" row of README's retrieval-quality
# table: "| **Overall** | **87.5%** | **0.5871** |" (or, since MRR is
# genuinely only accurate to this corpus-sensitivity floor, "| **Overall** |
# **87.5%** | **≈0.59** |" -- the optional leading '≈' is tolerated in both
# numeric groups so a doc author hedging precision doesn't break the guard).
# Historical note: this row read 77.8% / 0.558 before the 2026-08-06 label
# regeneration (see eval/README.md's "Update 2026-08-06" section).
# Not a general markdown table/number parser -- it matches nothing else in
# the file.
_OVERALL_ROW_RE = re.compile(
    r"\|\s*\*\*Overall\*\*\s*\|\s*\*\*≈?\s*([\d.]+)%\*\*\s*\|\s*\*\*≈?\s*([\d.]+)\*\*\s*\|"
)


def _read_readme_headline() -> tuple[float, float]:
    """(recall_at_10 as a fraction, mrr) published in README's Overall row.

    Raises an assertion failure naming exactly what went wrong if the row
    can't be found -- a regex miss must never surface as a silent None or an
    unrelated TypeError several lines later; it must read as "the table
    format changed, update this guard" on its own.
    """
    text = README_PATH.read_text()
    match = _OVERALL_ROW_RE.search(text)
    assert match is not None, (
        f"Could not find the '| **Overall** | **X%** | **Y** |' row in "
        f"{README_PATH} -- either the retrieval-quality table's format "
        f"changed (update _OVERALL_ROW_RE in {__file__}) or the row was "
        f"removed (then this guard should be removed too, deliberately, not "
        f"left to fail forever)."
    )
    recall_pct, mrr = match.groups()
    return float(recall_pct) / 100.0, float(mrr)


def _read_artifact() -> dict:
    assert ARTIFACT_PATH.exists(), (
        f"{ARTIFACT_PATH} is missing -- README.md's headline retrieval "
        f"numbers have no committed artifact left to check against. Either "
        f"restore it or repoint ARTIFACT_PATH in {__file__} at whatever "
        f"artifact now backs README's numbers."
    )
    return json.loads(ARTIFACT_PATH.read_text())


def _recall_floor(artifact: dict) -> float:
    """1/n, n = the labelled query count actually recorded in the artifact
    being compared against -- not a hardcoded 36, so this stays correct if
    the label set ever grows past its current size."""
    n = artifact["summary"]["overall"]["n"]
    return 1 / n


# Bounds CORPUS DRIFT between two measurements taken at different times, not
# measurement error -- this pipeline has zero inherent noise (measured:
# 60 independent full rebuilds at an unchanged corpus were bit-identical on
# every metric; see eval/results/noise_floor_2026-07-27.json's
# `provenance.corpus_determinism_proof`). 0.012 is the measured delta
# between two same-code/same-config runs ~9 minutes apart, where the corpus
# was the only thing able to differ -- see `floors.mrr` in that same file
# for exactly how this was measured, and `floors.mrr.
# cross_experiment_30min_observed` (0.0162) for evidence this may still
# understate drift over longer gaps. Left at 0.012, not widened to match
# that larger figure: this guard is calibrated and working, and the
# 30-minute number is recorded as guidance for whoever next revisits this
# constant, not adopted here. Not derivable from a simple formula the way
# the recall floor is, so it is a named, commented constant rather than
# computed.
MRR_CROSS_EXPERIMENT_FLOOR = 0.012


class TestReadmeHeadlineMatchesArtifact:
    def test_readme_overall_recall_matches_artifact(self) -> None:
        readme_recall, _readme_mrr = _read_readme_headline()
        artifact = _read_artifact()
        artifact_recall = artifact["summary"]["overall"]["recall_at_10"]
        floor = _recall_floor(artifact)
        delta = abs(readme_recall - artifact_recall)
        assert delta <= floor, (
            f"README.md publishes overall recall@10 = {readme_recall:.4f}, but "
            f"{ARTIFACT_PATH.name} (the artifact those numbers cite) measures "
            f"{artifact_recall:.4f} -- a delta of {delta:.4f}, larger than the "
            f"measured corpus-sensitivity floor ({floor:.4f}, one query at "
            f"n={artifact['summary']['overall']['n']}). Either README.md is "
            f"stale and needs updating, or the artifact needs regenerating -- "
            f"this is exactly the drift eval/results/noise_floor_2026-07-27.json "
            f"exists to catch."
        )

    def test_readme_overall_mrr_matches_artifact(self) -> None:
        _readme_recall, readme_mrr = _read_readme_headline()
        artifact = _read_artifact()
        artifact_mrr = artifact["summary"]["overall"]["mrr"]
        delta = abs(readme_mrr - artifact_mrr)
        assert delta <= MRR_CROSS_EXPERIMENT_FLOOR, (
            f"README.md publishes overall MRR = {readme_mrr:.4f}, but "
            f"{ARTIFACT_PATH.name} (the artifact those numbers cite) measures "
            f"{artifact_mrr:.4f} -- a delta of {delta:.4f}, larger than the "
            f"measured cross-experiment corpus-sensitivity floor "
            f"(±{MRR_CROSS_EXPERIMENT_FLOOR}). Either README.md is stale "
            f"and needs updating, or the artifact needs regenerating."
        )

    def test_shipped_boost_matches_artifact_provenance(self) -> None:
        """The assertion that would have caught the original defect: docs
        describing a configuration the product does not ship.

        Exact equality, not tolerance -- MAIN_SESSION_BOOST is a discrete
        shipped constant. If it changes without regenerating (or repointing
        README and ARTIFACT_PATH at a fresh) baseline artifact, this fails
        loudly instead of leaving README's numbers silently attributed to a
        boost nothing ships anymore.
        """
        artifact = _read_artifact()
        artifact_boost = artifact["main_session_boost"]
        assert search.MAIN_SESSION_BOOST == artifact_boost, (
            f"Shipped search.MAIN_SESSION_BOOST ({search.MAIN_SESSION_BOOST}) "
            f"does not match the boost recorded in {ARTIFACT_PATH.name} "
            f"({artifact_boost}) -- README.md's headline numbers are "
            f"attributed to this artifact, which no longer describes what the "
            f"product ships."
        )


# ---------------------------------------------------------------------------
# noise_floor_2026-07-27.json's DERIVED gap fields vs. the source artifacts
# they're derived from.
#
# Scope, deliberately narrow: this catches one observed failure mode --
# committed_model_ab_gap/committed_chunking_ab_gap are pure
# arithmetic (retrieval-32M/per_turn's overall value minus base-8M/uniform's)
# read out of model_ab_2026-07-27.json / chunking_ab_2026-07-27.json, and
# went stale when those two files were regenerated without this file being
# updated to match. That's a numeric equality a test CAN check.
#
# It deliberately does NOT and CANNOT check the INTERPRETIVE fields next to
# those numbers (mrr_verdict, mrr_gap_as_multiple_of_own_spread): the
# *verdict* has been wrong while the *gap value* beside it was already
# correct -- the ratio used the wrong denominator (a single-arm instantaneous
# floor) because of an assumption (matched-pair builds cancel corpus
# sensitivity) that looked reasonable and was only settled by
# actually running a dedicated 36-sample study
# (matched_pair_gap_study_2026-07-28.json) that falsified it. No cheap,
# robust check distinguishes a well-reasoned-but-wrong methodological
# assumption from a correct one without doing that same work -- that's a
# job for a study, not a unit test. Considered writing one anyway (e.g.
# asserting the ratio field equals gap/denominator using whatever
# denominator is cited alongside it) and rejected it: that would only prove
# the arithmetic was carried out correctly, which was never what went wrong
# here, so it would pass on a stale-but-self-consistent verdict and create
# false confidence. Recorded here rather than built.
# ---------------------------------------------------------------------------

NOISE_FLOOR_PATH = REPO_ROOT / "eval" / "results" / "noise_floor_2026-07-27.json"
MODEL_AB_PATH = REPO_ROOT / "eval" / "results" / "model_ab_2026-07-27.json"
CHUNKING_AB_PATH = REPO_ROOT / "eval" / "results" / "chunking_ab_2026-07-27.json"
_GAP_FLOAT_TOLERANCE = 1e-9  # float round-trip slack, not a measurement floor


def _load(path: Path) -> dict:
    assert path.exists(), f"{path} is missing."
    return json.loads(path.read_text())


class TestNoiseFloorGapFieldsMatchSourceArtifacts:
    def test_model_ab_gap_matches_source_files(self) -> None:
        noise_floor = _load(NOISE_FLOOR_PATH)
        model_ab = _load(MODEL_AB_PATH)
        gap = noise_floor["corpus_sensitivity"]["committed_model_ab_gap"]
        base = model_ab["potion-base-8M"]["summary"]["overall"]
        retr = model_ab["potion-retrieval-32M"]["summary"]["overall"]

        for metric, recorded in (
            ("recall_at_10", gap["overall_recall_at_10"]),
            ("mrr", gap["overall_mrr"]),
        ):
            expected = retr[metric] - base[metric]
            delta = abs(recorded - expected)
            assert delta <= _GAP_FLOAT_TOLERANCE, (
                f"{NOISE_FLOOR_PATH.name}'s corpus_sensitivity."
                f"committed_model_ab_gap.overall_{metric} ({recorded}) no "
                f"longer matches (retrieval-32M - base-8M) freshly computed "
                f"from {MODEL_AB_PATH.name} ({expected}) -- delta {delta}. "
                f"{MODEL_AB_PATH.name} was regenerated after this gap field "
                f"was last computed; recompute committed_model_ab_gap "
                f"against the current file."
            )

    def test_chunking_ab_gap_matches_source_files(self) -> None:
        noise_floor = _load(NOISE_FLOOR_PATH)
        chunking_ab = _load(CHUNKING_AB_PATH)
        gap = noise_floor["corpus_sensitivity"]["committed_chunking_ab_gap"]
        uniform = chunking_ab["uniform"]["summary"]["overall"]
        per_turn = chunking_ab["per_turn"]["summary"]["overall"]

        for metric, recorded in (
            ("recall_at_10", gap["overall_recall_at_10"]),
            ("mrr", gap["overall_mrr"]),
        ):
            expected = per_turn[metric] - uniform[metric]
            delta = abs(recorded - expected)
            assert delta <= _GAP_FLOAT_TOLERANCE, (
                f"{NOISE_FLOOR_PATH.name}'s corpus_sensitivity."
                f"committed_chunking_ab_gap.overall_{metric} ({recorded}) no "
                f"longer matches (per_turn - uniform) freshly computed from "
                f"{CHUNKING_AB_PATH.name} ({expected}) -- delta {delta}. "
                f"{CHUNKING_AB_PATH.name} was regenerated after this gap "
                f"field was last computed; recompute "
                f"committed_chunking_ab_gap against the current file."
            )


# ---------------------------------------------------------------------------
# eval/README.md's Result 2 snapshot table (model A/B) vs. model_ab_2026-07-27.json
#
# This guard's scope was once too narrow -- it checked README.md's Overall
# row and noise_floor.json's derived gap fields, but not eval/README.md's own
# A/B table, so when model_ab_2026-07-27.json was regenerated (repairing a
# mutation-testing corruption), every cell went stale silently, including a
# bolded "win" that the artifact had turned into a regression.
#
# What's covered, and why it stops there:
#
# - Dimension (256 / 512) and the shipped boost (both arms' provenance):
#   exact equality. These are architectural/config facts, not corpus-
#   sensitive measurements -- there is no legitimate reason for them to
#   differ from the artifact, ever.
# - Overall recall@10 and per-class recall@10 for all 5 classes, both arms:
#   tolerance = 1/n (n read from the artifact's own recorded `n` per class,
#   not hardcoded -- subagent-only has n=13, not n=9 like the other four).
#   Exact, not estimated: recall@10 is a step function, so 1/n is the
#   smallest possible non-zero move by construction, not a measured guess.
#
# What's deliberately NOT covered, and why a tolerance check there would be
# worse than no check:
#
# - Per-class MRR (10 cells: 5 classes x 2 arms). Measured floors (spread
#   across the 36-run matched_pair_gap_study, same methodology as every
#   other floor in this file) range from 0.0063 (paraphrase) to 0.1051
#   (subagent-only) -- a 17x range. Critically, the classes with the
#   LARGEST floors are exactly the ones where the published table's winner
#   flipped after the corruption-repair regeneration (error-string MRR:
#   published gap 0.075 < its own measured floor 0.085; subagent-only MRR:
#   same pattern) -- a tolerance check using each class's own honestly-
#   measured floor would have PASSED both of those flips as "not stale,"
#   because they're genuinely within normal corpus-to-corpus variation for
#   those specific classes. That is a check which looks like verification
#   and provides none for exactly the failure mode it would exist to catch.
#   Overall MRR is still covered above (README.md's own guard) with its
#   established, meaningfully tight ~0.012 floor; per-class MRR is not.
# - The bold/emphasis markup marking which arm "wins" a cell: as of this
#   writing eval/README.md's table carries no bold markup at all (removed
#   in a later revision, alongside an explicit reader-facing disclaimer
#   that "no per-class finding in this document rests on this table by
#   itself"), so there is nothing to check today -- and if it returns,
#   verifying "is this specific character span bold" reliably across cells
#   that can have 0, 1, or 2 bolded numbers is exactly the kind of fragile,
#   confusing-failure-mode check this module has already declined once
#   (the interpretive-verdict-field check, above).
# - The W/T/L distribution table right after the snapshot (eval/README.md
#   lines ~403-410): independently re-derived from
#   matched_pair_gap_study_2026-07-28.json's raw per-run records before
#   writing this comment and found to match exactly. That file is a dated,
#   one-time historical study, not something `-m eval.model_ab` regenerates
#   -- it does not carry the same silent-staleness risk the snapshot table
#   does, so it isn't included here.
# ---------------------------------------------------------------------------

EVAL_README_PATH = REPO_ROOT / "eval" / "README.md"

_MODEL_AB_TABLE_ROW_LABELS = (
    "overall",
    "error-string",
    "exact-identifier",
    "multi-hop",
    "paraphrase",
    "subagent-only",
)
_MODEL_AB_ARTIFACT_KEYS = {
    "overall": "overall",
    "error-string": "class:error-string",
    "exact-identifier": "class:exact-identifier",
    "multi-hop": "class:multi-hop",
    "paraphrase": "class:paraphrase",
    "subagent-only": "subagent_only",
}


_RESULT_2_SECTION_RE = re.compile(r"^## Result 2 —.*?(?=^## Result 3 —)", re.MULTILINE | re.DOTALL)


def _read_result_2_section() -> str:
    """Isolate the Result 2 (model A/B) section's text before running any
    row regex within it. eval/README.md has multiple sections whose tables
    reuse the EXACT SAME row labels ('subagent-only recall@10 / MRR' appears
    in Results 1, 2, and 3) -- a plain .search() over the whole file would
    silently lock onto whichever section happens to come first in document
    order, which today is Result 2 by luck, not by anything this function
    verifies. Confirmed this matters, not just theoretically: Result 3's
    row for that exact label is byte-for-byte format-identical to Result
    2's and would have matched just as validly."""
    text = EVAL_README_PATH.read_text()
    match = _RESULT_2_SECTION_RE.search(text)
    assert match is not None, (
        f"Could not isolate the '## Result 2 —...## Result 3 —' section in "
        f"{EVAL_README_PATH} -- section headings changed, update "
        f"_RESULT_2_SECTION_RE in {__file__}."
    )
    return match.group(0)


def _model_ab_row_re(label: str) -> re.Pattern:
    """'| <label> recall@10 / MRR | X% / ≈Y | A% / ≈B |' -- captures only
    the two recall percentages, deliberately ignoring the MRR halves (see
    class comment above for why). Bold markers tolerated but not required,
    matching this file's other regexes' defensive style."""
    return re.compile(
        r"\|\s*" + re.escape(label) + r" recall@10 / MRR\s*\|\s*"
        r"\*{0,2}([\d.]+)%\*{0,2}\s*/[^|]*\|\s*"
        r"\*{0,2}([\d.]+)%\*{0,2}\s*/"
    )


def _read_eval_readme_model_ab_row(label: str) -> tuple[float, float]:
    section = _read_result_2_section()
    match = _model_ab_row_re(label).search(section)
    assert match is not None, (
        f"Could not find the '{label} recall@10 / MRR' row in "
        f"{EVAL_README_PATH}'s Result 2 section -- either the table format "
        f"changed (update _model_ab_row_re in {__file__}) or the row was "
        f"removed (then this guard should be removed too, deliberately)."
    )
    base_pct, retr_pct = match.groups()
    return float(base_pct) / 100.0, float(retr_pct) / 100.0


def _read_eval_readme_dimensions() -> tuple[int, int]:
    section = _read_result_2_section()
    match = re.search(
        r"\|\s*dimension\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\(measured, not assumed\)\s*\|",
        section,
    )
    assert match is not None, (
        f"Could not find the '| dimension | X | Y (measured, not assumed) |' "
        f"row in {EVAL_README_PATH}'s Result 2 section -- table format "
        f"changed, update the regex in {__file__}."
    )
    return int(match.group(1)), int(match.group(2))


def _read_model_ab_artifact() -> dict:
    assert MODEL_AB_PATH.exists(), f"{MODEL_AB_PATH} is missing."
    return json.loads(MODEL_AB_PATH.read_text())


class TestEvalReadmeModelABTableMatchesArtifact:
    def test_shipped_boost_matches_this_artifact(self) -> None:
        artifact = _read_model_ab_artifact()
        for label in ("potion-base-8M", "potion-retrieval-32M"):
            artifact_boost = artifact[label]["provenance"]["tuning_constants"]["MAIN_SESSION_BOOST"]
            assert search.MAIN_SESSION_BOOST == artifact_boost, (
                f"Shipped search.MAIN_SESSION_BOOST ({search.MAIN_SESSION_BOOST}) "
                f"does not match the boost {MODEL_AB_PATH.name}'s {label!r} "
                f"arm was built at ({artifact_boost}) -- eval/README.md's "
                f"Result 2 table is attributed to this artifact at "
                f"'the shipped value'."
            )

    def test_dimensions_match_artifact(self) -> None:
        readme_base_dim, readme_retr_dim = _read_eval_readme_dimensions()
        artifact = _read_model_ab_artifact()
        artifact_base_dim = artifact["potion-base-8M"]["dimension"]
        artifact_retr_dim = artifact["potion-retrieval-32M"]["dimension"]
        assert (readme_base_dim, readme_retr_dim) == (artifact_base_dim, artifact_retr_dim), (
            f"eval/README.md publishes dimensions ({readme_base_dim}, "
            f"{readme_retr_dim}) but {MODEL_AB_PATH.name} records "
            f"({artifact_base_dim}, {artifact_retr_dim}). This is an exact "
            f"architectural fact, not a corpus-sensitive measurement -- any "
            f"disagreement is a real defect, not drift."
        )

    def test_recall_rows_match_artifact(self) -> None:
        artifact = _read_model_ab_artifact()
        base_summary = artifact["potion-base-8M"]["summary"]
        retr_summary = artifact["potion-retrieval-32M"]["summary"]

        for label in _MODEL_AB_TABLE_ROW_LABELS:
            readme_base, readme_retr = _read_eval_readme_model_ab_row(label)
            key = _MODEL_AB_ARTIFACT_KEYS[label]
            artifact_base = base_summary[key]["recall_at_10"]
            artifact_retr = retr_summary[key]["recall_at_10"]
            n = base_summary[key]["n"]
            floor = 1 / n

            for arm_label, readme_val, artifact_val in (
                ("potion-base-8M", readme_base, artifact_base),
                ("potion-retrieval-32M", readme_retr, artifact_retr),
            ):
                delta = abs(readme_val - artifact_val)
                assert delta <= floor, (
                    f"eval/README.md's Result 2 table publishes {label!r} "
                    f"recall@10 = {readme_val:.4f} for {arm_label}, but "
                    f"{MODEL_AB_PATH.name} measures {artifact_val:.4f} -- a "
                    f"delta of {delta:.4f}, larger than the exact "
                    f"corpus-sensitivity floor for n={n} (1/{n} = "
                    f"{floor:.4f}, one query). Either eval/README.md is "
                    f"stale and needs updating, or the artifact needs "
                    f"regenerating."
                )


# ---------------------------------------------------------------------------
# Declined: a guard on eval/README.md's episode-count prose. Optional, not
# blocking, and declining is the answer after actually looking, not by
# default.
#
# Three prose passages in eval/README.md cite episode counts, and the
# obvious fix is shaped like the checks above: assert an episode-range
# sentence's two bounds equal the min and max of the four committed
# artifacts' recorded `episode_count`. Each of the three was read before
# deciding, rather than reasoning about it in the abstract, and that design
# fits none of them, for two different reasons -- not "the wording might
# change" as a hypothetical, but as directly observed:
#
# 1. Two of the three are explicitly historical, point-in-time narrations,
#    not current-state claims: "(3,360 in the boost files; 3,453/3,456 in
#    one model/chunking A/B regeneration; 3,530 in the next; ...)" and
#    "(3,530 at the time of the committed run above)". Both are scoped in
#    their own text to a specific past moment. Pinning either to "does this
#    match a CURRENT artifact's episode_count" would be checking the wrong
#    thing on purpose -- these numbers are correct precisely because they
#    no longer match anything current; that's the point being made.
# 2. The third -- the one actually shaped like a range claim -- was
#    rewritten while this was being considered, from an exact
#    range to an open-ended bound: "stable
#    across at least a 3,360-to-4,157-episode range ... so far, an
#    open-ended bound rather than a fixed one ... any future regeneration
#    can extend it further without ever falsifying it." Its upper number is
#    explicitly sourced from an ad hoc, uncommitted `eval.harness` run
#    "taken while writing this sentence," not from any of the four
#    committed artifacts that design would check against -- so that
#    design is not just risky, it is inapplicable to this sentence as
#    currently written, by the sentence's own stated methodology.
#
# eval/README.md changed twice in the few minutes spent looking at it for
# this decision (confirmed: two reads of the same section, minutes apart,
# returned different episode numbers) -- direct, observed evidence for
# the standing caution to settle the final wording before pinning anything
# to it, not a reason invented to justify declining. Revisit if
# a future revision states a genuine current-state range sourced from
# committed artifacts and holds still long enough to pin.
