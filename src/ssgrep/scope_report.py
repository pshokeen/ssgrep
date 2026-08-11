"""Why discovery found nothing: the census behind "Indexed 0 sessions".

"Indexed 0 sessions, 0 episodes, 0 chunks. (exit 0)" is the single observable
common to every scoping failure this tool has had -- the moved-project bug,
the case-folding bug, the symlink bug, the subdirectory bug, and the
CLAUDE_CONFIG_DIR bug all surface as exactly that line. It is also what a
genuinely empty project prints. A buyer cannot tell the two apart, and the
natural response to "my index looks empty" is ``ssgrep index --rebuild``,
which is precisely the command that destroys a healthy index (see
rebuild_guard). Making zero self-explaining is therefore not a nicety; it is
what stops the user from reaching for the destructive remedy.

The census answers the five questions that separate "misconfigured" from
"empty", in the order a user needs them:

1. Which scope was actually matched against, *after* canonicalization -- the
   case bug is invisible unless the folded form is shown.
2. Which transcript root was actually scanned, *after* CLAUDE_CONFIG_DIR
   resolution -- the relocation bug is invisible unless the root is named.
3. How many transcripts exist under that root at all. Zero here means
   "nothing recorded yet"; a large number means "recorded, but not yours".
4. How many of them scope rejected.
5. The most common ``cwd`` those rejected transcripts recorded, with counts.

(5) is what makes the move bug self-diagnosing. A user who reads
``2041 transcripts recorded cwd=/Users/me/code/old`` needs no further
explanation and no support ticket: the old path is right there, and it is
also the argument to the command that fixes it -- shell-quoted, and offered
only when one cwd genuinely dominates the histogram (see :func:`remedy_cwd`).

The census is data, not only prose. :func:`ScopeReport.as_payload` is the same
evidence as a JSON-serializable dict, so ``--json`` callers and the MCP server
get it too. They are not a lesser audience here: ssgrep is consumed by Claude
Code over MCP, so the caller most likely to see "empty" and reach for
``ssgrep index --rebuild`` is the one that cannot read a terminal message.

Enumeration deliberately goes through :func:`discovery.iter_transcript_files`
and :func:`discovery.cwd_index_for` rather than re-walking the corpus here.
A second, independent walk would be free to disagree with the one discovery
performed -- and a diagnostic that reports a total the indexer never saw is
worse than no diagnostic, because it is believed.
"""

from __future__ import annotations

import shlex
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ssgrep import discovery, paths, textsafe

NO_CWD_RECORDED = "(no cwd recorded)"
"""Histogram label for transcripts that record no ``cwd`` at all.

Real on real corpora: summary- and compaction-only transcript files carry no
``cwd`` key. It belongs in the histogram, where it is informative, and must
never reach the suggested command -- ``--project-dir (no cwd recorded)`` is
not a wrong path, it is a bash syntax error. See :func:`remedy_cwd`.
"""

DOMINANCE = 0.5
"""How much of the rejected corpus one ``cwd`` must hold to be called a move.

The "your project moved" remedy names a single directory and tells the buyer
their sessions are there. That claim is only true when the histogram actually
has a dominant value, which is what a move produces. The common benign case --
a buyer with many projects running ``ssgrep index`` in a brand-new directory --
produces a FLAT histogram, where ``Counter.most_common`` breaks the tie by
filesystem walk order and would confidently point them at an arbitrary
unrelated project. Below this share, the census is still shown; the remedy is
not.
"""

TOP_REJECTED_CWDS = 3
"""How many distinct rejected ``cwd`` values to name.

Enough to expose a move (which produces one dominant value) and to show that
the rest of the corpus belongs to other projects, without turning an error
message into a corpus dump. A buyer with fifty projects would otherwise get
fifty lines and read none of them.
"""


@dataclass(frozen=True)
class ScopeReport:
    """The evidence behind a zero-discovery result.

    ``scope`` is the live, on-disk spelling handed to discovery;
    ``scope_canonical`` is what matching actually compared against. They
    differ on a case-insensitive filesystem, and that difference IS the case
    bug -- reporting only one of them hides it.

    ``total_transcripts`` counts every file discovery was willing to look at
    under ``transcript_root`` (sidecars already excluded). ``rejected_count``
    is how many of those scope turned away. ``top_rejected_cwds`` maps a
    recorded ``cwd`` to the number of rejected transcripts recording it,
    highest first; a transcript recording several distinct cwds counts once
    against each.
    """

    scope: str
    scope_canonical: str
    transcript_root: Path
    root_exists: bool
    total_transcripts: int
    rejected_count: int
    top_rejected_cwds: tuple[tuple[str, int], ...]
    #: Rejected cwds whose final path component equals the scope's -- the
    #: moved-repo / org-rename signature (same project name, different
    #: parent). Computed from the FULL rejection histogram, not the top-N
    #: truncation, because the sibling that matters may not be a top
    #: rejector on a busy machine. Empty tuple when there is none.
    same_basename_rejected: tuple[tuple[str, int], ...] = ()

    @property
    def root_is_empty(self) -> bool:
        """True when there is nothing under the root to have rejected.

        The "you have not recorded any sessions yet" case, which must be said
        plainly instead of being dressed up as a scope failure with an empty
        histogram beneath it.
        """
        return self.total_transcripts == 0

    def as_payload(self) -> dict[str, object]:
        """The census as JSON-serializable plain data.

        For ``--json`` and MCP callers, which get no rendered text at all and
        would otherwise see ``session_count: 0`` / ``index_empty: true`` with
        no explanation on any stream. ``remedy_command`` is included because
        the whole point is to hand back something actionable, and is None
        whenever there is no honest one to give.
        """
        return {
            "scope": self.scope,
            "scope_canonical": self.scope_canonical,
            "transcript_root": str(self.transcript_root),
            "root_exists": self.root_exists,
            "total_transcripts": self.total_transcripts,
            "rejected_count": self.rejected_count,
            "top_rejected_cwds": [
                {"cwd": cwd, "count": count} for cwd, count in self.top_rejected_cwds
            ],
            "same_basename_rejected": [
                {"cwd": cwd, "count": count} for cwd, count in self.same_basename_rejected
            ],
            "remedy_command": remedy_command(self),
        }


def build_scope_report(scope: str | Path, *, top_n: int = TOP_REJECTED_CWDS) -> ScopeReport:
    """Census the transcript root for a scope that discovered nothing.

    Resolves the transcript root the same way indexing does (through
    :func:`paths.resolve_claude_dir`, so CLAUDE_CONFIG_DIR is honoured) and
    walks it with discovery's own enumeration and cwd index. On the index
    command's zero path that index is already warm from the
    :func:`discovery.discover_sessions` call that returned nothing, so this
    costs a dictionary lookup and a re-walk of the directory tree rather than
    a second full parse of the corpus.

    Never raises for a missing or unreadable root: this runs when something
    has already gone wrong, and a diagnostic that itself fails leaves the
    user with strictly less than the bare "0 sessions" line they started with.
    """
    scope_str = str(scope)
    transcript_root = paths.resolve_claude_dir() / "projects"

    if not transcript_root.exists():
        return ScopeReport(
            scope=scope_str,
            scope_canonical=str(paths.canonical(scope_str)),
            transcript_root=transcript_root,
            root_exists=False,
            total_transcripts=0,
            rejected_count=0,
            top_rejected_cwds=(),
        )

    cwd_index = discovery.cwd_index_for(transcript_root)

    total = 0
    rejected = 0
    rejected_cwds: Counter[str] = Counter()
    for jsonl_file in discovery.iter_transcript_files(transcript_root):
        total += 1
        cwds = cwd_index.get(str(jsonl_file), set())
        if any(paths.is_at_or_beneath(cwd, scope_str) for cwd in cwds):
            continue
        rejected += 1
        # A transcript with no recorded cwd at all was rejected too, but it
        # has no prefix to report and would otherwise be silently dropped
        # from the histogram; count it under an explicit label so the
        # per-cwd counts still add up to something a user can reconcile.
        for cwd in cwds or {NO_CWD_RECORDED}:
            rejected_cwds[cwd] += 1

    # The moved-repo / org-rename signature: a rejected cwd whose final
    # path component matches the scope's. Matched on the full histogram
    # (before the top-N truncation) and compared case-insensitively, the
    # same folding paths.canonical applies to the scope itself -- the case
    # bug must not be able to hide a sibling.
    scope_basename = Path(scope_str).name.lower()
    same_basename = tuple(
        (cwd, count)
        for cwd, count in rejected_cwds.most_common()
        if cwd != NO_CWD_RECORDED and Path(cwd).name.lower() == scope_basename
    )

    return ScopeReport(
        scope=scope_str,
        scope_canonical=str(paths.canonical(scope_str)),
        transcript_root=transcript_root,
        root_exists=True,
        total_transcripts=total,
        rejected_count=rejected,
        top_rejected_cwds=tuple(rejected_cwds.most_common(top_n)),
        same_basename_rejected=same_basename,
    )


def remedy_cwd(report: ScopeReport) -> str | None:
    """The recorded ``cwd`` this project most likely moved FROM, or None.

    None -- meaning "say nothing, offer nothing" -- in three cases, each of
    which produced a remedy that was worse than silence:

    - No rejected transcripts at all: there is nothing to point at.
    - The dominant entry is the :data:`NO_CWD_RECORDED` sentinel: it is a
      label, not a path, and interpolating it emits an unbalanced ``(`` that
      is a bash syntax error rather than merely a wrong directory.
    - No entry clears :data:`DOMINANCE`: a flat histogram is many unrelated
      projects, not one moved project, and naming its arbitrary winner tells
      the buyer to index somebody else's work.

    The sentinel is skipped rather than disqualifying, so a corpus of mostly
    cwd-less transcripts with a genuine dominant path behind them still gets
    its remedy; dominance is measured against the full rejected count either
    way, so skipping never inflates a minority into a majority.
    """
    for cwd, count in report.top_rejected_cwds:
        if cwd == NO_CWD_RECORDED:
            continue
        if count >= report.rejected_count * DOMINANCE:
            return cwd
        return None
    return None


def remedy_command(report: ScopeReport) -> str | None:
    """The shell command that re-points ssgrep at where the sessions are.

    Shell-quoted, because recorded cwds routinely contain spaces on macOS
    (``~/Documents/My Project``, iCloud paths). Unquoted, the one line the
    whole diagnostic exists to produce was un-runnable for those buyers --
    ``Got unexpected extra argument(s)`` -- so the message written to prevent
    a support ticket generated one. ``shlex.quote`` is a no-op for every
    space-free path, so existing output is unchanged to the byte.
    """
    cwd = remedy_cwd(report)
    if cwd is None:
        return None
    # --scope, not --project-dir: since the scope flag exists, the right
    # remedy for a moved project is "index the old history into THIS
    # project's index" (one index, searchable from here, staleness-coherent
    # via the persisted scope) -- not "build a second index at the old
    # path", which strands the index somewhere searches must then be
    # re-pointed at. The old remedy was the pre---scope workaround.
    return f"ssgrep index --scope {shlex.quote(cwd)}"


ZERO_DISCOVERY_HEADLINE = "ssgrep found no session transcripts for this project."
"""Default headline: the case this census was built for.

Only true when discovery genuinely returned nothing. Callers that attach the
census to some OTHER failure must pass their own headline -- see
:func:`render_scope_report`.
"""


def _display(value: str) -> str:
    """Render an untrusted recorded ``cwd`` as one printable line.

    A ``cwd`` is transcript-derived, so it is attacker-influenced input that
    this module interpolates straight into a message the buyer is being asked
    to trust and copy from. Newlines are legal in directory names on macOS and
    Linux, and a ``cwd`` containing them let the histogram forge a complete
    second "Point ssgrep there:" block -- unquoted, and printed ABOVE the
    genuine one, so it is the one a reader copies. Raw ESC bytes passed
    through too, so ``\\x1b[2J`` could erase the real remedy underneath it.

    Escaping rather than stripping keeps the line honest: the buyer still sees
    that the recorded path contains something odd, instead of a silently
    different path. metadata.normalize_title() already does this for
    transcript-derived titles; ``cwd`` is the same class of untrusted string
    and simply never got the same treatment.
    """
    return textsafe.printable(value)


def _render_header(report: ScopeReport, headline: str) -> list[str]:
    """The scope/root facts, shown in every variant of the message.

    The folded (case-normalized) scope is deliberately NOT printed. It used to
    appear whenever it differed from the live spelling, on the reasoning that
    a case-insensitive filesystem made it "the entire explanation". Both
    halves of that were wrong once paths.is_at_or_beneath() started
    canonicalizing BOTH sides: a case mismatch can no longer cause a
    rejection, so it can never be the reason this message is printing. And
    since resolve_project_dir() hands in an already expanded, resolved,
    normalized path, case is the only thing canonicalization can still
    change -- which means the line printed for every macOS buyer (every path
    under ``/Users`` has an uppercase char) and never for anyone else. A
    constant is not evidence. It showed a lowercased path the buyer has never
    typed, directly beneath their real one, inside an error message, including
    in the "your index is not broken, this is a fresh install" branch where
    there was nothing to explain. It remains in as_payload() for machine
    consumers, where it is data rather than alarm.
    """
    lines = [
        headline,
        "",
        f"  scope scanned:    {report.scope}",
    ]
    root_note = "" if report.root_exists else "  (does not exist)"
    lines.append(f"  transcript root:  {report.transcript_root}{root_note}")
    return lines


def render_scope_report(report: ScopeReport, *, headline: str | None = None) -> str:
    """Render the census as the message a user reads on a zero result.

    Three shapes, because the three situations have different remedies:

    - Root missing: CLAUDE_CONFIG_DIR, or Claude Code never ran here.
    - Root present but holding no transcripts: nothing is wrong, say so
      plainly. Printing a "0 rejected by scope" line and an empty histogram
      here would invent a scoping problem that does not exist and push the
      user toward --rebuild for no reason.
    - Root holding transcripts that scope rejected: the actual bug. Name the
      counts and the cwds, and hand back the exact command that indexes the
      path the sessions were really recorded under.

    ``headline`` overrides the opening sentence for callers attaching this
    census to a failure that is not zero discovery. The default asserts
    "ssgrep found no session transcripts for this project", which is simply
    false when the caller found some -- and this census was being stapled
    verbatim onto every shrink refusal, so a buyer read "rebuilt: 2 sessions"
    and, three lines later, that nothing was found at all, alongside
    "rejected by scope: 0". They were told they had a scope bug they did not
    have, on the one message that most needed to be believed.
    """
    lines = _render_header(report, headline or ZERO_DISCOVERY_HEADLINE)

    if not report.root_exists:
        lines += [
            "",
            "Claude Code's config directory was not found at the path above.",
            "If it lives elsewhere, set CLAUDE_CONFIG_DIR to point at it.",
        ]
        return "\n".join(lines)

    if report.root_is_empty:
        lines += [
            "",
            "That root holds no session transcripts at all, so there is nothing",
            "to index yet. This is normal for a fresh install: transcripts appear",
            "once you have used Claude Code. Your index is not broken, and",
            "`ssgrep index --rebuild` will not help.",
        ]
        return "\n".join(lines)

    lines += [
        f"  transcripts there: {report.total_transcripts}",
        f"  rejected by scope: {report.rejected_count}",
    ]

    if report.top_rejected_cwds:
        width = max(len(str(count)) for _, count in report.top_rejected_cwds)
        lines += ["", "Working directories recorded by the rejected transcripts:"]
        lines += [
            f"  {count:>{width}} transcripts recorded cwd={_display(cwd)}"
            for cwd, count in report.top_rejected_cwds
        ]

    command = remedy_command(report)
    if command is not None and report.rejected_count == report.total_transcripts:
        cwd = remedy_cwd(report)
        assert cwd is not None
        lines += [
            "",
            # "recorded at", not "indexed under": at this moment nothing is
            # indexed anywhere for that path. Saying "still indexed" told the
            # buyer their history was already searchable somewhere, which is
            # the opposite of why they are being handed a command to run.
            "If this project moved or was renamed, your sessions are still recorded",
            "under the path they were recorded at. Index and search them there:",
            f"  {_display(command)}",
            f"  ssgrep search 'your query' --project-dir {_display(shlex.quote(cwd))}",
            # The second line is not decoration. Buyers ran the first command
            # verbatim, saw "Indexed 7 sessions", believed they were fixed --
            # and then got this identical message again from their real
            # project, because the index was built at the OLD path and
            # `ssgrep search` in the new one still reads the new one's empty
            # index. The census named the only verb that cannot finish the
            # job. Both steps, or neither.
            "Both steps use the old path: the index is built there, so searches",
            "must be pointed there too until Claude Code records sessions under",
            "the new one.",
            "Do NOT run `ssgrep index --rebuild` to fix this -- it cannot recover",
            "sessions recorded under a different path, and would discard the ones",
            "you already have.",
        ]
    elif report.rejected_count < report.total_transcripts:
        # Scope DID match transcripts for this project -- the rejected ones
        # below simply belong to the buyer's other projects. Saying "this
        # project moved" here, or naming the biggest other project as the
        # place to point ssgrep, is a false diagnosis drawn from a histogram
        # the report's own numbers already contradict.
        matched = report.total_transcripts - report.rejected_count
        lines += [
            "",
            f"Scope matched {matched} of those transcripts, so this project's own",
            "sessions were found. The rejected ones above belong to your other",
            "projects and are nothing to do with this result.",
        ]
    elif report.top_rejected_cwds:
        # No dominant cwd: the transcripts above belong to other projects, and
        # claiming otherwise would send the buyer to index an unrelated one.
        lines += [
            "",
            "No single working directory accounts for most of those, so this does",
            "not look like a moved project -- it looks like transcripts belonging",
            "to your other projects. Nothing has been recorded for this one yet.",
            "`ssgrep index --rebuild` will not change that.",
        ]

    return "\n".join(lines)


def describe_zero_discovery(scope: str | Path) -> str:
    """Convenience: census ``scope`` and render it in one call."""
    return render_scope_report(build_scope_report(scope))
