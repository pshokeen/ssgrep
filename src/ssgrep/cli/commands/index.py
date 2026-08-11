"""Index command for ssgrep."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import NoReturn

from usecli.cli.core.base_command import BaseCommand
from usecli.cli.core.runtime import is_json_mode, is_quiet

from ssgrep import api, indexer_support, scope_report
from ssgrep.cli import exit_codes
from ssgrep.cli.commands import resolve_project_dir, to_jsonable
from ssgrep.types import (
    IndexNotFoundError,
    IndexNotReadyError,
    IndexStats,
    RebuildWouldShrinkError,
    SearchException,
)


def _fail(error: SearchException, code: int) -> NoReturn:
    """Report a structured failure and exit with `code`, in JSON or text mode.

    In JSON mode the document is written straight to the real stdout and the
    process exits via os._exit(), bypassing Python exception handling and
    usecli's stdout redirection so the envelope cannot be wrapped or swallowed.
    """
    if is_json_mode():
        error_doc = {
            "ok": False,
            "condition": error.condition,
            "message": str(error),
            "command": error.command,
        }
        sys.__stdout__.write(json.dumps(error_doc) + "\n")  # type: ignore
        sys.__stdout__.flush()  # type: ignore
        os._exit(code)
    print(error, file=sys.stderr)
    raise SystemExit(code)


_REFUSAL_CENSUS_HEADLINE = "Census of the transcripts ssgrep can see for this project:"


def _attach_census(
    error: RebuildWouldShrinkError, project: Path, census_scope: str | None = None
) -> None:
    """Fold the transcript census into a refusal, best-effort.

    Appends the rendered census to the message (so text-mode users see the
    evidence for "scope mismatch" instead of just the claim) and, when scope
    matched NOTHING and the census found a single directory the project
    plausibly moved from, repoints ``error.command`` at that corrective,
    non-destructive command.

    Both halves are conditional, and both used not to be:

    - The census was rendered with its default headline, "ssgrep found no
      session transcripts for this project", stapled onto refusals where
      discovery had plainly found some. Buyers read "rebuilt: 2 sessions"
      and, three lines later, that nothing was found at all -- next to
      "rejected by scope: 0", which contradicts it again. A refusal is
      already the most alarming message this tool prints; it must not also
      be the least accurate one.
    - ``error.command`` was repointed whenever ONE rejected cwd dominated the
      histogram, with no check that scope had actually failed. A buyer whose
      transcripts merely aged out got pointed at whichever OTHER project of
      theirs happened to be busiest -- so a caller following the field would
      index an unrelated directory, leave a stray .ssgrep in it, and not
      touch the real problem. Requiring `rejected == total` restricts the
      repoint to the case the remedy is actually about: scope matched
      nothing, so everything under the root was rejected.

    Swallows its own failures: a diagnostic that raises would turn a clean,
    non-destructive refusal into a crash, which is strictly worse than a
    refusal with less detail.
    """
    try:
        # Census the scope the index call actually used (a blind review found
        # this as the fourth call site still censusing raw project_dir).
        report = scope_report.build_scope_report(census_scope or project)
        rendered = scope_report.render_scope_report(report, headline=_REFUSAL_CENSUS_HEADLINE)
        error.args = (f"{error.args[0]}\n\n{rendered}",)
        scope_found_nothing = report.rejected_count == report.total_transcripts
        remedy = scope_report.remedy_command(report) if scope_found_nothing else None
        if remedy is not None:
            error.command = remedy
    except Exception:  # pragma: no cover - defensive; census must never crash a refusal
        pass


class IndexCommand(BaseCommand):
    """Build or update the local session index."""

    def visible(self) -> bool:
        """Show this command in help."""
        return True

    def signature(self) -> str:
        """Command name."""
        return "index"

    def description(self) -> str:
        """Command description."""
        return "Build or update the local session index"

    def handle(
        self,
        project_dir: str = ".",
        rebuild: bool = False,
        no_subagents: bool = False,
        quiet: bool = False,
        allow_shrink: bool = False,
        scope: str = "",
    ) -> object:
        """Index session transcripts for the resolved project.

        --scope decouples WHICH transcripts are discovered (cwd containment
        against this path) from WHERE the index lives (--project-dir). The
        moved-repo / org-rename remedy: `ssgrep index --scope /old/path`
        indexes the stranded history into the current project's index. The
        scope path may no longer exist on disk -- it is matched against the
        cwd strings recorded inside transcripts, not the filesystem.
        """
        try:
            project = resolve_project_dir(project_dir)
            scope_arg = str(resolve_project_dir(scope)) if scope else None
            # The census scope must be the EFFECTIVE scope the index run
            # uses -- persisted when the flag is omitted (the sticky-scope
            # rule) -- not the raw flag. With scope_arg alone, an omitted
            # flag made every post-run diagnostic scan project_dir: the
            # refusal census then re-recommended the already-active scope
            # (masking real causes), and the suspicious-discovery
            # remedy_command could point an auto-executing agent at an
            # UNRELATED same-basename directory (blind-review blocker,
            # reproduced with both shapes).
            census_scope = indexer_support.effective_scope(project, scope_arg)
            if not project.exists():
                # Indexing a path that is not there succeeds and reports a
                # real count, because the transcripts live under ~/.claude,
                # not here -- so the buyer who ran the census's
                # moved-project remedy saw "Indexed 7 sessions." and
                # reasonably concluded they were fixed. They were not: the
                # index was built at the old, absent path, and searching
                # from their real project still finds nothing. Say where the
                # index is actually going, before the success line claims
                # otherwise.
                print(
                    f"ssgrep: note: {project} does not exist. The index will be created "
                    f"there, so searches must use --project-dir {project} to read it.",
                    file=sys.stderr,
                )
            stats: IndexStats = api.index(
                project,
                rebuild=rebuild,
                no_subagents=no_subagents,
                quiet=quiet or is_quiet() or is_json_mode(),
                allow_shrink=allow_shrink,
                scope=scope_arg,
            )
        except RebuildWouldShrinkError as error:
            # USAGE_ERROR, not INTERNAL_FAILURE: nothing broke and nothing was
            # written. This is the "required confirmation" arm of the exit-code
            # contract -- the operation is refused pending --allow-shrink, and
            # the existing index is still there, intact.
            #
            # This is the one path that has definitively detected impending
            # data loss, so it must not also be the least informative one. The
            # refusal raises before handle() ever reaches the session_count==0
            # branch below, so the census has to be attached here or it never
            # runs at all -- leaving "the most likely cause is a scope
            # mismatch" asserted with none of the evidence for it. And the
            # machine-readable `command` must not stay pointed at
            # --allow-shrink when a corrective command exists: an agent that
            # auto-executes that field on failure would destroy the very index
            # this guard just saved.
            _attach_census(error, project, census_scope=census_scope)
            _fail(error, exit_codes.USAGE_ERROR)
        except (IndexNotFoundError, IndexNotReadyError) as error:
            # MISSING_INDEX (4) for missing/corrupt index errors
            _fail(error, exit_codes.MISSING_INDEX)
        except Exception as error:
            if is_json_mode():
                raise
            print(f"Index operation failed: {error}", file=sys.stderr)

            # Suppress generic rebuild advice when:
            # 1. User already passed --rebuild (re-running won't help)
            # 2. Error carries its own structured remedy (command field)
            carries_remedy = getattr(error, "command", None) is not None
            should_advise_rebuild = not rebuild and not carries_remedy

            if should_advise_rebuild:
                print("Try `ssgrep index --rebuild` to force a full rebuild.", file=sys.stderr)
            raise SystemExit(exit_codes.INTERNAL_FAILURE) from error

        if is_json_mode():
            payload = to_jsonable(stats)
            if stats.corpus_session_count == 0:
                # A structured caller gets `session_count: 0` and, before
                # this, nothing at all on either stream to explain it. The
                # explanation is not a rendering concern -- it is the answer
                # to the question the caller just asked -- so it travels in
                # the document rather than being dropped for lack of a
                # terminal to print to.
                payload["zero_discovery"] = scope_report.build_scope_report(
                    census_scope
                ).as_payload()
            elif stats.corpus_session_count == 1 and census_scope == str(project):
                # The suspiciously-small case: "Indexed 1 sessions" from a
                # freshly-moved or org-renamed repo looks like a successful
                # run while 99% of the real history sits orphaned under the
                # old path (documented verbatim in a field deployment: 579
                # sessions invisible, zero warning). Attach the census only
                # when a same-named directory actually appears among the
                # rejected cwds -- a plain small project stays unbothered.
                # Suppressed entirely when a foreign scope is in effect
                # (explicit or persisted): the heuristic's premise -- a user
                # UNAWARE of stranded history -- is false once a scope was
                # deliberately set, and recommending another same-basename
                # sibling there handed auto-executing agents an unrelated
                # directory to merge (blind-review blocker). The
                # persisted!=project notice covers that state instead.
                report = scope_report.build_scope_report(census_scope)
                if report.same_basename_rejected:
                    payload["suspicious_discovery"] = report.as_payload()
            return payload

        if not (quiet or is_quiet()):
            print(
                f"Indexed {stats.session_count} sessions, "
                f"{stats.episode_count} episodes, "
                f"{stats.chunk_count} chunks."
            )
        if stats.corpus_session_count == 0:
            # The count line above, on its own, is the exact observable every
            # scoping bug produces and is indistinguishable from an empty
            # project -- which is what sends users to --rebuild. Say why zero,
            # on stderr so piped stdout keeps its shape.
            #
            # Deliberately OUTSIDE the quiet check: --quiet suppresses
            # progress chatter, not diagnostics. Suppressing it here left the
            # quiet caller with total silence and exit 0 on the one outcome
            # that most needs explaining.
            print(scope_report.describe_zero_discovery(census_scope), file=sys.stderr)
        elif stats.corpus_session_count == 1 and census_scope == str(project):
            # "Indexed 1 sessions" is the moved-repo bug's OTHER disguise:
            # unlike zero it looks like success, so it earns a warning only
            # on positive evidence -- a rejected transcript recording a
            # directory with this project's exact name somewhere else. A
            # genuinely small project (no same-named sibling) stays silent.
            report = scope_report.build_scope_report(census_scope)
            if report.same_basename_rejected:
                cwd, count = report.same_basename_rejected[0]
                quoted = shlex.quote(cwd)
                print(
                    f"ssgrep: note: only 1 session matched this project, but "
                    f"{count} rejected transcript(s) record a same-named "
                    f"directory elsewhere: {cwd}. If this project moved or "
                    f"was renamed, that history can be indexed into THIS "
                    f"project's index with: ssgrep index --scope {quoted}",
                    file=sys.stderr,
                )
        return None
