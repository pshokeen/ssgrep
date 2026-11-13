"""Idempotent skill installation for supported coding-agent harnesses."""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

from ssgrep.utilities.paths import (
    resolve_claude_dir,
    resolve_codex_dir,
    resolve_omp_agent_dir,
    resolve_opencode_dir,
    resolve_pi_agent_dir,
    resolve_prime_agent_dir,
)

_MARKER_RE = re.compile(r"\n?<!-- ssgrep-template: hash=([0-9a-f]{64}) bin=(.*?) -->\n?")
_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)
_SHORT_RE = re.compile(r"<!-- ssgrep-rules:short -->\n(.*?)\n<!-- /ssgrep-rules:short -->", re.S)


def ssgrep_executable() -> str:
    """Return a shell-safe executable that non-interactive agents can invoke."""
    beside_python = Path(sys.executable).with_name("ssgrep")
    candidate = str(beside_python) if beside_python.exists() else shutil.which("ssgrep") or "ssgrep"
    return shlex.quote(candidate)


def _template_hash(content: str) -> str:
    return hashlib.sha256(content.rstrip("\n").encode()).hexdigest()


def _render_template(content: str) -> str:
    executable = ssgrep_executable()
    body = content.replace("SSGREP_BIN_PLACEHOLDER", executable)
    marker = f"<!-- ssgrep-template: hash={_template_hash(content)} bin={executable} -->"
    return body.rstrip("\n") + "\n\n" + marker + "\n"


def _install_template(content: str, destination: Path) -> str:
    """Install/update a managed template without overwriting user edits."""
    rendered = _render_template(content)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered)
        return "installed"

    existing = destination.read_text()
    if existing == rendered:
        return "already_installed"
    match = _MARKER_RE.search(existing)
    if match:
        recorded_hash, recorded_executable = match.groups()
        body = _MARKER_RE.sub("\n", existing).rstrip("\n")
        original = body.replace(recorded_executable, "SSGREP_BIN_PLACEHOLDER")
        if _template_hash(original) == recorded_hash:
            destination.write_text(rendered)
            return "updated"
        return "user_modified"
    if existing == content.replace("SSGREP_BIN_PLACEHOLDER", ssgrep_executable()):
        destination.write_text(rendered)
        return "adopted"
    return "user_modified"


def _skill_template() -> str:
    resource = files("ssgrep").joinpath("templates/skills/ssgrep/SKILL.md")
    return resource.read_text()


#: Claude Code routes skills by these phrases; other runtimes ignore the key.
#: Frontmatter is the ONLY per-runtime variation -- the body below it is one
#: shared document, so the rules an agent reads never depend on its harness.
_CLAUDE_TRIGGERS = (
    "how was X solved before",
    "find past session about Y",
    "search sessions for Z",
    "ssgrep",
)

#: Adapter name -> the product name the guidance uses for it. Kept beside the
#: guidance because a new adapter must be NAMED in the rules to be discoverable
#: by an agent reading them; a test pins these keys to the live registry.
RUNTIME_LABELS: dict[str, str] = {
    "native": "Claude Code",
    "opencode": "OpenCode",
    "codex": "Codex",
    "pi": "Pi",
    "prime-agent": "Prime Agent",
    "omp": "omp",
}


def _claude_template(content: str) -> str:
    """Avoid duplicate named-skill loading while retaining Claude compatibility."""
    triggers = "".join(f'  - "{phrase}"\n' for phrase in _CLAUDE_TRIGGERS)
    without_name = content.replace("name: ssgrep\n", "", 1)
    return without_name.replace("---\n\n", f"triggers:\n{triggers}---\n\n", 1)


def guidance_body(content: str | None = None) -> str:
    """Return the shared guidance body: no frontmatter, no marker, real binary.

    This is the exact text installed under every runtime's frontmatter, so the
    ``rules`` command and the installed skills can never disagree about what
    the rules are.
    """
    template = _skill_template() if content is None else content
    body = _FRONTMATTER_RE.sub("", template, count=1)
    body = _MARKER_RE.sub("\n", body)
    return body.replace("SSGREP_BIN_PLACEHOLDER", ssgrep_executable()).strip() + "\n"


def guidance_short(content: str | None = None) -> str:
    """Return only the fenced short block, for pasting into project instructions."""
    body = guidance_body(content)
    match = _SHORT_RE.search(body)
    if match is None:
        raise ValueError("guidance is missing its ssgrep-rules:short fenced region")
    return match.group(1).strip() + "\n"


_PROJECT_RULE_START = "<!-- ssgrep-rules -->"
_PROJECT_RULE_END = "<!-- /ssgrep-rules -->"
_PROJECT_RULE_RE = re.compile(
    rf"{re.escape(_PROJECT_RULE_START)}\n(.*?)\n{re.escape(_PROJECT_RULE_END)}", re.S
)


def project_rule_block() -> str:
    """The marked block `init --apply-rules` writes into a project's instructions.

    One source of truth: the body is the shared short guidance, so `ssgrep rules --short`,
    the installed skill, and every project instruction file carry the same words. The
    marker comments let re-runs replace the block in place without duplicating it.
    """
    body = guidance_short().rstrip("\n")
    return f"{_PROJECT_RULE_START}\n{body}\n{_PROJECT_RULE_END}\n"


def project_rule_instruction_file(cwd: Path | None = None) -> Path:
    """The instruction file agents read in this repo, preferring CLAUDE.md then AGENTS.md.

    Claude Code loads CLAUDE.md; opencode/codex/pi/prime load AGENTS.md. When neither
    exists, CLAUDE.md is created (Claude Code is the dominant harness here; AGENTS.md
    remains a manual apply away for repos standardized on it).
    """
    root = Path.cwd() if cwd is None else cwd
    for name in ("CLAUDE.md", "AGENTS.md"):
        target = root / name
        if target.exists():
            return target
    return root / "CLAUDE.md"


def apply_project_rule(path: Path) -> str:
    """Idempotently install the search-first block into a project instruction file.

    Returns one of ``installed`` (created/adopted), ``updated`` (block replaced in
    place), ``already_installed`` (byte-identical), or ``user_modified`` (the file
    carries a non-canonical block and is never overwritten).
    """
    canonical = project_rule_block()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Agent instructions\n\n{canonical}")
        return "installed"
    existing = path.read_text()
    match = _PROJECT_RULE_RE.search(existing)
    if match is None:
        if existing.rstrip("\n") and not existing.endswith("\n"):
            existing = existing + "\n"
        path.write_text(f"{existing}\n{canonical}")
        return "installed"
    if match.group(0) == canonical.rstrip("\n"):
        return "already_installed"
    if match.group(1).strip() == guidance_short().strip():
        # same body, different whitespace: safe to refresh in place
        path.write_text(_PROJECT_RULE_RE.sub(canonical.rstrip("\n"), existing))
        return "updated"
    return "user_modified"


def remove_project_rule(path: Path) -> str:
    """Strip the managed block (and its leading blank lines), leaving the rest untouched.

    The user's own formatting survives verbatim: only the marked block and up to the
    blank lines that separated it from the preceding text are removed. A file that
    held nothing but the block is deleted outright.
    """
    if not path.exists():
        return "absent"
    existing = path.read_text()
    match = _PROJECT_RULE_RE.search(existing)
    if match is None:
        return "absent"
    before = existing[: match.start()]
    after = existing[match.end() :]
    cleaned = before.rstrip("\n") + after if before.strip() else after.lstrip("\n")
    cleaned = cleaned.lstrip("\n") if not before.strip() else cleaned
    cleaned = cleaned.rstrip("\n")
    if not cleaned.strip():
        path.unlink()
        return "removed"
    path.write_text(cleaned + "\n")
    return "removed"


def render_guidance(*, short: bool = False) -> str:
    """Return the guidance text, either the short block or the full body."""
    return guidance_short() if short else guidance_body()


def guidance_document() -> dict[str, str]:
    """Return the machine-readable guidance payload with its provenance."""
    return {
        "version": version("ssgrep"),
        "executable": ssgrep_executable(),
        "short": guidance_short(),
        "full": guidance_body(),
    }


def skill_destinations() -> tuple[tuple[str, Path], ...]:
    """Return (runtime name, SKILL.md destination) for each supported agent.

    Each agent loads skills from its own canonical global directory, so ssgrep
    installs an idempotent copy per agent so the skill applies to whichever of
    the supported coding agents the user has available.
    """
    return (
        ("claude", resolve_claude_dir() / "skills" / "ssgrep" / "SKILL.md"),
        ("opencode", resolve_opencode_dir() / "skills" / "ssgrep" / "SKILL.md"),
        ("codex", resolve_codex_dir() / "skills" / "ssgrep" / "SKILL.md"),
        ("pi", resolve_pi_agent_dir() / "skills" / "ssgrep" / "SKILL.md"),
        ("prime-agent", resolve_prime_agent_dir() / "skills" / "ssgrep" / "SKILL.md"),
        ("omp", resolve_omp_agent_dir() / "skills" / "ssgrep" / "SKILL.md"),
    )


def install_skills() -> tuple[tuple[str, str], ...]:
    """Install the ssgrep skill into every supported agent harness, independently."""
    standard = _skill_template()
    results: list[tuple[str, str]] = []
    for name, destination in skill_destinations():
        content = _claude_template(standard) if name == "claude" else standard
        try:
            status = _install_template(content, destination)
        except OSError as error:
            status = f"error: {error}"
        results.append((name, status))
    return tuple(results)


# --- pre-edit hook (Claude Code PreToolUse; the one harness with a hook bus) ------

_HOOK_SCRIPT = "ssgrep-search-first.sh"


def hook_destinations() -> tuple[tuple[str, Path], ...]:
    """(runtime, hook script path) for every harness with a pre-tool hook bus.

    Claude Code is the only supported harness with a settings-driven PreToolUse
    hook bus today; the other runtimes rely on the rule block in their project
    instructions, which is the primary mechanism anyway. A new harness gains a
    hook by adding an entry here plus its settings wiring below.
    """
    return (("claude", resolve_claude_dir() / "hooks" / _HOOK_SCRIPT),)


def _hook_template() -> str:
    resource = files("ssgrep").joinpath(f"templates/hooks/{_HOOK_SCRIPT}")
    return resource.read_text()


def install_hooks() -> tuple[tuple[str, str], ...]:
    """Install the pre-edit hook script and its Claude settings wiring.

    The script is the rendered template with the executable substituted but WITHOUT
    the markdown template-hash marker (that convention is for SKILL.md files; a
    trailing HTML comment is a syntax error in a sourced shell script), made
    executable. The settings entry is added only when absent, keyed by the hook
    command string, and never duplicated; removing hooks removes exactly the entry
    ssgrep added.
    """
    results: list[tuple[str, str]] = []
    for name, destination in hook_destinations():
        rendered = _hook_template().replace(
            "SSGREP_BIN_PLACEHOLDER", ssgrep_executable().strip("'\"")
        )
        try:
            existing = destination.read_text() if destination.exists() else None
            if existing == rendered:
                status = "already_installed"
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(rendered)
                status = "installed"
            destination.chmod(0o755)
        except OSError as error:
            status = f"error: {error}"
        results.append((name, status))
    results.append(("claude-settings", _wire_claude_hook()))
    return tuple(results)


def remove_hooks() -> tuple[tuple[str, str], ...]:
    """Remove the hook script and its settings wiring; rule blocks stay."""
    results: list[tuple[str, str]] = []
    for name, destination in hook_destinations():
        if destination.exists():
            try:
                destination.unlink()
                status = "removed"
            except OSError as error:
                status = f"error: {error}"
        else:
            status = "absent"
        results.append((name, status))
    results.append(("claude-settings", _unwire_claude_hook()))
    return tuple(results)


def _claude_settings_path() -> Path:
    return resolve_claude_dir() / "settings.json"


def _hook_command() -> str:
    return str(resolve_claude_dir() / "hooks" / _HOOK_SCRIPT)


def _load_settings(path: Path) -> dict:
    import json

    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text())
    except ValueError:
        return {}
    return document if isinstance(document, dict) else {}


def _save_settings(path: Path, document: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")


def _hook_entries(document: dict) -> list[dict]:
    hooks = document.get("hooks")
    if not isinstance(hooks, dict):
        return []
    groups = hooks.get("PreToolUse")
    if not isinstance(groups, list):
        return []
    return [group for group in groups if isinstance(group, dict)]


def _wire_claude_hook() -> str:
    """Add the PreToolUse matcher group for Edit|Write once; keep everything else."""
    path = _claude_settings_path()
    document = _load_settings(path)
    if path.exists() and not document:
        return "error: settings unreadable (invalid JSON); nothing changed"
    command = _hook_command()
    for group in _hook_entries(document):
        handlers = group.get("hooks")
        if not isinstance(handlers, list):
            continue
        for handler in handlers:
            if isinstance(handler, dict) and handler.get("command") == command:
                return "already_installed"
    hooks = document.setdefault("hooks", {})
    groups = hooks.setdefault("PreToolUse", [])
    groups.append(
        {
            "matcher": "Edit|Write|NotebookEdit",
            "hooks": [{"type": "command", "command": command}],
        }
    )
    _save_settings(path, document)
    return "installed"


def _unwire_claude_hook() -> str:
    """Remove exactly ssgrep's matcher group; every other hook stays."""
    path = _claude_settings_path()
    document = _load_settings(path)
    if not document:
        return "absent"
    command = _hook_command()
    groups = document.get("hooks", {}).get("PreToolUse")
    if not isinstance(groups, list):
        return "absent"
    kept = [
        group
        for group in groups
        if not (
            isinstance(group, dict)
            and group.get("matcher") == "Edit|Write|NotebookEdit"
            and isinstance(group.get("hooks"), list)
            and any(
                isinstance(handler, dict) and handler.get("command") == command
                for handler in group["hooks"]
            )
        )
    ]
    if len(kept) == len(groups):
        return "absent"
    if kept:
        document["hooks"]["PreToolUse"] = kept
    else:
        document["hooks"].pop("PreToolUse", None)
        if not document["hooks"]:
            document.pop("hooks", None)
    _save_settings(path, document)
    return "removed"
