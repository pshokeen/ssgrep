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
