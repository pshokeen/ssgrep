"""Policy for the dependency environment, kept free of Dagger imports so it can be unit tested.

`tests/test_recipe.py` runs these under the `module-quality` check; the Dagger `tests` shard cannot
see `.dagger`.
"""

from __future__ import annotations


def admitted_seed(image: str, built_base: str, current_base: str) -> str:
    """Return the published image to start from, or "" to build in the engine.

    A seed carries its own interpreter, libc and OS packages, which the locked sync, the apt check
    and the model prefetch cannot rebase. It is therefore admitted only when it was built from the
    exact base image this checkout pins; a base bump (or missing provenance) ignores the seed
    instead of silently testing on the older base.
    """
    if image and built_base == current_base:
        return image
    return ""


def apt_install(packages: tuple[str, ...], lists_dir: str = "/var/lib/apt/lists") -> list[str]:
    """Install `packages` only when `dpkg` reports one missing.

    Every step is chained with `&&` so a failed `apt-get update` or install fails the exec; a
    trailing cleanup joined with `;` would turn that failure into success.
    """
    names = " ".join(packages)
    return [
        "sh",
        "-ec",
        f"dpkg -s {names} >/dev/null 2>&1 || "
        f"{{ apt-get update && apt-get install -y --no-install-recommends {names} "
        f"&& rm -rf {lists_dir}; }}",
    ]


# Run with `uv run python -c` after the locked sync: the project interpreter, not the system one,
# must be the version the lane asks for (UV_PYTHON), whatever image supplied the environment.
INTERPRETER_CHECK = (
    "import os, sys\n"
    "want = os.environ['UV_PYTHON']\n"
    "got = f'{sys.version_info.major}.{sys.version_info.minor}'\n"
    "assert got == want, f'project interpreter is {got}, expected {want}'\n"
)
