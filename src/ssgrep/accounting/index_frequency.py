"""Read-only, bounded lexical statistics with verified profile-corpus authorization.

The existing lexical_stats counts casefolded ASCII tokens per indexed chunk,
not whole paths, flags, episodes or conversations. Guaranteed lexical components
provide upper bounds on a whole candidate's df and therefore LOWER rarity bounds.
Only proven threshold bands may support credit; inconclusive bands remain unknown.
No exact whole-path/flag df is invented. No plaintext source rescan, index mutation,
rebuild or raw mirror is created. Only keyed metadata and compact provenance leave here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from ssgrep.accounting.contracts import digest_of
from ssgrep.accounting.frequency import FrequencyView, frequency_key_id

Authority = Callable[[str | None], bool]


class Repository(Protocol):
    """Read-only metadata and immutable direct-page snapshot seam."""

    def exists(self) -> bool: ...
    def get_meta(self, key: str) -> str | None: ...
    def count(self, name: str) -> int: ...
    def snapshot(self, generation: str) -> Repository: ...
    def rows_page(self, *, start: int, limit: int) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


def _generation(repository: Repository) -> str | None:
    """Include immutable Lance table versions; detect damage without reading full df."""
    generation = repository.get_meta("last_index_time")
    if not generation:
        return None
    table = getattr(repository, "table", None)
    if table is None:
        return generation  # Other read interfaces supply their own immutable generation.
    versions = []
    for name in ("metadata", "chunks"):
        value = table(name)
        version = value.version if value is not None else None
        if type(version) is not int or version < 0:
            return None
        versions.append(f"{name}:{version}")
    return f"{generation}@{','.join(versions)}"


def _authority(profile: str) -> tuple[Authority, str]:
    from ssgrep.accounting.policy import read_policy_file
    from ssgrep.accounting.scope_events import Scoper

    ownership = read_policy_file()
    policy = ownership.profiles.get(profile)
    if policy is None or not policy.collection_enabled or policy.paused:
        raise ValueError("no active corpus authority")
    scoper = Scoper(policy, ownership)
    return scoper.in_scope, digest_of([policy, ownership])


def _read(
    profile: str, repository: Repository, authority: Authority, authority_id: str
) -> FrequencyView:
    from ssgrep.accounting.fingerprint import load_key
    from ssgrep.accounting.index_frequency_build import advance

    if not repository.exists():
        return FrequencyView.unknown("frequency_index_absent", profile=profile)
    try:
        key = load_key(profile, create=False)
    except (OSError, ValueError):
        key = None
    if key is None:
        return FrequencyView.unknown("frequency_key_unavailable", profile=profile)
    return advance(profile, repository, authority, authority_id, key, _generation)


def read_index_view(
    profile: str,
    *,
    repository: Repository | None = None,
    authority: Authority | None = None,
    authority_id: str | None = None,
    cached: FrequencyView | None = None,
) -> FrequencyView:
    """Read compatible current statistics, advancing private bounded build work if needed.

    Explicit authority is an integration seam for an already approved ownership
    snapshot, not authority carried by index data. Supply its stable identity too.
    Default authority uses current local profile policy, never organization enrollment
    alone. Any unapproved chunk invalidates the global df; filtered df is not invented.
    """
    owned = repository is None
    try:
        if authority is None:
            try:
                authority, authority_id = _authority(profile)
            except (OSError, ValueError):
                return FrequencyView.unknown("frequency_scope_unapproved", profile=profile)
        if not authority_id:
            return FrequencyView.unknown("frequency_scope_unapproved", profile=profile)
        if repository is None:
            from ssgrep.accounting.index_frequency_repository import IndexRepository

            repository = IndexRepository()
        from ssgrep.accounting.fingerprint import load_key

        key = load_key(profile, create=False)
        reuse = (
            cached is not None
            and bool(cached.generation)
            and cached.profile_scope == profile
            and cached.authority_id == authority_id
            and key is not None
            and cached.key_id == frequency_key_id(key)
            and cached.unavailable_reason
            not in (
                "frequency_generation_changed",
                "frequency_index_unreadable",
                "frequency_build_pending",
                "frequency_snapshot_unavailable",
            )
            and repository.exists()
            and repository.get_meta("index_state") == "ready"
            and _generation(repository) == cached.generation
        )
        # A ready index generation is immutable. Unchanged generation, authority and
        # key require no lexical_stats read, df copy or chunk authorization projection.
        result = cached if reuse else _read(profile, repository, authority, authority_id)
    except Exception:
        # An unreadable local index/key must not trigger rebuild or enable rarity.
        result = FrequencyView.unknown("frequency_index_unreadable", profile=profile)
    finally:
        if owned and repository is not None:
            try:
                repository.close()
            except Exception:
                result = FrequencyView.unknown("frequency_index_unreadable", profile=profile)
    return result
