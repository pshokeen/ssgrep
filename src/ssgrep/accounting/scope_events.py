"""Per-EVENT workspace scope and separate consumer/source permissions (task 5.5).

Scope is decided from each event's own working directory through the profile
policy, never from a transcript header: a session can change cwd and children can
work in other repositories. An organization profile (any profile with roots, or
any non-personal profile) never treats an unknown cwd as in scope. A consumer may be
in scope while its source's cost is not: that source contribution is excluded from
the organization's valuation with a missing-coverage reason, and the existence of a
personal-local copy of the same transcript never widens an organization's scope.
Capture, consumer reduction and source valuation use the same effective ownership map.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ssgrep.accounting.contracts import PERSONAL_PROFILE
from ssgrep.accounting.ledger_contracts import ProfilePolicy
from ssgrep.accounting.policy import effective_ownership
from ssgrep.accounting.policy_rules import PolicyError, PolicyFile, workspace_owner
from ssgrep.accounting.valuation_inputs import AtomStatus

SOURCE_OUT_OF_SCOPE = "source_out_of_scope"
SOURCE_SCOPE_UNKNOWN = "source_scope_unknown"


@dataclass
class Scoper:
    """Cached per-cwd scope decisions for one profile policy."""

    policy: ProfilePolicy
    ownership: PolicyFile | None = None
    _cache: dict[str | None, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self.ownership = effective_ownership(self.policy, self.ownership)
        except PolicyError:
            self.ownership = PolicyFile()

    def in_scope(self, cwd: str | None) -> bool:
        """Whether an event that ran in ``cwd`` belongs to this profile."""
        assert self.ownership is not None
        if cwd in self._cache:
            return self._cache[cwd]
        if cwd is None:
            allowed = (
                self.policy.profile_id == PERSONAL_PROFILE
                and not self.policy.roots
                and PERSONAL_PROFILE in self.ownership.profiles
            )
        else:
            allowed = workspace_owner(self.ownership, cwd) == self.policy.profile_id
        self._cache[cwd] = allowed
        return allowed

    def source_status(self, source_cwd: str | None) -> tuple[AtomStatus, str]:
        """Status of a source atom whose content was produced in ``source_cwd``.

        The consumer's own scope is checked separately (and first); this only
        answers whether the SOURCE cost may be valued under this profile.
        """
        if self.in_scope(source_cwd):
            return AtomStatus.OK, ""
        reason = SOURCE_SCOPE_UNKNOWN if source_cwd is None else SOURCE_OUT_OF_SCOPE
        return AtomStatus.OUT_OF_SCOPE, reason
