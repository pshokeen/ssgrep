"""Service-layer fakes shared by the mirrored service tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


class FakeStore:
    """Small, configurable LanceStore stand-in that records every query."""

    def __init__(
        self,
        root: Path,
        *,
        exists: bool,
        metadata: dict[str, str | None] | None = None,
        counts: dict[tuple[str, str | None], int] | None = None,
        schema_version: int = 7,
    ) -> None:
        self.root = root
        self.path = root / "lance"
        self._exists = exists
        self._metadata = metadata or {}
        self._counts = counts or {}
        self.schema_version = schema_version
        self.count_calls: list[tuple[str, str | None]] = []
        self.meta_calls: list[str] = []
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self.rows_calls: list[tuple[str, dict[str, Any]]] = []

    def seed_rows(self, table: str, rows: list[dict[str, Any]]) -> None:
        self._rows[table] = rows

    def rows(
        self,
        table: str,
        *,
        columns: list[str] | None = None,
        limit: int | None = None,
        where: str | None = None,
    ) -> list[dict[str, Any]]:
        self.rows_calls.append((table, {"columns": columns, "limit": limit, "where": where}))
        return self._rows.get(table, [])

    def exists(self) -> bool:
        return self._exists

    def schema_matches(self) -> bool:
        return True

    def count(self, table: str, where: str | None = None) -> int:
        self.count_calls.append((table, where))
        return self._counts[(table, where)]

    def get_meta(self, key: str) -> str | None:
        self.meta_calls.append(key)
        return self._metadata.get(key)


class FakeFastMCP:
    """FastMCP double that captures constructor and tool registration data."""

    instances: list[FakeFastMCP] = []

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self.registrations: list[tuple[Any, dict[str, Any]]] = []
        type(self).instances.append(self)

    def tool(self, **kwargs: Any):
        def register(function: Any) -> Any:
            self.registrations.append((function, kwargs))
            return function

        return register


@pytest.fixture
def fake_store_factory(tmp_path: Path):
    """Build a fake repository rooted wholly inside a test's temp directory."""

    def build(**kwargs: Any) -> FakeStore:
        return FakeStore(tmp_path / "data", **kwargs)

    return build


@pytest.fixture
def fake_fastmcp_module():
    """Return a module-shaped fake with mutable FastMCP settings."""
    FakeFastMCP.instances.clear()
    return SimpleNamespace(
        FastMCP=FakeFastMCP,
        settings=SimpleNamespace(show_server_banner=True, check_for_updates="on"),
    )
