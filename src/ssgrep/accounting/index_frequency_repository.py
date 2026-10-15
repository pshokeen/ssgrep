"""Read-only, version-pinned metadata projections with direct bounded row takes."""

from __future__ import annotations

import re
from typing import Any

from ssgrep.store import LanceStore

_VERSION = re.compile(r".+@metadata:([0-9]+),chunks:([0-9]+)$")


class IndexRepository:
    """Private accounting adapter; checkout changes handles, never index contents.

    A page cursor is a logical row offset in ONE immutable chunks table version.
    take_offsets requests only that range, unlike a query OFFSET prefix scan. Results
    carry _rowoffset and are sorted within the bounded page, not over the corpus.
    """

    def __init__(self) -> None:
        self.store = LanceStore()
        self.tables: dict[str, Any] = {}

    def exists(self) -> bool:
        return self.store.exists()

    def table(self, name: str):
        return self.tables[name] if name in self.tables else self.store.table(name)

    def get_meta(self, key: str) -> str | None:
        if not self.tables:
            return self.store.get_meta(key)
        from ssgrep.store import quote

        rows = self.tables["metadata"].search().where(f"key = {quote(key)}").select(["value"])
        values = rows.limit(1).to_list()
        return str(values[0]["value"]) if values else None

    def count(self, name: str) -> int:
        table = self.table(name)
        return int(table.count_rows()) if table is not None else 0

    def rows_page(self, *, start: int, limit: int) -> list[dict[str, Any]]:
        table = self.table("chunks")
        rows = (
            table.take_offsets(list(range(start, start + limit)))
            .select(["chunk_id", "cwd", "_rowoffset"])
            .to_arrow()
            .sort_by("_rowoffset")
        )
        if rows["_rowoffset"].to_pylist() != list(range(start, start + limit)):
            raise ValueError("frequency page does not cover its immutable offset range")
        return rows.select(["chunk_id", "cwd"]).to_pylist()

    def snapshot(self, generation: str) -> IndexRepository:
        match = _VERSION.fullmatch(generation)
        if match is None:
            raise ValueError("frequency snapshot needs immutable table versions")
        frozen = IndexRepository()
        try:
            for name, version in zip(("metadata", "chunks"), match.groups(), strict=True):
                table = frozen.store.table(name)
                if table is None:
                    raise ValueError("frequency snapshot table is absent")
                table.checkout(int(version))
                frozen.tables[name] = table
            return frozen
        except Exception:
            frozen.close()
            raise

    def close(self) -> None:
        self.store.close()
