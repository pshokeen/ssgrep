"""Rich table rendering for ssgrep CLI commands."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.table import Table

from ssgrep.utilities.types import IndexStats


def render_index_stats_table(stats: IndexStats) -> None:
    """Print global index state as a Rich table.

    Shared by the ``index`` and ``status`` commands so both report counts,
    size, freshness, model binding, and archive status identically.
    """
    size_mib = stats.index_size_bytes / (1024 * 1024)
    indexed = stats.last_index_time.isoformat() if stats.last_index_time else "never"

    table = Table(
        title="ssgrep index",
        title_justify="left",
        title_style="bold",
        show_header=False,
        box=box.HEAVY,
        padding=(0, 2),
    )
    table.add_column("", style="cyan", no_wrap=True)
    table.add_column("")
    table.add_row("Sessions", str(stats.session_count))
    table.add_row("Episodes", str(stats.episode_count))
    table.add_row("Chunks", str(stats.chunk_count))
    table.add_row(
        "Size",
        f"{stats.index_size_bytes} bytes ({size_mib:.1f} MiB)",
    )
    table.add_row("Indexed", indexed)
    table.add_row("Model", f"{stats.model_id} (dim {stats.vector_dimension})")
    table.add_row(
        "Records",
        f"{stats.skipped_records} skipped, {stats.malformed_records} malformed",
    )
    if stats.runtime_counts:
        runtimes = ", ".join(f"{name}={count}" for name, count in stats.runtime_counts)
        table.add_row("Runtimes", runtimes)
    if stats.tombstoned_chunk_count:
        table.add_row(
            "Tombstone",
            f"{stats.tombstoned_source_count} sources, {stats.tombstoned_chunk_count} chunks",
        )
    if stats.archived_source_count:
        table.add_row(
            "Archived",
            f"{stats.archived_source_count} sources served from their indexed snapshot "
            "(transcript files no longer on disk)",
        )
    Console().print(table)
