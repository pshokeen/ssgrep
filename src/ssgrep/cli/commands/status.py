"""Status command for ssgrep."""

from __future__ import annotations

import sys

from usecli.cli.core.base_command import BaseCommand
from usecli.cli.core.runtime import is_json_mode

from ssgrep import api
from ssgrep.cli import exit_codes
from ssgrep.cli.commands import resolve_project_dir, to_jsonable
from ssgrep.types import ErrorResponse, IndexNotReadyError, IndexStats


class StatusCommand(BaseCommand):
    """Report index observability data."""

    def visible(self) -> bool:
        """Show this command in help."""
        return True

    def signature(self) -> str:
        """Command name."""
        return "status"

    def description(self) -> str:
        """Command description."""
        return "Show index counts, freshness, model binding, and staleness"

    def handle(self, project_dir: str = ".") -> object:
        """Print index observability data; succeeds even with no index."""
        try:
            project = resolve_project_dir(project_dir)
            stats: IndexStats = api.status(project)
        except IndexNotReadyError as error:
            if is_json_mode():
                return to_jsonable(
                    ErrorResponse(
                        ok=False,
                        condition=error.condition,
                        message=str(error),
                        command=error.command,
                    )
                )
            else:
                print(error, file=sys.stderr)
                raise SystemExit(exit_codes.INTERNAL_FAILURE) from error
        except Exception as error:
            if is_json_mode():
                raise
            print(f"Status failed: {error}", file=sys.stderr)

            # Suppress generic rebuild advice when error carries its own remedy
            carries_remedy = getattr(error, "command", None) is not None

            if not carries_remedy:
                print("Try `ssgrep index --rebuild` to recover.", file=sys.stderr)
            raise SystemExit(exit_codes.INTERNAL_FAILURE) from error

        if is_json_mode():
            return to_jsonable(stats)

        if not stats.index_exists:
            print("No index found for this project. Run `ssgrep index` to build one.")
            return None

        size_mib = stats.index_size_bytes / (1024 * 1024)
        last_indexed = stats.last_index_time.isoformat() if stats.last_index_time else "never"
        print(f"Sessions:  {stats.session_count}")
        print(f"Episodes:  {stats.episode_count}")
        print(f"Chunks:    {stats.chunk_count}")
        print(f"Size:      {stats.index_size_bytes} bytes ({size_mib:.1f} MiB)")
        print(f"Indexed:   {last_indexed}")
        print(f"Model:     {stats.model_id} (dim {stats.vector_dimension})")
        print(
            f"Records:   {stats.skipped_records} skipped, " f"{stats.malformed_records} malformed"
        )
        if stats.queue_items_out_of_scope:
            # Nonzero only when the drain's fail-closed scope filter dropped
            # hints -- worth a line precisely because it is rare (something
            # enqueued out-of-scope items and the filter caught them).
            print(
                f"Queue:     {stats.queue_items_out_of_scope} out-of-scope "
                f"hint(s) dropped by the scope filter last index run"
            )
        if stats.tombstoned_chunk_count:
            print(
                f"Tombstone: {stats.tombstoned_source_count} sources, "
                f"{stats.tombstoned_chunk_count} chunks"
            )
        if stats.stale:
            print(f"Stale:     yes ({stats.stale_count} transcripts changed)")
        else:
            print("Stale:     no")
        if stats.cwd_cache_degraded:
            print(
                f"Cwd cache: degraded ({stats.cwd_cache_fallback_scans} full transcript scans; "
                "results correct but slower)"
            )
        return None
