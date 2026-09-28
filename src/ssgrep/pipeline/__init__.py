"""CocoIndex-powered ingestion for the global transcript index."""

import os

# cocoindex sends a usage-tracking event to https://cocoindex.gateway.scarf.sh
# the moment it is imported (and again on later app create/update), so on
# every `ssgrep index`, `ssgrep note`, and MCP-server startup reconciliation
# -- contradicting the README's "fully local and offline" promise. Set the
# documented opt-out before the import below makes cocoindex's own first
# import, unless the user already set it themselves. Mirrors the FastMCP
# update-check opt-out in ``ssgrep.services.mcp_server``.
os.environ.setdefault("COCOINDEX_DISABLE_USAGE_TRACKING", "1")

from ssgrep.pipeline.app import run

__all__ = ["run"]
