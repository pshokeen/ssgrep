#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Clean all ssgrep state so the next `ssgrep init` runs as if from a fresh
# install: application data, installed skills, and cached Hugging Face models.
#
# Usage:  scripts/clean-install.sh [--force]
#
# --force  skip the confirmation prompt
#
# After cleaning, run  uv run ssgrep init  to re-install and re-download.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- paths ---------------------------------------------------------------

# Application data (platformdirs on macOS: ~/Library/Application Support/ssgrep)
SSGREP_DATA_DIR="${SSGREP_DATA_DIR:-$HOME/Library/Application Support/ssgrep}"

# Agent skill directories (one per supported runtime)
CLAUDE_SKILLS="${CLAUDE_CONFIG_DIR:-$HOME/.claude}/skills/ssgrep"
OPENCODE_SKILLS="${XDG_CONFIG_HOME:-$HOME/.config}/opencode/skills/ssgrep"
PI_SKILLS="${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}/skills/ssgrep"
PRIME_SKILLS="${PRIME_AGENT_CODING_AGENT_DIR:-$HOME/.prime/agent}/skills/ssgrep"

# Hugging Face cache entries for the default models
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface/hub}"
EMBED_CACHE="$HF_CACHE/models--sentence-transformers--all-mpnet-base-v2"
RERANKER_CACHE="$HF_CACHE/models--cross-encoder--ms-marco-TinyBERT-L-6"

# ---- dry-run / confirmation ----------------------------------------------

if [ "${1:-}" != "--force" ]; then
  echo "This will permanently DELETE:"
  echo "  $SSGREP_DATA_DIR  (index database, registry, cursors)"
  echo "  $CLAUDE_SKILLS         (Claude Code skill)"
  echo "  $OPENCODE_SKILLS       (OpenCode skill)"
  echo "  $PI_SKILLS             (Pi skill)"
  echo "  $PRIME_SKILLS          (Prime Agent skill)"
  echo "  $EMBED_CACHE           (embedding model — will re-download)"
  echo "  $RERANKER_CACHE        (reranker model — will re-download)"
  echo ""
  read -rp "Proceed? [y/N] " REPLY
  if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
  fi
fi

# ---- remove --------------------------------------------------------------

remove() {
  local path="$1"
  if [ -e "$path" ]; then
    rm -rf "$path"
    echo "  removed  $path"
  else
    echo "  skipped  $path (not found)"
  fi
}

echo ""
echo "Cleaning ssgrep state …"
remove "$SSGREP_DATA_DIR"
remove "$CLAUDE_SKILLS"
remove "$OPENCODE_SKILLS"
remove "$PI_SKILLS"
remove "$PRIME_SKILLS"
remove "$EMBED_CACHE"
remove "$RERANKER_CACHE"

echo ""
echo "Done. Next step:"
echo "  uv run ssgrep init"