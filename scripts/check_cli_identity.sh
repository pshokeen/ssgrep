#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "usage: sh scripts/check_cli_identity.sh /path/to/ssgrep" >&2
  exit 2
fi

cli=$1
if [ ! -x "$cli" ]; then
  echo "FAIL: ssgrep executable is missing or not executable: $cli" >&2
  exit 1
fi

version_output=$("$cli" --version)
version_name=$(printf '%s\n' "$version_output" | sed 's/ *(.*//')
printf '%s\n' "$version_name" | grep -qi ssgrep || {
  echo "FAIL: --version does not identify ssgrep" >&2
  exit 1
}
if printf '%s\n' "$version_name" | grep -qi usecli; then
  echo "FAIL: --version exposes usecli identity" >&2
  exit 1
fi

description_output=$("$cli" version 2>&1)
printf '%s\n' "$description_output" | grep -qi ssgrep || {
  echo "FAIL: version command does not identify ssgrep" >&2
  exit 1
}
if printf '%s\n' "$description_output" | grep -qi usecli; then
  echo "FAIL: version command exposes usecli identity" >&2
  exit 1
fi

about_output=$("$cli" about)
cli_name=$(printf '%s\n' "$about_output" | grep -i 'Cli Name' || true)
printf '%s\n' "$cli_name" | grep -qi ssgrep || {
  echo "FAIL: about does not identify ssgrep as the CLI name" >&2
  exit 1
}
if printf '%s\n' "$cli_name" | grep -qi usecli; then
  echo "FAIL: about exposes usecli identity" >&2
  exit 1
fi

help_output=$("$cli" --help)
if printf '%s\n' "$help_output" | grep -qi inspire; then
  echo "FAIL: --help exposes the usecli demo command" >&2
  exit 1
fi

echo "PASS: installed CLI identity is ssgrep"
