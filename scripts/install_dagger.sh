#!/bin/sh
# Install the pinned Dagger CLI for the current OS/architecture into DIR and
# verify it against checksums committed here (from the release's checksums.txt).
# Usage: sh scripts/install_dagger.sh DIR
set -eu

version=1.0.0-beta.21

if [ "$#" -ne 1 ]; then
  echo "usage: sh scripts/install_dagger.sh DIR" >&2
  exit 2
fi
dir=$1

os=$(uname -s | tr '[:upper:]' '[:lower:]')
case $(uname -m) in
  x86_64 | amd64) arch=amd64 ;;
  arm64 | aarch64) arch=arm64 ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

case "$os/$arch" in
  darwin/amd64) sha=fdbe191c965d240ee9cc90719fa9dca1586805cbb89b0a0087314a84a0937354 ;;
  darwin/arm64) sha=781f4862c022a19b9a4c498b59e1d9cfda1ee3cdc8ce210e4545bd8842a7416d ;;
  linux/amd64) sha=c5ac777a5c99089869660943381136e22177b59a46412e310c2fef594760a041 ;;
  linux/arm64) sha=ba9d55638636918a3404313511bceba5cc0af22f487c1c1b6a5ba60ce70a61bb ;;
  *) echo "unsupported platform: $os/$arch" >&2; exit 1 ;;
esac

name="dagger_v${version}_${os}_${arch}.tar.gz"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL --proto '=https' -o "$tmp/$name" "https://dl.dagger.io/dagger/releases/$version/$name"

if command -v sha256sum >/dev/null 2>&1; then
  actual=$(sha256sum "$tmp/$name" | cut -d' ' -f1)
else
  actual=$(shasum -a 256 "$tmp/$name" | cut -d' ' -f1)
fi
if [ "$actual" != "$sha" ]; then
  echo "FAIL: $name sha256 $actual does not match $sha" >&2
  exit 1
fi

mkdir -p "$dir"
tar -xzf "$tmp/$name" -C "$dir" dagger
"$dir/dagger" version | grep -F "v$version"
