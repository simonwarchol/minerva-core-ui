#!/usr/bin/env bash
# Builds kiln-render from upstream (github.com/MPanknin/kiln-render) at
# KILN_SHA with minerva's patches (patches/*.patch, applied in order) into
# vendor/kiln/kiln-render-upstream-<sha>.tgz — the `kiln-upstream` dependency.
#
# To move to a newer kiln: set KILN_SHA, run this, fix any patch that no longer
# applies, point package.json at the new tarball and `pnpm install`.
set -euo pipefail

KILN_SHA=edd7620abcc1f903481a2497648a23c7b9a41015

here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

git clone --quiet https://github.com/MPanknin/kiln-render "$work/kiln"
git -C "$work/kiln" checkout --quiet "$KILN_SHA"
for patch in "$here"/patches/*.patch; do
  git -C "$work/kiln" apply "$patch"
done

cd "$work/kiln"
# Upstream builds with bun; pnpm works once zarrita's numcodecs is hoisted.
export npm_config_package_manager_strict=false CI=true
pnpm install --ignore-scripts --shamefully-hoist
pnpm exec vitest run
pnpm run build:lib
pnpm pack --pack-destination "$work"
mv "$work"/kiln-render-*.tgz "$here/kiln-render-upstream-${KILN_SHA:0:7}.tgz"
echo "Built $here/kiln-render-upstream-${KILN_SHA:0:7}.tgz"
