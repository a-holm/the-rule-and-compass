#!/usr/bin/env bash
# Link and contract check of verksted/: runs the crate test through the machine lock.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOCK=${RCSR_LOCK_SCRIPT:-/mnt/lager/tmp/opencode/one-full-suite.sh}
run=(bash -c "cd '$ROOT/my-version' && CARGO_BUILD_JOBS=2 timeout 580 cargo test --release --test verksted_links")
if [ -x "$LOCK" ]; then
  exec "$LOCK" "${run[@]}"
fi
exec "${run[@]}"
