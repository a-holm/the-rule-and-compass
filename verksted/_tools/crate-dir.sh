#!/usr/bin/env bash
# Usage: crate-dir.sh <worktree>
# Prints the engine crate directory of the worktree: the one first-level directory whose Cargo.toml declares
# package rule-and-compass (RCSR_CRATE names it instead). Thin wrapper over crate_dir in lib.sh.
# Writes: nothing.
# Exit: 0 printed; 1 none or several match; 2 usage.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

case ${1:-} in
  -h | --help) usage; exit 0 ;;
esac
[ $# -eq 1 ] || { usage >&2; exit 2; }
[ -d "$1" ] || { echo "crate-dir.sh: $1 is not a directory" >&2; exit 2; }
crate_dir "$1"
