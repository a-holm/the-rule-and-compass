#!/usr/bin/env bash
# Usage: lints.sh [--tree <worktree>]
# Prints the `--test` arguments of the fast lint set on one line, for `cargo test --release <arguments>`. This file is the one home of the list.
# With --tree, a name whose test file is missing in the engine crate directory of that worktree (crate-dir.sh) is left out and named on stderr.
# Writes: nothing.
# Exit: 0 printed; 1 the worktree has no unique engine crate directory; 2 usage.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

LINTS=(
  test_attributes rcsr_command_lint test_hygiene device_skip_lint no_python sanitizer_plan
  sanitizer_child_env stdin_write_lint conflict_markers no_banned_words lockbox_literal cpu_config_lint
  formal_sync
)

tree=
case ${1:-} in
  -h | --help) usage; exit 0 ;;
  --tree)
    [ $# -eq 2 ] || { usage >&2; exit 2; }
    tree=$2 ;;
  '') ;;
  *) usage >&2; exit 2 ;;
esac
[ $# -le 2 ] || { usage >&2; exit 2; }

crate=
if [ -n "$tree" ]; then
  [ -d "$tree" ] || { echo "lints.sh: $tree is not a directory" >&2; exit 2; }
  crate=$(crate_dir "$tree") || exit 1
fi
args=()
for name in "${LINTS[@]}"; do
  if [ -n "$crate" ] && [ ! -f "$crate/tests/$name.rs" ]; then
    echo "lints.sh: skipped $name (no tests/$name.rs in $tree)" >&2
    continue
  fi
  args+=(--test "$name")
done
echo "${args[*]}"
