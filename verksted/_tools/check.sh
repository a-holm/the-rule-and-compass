#!/usr/bin/env bash
# Usage: check.sh --stage <pipeline>/<stage> | check.sh <output-dir> '<check command>'
# --stage reads the one fenced sh block under `## Check` of $V/<pipeline>/stages/<stage>/CONTEXT.md (a pipeline name alone
# reads $V/<pipeline>/CONTEXT.md), sources lib.sh, exports every upper-case variable the block reads and runs it with `bash -u`.
# A variable that is still unset after lib.sh (N, SLUG, W, V, PR, ...) is refused: exit 2, the name on stderr, and a `fail:` line in check.md.
# A contract with no Check block, or with two, exits 2. Variables the block assigns itself, `${X:-...}` defaults and shell specials are not required.
# Writes: <output-dir>/check.md (--stage: $V/<pipeline>/stages/<stage>/output/check.md); line 1 is `pass` or `fail: ...`.
# Exit: the command's exit code; 2 usage, refusal or missing contract.
set -u
usage_exit() {
  echo "usage: check.sh --stage <pipeline>/<stage> | check.sh <output-dir> '<check command>'" >&2
  exit 2
}
record() { # record <dir> <head> <detail line> <exit code>
  mkdir -p "$1"
  printf '%s\n\n%s\n- exit: %s\n' "$2" "$3" "$4" > "$1/check.md"
}
run_cmd() { # run_cmd <dir> <detail> <bash flags> <command>
  local out code head last
  out=$(bash "$3" "$4" 2>&1)
  code=$?
  if [ "$code" -eq 0 ]; then
    head="pass"
  else
    last=$(printf '%s\n' "$out" | grep . | tail -n 1 | cut -c1-200)
    head="fail: exit $code: ${last:-no output}"
  fi
  record "$1" "$head" "$2" "$code"
  printf '%s\n' "$head"
  exit "$code"
}

if [ "${1:-}" = "--stage" ]; then
  [ $# -eq 2 ] || usage_exit
  stage=$2
  case "$stage" in
    */*/* | */ | /* | *..*) usage_exit ;;
    */*) rel=${stage%%/*}/stages/${stage#*/} ;;
    *) rel=$stage ;;
  esac
  [ -n "${V:-}" ] || { echo "check.sh: variable V is unset (the run's verksted directory)" >&2; exit 2; }
  contract=$V/$rel/CONTEXT.md
  [ -f "$contract" ] || { echo "check.sh: no contract $contract" >&2; exit 2; }
  dir=$V/$rel/output
  blocks=$(awk '/^## / { c = ($0 == "## Check") } c && /^```sh$/ { n++ } END { print n + 0 }' "$contract")
  [ "$blocks" -eq 1 ] || { echo "check.sh: $contract has $blocks Check blocks, want 1" >&2; exit 2; }
  cmd=$(awk '/^## / { c = ($0 == "## Check") } c && /^```sh$/ { b = 1; next } b && /^```$/ { exit } b { print }' "$contract")
  # shellcheck source=lib.sh
  . "$(dirname "${BASH_SOURCE[0]}")/lib.sh" || exit 2
  assigned=$(printf '%s\n' "$cmd" | grep -oE '(^|[^A-Za-z0-9_$])[A-Z][A-Z0-9_]*=|for [A-Z][A-Z0-9_]* in' | grep -oE '[A-Z][A-Z0-9_]*' | sort -u)
  needed=$(printf '%s\n' "$cmd" | sed -E 's/\$\{[A-Z][A-Z0-9_]*:?[-=+?]//g' | grep -oE '\$\{?[A-Z][A-Z0-9_]*' | tr -d '${' | sort -u)
  for name in V $needed; do
    case "$name" in IFS | PATH | HOME | PWD | OLDPWD | TMPDIR | USER | SHELL | LANG | RANDOM) continue ;; esac
    printf '%s\n' "$assigned" | grep -qxF "$name" && continue
    if [ -z "${!name+x}" ]; then
      echo "check.sh: variable $name is unset (needed by the Check of $stage); export it first" >&2
      record "$dir" "fail: variable $name is unset" "- stage: $stage" 2
      exit 2
    fi
    export "${name?}"
  done
  run_cmd "$dir" "- stage: $stage" -uc "$cmd"
fi

[ $# -eq 2 ] || usage_exit
run_cmd "$1" "- command: \`$2\`" -c "$2"
