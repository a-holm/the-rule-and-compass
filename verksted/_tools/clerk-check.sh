#!/usr/bin/env bash
# Usage: clerk-check.sh <pr> [--base <branch>] [--claims <file>] [--scope <glob>]...
# Read-only evidence check for the merge clerk. git fetch origin, then prints:
#   state: <OPEN|MERGED|CLOSED>  base: <name>  mergeable: <MERGEABLE|CONFLICTING|UNKNOWN>  head: <sha>
#   files: one changed path per line, `out of scope` after a path that matches no --scope glob
#   stat:  git diff --stat origin/<base>...<head>
#   conflicts: on CONFLICTING, git merge-tree --write-tree --name-only origin/<base> <head>
#   claims: one row per line `<label>|<log path>|<ERE>` of the claims file: found | no match | no log
#   merge: the command to run, with a subject placeholder; never executed
# Retries while an OPEN PR has mergeable UNKNOWN, every 5 s for up to 60 s.
# Writes: nothing (git fetch only).
# Exit: 0 when state OPEN, base matches (default $BASE), MERGEABLE, every claim found and nothing out of scope; 1 otherwise; 2 usage.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
base=$BASE
claims=
scopes=()
pr=
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --base) [ $# -ge 2 ] || { usage >&2; exit 2; }; base=$2; shift ;;
    --claims) [ $# -ge 2 ] || { usage >&2; exit 2; }; claims=$2; shift ;;
    --scope) [ $# -ge 2 ] || { usage >&2; exit 2; }; scopes+=("$2"); shift ;;
    -*) usage >&2; exit 2 ;;
    *) [ -z "$pr" ] || { usage >&2; exit 2; }; pr=$1 ;;
  esac
  shift
done
case "$pr" in ''|*[!0-9]*) usage >&2; exit 2 ;; esac
if [ -n "$claims" ] && [ ! -r "$claims" ]; then
  echo "claims file $claims unreadable" >&2
  exit 2
fi
sleep_s=${RCSR_RETRY_SLEEP:-5}
bad=0
git -C "$ROOT" fetch -q origin || { echo "git fetch origin failed" >&2; exit 1; }
tries=0
while :; do
  json=$(cd "$ROOT" && gh pr view "$pr" --json state,baseRefName,headRefOid,mergeable,files) \
    || { echo "gh pr view $pr failed" >&2; exit 1; }
  mergeable=$(jq -r .mergeable <<<"$json")
  [ "$mergeable" = UNKNOWN ] && [ "$(jq -r .state <<<"$json")" = OPEN ] && [ "$tries" -lt 12 ] || break
  tries=$((tries + 1))
  sleep "$sleep_s"
done
state=$(jq -r .state <<<"$json")
pbase=$(jq -r .baseRefName <<<"$json")
head=$(jq -r .headRefOid <<<"$json")
echo "state: $state  base: $pbase  mergeable: $mergeable  head: $head"
[ "$state" = OPEN ] || bad=1
[ "$pbase" = "$base" ] || { echo "base: PR targets $pbase, expected $base"; bad=1; }
[ "$mergeable" = MERGEABLE ] || bad=1

echo "files:"
while IFS= read -r f; do
  [ -n "$f" ] || continue
  mark=
  if [ "${#scopes[@]}" -gt 0 ]; then
    mark=" out of scope"
    for g in "${scopes[@]}"; do
      # shellcheck disable=SC2254
      case "$f" in $g) mark=; break ;; esac
    done
    [ -z "$mark" ] || bad=1
  fi
  echo "  $f$mark"
done < <(jq -r '.files[].path' <<<"$json")

if ! git -C "$ROOT" cat-file -e "$head^{commit}" 2>/dev/null; then
  git -C "$ROOT" fetch -q origin "pull/$pr/head" 2>/dev/null
fi
echo "stat:"
git -C "$ROOT" diff --stat "origin/$base...$head" | sed 's/^/  /'
if [ "$mergeable" = CONFLICTING ]; then
  echo "conflicts:"
  git -C "$ROOT" merge-tree --write-tree --name-only "origin/$base" "$head" | sed 's/^/  /'
fi

echo "claims:"
if [ -n "$claims" ]; then
  while IFS='|' read -r label log ere; do
    case "$label" in ''|'#'*) continue ;; esac
    if [ ! -f "$log" ]; then
      res="no log"
    elif grep -qE -- "$ere" "$log"; then
      res=found
    else
      res="no match"
    fi
    [ "$res" = found ] || bad=1
    echo "  $label|$log|$ere|$res"
  done < "$claims"
fi
echo "merge: gh pr merge $pr --squash --subject '<subject>' --body '<body>'"
exit "$bad"
