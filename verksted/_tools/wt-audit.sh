#!/usr/bin/env bash
# Usage: wt-audit.sh [--old <minutes>]
# Read-only audit of every linked worktree of the repository (the main checkout is not listed), one row per worktree:
#   <path> branch=<branch|detached> age=<n>min state=<clean|dirty> owner=<owner|none> pr=<OPEN|MERGED|CLOSED|none|unknown> flags=<list|->
# age is the minutes since the newest file change outside the engine crate target directory and .git. owner is the `owner:` line
# of output/worktree.md (the file `01_issue/stages/01_worktree` writes), else none. pr is the state of the pull request of the branch (`gh pr view`).
# flags: no-owner; dirty-old (dirty and age of --old minutes or more, default 180); merged (the pull request is merged and the
# worktree is still there); stray (the path is outside $WT, rule 59); missing (the registered path is gone).
# Never removes anything: clean-up stays a path-by-path step (rule 7). RCSR_NOW replaces the clock (epoch seconds); tests use it.
# Writes: nothing.
# Exit: 0 audited; 2 usage.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

old=180
while [ $# -gt 0 ]; do
  case $1 in
    -h | --help) usage; exit 0 ;;
    --old)
      [ $# -ge 2 ] || { usage >&2; exit 2; }
      case $2 in '' | *[!0-9]*) usage >&2; exit 2 ;; esac
      old=$2
      shift ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done
now=${RCSR_NOW:-$(date +%s)}

first=1
while IFS= read -r w; do
  [ -n "$w" ] || continue
  if [ "$first" -eq 1 ]; then
    first=0
    continue
  fi
  flags=()
  case $w in "$WT"/*) ;; *) flags+=(stray) ;; esac
  if [ ! -d "$w" ]; then
    echo "$w branch=? age=? state=? owner=none pr=unknown flags=missing"
    continue
  fi
  branch=$(git -C "$w" symbolic-ref -q --short HEAD 2> /dev/null) || branch=detached
  if [ -n "$(git -C "$w" status --porcelain 2> /dev/null | head -n 1)" ]; then state=dirty; else state=clean; fi
  newest=$(find "$w" -name target -type d -prune -o -name .git -prune -o -type f -printf '%T@\n' 2> /dev/null | sort -n | tail -n 1 | cut -d. -f1)
  age=$(((now - ${newest:-0}) / 60))
  owner=
  f=$w/output/worktree.md
  [ ! -f "$f" ] || owner=$(sed -n 's/^owner: *//p' "$f" | head -n 1 | sed 's/[[:space:]]*$//')
  if [ -z "$owner" ]; then
    owner=none
    flags+=(no-owner)
  fi
  pr=none
  if [ "$branch" != detached ]; then
    pr=$(cd "$ROOT" && gh pr view "$branch" --json state --jq .state 2> /dev/null) || pr=unknown
    [ -n "$pr" ] || pr=unknown
  fi
  [ "$state" = clean ] || [ "$age" -lt "$old" ] || flags+=(dirty-old)
  [ "$pr" != MERGED ] || flags+=(merged)
  list=-
  if [ "${#flags[@]}" -gt 0 ]; then list=$(IFS=,; echo "${flags[*]}"); fi
  echo "$w branch=$branch age=${age}min state=$state owner=$owner pr=$pr flags=$list"
done < <(git -C "$RCSR_ROOT" worktree list --porcelain 2> /dev/null | sed -n 's/^worktree //p' | { IFS= read -r head; printf '%s\n' "$head"; sort; })
exit 0
