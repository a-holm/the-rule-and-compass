#!/usr/bin/env bash
# Usage: wt-new.sh <issue> <slug> [--base <branch>]
#        wt-new.sh --research <slug> [--base <branch>]
# Fetches origin, then adds $WT/issue-<issue>-<slug> on a new branch issue/<issue>-<slug> from origin/<base>
# (default $BASE) with --no-track, or a detached read-only $WT/research-<slug>. Creates the scratch dir
# $G/issue<issue>-scratch/ (or $G/research-<slug>-scratch/). Prints W=, BRANCH=, BASE_SHA=, SCRATCH= lines.
# Writes: one worktree, one local branch (not for --research), one scratch directory.
# Exit: 0 created; 1 refused (worktree or branch exists, slug not kebab-case, base missing); 2 usage.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
research=0
base=$BASE
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --research) research=1 ;;
    --base) [ $# -ge 2 ] || { usage >&2; exit 2; }; base=$2; shift ;;
    -*) usage >&2; exit 2 ;;
    *) args+=("$1") ;;
  esac
  shift
done
if [ "$research" -eq 1 ]; then
  [ "${#args[@]}" -eq 1 ] || { usage >&2; exit 2; }
  slug=${args[0]}
else
  [ "${#args[@]}" -eq 2 ] || { usage >&2; exit 2; }
  issue=${args[0]}
  slug=${args[1]}
  case "$issue" in ''|*[!0-9]*) usage >&2; exit 2 ;; esac
fi
if ! [[ $slug =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
  echo "refused: slug '$slug' is not kebab-case" >&2
  exit 1
fi
need_gates
git -C "$ROOT" fetch -q origin || { echo "refused: git fetch origin failed" >&2; exit 1; }
if ! sha=$(git -C "$ROOT" rev-parse --verify -q "refs/remotes/origin/$base^{commit}"); then
  echo "refused: origin/$base does not exist" >&2
  exit 1
fi
if [ "$research" -eq 1 ]; then
  w=$WT/research-$slug
  scratch=$G/research-$slug-scratch
  branch=
else
  w=$WT/issue-$issue-$slug
  scratch=$G/issue$issue-scratch
  branch=issue/$issue-$slug
fi
if [ -e "$w" ]; then
  echo "refused: $w exists" >&2
  exit 1
fi
if [ -n "$branch" ] && git -C "$ROOT" show-ref --verify -q "refs/heads/$branch"; then
  echo "refused: branch $branch exists" >&2
  exit 1
fi
if [ -n "$branch" ]; then
  git -C "$ROOT" worktree add -q --no-track -b "$branch" "$w" "origin/$base" || exit 1
else
  git -C "$ROOT" worktree add -q --detach "$w" "origin/$base" || exit 1
fi
mkdir -p "$scratch"
echo "W=$w"
echo "BRANCH=$branch"
echo "BASE_SHA=$sha"
echo "SCRATCH=$scratch"
