#!/usr/bin/env bash
# Usage: wt-done.sh <issue> [--dry-run]
#        wt-done.sh --research <slug> [--dry-run]
# Removes one finished ticket. Refuses unless exactly one worktree matches $WT/issue-<issue>-*, its
# `git status --porcelain` is empty, no process has its cwd inside it (read from /proc/*/cwd, no pgrep), and
# `gh pr view <branch> --json state` says MERGED. Then: git worktree remove <path> (no --force);
# git branch -D <branch>; git push origin --delete <branch> when `git ls-remote --heads origin <branch>` lists it;
# git worktree prune. A research worktree needs no PR. --dry-run prints the actions only.
# Writes: removes that worktree, its local branch and its remote branch; nothing else.
# Exit: 0 removed (or would be); 1 refused with the reason; 2 usage.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
research=0
dry=0
args=()
for a in "$@"; do
  case "$a" in
    -h|--help) usage; exit 0 ;;
    --research) research=1 ;;
    --dry-run) dry=1 ;;
    -*) usage >&2; exit 2 ;;
    *) args+=("$a") ;;
  esac
done
[ "${#args[@]}" -eq 1 ] || { usage >&2; exit 2; }
key=${args[0]}
refuse() { echo "refused: $*" >&2; exit 1; }
if [ "$research" -eq 1 ]; then
  pattern=$WT/research-$key
else
  case "$key" in ''|*[!0-9]*) usage >&2; exit 2 ;; esac
  pattern=$WT/issue-$key-*
fi
shopt -s nullglob
# shellcheck disable=SC2206
matches=($pattern)
shopt -u nullglob
[ "${#matches[@]}" -eq 1 ] || refuse "${#matches[@]} worktrees match $pattern, need exactly one"
w=${matches[0]}
[ "$(basename "$w")" != default-ab-run ] || refuse "$w is protected"
registered=$(git -C "$ROOT" worktree list --porcelain)
case $'\n'"$registered"$'\n' in *$'\nworktree '"$w"$'\n'*) ;; *) refuse "$w is not a registered worktree" ;; esac
dirty=$(git -C "$w" status --porcelain) || refuse "git status failed in $w"
[ -z "$dirty" ] || refuse "$w is not clean (git status --porcelain is not empty)"
for c in /proc/[0-9]*/cwd; do
  cwd=$(readlink "$c" 2>/dev/null) || continue
  case "$cwd" in
    "$w"|"$w"/*) pid=${c#/proc/}; refuse "process ${pid%/cwd} has its cwd inside $w" ;;
  esac
done
branch=
if [ "$research" -eq 0 ]; then
  branch=$(git -C "$w" symbolic-ref -q --short HEAD) || refuse "$w has no branch checked out"
  state=$(cd "$ROOT" && gh pr view "$branch" --json state --jq .state 2>/dev/null) \
    || refuse "gh pr view $branch failed (branch pushed? PR exists?)"
  [ "$state" = MERGED ] || refuse "PR for $branch is $state, not MERGED"
fi
remote=
if [ -n "$branch" ]; then
  remote=$(git -C "$ROOT" ls-remote --heads origin "$branch") || refuse "git ls-remote origin failed"
fi
if [ "$dry" -eq 1 ]; then
  echo "would: git worktree remove $w"
  [ -z "$branch" ] || echo "would: git branch -D $branch"
  [ -z "$remote" ] || echo "would: git push origin --delete $branch"
  echo "would: git worktree prune"
  exit 0
fi
git -C "$ROOT" worktree remove "$w" || refuse "git worktree remove $w failed"
if [ -n "$branch" ]; then
  git -C "$ROOT" branch -D "$branch" || refuse "git branch -D $branch failed"
  if [ -n "$remote" ]; then
    git -C "$ROOT" push -q origin --delete "$branch" || refuse "git push origin --delete $branch failed"
  fi
fi
git -C "$ROOT" worktree prune
echo "removed: $w"
