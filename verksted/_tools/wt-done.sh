#!/usr/bin/env bash
# Usage: wt-done.sh <issue> [--pr <number>] [--dry-run]
#        wt-done.sh --research <slug> [--dry-run]
# Removes one finished ticket. Refuses unless exactly one worktree matches $WT/issue-<issue>-*, its
# `git status --porcelain` is empty, no process has its cwd inside it (read from /proc/*/cwd, no pgrep), and
# `gh pr view <branch> --json state` says MERGED. Then: git worktree remove <path> (no --force);
# git branch -D <branch>; git push origin --delete <branch> when `git ls-remote --heads origin <branch>` lists it;
# git worktree prune. A research worktree needs no PR. --dry-run prints the actions only.
# No worktree is expected after `gh pr merge --delete-branch`, which removes the worktree that has the head branch
# checked out. Then the PR must be MERGED (`gh pr view <branch>` for a leftover local or remote branch matching
# issue/<issue>-*, or --pr <number> when none is left); the leftover branches are deleted, and with --pr an
# already clean state prints `already gone` and exits 0. Without a worktree, a branch or --pr it refuses.
# Writes: removes that worktree, its local branch and its remote branch; nothing else.
# Exit: 0 removed, already gone (or would be); 1 refused with the reason; 2 usage.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
research=0
dry=0
pr=
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --research) research=1 ;;
    --dry-run) dry=1 ;;
    --pr) [ $# -ge 2 ] || { usage >&2; exit 2; }; pr=$2; shift ;;
    -*) usage >&2; exit 2 ;;
    *) args+=("$1") ;;
  esac
  shift
done
[ "${#args[@]}" -eq 1 ] || { usage >&2; exit 2; }
case "$pr" in *[!0-9]*) usage >&2; exit 2 ;; esac
[ -z "$pr" ] || [ "$research" -eq 0 ] || { usage >&2; exit 2; }
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

if [ "${#matches[@]}" -eq 0 ] && [ "$research" -eq 0 ]; then
  leftover=()
  while IFS= read -r b; do
    [ -n "$b" ] && leftover+=("$b")
  done < <({
    git -C "$ROOT" for-each-ref --format='%(refname:short)' "refs/heads/issue/$key-*"
    git -C "$ROOT" ls-remote --heads origin "issue/$key-*" | sed 's|^[^[:space:]]*[[:space:]]*refs/heads/||'
  } | sort -u)
  [ "${#leftover[@]}" -gt 0 ] || [ -n "$pr" ] || refuse "no worktree matches $pattern and no issue/$key-* branch is left; pass --pr <number> to confirm a ticket that gh pr merge --delete-branch already removed"
  if [ -n "$pr" ]; then
    state=$(cd "$ROOT" && gh pr view "$pr" --json state --jq .state 2>/dev/null) || refuse "gh pr view $pr failed"
  else
    [ "${#leftover[@]}" -eq 1 ] || refuse "${#leftover[@]} branches match issue/$key-*: ${leftover[*]}"
    state=$(cd "$ROOT" && gh pr view "${leftover[0]}" --json state --jq .state 2>/dev/null) \
      || refuse "gh pr view ${leftover[0]} failed (branch pushed? PR exists?)"
  fi
  [ "$state" = MERGED ] || refuse "PR for issue $key is $state, not MERGED"
  for b in ${leftover[@]+"${leftover[@]}"}; do
    if [ "$dry" -eq 1 ]; then
      echo "would: git branch -D $b (local, when present)"
      echo "would: git push origin --delete $b (when listed)"
      continue
    fi
    if git -C "$ROOT" show-ref -q --verify "refs/heads/$b"; then
      git -C "$ROOT" branch -D "$b" || refuse "git branch -D $b failed"
    fi
    if [ -n "$(git -C "$ROOT" ls-remote --heads origin "$b")" ]; then
      git -C "$ROOT" push -q origin --delete "$b" || refuse "git push origin --delete $b failed"
    fi
  done
  [ "$dry" -eq 1 ] || git -C "$ROOT" worktree prune
  echo "already gone: no worktree for issue $key (gh pr merge --delete-branch removes it)"
  exit 0
fi

[ "${#matches[@]}" -eq 1 ] || refuse "${#matches[@]} worktrees match $pattern, need exactly one"
w=${matches[0]}
[ "$(basename "$w")" != default-ab-run ] || refuse "$w is protected"
registered=$(git -C "$ROOT" worktree list --porcelain)
case $'\n'"$registered"$'\n' in *$'\nworktree '"$w"$'\n'*) ;; *) refuse "$w is not a registered worktree" ;; esac
dirty=$(git -C "$w" status --porcelain) || refuse "git status failed in $w"
[ -z "$dirty" ] || refuse "$w is not clean (git status --porcelain is not empty)"
pat=$(printf '%s' "$w" | sed 's/[][*?\\]/\\&/g')
busy=$(find /proc/[0-9]*/cwd -maxdepth 0 \( -lname "$pat" -o -lname "$pat/*" \) -printf '%h\n' 2>/dev/null | head -n 1)
[ -z "$busy" ] || refuse "process ${busy#/proc/} has its cwd inside $w"
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
