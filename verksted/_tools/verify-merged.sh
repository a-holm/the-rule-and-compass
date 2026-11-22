#!/usr/bin/env bash
# Usage: verify-merged.sh <PR> [--tip] [--lints] [--build] [--goldens] [--walk] [--worktree <path>]
#        verify-merged.sh <PR> --done [--worktree <path>]
# Merged-tree checks of the verifier (rules 56 and 68, routine 08_close/stages/01_verifier). The worktree $WT/verifier-<PR> is created, or
# reused, detached at origin/$BASE. Without --tip the PR head is merged into it with `git merge --no-ff --no-commit` (a conflict prints
# `conflict: <file>` per file, aborts the merge and exits 1 before any build). With --tip the worktree stays at the base tip, for the check after the merge.
# One `rc-run.sh build` job (CPU, log $G/verifier<PR>-merged.log or -tip.log) runs every selected check and keeps going after a red one:
#   --lints     cargo test of the lint set (lints.sh --tree) and of the verksted-check crate
#   --build     cargo test --release --no-run
#   --goldens   cargo test of every test file of the engine crate whose name has `golden` or that pins a sha256 or view id
#   --walk      verksted-check walk (non-strict) on the verksted tree
# Then prints the test-summary.sh block of the log; a verdict other than ok (an empty run included) is exit 1.
# --done aborts an unfinished merge and removes the worktree (nothing else); a missing worktree prints `already gone`.
# --worktree names another path; a path outside $WT is refused.
# Writes: the worktree, the job log.
# Exit: 0 green; 1 red, conflict or setup failure; 2 usage; 3 the same job already runs (rc-run.sh); 75 the build slot gave up (rc-run.sh).
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RC_RUN=$(dirname "${BASH_SOURCE[0]}")/rc-run.sh
SUMMARY=$(dirname "${BASH_SOURCE[0]}")/test-summary.sh

fail() { echo "verify-merged: $1" >&2; exit 1; }
bad_usage() { echo "verify-merged: $1" >&2; usage >&2; exit 2; }

pr= tip=0 lints=0 build=0 goldens=0 walk=0 done_=0 w=
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --tip) tip=1 ;;
    --lints) lints=1 ;;
    --build) build=1 ;;
    --goldens) goldens=1 ;;
    --walk) walk=1 ;;
    --done) done_=1 ;;
    --worktree) [ $# -ge 2 ] || bad_usage "--worktree needs a path"; w=$2; shift ;;
    -*) bad_usage "unknown option $1" ;;
    *) [ -z "$pr" ] || bad_usage "one pull request number"; pr=$1 ;;
  esac
  shift
done
case "$pr" in ''|*[!0-9]*) bad_usage "a pull request number is required" ;; esac
checks=$((lints + build + goldens + walk))
if [ "$done_" -eq 1 ]; then
  [ "$checks" -eq 0 ] && [ "$tip" -eq 0 ] || bad_usage "--done takes no check option"
else
  [ "$checks" -gt 0 ] || bad_usage "select at least one of --lints, --build, --goldens, --walk"
fi
w=${w:-$WT/verifier-$pr}
wt_real=$(realpath -m -- "$WT")
w_real=$(realpath -m -- "$w")
case "$w_real" in
  "$wt_real"/?*) ;;
  *) echo "verify-merged: $w is outside $WT" >&2; exit 2 ;;
esac
w=$w_real

in_merge() { git -C "$w" rev-parse -q --verify MERGE_HEAD > /dev/null 2>&1; }

if [ "$done_" -eq 1 ]; then
  if [ ! -e "$w" ]; then
    echo "already gone: $w"
    exit 0
  fi
  ! in_merge || git -C "$w" merge --abort || fail "cannot abort the merge in $w"
  git -C "$RCSR_ROOT" worktree remove "$w" || fail "cannot remove $w"
  echo "removed: $w"
  exit 0
fi

git -C "$RCSR_ROOT" fetch -q origin || fail "git fetch origin failed"
if [ -e "$w" ]; then
  ! in_merge || git -C "$w" merge --abort || fail "cannot abort the merge in $w"
  [ -z "$(git -C "$w" status --porcelain)" ] || fail "$w is not clean"
  git -C "$w" checkout -q --detach "origin/$BASE" || fail "cannot check out origin/$BASE in $w"
else
  git -C "$RCSR_ROOT" worktree add -q --detach "$w" "origin/$BASE" || fail "cannot create $w"
fi

if [ "$tip" -eq 0 ]; then
  head=$(cd "$RCSR_ROOT" && gh pr view "$pr" --json headRefOid -q .headRefOid) || fail "gh pr view $pr failed"
  git -C "$w" cat-file -e "$head^{commit}" 2> /dev/null || git -C "$RCSR_ROOT" fetch -q origin "pull/$pr/head" || fail "cannot fetch the head of pull request $pr"
  if ! merge_out=$(git -C "$w" merge --no-ff --no-commit "$head" 2>&1); then
    files=$(git -C "$w" diff --name-only --diff-filter=U)
    if [ -n "$files" ]; then
      while IFS= read -r f; do echo "conflict: $f"; done <<<"$files"
    else
      printf '%s\n' "$merge_out" >&2
    fi
    git -C "$w" merge --abort 2> /dev/null
    exit 1
  fi
fi

crate=$(crate_dir "$w") || fail "no single engine crate under $w"
check_crate=$w/verksted/_tools/verksted-check
script="rc=0"
if [ "$lints" -eq 1 ]; then
  lint_args=$("$(dirname "${BASH_SOURCE[0]}")/lints.sh" --tree "$w") || fail "no lint set for $w"
  [ -n "$lint_args" ] || fail "no lint test file exists in $crate/tests"
  script="$script
cargo test --release --no-fail-fast $lint_args || rc=1
(cd $(printf '%q' "$check_crate") && cargo test --release --no-fail-fast) || rc=1"
fi
if [ "$goldens" -eq 1 ]; then
  golden_args=
  for f in $( { ls "$crate"/tests/*golden*.rs; grep -lE 'sha256|view_id' "$crate"/tests/*.rs; } 2> /dev/null | sort -u); do
    golden_args="$golden_args --test $(basename "$f" .rs)"
  done
  if [ -z "$golden_args" ]; then
    echo "goldens: no golden or pinned-hash test file in $crate/tests"
  else
    script="$script
cargo test --release --no-fail-fast$golden_args || rc=1"
  fi
fi
[ "$build" -eq 0 ] || script="$script
cargo test --release --no-run || rc=1"
[ "$walk" -eq 0 ] || script="$script
(cd $(printf '%q' "$check_crate") && cargo run --release -q -- walk $(printf '%q' "$w/verksted")) || rc=1"
script="$script
exit \$rc"

tag=merged
[ "$tip" -eq 0 ] || tag=tip
"$RC_RUN" build "$w" "verifier$pr" "$tag" --timeout 3000 -- bash -c "$script" > /dev/null
rc=$?
log=$(log_path "verifier$pr" "$tag")
echo "log: $log"
status=0
[ ! -f "$log" ] || "$SUMMARY" "$log" || status=1
case "$rc" in
  0) exit "$status" ;;
  3 | 75) exit "$rc" ;;
  *) exit 1 ;;
esac
