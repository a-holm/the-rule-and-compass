#!/usr/bin/env bash
# Usage: test-stage-checks.sh
# Hermetic test of the Check commands of 08_close/stages/03_cleanup and 04_gate/stages/03_full_gate (02_close is in test-seven-checks.sh).
# Each Check is read from its CONTEXT.md and run on fixture outputs with a fake gh.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/machine.sh"
V=$HERE/../..
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
# check_cmd <relative CONTEXT.md>: the first fenced sh block under `## Check`
check_cmd() {
  awk '/^## Check$/ { c = 1; next } c && /^```sh$/ { b = 1; next } b && /^```$/ { exit } b { print }' "$V/$1"
}
mkdir -p "$T/bin" "$T/out/verifier" "$T/out/04_gate/stages/03_full_gate/output"
cat > "$T/bin/gh" <<'GH'
#!/usr/bin/env bash
args="$*"
case "$args" in
  "pr list"*) echo 3 ;;
  "pr view 3 --json mergeCommit"*) echo "$FAKE_MERGE_SHA" ;;
  "pr view"*) echo "${FAKE_PR_STATE:-MERGED}" ;;
  "issue view 7 "*) echo "${FAKE_OWNER_STATE:-OPEN}" ;;
  *"--json body"*) cat "$FAKE_BODY" ;;
  *"--json comments"*) cat "$FAKE_COMMENT" ;;
  *"--json state,stateReason"*) echo "${FAKE_ISSUE_STATE:-CLOSED COMPLETED}" ;;
  *) echo "fake gh: unexpected $args" >&2; exit 1 ;;
esac
GH
chmod +x "$T/bin/gh"
run_check() { # run_check <CONTEXT.md> [env assignments...]: exit code of the Check command
  local rel=$1
  shift
  (cd "$T" && env V="$T/out" OUT="$T/out/verifier" N=5 PR=9 WT="$T/wt" PATH="$T/bin:$PATH" FAKE_BODY="$T/body.md" FAKE_COMMENT="$T/out/verifier/verification.md" "$@" bash -c "$(check_cmd "$rel")" > /dev/null 2>&1)
}

# 03_cleanup: no ticket worktree, no verifier worktree, no branch
V3=08_close/stages/03_cleanup/CONTEXT.md
git init -q "$T/repo"
git -C "$T/repo" -c user.name=t -c user.email=t@t.invalid commit -q --allow-empty -m init
run3() { (cd "$T/repo" && env N=5 PR=9 WT="$T/wt" bash -c "$(check_cmd "$V3")" > /dev/null 2>&1); }
run3; check cleanup-clean "$?" 0
mkdir -p "$T/wt/verifier-9"
run3; check cleanup-verifier-worktree-left "$?" 1
rmdir "$T/wt/verifier-9"
git -C "$T/repo" branch issue/5-x
run3; check cleanup-branch-left "$?" 1
git -C "$T/repo" branch -q -D issue/5-x

# 04_gate/stages/03_full_gate: an ignored test that ran nothing fails the Check
V4=04_gate/stages/03_full_gate/CONTEXT.md
g=$T/out/04_gate/stages/03_full_gate/output/gate.md
sha=0123456789abcdef0123456789abcdef01234567
formal='formal: theorems=51 bv_decide=39 vectors=208192 mismatches=0 lean=x mathlib=none'
printf 'cpu: ok=10 failed=0\ncuda: ok=5 failed=0\nrecovered 27/27\ntested: %s\n%s\n' "$sha" "$formal" > "$g"
run_check "$V4"; check gate-ok "$?" 0
printf 'ignored: x passed=2\n' >> "$g"
run_check "$V4"; check gate-ignored-ran "$?" 0
printf 'ignored: x passed=0\n' >> "$g"
run_check "$V4"; check gate-ignored-zero "$?" 1
printf 'cpu: ok=10 failed=0\ncuda: ok=5 failed=0\nrecovered 27/27\ntested: %s\n%s\nignored: y passed=0\n' "$sha" "$formal" > "$g"
run_check "$V4"; check gate-ignored-zero-alone "$?" 1
printf 'cpu: ok=10 failed=0\ncuda: ok=5 failed=0\nrecovered 27/27\ntested: %s\n%s\nignored: y passed=10\n' "$sha" "$formal" > "$g"
run_check "$V4"; check gate-ignored-ten "$?" 0
printf 'cpu: ok=10 failed=0\ncuda: ok=5 failed=0\nrecovered 27/27\ntested: %s\n' "$sha" > "$g"
run_check "$V4"; check gate-formal-missing "$?" 1
printf 'cpu: ok=10 failed=0\ncuda: ok=5 failed=0\nrecovered 27/27\ntested: %s\n%s\n' "$sha" "${formal/mismatches=0/mismatches=3}" > "$g"
run_check "$V4"; check gate-formal-mismatches "$?" 1
printf 'cpu: ok=10 failed=0\ncuda: ok=5 failed=0\nrecovered 27/27\ntested: %s\n%s\n' "$sha" "${formal/theorems=51/theorems=0}" > "$g"
run_check "$V4"; check gate-formal-no-theorems "$?" 1
printf 'cpu: ok=10 failed=0\ncuda: ok=5 failed=0\nrecovered 27/27\ntested: %s\n%s\n' "$sha" "${formal/vectors=208192/vectors=0}" > "$g"
run_check "$V4"; check gate-formal-no-vectors "$?" 1

# 04_gate/stages/05_merge_main: the merge is on origin/main and the main checkout is at origin/main
V5=04_gate/stages/05_merge_main/CONTEXT.md
g5() { git -C "$1" -c user.name=t -c user.email=t@t.invalid "${@:2}"; }
git init -q --bare "$T/origin.git"
git clone -q "$T/origin.git" "$T/mroot" 2> /dev/null
g5 "$T/mroot" checkout -q -b main
g5 "$T/mroot" commit -q --allow-empty -m one
g5 "$T/mroot" push -q origin main
msha=$(git -C "$T/mroot" rev-parse HEAD)
run5() { (cd "$T/mroot" && env BASE=integration/x ROOT="$T/mroot" PATH="$T/bin:$PATH" FAKE_MERGE_SHA="$msha" bash -c "$(check_cmd "$V5")" > /dev/null 2>&1); }
run5; check merge-main-checkout-at-origin "$?" 0
git clone -q "$T/origin.git" "$T/other" 2> /dev/null
g5 "$T/other" commit -q --allow-empty -m two
g5 "$T/other" push -q origin HEAD:main
run5; check merge-main-checkout-behind "$?" 1
g5 "$T/mroot" merge -q --ff-only origin/main
run5; check merge-main-checkout-fast-forwarded "$?" 0

if [ "$fails" -ne 0 ]; then
  echo "test-stage-checks: $fails failed"
  exit 1
fi
echo "test-stage-checks: ok"
