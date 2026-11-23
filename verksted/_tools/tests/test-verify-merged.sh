#!/usr/bin/env bash
# Usage: test-verify-merged.sh
# Hermetic test of verify-merged.sh: a fixture origin with a base and a PR branch, a fake gh, a stub rc-run.sh in a copy of the tools.
# Cases: clean merge and the lint arguments of one rc-run job, a conflict before any job, --tip without a merge, --build, --goldens, --walk,
# reuse, --done and the worktree count, a path outside the worktrees folder, red and empty runs, exit codes passed through, usage.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/machine.sh"
. "$HERE/lib/fakes.sh"
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
git_q() { git -c user.name=t -c user.email=t@t.invalid "$@"; }
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t.invalid GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t.invalid

# the tools under test, with a stub rc-run.sh next to them
mkdir -p "$T/tools"
for f in lib.sh lints.sh test-summary.sh verify-merged.sh; do cp "$HERE/../$f" "$T/tools/$f"; done
cat > "$T/tools/rc-run.sh" <<'STUB'
#!/usr/bin/env bash
{ echo CALL; printf '%s\n' "$@"; } >> "$FAKE_DIR/rc-calls"
log=$RCSR_GATES/$3-$4.log
printf 'RC_HEAD sha=none dirty=1\n%s\nRC_DONE rc=%s at=x\n' \
  "${FAKE_RESULT:-test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out}" "${FAKE_RC:-0}" > "$log"
echo "$log"
echo "RC_DONE rc=${FAKE_RC:-0} at=x"
exit "${FAKE_RC:-0}"
STUB
chmod +x "$T/tools/rc-run.sh"

# fixture: origin, clone on main with a crate directory of test files, a clean PR branch, a conflicting PR branch
git init -q --bare -b main "$T/origin.git"
git clone -q "$T/origin.git" "$T/clone" 2> /dev/null
git_q -C "$T/clone" checkout -q -b main
mkdir -p "$T/clone/crate-x/tests" "$T/clone/verksted/_tools/verksted-check"
for n in $(RCSR_ROOT="$T/clone" bash "$T/tools/lints.sh" | sed 's/--test //g'); do : > "$T/clone/crate-x/tests/$n.rs"; done
: > "$T/clone/crate-x/tests/plain.rs"
: > "$T/clone/crate-x/tests/lock_record_golden.rs"
printf 'const H: &str = "sha256";\n' > "$T/clone/crate-x/tests/pinned.rs"
printf 'base\n' > "$T/clone/shared.txt"
printf 'ignored\n' > "$T/clone/.gitignore"
git_q -C "$T/clone" add -A
git_q -C "$T/clone" commit -q -m init
git_q -C "$T/clone" push -q origin main
git_q -C "$T/clone" checkout -q -b feat
printf 'feature\n' > "$T/clone/feat.txt"
git_q -C "$T/clone" add -A
git_q -C "$T/clone" commit -q -m feat
git_q -C "$T/clone" push -q origin feat
head=$(git -C "$T/clone" rev-parse HEAD)
git_q -C "$T/clone" checkout -q -b clash main
printf 'clash\n' > "$T/clone/shared.txt"
git_q -C "$T/clone" commit -q -am clash
clash=$(git -C "$T/clone" rev-parse HEAD)
git_q -C "$T/clone" checkout -q main
printf 'main change\n' > "$T/clone/shared.txt"
git_q -C "$T/clone" commit -q -am "diverge"
git_q -C "$T/clone" push -q origin main
tipsha=$(git -C "$T/clone" rev-parse HEAD)

mkdir -p "$T/bin" "$T/gates"
cat > "$T/bin/gh" <<'GH'
#!/bin/sh
echo "$FAKE_PR_HEAD"
GH
chmod +x "$T/bin/gh"
export FAKE_DIR=$T PATH="$T/bin:$PATH" RCSR_ROOT="$T/clone" RCSR_GATES="$T/gates" RCSR_BASE=main RCSR_CRATE=crate-x FAKE_PR_HEAD=$head
export RCSR_LOCAL="$T/local"
mkdir -p "$T/local"
printf -- "%s" "$FAKE_LIB_KEYS" > "$T/local/machine.md"
V=$T/tools/verify-merged.sh
WTP=$T/clone/worktrees/verifier-7
run() { bash "$V" "$@" 2>&1; }
calls() { grep -c '^CALL$' "$T/rc-calls" 2> /dev/null || true; }
: > "$T/rc-calls"
n0=$(git -C "$T/clone" worktree list | wc -l | tr -d ' ')

# clean merge: one rc-run job with every lint argument, the worktree holds the merge without a commit
out=$(run 7 --lints); rc=$?
check merge-rc "$rc" 0
check merge-one-job "$(calls)" 1
lints=$(bash "$T/tools/lints.sh" --tree "$WTP" 2> /dev/null)
check merge-lint-count "$(wc -w <<<"$lints" | tr -d ' ')" 24
check merge-every-lint-arg "$(grep -cF -- "cargo test --release --no-fail-fast $lints || rc=1" "$T/rc-calls")" 1
check merge-verksted-check "$(grep -c 'verksted/_tools/verksted-check && cargo test --release --no-fail-fast' "$T/rc-calls")" 1
check merge-build-mode "$(sed -n '2p' "$T/rc-calls")" "build"
check merge-id-tag "$(sed -n '4,5p' "$T/rc-calls" | tr '\n' ' ')" "verifier7 merged "
check merge-worktree "$(sed -n '3p' "$T/rc-calls")" "$WTP"
check merge-file-present "$(cat "$WTP/feat.txt" 2> /dev/null)" "feature"
check merge-in-progress "$(git -C "$WTP" rev-parse -q --verify MERGE_HEAD > /dev/null && echo yes || echo no)" "yes"
check merge-base-tip "$(git -C "$WTP" rev-parse HEAD)" "$tipsha"
check merge-count "$(git -C "$T/clone" worktree list | wc -l | tr -d ' ')" "$((n0 + 1))"
check merge-log-line "$(grep -c "^log: $T/gates/verifier7-merged.log$" <<<"$out")" 1
check merge-summary-block "$(grep -c '^verdict: ok$' <<<"$out")" 1
check merge-no-goldens-arg "$(grep -c -- '--test pinned' "$T/rc-calls")" 0

# reuse: a second run aborts the merge in progress and merges again
: > "$T/rc-calls"
out=$(run 7 --lints --build); rc=$?
check reuse-rc "$rc" 0
check reuse-one-job "$(calls)" 1
check reuse-no-extra-worktree "$(git -C "$T/clone" worktree list | wc -l | tr -d ' ')" "$((n0 + 1))"
check reuse-build "$(grep -c '^cargo test --release --no-run || rc=1$' "$T/rc-calls")" 1

# --goldens: golden and pinned-hash test files, not the others
: > "$T/rc-calls"
run 7 --goldens > /dev/null; rc=$?
check goldens-rc "$rc" 0
check goldens-args "$(grep -c -- 'cargo test --release --no-fail-fast --test lock_record_golden --test pinned || rc=1' "$T/rc-calls")" 1
check goldens-not-plain "$(grep -c -- '--test plain' "$T/rc-calls")" 0

# --walk runs the strict walk on the verksted tree of the merged worktree
: > "$T/rc-calls"
run 7 --walk > /dev/null
check walk-cmd "$(grep -c "cargo run --release -q -- walk --strict $WTP/verksted" "$T/rc-calls")" 1
check walk-strict-flag "$(grep -c -- "--strict" "$T/rc-calls")" 1

# --done removes the worktree and leaves the count as before; a second --done says already gone
out=$(run 7 --done); rc=$?
check done-rc "$rc" 0
check done-count "$(git -C "$T/clone" worktree list | wc -l | tr -d ' ')" "$n0"
check done-gone "$([ -e "$WTP" ] && echo yes || echo no)" no
out=$(run 7 --done); rc=$?
check done-twice-rc "$rc" 0
check done-twice-msg "$(grep -c '^already gone: ' <<<"$out")" 1

# --done --out: refuses while a record file is missing or empty, names it; removes the worktree when all three are present
run 7 --lints > /dev/null
od=$T/record
mkdir -p "$od"
printf 'x\n' > "$od/verification.md"; printf 'x\n' > "$od/report.md"
out=$(run 7 --done --out "$od"); rc=$?
check doneout-missing-rc "$rc" 1
check doneout-missing-names "$(grep -c 'check.md' <<<"$out")" 1
check doneout-missing-kept "$([ -e "$WTP" ] && echo yes || echo no)" yes
: > "$od/check.md"
out=$(run 7 --done --out "$od"); rc=$?
check doneout-empty-rc "$rc" 1
check doneout-empty-names "$(grep -c 'check.md' <<<"$out")" 1
rm "$od/report.md"; printf 'pass\n' > "$od/check.md"
out=$(run 7 --done --out "$od"); check doneout-report-names "$(grep -c 'report.md' <<<"$out")" 1
check doneout-report-kept "$([ -e "$WTP" ] && echo yes || echo no)" yes
out=$(run 7 --done --out "$T/nonexistent-dir"); rc=$?
check doneout-nodir-rc "$rc" 1
printf 'x\n' > "$od/report.md"
out=$(run 7 --done --out "$od"); rc=$?
check doneout-ok-rc "$rc" 0
check doneout-ok-removed "$([ -e "$WTP" ] && echo yes || echo no)" no
check doneout-ok-files-kept "$(ls "$od" | wc -l | tr -d ' ')" 3
out=$(run 7 --done --out "$od"); check doneout-gone-rc "$?" 0
run 7 --lints --out "$od" > /dev/null; check out-without-done-usage "$?" 2
run 7 --done --out > /dev/null; check out-no-value-usage "$?" 2
run 7 --lints > /dev/null
run 7 --done > /dev/null; check done-without-out-rc "$?" 0
check done-without-out-removed "$([ -e "$WTP" ] && echo yes || echo no)" no

# --tip: the base tip without a merge
: > "$T/rc-calls"
out=$(run 7 --tip --lints --build); rc=$?
check tip-rc "$rc" 0
check tip-one-job "$(calls)" 1
check tip-tag "$(sed -n '5p' "$T/rc-calls")" "tip"
check tip-no-merge "$(git -C "$WTP" rev-parse -q --verify MERGE_HEAD > /dev/null && echo yes || echo no)" "no"
check tip-no-pr-file "$([ -e "$WTP/feat.txt" ] && echo yes || echo no)" "no"
check tip-clean "$(git -C "$WTP" status --porcelain)" ""
check tip-log "$(grep -c "^log: $T/gates/verifier7-tip.log$" <<<"$out")" 1
# reuse after a merge run: --tip drops the merge in progress
run 7 --lints > /dev/null
run 7 --tip --build > /dev/null
check tip-after-merge "$(git -C "$WTP" rev-parse -q --verify MERGE_HEAD > /dev/null && echo yes || echo no)" "no"
run 7 --done > /dev/null

# conflict: exit 1 naming the file before any job; the merge is aborted
: > "$T/rc-calls"
out=$(FAKE_PR_HEAD=$clash run 7 --lints); rc=$?
check conflict-rc "$rc" 1
check conflict-file "$(grep -c '^conflict: shared.txt$' <<<"$out")" 1
check conflict-no-job "$(calls)" 0
check conflict-aborted "$(git -C "$WTP" rev-parse -q --verify MERGE_HEAD > /dev/null && echo yes || echo no)" "no"
run 7 --done > /dev/null

# a job that ends red or empty is exit 1; slot give-up (75) and a duplicate job (3) pass through
out=$(FAKE_RC=1 FAKE_RESULT='test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out' run 7 --lints); rc=$?
check red-rc "$rc" 1
check red-verdict "$(grep -c '^verdict: fail$' <<<"$out")" 1
out=$(FAKE_RESULT='test result: ok. 0 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out' run 7 --lints); rc=$?
check empty-rc "$rc" 1
check empty-verdict "$(grep -c '^verdict: empty$' <<<"$out")" 1
FAKE_RC=75 run 7 --lints > /dev/null; check giveup-rc "$?" 75
FAKE_RC=3 run 7 --lints > /dev/null; check duplicate-rc "$?" 3
run 7 --done > /dev/null

# a worktree path outside the worktrees folder is refused with exit 2 and nothing is created
: > "$T/rc-calls"
run 7 --lints --worktree "$T/outside" > /dev/null; check outside-rc "$?" 2
run 7 --done --worktree "$T/clone/worktrees/../escape" > /dev/null; check dotdot-rc "$?" 2
run 7 --done --worktree "$T/clone/worktrees" > /dev/null; check folder-itself-rc "$?" 2
check outside-nothing-created "$([ -e "$T/outside" ] && echo yes || echo no)" no
check outside-no-job "$(calls)" 0
check outside-count "$(git -C "$T/clone" worktree list | wc -l | tr -d ' ')" "$n0"
run 7 --lints --worktree "$T/clone/worktrees/other-7" > /dev/null; check named-inside-rc "$?" 0
run 7 --done --worktree "$T/clone/worktrees/other-7" > /dev/null
check named-inside-count "$(git -C "$T/clone" worktree list | wc -l | tr -d ' ')" "$n0"

# usage
run > /dev/null; check usage-none "$?" 2
run abc --lints > /dev/null; check usage-nonnumeric "$?" 2
run 7 > /dev/null; check usage-no-check "$?" 2
run 7 --done --lints > /dev/null; check usage-done-with-check "$?" 2
run 7 --bogus > /dev/null; check usage-unknown "$?" 2
run 7 8 --lints > /dev/null; check usage-two-prs "$?" 2
bash "$V" -h > /dev/null 2>&1; check help "$?" 0

if [ "$fails" -ne 0 ]; then
  echo "test-verify-merged: $fails failed"
  exit 1
fi
echo "test-verify-merged: ok"
