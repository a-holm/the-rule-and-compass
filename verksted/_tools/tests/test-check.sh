#!/usr/bin/env bash
# Usage: test-check.sh
# Hermetic test of check.sh: --stage with an unset variable, a pass, a failing clause, a missing or double Check block,
# and the plain `<output-dir> '<command>'` form. The contract is the real 01_issue/04_implement one, run against a fixture
# clone with a fake origin.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/machine.sh"
REPO=$(cd "$HERE/../../.." && pwd)
CHECK=$REPO/verksted/_tools/check.sh
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

STAGE=01_issue/stages/04_implement
mkdir -p "$T/v/$STAGE" "$T/gates"
cp "$REPO/verksted/$STAGE/CONTEXT.md" "$T/v/$STAGE/CONTEXT.md"
OUTDIR=$T/v/$STAGE/output
printf 'test a ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\nRC_DONE rc=0 at=2026-01-01T00:00:00Z\n' > "$T/gates/issue5-green.log"
git init -q --bare "$T/origin.git"
git init -q "$T/w"
git_q -C "$T/w" checkout -q -b base1
git_q -C "$T/w" commit -q --allow-empty -m init
git_q -C "$T/w" remote add origin "$T/origin.git"
git_q -C "$T/w" push -q origin base1
git_q -C "$T/w" checkout -q -b issue/5-x
git_q -C "$T/w" commit -q --allow-empty -m "feat(x): add y"

# stage [-u NAME] [VAR=value...]: the --stage call from the repo root with the fixture variables; -u leaves one of them unset
stage() {
  local drop=- a vars=()
  [ "${1:-}" = -u ] && { drop=$2; shift 2; }
  for a in W="$T/w" N=5 V="$T/v" RCSR_BASE=base1 RCSR_GATES="$T/gates"; do
    [ "${a%%=*}" = "$drop" ] || vars+=("$a")
  done
  (cd "$REPO" && env -u W -u N -u V "${vars[@]}" "$@" "$CHECK" --stage 01_issue/04_implement 2> "$T/err")
}

# unset variables are refused by name, exit 2
rm -rf "$OUTDIR"
out=$(stage -u W); check unset-W-exit "$?" 2
check unset-W-names-W "$(grep -c 'variable W is unset' "$T/err")" 1
check unset-W-no-pass "$(head -1 "$OUTDIR/check.md")" "fail: variable W is unset"
out=$(stage -u N); check unset-N-exit "$?" 2
check unset-N-names-N "$(grep -c 'variable N is unset' "$T/err")" 1
out=$(stage -u V); check unset-V-exit "$?" 2

# clean fixture: pass, check.md written
out=$(stage); check pass-exit "$?" 0
check pass-line "$out" pass
check pass-check-md "$(head -1 "$OUTDIR/check.md")" pass
check pass-stage-line "$(grep -c -- '- stage: 01_issue/04_implement' "$OUTDIR/check.md")" 1

# a closing keyword in a commit fails, exit 1
git_q -C "$T/w" commit -q --allow-empty -m "fix(x): z" -m "Closes #5"
out=$(stage); check keyword-exit "$?" 1
check keyword-check-md "$(head -1 "$OUTDIR/check.md" | cut -c1-12)" "fail: exit 1"
git_q -C "$T/w" reset -q --hard HEAD~1

# a dirty tree fails
touch "$T/w/dirty"
out=$(stage); check dirty-exit "$?" 1
rm "$T/w/dirty"

# a base branch that origin lacks fails (the fetch error is not swallowed)
out=$(stage RCSR_BASE=nobranch); check bad-base-exit "$?" 128

# the local base branch is ignored: origin/base1 has a closing keyword the local branch lacks
git_q -C "$T/w" checkout -q base1
git_q -C "$T/w" commit -q --allow-empty -m "fix: q" -m "Fixes #9"
git_q -C "$T/w" push -q origin base1
git_q -C "$T/w" reset -q --hard HEAD~1
git_q -C "$T/w" checkout -q issue/5-x
out=$(stage); check origin-not-local-exit "$?" 0

# no Check block, two Check blocks, unknown stage
mkdir -p "$T/v/01_issue/stages/09_none" "$T/v/01_issue/stages/09_two"
printf '# 09_none: x\n\n## Check\n\nprose only\n' > "$T/v/01_issue/stages/09_none/CONTEXT.md"
printf '# 09_two: x\n\n## Check\n\n```sh\ntrue\n```\n\n```sh\ntrue\n```\n' > "$T/v/01_issue/stages/09_two/CONTEXT.md"
for s in 09_none 09_two 99_missing; do
  (cd "$REPO" && V="$T/v" RCSR_GATES="$T/gates" "$CHECK" --stage "01_issue/$s" > /dev/null 2>&1); check "stage-$s-exit" "$?" 2
done
(cd "$REPO" && "$CHECK" --stage > /dev/null 2>&1); check stage-usage "$?" 2

# a variable the block assigns itself or defaults is not required
mkdir -p "$T/v/01_issue/stages/09_own"
printf '# 09_own: x\n\n## Check\n\n```sh\nA=1; test "$A" = 1 && test "${OPTIONAL:-x}" = x\n```\n' > "$T/v/01_issue/stages/09_own/CONTEXT.md"
(cd "$REPO" && V="$T/v" RCSR_GATES="$T/gates" "$CHECK" --stage 01_issue/09_own > /dev/null 2>&1); check own-vars-exit "$?" 0

# plain form still works
mkdir -p "$T/plain"
"$CHECK" "$T/plain" 'echo hi; exit 3' > /dev/null 2>&1; check plain-fail-exit "$?" 3
check plain-fail-head "$(head -1 "$T/plain/check.md")" "fail: exit 3: hi"
"$CHECK" "$T/plain" true > /dev/null 2>&1; check plain-pass-exit "$?" 0
check plain-pass-head "$(head -1 "$T/plain/check.md")" pass
"$CHECK" > /dev/null 2>&1; check plain-usage "$?" 2

if [ "$fails" -ne 0 ]; then
  echo "test-check: $fails failed"
  exit 1
fi
echo "test-check: ok"
