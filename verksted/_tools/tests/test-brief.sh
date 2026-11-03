#!/usr/bin/env bash
# Usage: test-brief.sh
# Hermetic test of brief.sh: all kinds, the no-resume line, model line, refusals, usage errors, named paths exist.
# Writes: nothing.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
S=$HERE/../brief.sh
V=$HERE/../..
export RCSR_GATES=/nonexistent-gates
LINE1="You are not resumed automatically; block in your own turn until everything is done."
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}

for k in impl research clerk; do
  case "$k" in
    clerk) args=(--issue 7 --pr 9) ;;
    *) args=(--issue 7 --slug demo-slug --epic 3) ;;
  esac
  out=$(bash "$S" "$k" "${args[@]}"); rc=$?
  check "$k-rc" "$rc" 0
  check "$k-line1" "$(printf '%s\n' "$out" | head -1)" "$LINE1"
  check "$k-no-placeholder" "$(printf '%s\n' "$out" | grep -cE '<[A-Za-z_][^>]*>')" 0
  check "$k-gates" "$(printf '%s\n' "$out" | grep -c '^G=/nonexistent-gates$')" 1
  check "$k-no-tractatus" "$(printf '%s\n' "$out" | grep -c tractatus)" 0
  for key in N SLUG W BRANCH BASE G EPIC PR; do
    check "$k-key-$key" "$(printf '%s\n' "$out" | grep -c "^$key=")" 1
  done
  # every verksted path the prompt names exists
  for p in $(printf '%s\n' "$out" | grep -o 'verksted/[A-Za-z0-9_./-]*'); do
    [ -e "$V/${p#verksted/}" ] || { echo "FAIL $k-path: $p missing"; fails=$((fails + 1)); }
  done
  op=$(bash "$S" "$k" "${args[@]}" --model opus)
  check "$k-opus" "$(printf '%s\n' "$op" | grep -c 'mcp__tractatus_thinking__tractatus_thinking and context7 only; no WebSearch, no WebFetch')" 1
done

out=$(bash "$S" impl --issue 7 --slug demo-slug --epic 3)
check impl-w "$(printf '%s\n' "$out" | grep -c '^W=.*/worktrees/issue-7-demo-slug$')" 1
check impl-branch "$(printf '%s\n' "$out" | grep -c '^BRANCH=issue/7-demo-slug$')" 1
check impl-epic "$(printf '%s\n' "$out" | grep -c '^EPIC=3$')" 1

bash "$S" clerk --issue 7 >/dev/null 2>&1; check clerk-no-pr $? 1
bash "$S" impl --issue 7 >/dev/null 2>&1; check impl-no-slug $? 1
bash "$S" research --slug x >/dev/null 2>&1; check research-no-issue $? 1
bash "$S" impl --issue 7 --slug Bad_Slug >/dev/null 2>&1; check bad-slug $? 1
bash "$S" impl --issue 7 --slug ok --owned '<paths>' >/dev/null 2>&1; check placeholder $? 1
bash "$S" >/dev/null 2>&1; check no-kind $? 2
bash "$S" bogus >/dev/null 2>&1; check bad-kind $? 2
bash "$S" impl --issue x --slug ok >/dev/null 2>&1; check bad-issue $? 2
bash "$S" impl --issue 7 --slug ok --model gpt >/dev/null 2>&1; check bad-model $? 2
bash "$S" impl --issue 7 --slug ok --bogus >/dev/null 2>&1; check bad-flag $? 2
bash "$S" -h >/dev/null 2>&1; check help $? 0

[ "$fails" -eq 0 ] && echo "test-brief: ok"
exit "$fails"
