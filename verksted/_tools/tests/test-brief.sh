#!/usr/bin/env bash
# Usage: test-brief.sh
# Hermetic test of brief.sh: all kinds, the no-resume line, model line, refusals, usage errors, named paths exist.
# Writes: nothing.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
S=$HERE/../brief.sh
V=$HERE/../..
export RCSR_GATES=/nonexistent-gates
LOCAL=$(mktemp -d)
trap 'rm -rf "$LOCAL"' EXIT
export RCSR_LOCAL=$LOCAL
LINE1="You are not resumed automatically; block in your own turn until everything is done."
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}

for k in impl research verifier; do
  case "$k" in
    verifier) args=(--issue 7 --pr 9) ;;
    *) args=(--issue 7 --slug demo-slug --epic 3) ;;
  esac
  out=$(bash "$S" "$k" "${args[@]}"); rc=$?
  check "$k-rc" "$rc" 0
  check "$k-line1" "$(printf '%s\n' "$out" | head -1)" "$LINE1"
  check "$k-no-placeholder" "$(printf '%s\n' "$out" | grep -cE '<[A-Za-z_][^>]*>')" 0
  check "$k-gates" "$(printf '%s\n' "$out" | grep -c '^G=/nonexistent-gates$')" 1
  check "$k-no-tractatus" "$(printf '%s\n' "$out" | grep -c tractatus)" 0
  check "$k-no-fable-line" "$(printf '%s\n' "$out" | grep -c fable-52)" 0
  check "$k-local-line" "$(printf '%s\n' "$out" | tail -1)" "Read $LOCAL/rules-local.md when it exists."
  for key in N SLUG W BRANCH BASE G EPIC PR; do
    check "$k-key-$key" "$(printf '%s\n' "$out" | grep -c "^$key=")" 1
  done
  # every verksted path the prompt names exists
  for p in $(printf '%s\n' "$out" | grep -o 'verksted/[A-Za-z0-9_./-]*'); do
    [ -e "$V/${p#verksted/}" ] || { echo "FAIL $k-path: $p missing"; fails=$((fails + 1)); }
  done
  op=$(bash "$S" "$k" "${args[@]}" --model opus)
  check "$k-opus" "$(printf '%s\n' "$op" | grep -c '^Use tractatus thinking, context7 and web search\.$')" 1
  check "$k-opus-no-fable-line" "$(printf '%s\n' "$op" | grep -c fable-52)" 0
  check "$k-opus-no-ban" "$(printf '%s\n' "$op" | grep -ci websearch)" 0
  fb=$(bash "$S" "$k" "${args[@]}" --model fable); check "$k-fable-rc" $? 0
  check "$k-fable" "$(printf '%s\n' "$fb" | grep -c '^Use tractatus thinking, context7 and web search\.$')" 1
  check "$k-fable-line" "$(printf '%s\n' "$fb" | grep -c '^Dispatch with subagent_type fable-52\.$')" 1
done

out=$(bash "$S" impl --issue 7 --slug demo-slug --epic 3)
check impl-w "$(printf '%s\n' "$out" | grep -c '^W=.*/worktrees/issue-7-demo-slug$')" 1
check impl-branch "$(printf '%s\n' "$out" | grep -c '^BRANCH=issue/7-demo-slug$')" 1
check impl-epic "$(printf '%s\n' "$out" | grep -c '^EPIC=3$')" 1

out=$(bash "$S" verifier --issue 7 --pr 9)
check verifier-w "$(printf '%s\n' "$out" | grep -c '^W=.*/worktrees/verifier-9$')" 1
bash "$S" verifier --issue 7 >/dev/null 2>&1; check verifier-no-pr $? 1
bash "$S" impl --issue 7 >/dev/null 2>&1; check impl-no-slug $? 1
bash "$S" research --slug x >/dev/null 2>&1; check research-no-issue $? 1
bash "$S" impl --issue 7 --slug Bad_Slug >/dev/null 2>&1; check bad-slug $? 1
bash "$S" impl --issue 7 --slug ok --owned '<paths>' >/dev/null 2>&1; check placeholder $? 1
bash "$S" >/dev/null 2>&1; check no-kind $? 2
bash "$S" bogus >/dev/null 2>&1; check bad-kind $? 2
bash "$S" impl --issue x --slug ok >/dev/null 2>&1; check bad-issue $? 2
bash "$S" impl --issue 7 --slug ok --model gpt >/dev/null 2>&1; check bad-model $? 2
bash "$S" impl --issue 7 --slug ok --model x >/dev/null 2>&1; check bad-model-x $? 2
bash "$S" impl --issue 7 --slug ok --model >/dev/null 2>&1; check model-no-value $? 2
old_role=$(printf "cler%s" k)
bash "$S" "$old_role" --issue 7 --pr 9 >/dev/null 2>&1; check old-role-refused $? 2
ROOT=$(dirname "$(git -C "$V" rev-parse --path-format=absolute --git-common-dir)")
out=$(env -u RCSR_LOCAL bash "$S" impl --issue 7 --slug demo-slug --epic 3)
check default-local-line "$(printf '%s\n' "$out" | tail -1)" "Read $ROOT/verksted/_local/rules-local.md when it exists."
bash "$S" impl --issue 7 --slug ok --bogus >/dev/null 2>&1; check bad-flag $? 2
bash "$S" -h >/dev/null 2>&1; check help $? 0

[ "$fails" -eq 0 ] && echo "test-brief: ok"
exit "$fails"
