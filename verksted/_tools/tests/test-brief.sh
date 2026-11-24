#!/usr/bin/env bash
# Usage: test-brief.sh
# Hermetic test of brief.sh in a fixture repo (tools copied in, origin ref, worktrees): all kinds, the no-resume line, model line,
# refusals, usage errors, absolute tool paths that exist, refusal from the main checkout and from a stale tools worktree, dispatcher note on stderr.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/machine.sh"
V=$HERE/../..
export RCSR_GATES=/nonexistent-gates
export RCSR_BASE=integration/gpu-hardening
LOCAL=$RCSR_LOCAL
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
LINE1="You are not resumed automatically; block in your own turn until everything is done."
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
g() { git -c user.name=t -c user.email=t@t.invalid "$@"; }

# fixture: main checkout with the tools copied in, origin/integration/gpu-hardening at its tip, a stale and a current worktree
ROOT=$T/repo
g init -q -b main "$ROOT"
mkdir -p "$ROOT/verksted/_tools"
cp "$V"/_tools/*.sh "$ROOT/verksted/_tools/"
cp -r "$V/_shared" "$V/_templates" "$ROOT/verksted/"
g -C "$ROOT" add -A
g -C "$ROOT" commit -q -m tools
g -C "$ROOT" update-ref refs/remotes/origin/integration/gpu-hardening HEAD
mkdir -p "$ROOT/worktrees"
g -C "$ROOT" worktree add -q -b tools-stale "$ROOT/worktrees/stale"
echo x >> "$ROOT/verksted/_shared/rules.md"
g -C "$ROOT" commit -q -am "move verksted"
g -C "$ROOT" update-ref refs/remotes/origin/integration/gpu-hardening HEAD
g -C "$ROOT" worktree add -q -b tools-tip "$ROOT/worktrees/tip"
W=$ROOT/worktrees/tip
S=$W/verksted/_tools/brief.sh
VK=$W/verksted

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
  for key in N SLUG W BRANCH BASE G EPIC PR VERKSTED; do
    check "$k-key-$key" "$(printf '%s\n' "$out" | grep -c "^$key=")" 1
  done
  check "$k-verksted-value" "$(printf '%s\n' "$out" | grep -c "^VERKSTED=$VK$")" 1
  # every verksted path the prompt names is absolute, inside this checkout's verksted/, and exists
  check "$k-read-line" "$(printf '%s\n' "$out" | grep -c "^Read and follow $VK/_shared/brief.md and $VK/_templates/[a-z-]*.md exactly.$")" 1
  for p in $(printf '%s\n' "$out" | grep -E '^(Read and follow|Create your worktree)' | grep -o '[^ `]*verksted/[A-Za-z0-9_./-]*'); do
    case "$p" in "$VK"/*) ;; *) echo "FAIL $k-abs: $p is not an absolute path of this checkout"; fails=$((fails + 1)) ;; esac
    [ -e "$p" ] || { echo "FAIL $k-path: $p missing"; fails=$((fails + 1)); }
  done
  check "$k-no-relative-verksted" "$(printf '%s\n' "$out" | grep -E '^(Read and follow|Create your worktree)' | grep -cE '(^|[ `])verksted/')" 0
  op=$(bash "$S" "$k" "${args[@]}" --model opus)
  check "$k-opus" "$(printf '%s\n' "$op" | grep -c '^Use tractatus thinking, context7 and web search\.$')" 1
  check "$k-opus-no-fable-line" "$(printf '%s\n' "$op" | grep -c fable-52)" 0
  check "$k-opus-no-ban" "$(printf '%s\n' "$op" | grep -ci websearch)" 0
  fb=$(bash "$S" "$k" "${args[@]}" --model fable 2> "$T/fable.err"); check "$k-fable-rc" $? 0
  check "$k-fable" "$(printf '%s\n' "$fb" | grep -c '^Use tractatus thinking, context7 and web search\.$')" 1
  check "$k-fable-stdout-clean" "$(printf '%s\n' "$fb" | grep -c fable-52)" 0
  check "$k-fable-stderr-note" "$(grep -c '^Dispatch with subagent_type fable-52\.$' "$T/fable.err")" 1
done

# the exact setup command is named by absolute path
out=$(bash "$S" impl --issue 7 --slug demo-slug --epic 3)
check impl-setup-cmd "$(printf '%s\n' "$out" | grep -c "^Create your worktree: $VK/_tools/wt-new.sh 7 demo-slug$")" 1
out=$(bash "$S" research --issue 7 --slug demo-slug)
check research-setup-cmd "$(printf '%s\n' "$out" | grep -c "^Create your worktree: $VK/_tools/wt-new.sh --research demo-slug$")" 1
out=$(bash "$S" verifier --issue 7 --pr 9)
check verifier-no-setup-cmd "$(printf '%s\n' "$out" | grep -c '^Create your worktree')" 0

out=$(bash "$S" impl --issue 7 --slug demo-slug --epic 3)
check impl-w "$(printf '%s\n' "$out" | grep -c "^W=$ROOT/worktrees/issue-7-demo-slug$")" 1
check impl-branch "$(printf '%s\n' "$out" | grep -c '^BRANCH=issue/7-demo-slug$')" 1
check impl-epic "$(printf '%s\n' "$out" | grep -c '^EPIC=3$')" 1

out=$(bash "$S" verifier --issue 7 --pr 9)
check verifier-w "$(printf '%s\n' "$out" | grep -c '^W=.*/worktrees/verifier-9$')" 1
check verifier-out "$(printf '%s\n' "$out" | grep -c "^OUT=$LOCAL/verifier/9$")" 1
check verifier-n "$(printf '%s\n' "$out" | grep -c "^N=7$")" 1
check verifier-pr "$(printf '%s\n' "$out" | grep -c "^PR=9$")" 1
check impl-no-out "$(bash "$S" impl --issue 7 --slug demo-slug | grep -c "^OUT=")" 0
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
bash "$S" impl --issue 7 --slug ok --bogus >/dev/null 2>&1; check bad-flag $? 2
bash "$S" -h >/dev/null 2>&1; check help $? 0

# default local area: $ROOT/verksted/_local of the main checkout
mkdir -p "$ROOT/verksted/_local"
cp "$LOCAL/machine.md" "$ROOT/verksted/_local/machine.md"
out=$(env -u RCSR_LOCAL bash "$S" impl --issue 7 --slug demo-slug --epic 3)
check default-local-line "$(printf '%s\n' "$out" | tail -1)" "Read $ROOT/verksted/_local/rules-local.md when it exists."

# refusals: main checkout, tools worktree behind origin/$BASE in verksted/, missing origin ref
err=$(bash "$ROOT/verksted/_tools/brief.sh" impl --issue 7 --slug demo-slug 2>&1 >/dev/null); rc=$?
check main-checkout-rc "$rc" 1
check main-checkout-msg "$(printf '%s\n' "$err" | grep -c '^refused: .*main checkout')" 1
out=$(bash "$ROOT/verksted/_tools/brief.sh" impl --issue 7 --slug demo-slug 2>/dev/null)
check main-checkout-no-stdout "$out" ""
err=$(bash "$ROOT/worktrees/stale/verksted/_tools/brief.sh" impl --issue 7 --slug demo-slug 2>&1 >/dev/null); rc=$?
check stale-rc "$rc" 1
check stale-msg "$(printf '%s\n' "$err" | grep -c '^refused: .*1 commit.* behind origin/integration/gpu-hardening in verksted/')" 1
out=$(bash "$ROOT/worktrees/stale/verksted/_tools/brief.sh" impl --issue 7 --slug demo-slug 2>/dev/null)
check stale-no-stdout "$out" ""
# a worktree ahead of origin in verksted/ is fine
g -C "$W" commit -q --allow-empty -m "ahead"
echo y >> "$W/verksted/_shared/rules.md"
g -C "$W" commit -q -am "ahead in verksted"
bash "$S" impl --issue 7 --slug demo-slug >/dev/null 2>&1; check ahead-ok $? 0
g -C "$ROOT" update-ref -d refs/remotes/origin/integration/gpu-hardening
err=$(bash "$S" impl --issue 7 --slug demo-slug 2>&1 >/dev/null); rc=$?
check no-origin-rc "$rc" 1
check no-origin-msg "$(printf '%s\n' "$err" | grep -c '^refused: .*origin/integration/gpu-hardening')" 1
err=$(env -u RCSR_BASE bash "$S" impl --issue 7 --slug demo-slug 2>&1 >/dev/null); rc=$?
check no-base-rc "$rc" 2
check no-base-names-key "$(printf '%s\n' "$err" | grep -c 'key BASE missing in')" 1

# no text sends an agent to the main checkout: the phrase is gone from tracked text
phrase="from the repo"" root"
hits=$(grep -rIn --exclude-dir=target --exclude-dir=tests "$phrase" "$V" "$V/../AGENTS.md" "$V/../docs/issue-workflow.md" | wc -l)
check no-from-repo-root "$hits" 0

[ "$fails" -eq 0 ] && echo "test-brief: ok"
exit "$fails"
