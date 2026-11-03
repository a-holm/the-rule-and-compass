#!/usr/bin/env bash
# Usage: test-wt.sh
# Hermetic test of wt-new.sh and wt-done.sh against a temp bare origin and clone, with a fake gh.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS=$HERE/..
T=$(cd "$(mktemp -d)" && pwd -P)
sleeper=
cleanup() {
  [ -z "$sleeper" ] || kill "$sleeper" 2>/dev/null
  rm -rf "$T"
}
trap cleanup EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
git_q() { git -c user.name=t -c user.email=t@t.invalid "$@"; }

git init -q --bare -b main "$T/origin.git"
git clone -q "$T/origin.git" "$T/clone" 2>/dev/null
git_q -C "$T/clone" checkout -q -b main
git_q -C "$T/clone" commit -q --allow-empty -m init
git_q -C "$T/clone" push -q origin main
mkdir -p "$T/gates" "$T/bin"
cat > "$T/bin/gh" <<'GH'
#!/bin/sh
printf '%s\n' "${FAKE_GH_STATE:-OPEN}"
GH
chmod +x "$T/bin/gh"
export PATH="$T/bin:$PATH" RCSR_ROOT="$T/clone" RCSR_GATES="$T/gates" RCSR_BASE=main
WT=$T/clone/worktrees
sha=$(git -C "$T/clone" rev-parse origin/main)

new() { bash "$TOOLS/wt-new.sh" "$@" 2>&1; }
done_() { bash "$TOOLS/wt-done.sh" "$@" 2>&1; }

# wt-new
out=$(new 5 alpha-one); rc=$?
check new-rc "$rc" 0
check new-W "$(grep '^W=' <<<"$out")" "W=$WT/issue-5-alpha-one"
check new-BRANCH "$(grep '^BRANCH=' <<<"$out")" "BRANCH=issue/5-alpha-one"
check new-BASE_SHA "$(grep '^BASE_SHA=' <<<"$out")" "BASE_SHA=$sha"
check new-SCRATCH "$(grep '^SCRATCH=' <<<"$out")" "SCRATCH=$T/gates/issue5-scratch"
check new-scratch-dir "$([ -d "$T/gates/issue5-scratch" ] && echo yes)" yes
check new-head "$(git -C "$WT/issue-5-alpha-one" rev-parse HEAD)" "$sha"
check new-no-upstream "$(git -C "$T/clone" config --get branch.issue/5-alpha-one.merge)" ""
out=$(new 5 alpha-one); rc=$?
check new-again-rc "$rc" 1
out=$(new 6 Bad_Slug); rc=$?
check new-badslug-rc "$rc" 1
check new-badslug-nothing "$([ -e "$WT/issue-6-Bad_Slug" ] && echo yes || echo no)" no
out=$(new 7 x --base nosuch); rc=$?
check new-nobase-rc "$rc" 1
out=$(new 7); rc=$?
check new-usage-rc "$rc" 2
git_q -C "$T/clone" branch --no-track issue/8-taken origin/main
out=$(new 8 taken); rc=$?
check new-branch-exists-rc "$rc" 1
out=$(new --research probe-a); rc=$?
check research-rc "$rc" 0
check research-detached "$(git -C "$WT/research-probe-a" symbolic-ref -q HEAD || echo detached)" detached
check research-scratch "$([ -d "$T/gates/research-probe-a-scratch" ] && echo yes)" yes
check research-no-branch "$(git -C "$T/clone" branch --list 'issue/*probe*')" ""

# wt-done refusals
printf 'x\n' > "$WT/issue-5-alpha-one/untracked.txt"
out=$(FAKE_GH_STATE=MERGED done_ 5); rc=$?
check done-dirty-rc "$rc" 1
check done-dirty-msg "$(grep -c 'not clean' <<<"$out")" 1
rm "$WT/issue-5-alpha-one/untracked.txt"
out=$(FAKE_GH_STATE=OPEN done_ 5); rc=$?
check done-open-rc "$rc" 1
check done-open-msg "$(grep -c 'not MERGED' <<<"$out")" 1
(cd "$WT/issue-5-alpha-one" && exec sleep 300) &
sleeper=$!
sleep 0.3
out=$(FAKE_GH_STATE=MERGED done_ 5); rc=$?
check done-busy-rc "$rc" 1
check done-busy-msg "$(grep -c "process $sleeper has its cwd" <<<"$out")" 1
kill "$sleeper"; wait "$sleeper" 2>/dev/null; sleeper=
out=$(FAKE_GH_STATE=MERGED done_ 99); rc=$?
check done-none-rc "$rc" 1
out=$(done_); rc=$?
check done-usage-rc "$rc" 2

# dry run changes nothing
git_q -C "$WT/issue-5-alpha-one" push -q -u origin issue/5-alpha-one 2>/dev/null
out=$(FAKE_GH_STATE=MERGED done_ 5 --dry-run); rc=$?
check dry-rc "$rc" 0
check dry-remote-line "$(grep -c 'would: git push origin --delete issue/5-alpha-one' <<<"$out")" 1
check dry-worktree "$([ -d "$WT/issue-5-alpha-one" ] && echo yes)" yes
check dry-branch "$(git -C "$T/clone" branch --list issue/5-alpha-one | wc -l)" 1
check dry-remote "$(git -C "$T/origin.git" branch --list issue/5-alpha-one | wc -l)" 1

# real removal
out=$(FAKE_GH_STATE=MERGED done_ 5); rc=$?
check done-rc "$rc" 0
check done-worktree "$([ -e "$WT/issue-5-alpha-one" ] && echo yes || echo no)" no
check done-branch "$(git -C "$T/clone" branch --list issue/5-alpha-one | wc -l)" 0
check done-remote "$(git -C "$T/origin.git" branch --list issue/5-alpha-one | wc -l)" 0
check done-other-kept "$([ -d "$WT/research-probe-a" ] && echo yes)" yes

# research needs no PR
out=$(FAKE_GH_STATE=OPEN done_ --research probe-a); rc=$?
check research-done-rc "$rc" 0
check research-done-gone "$([ -e "$WT/research-probe-a" ] && echo yes || echo no)" no

if [ "$fails" -ne 0 ]; then
  echo "test-wt: $fails failed"
  exit 1
fi
echo "test-wt: ok"
