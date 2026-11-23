#!/usr/bin/env bash
# Usage: test-wt-audit.sh
# Hermetic test of wt-audit.sh against a temp repo with four worktrees (clean, dirty, owner-less, merged), a stray one
# outside the worktrees folder, a missing one and a fake gh: the rows and their fields, the flags, and that nothing is removed.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/machine.sh"
. "$HERE/lib/fakes.sh"
TOOLS=$HERE/..
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
g() { git -c user.name=t -c user.email=t@t.invalid "$@"; }

mkdir -p "$T/bin" "$T/repo"
cat > "$T/bin/gh" <<'GH'
#!/bin/sh
case "$3" in
  issue/4-merged) echo MERGED ;;
  issue/6-nopr) exit 1 ;;
  *) echo OPEN ;;
esac
GH
chmod +x "$T/bin/gh"
g -C "$T/repo" init -q -b main
echo x > "$T/repo/f"
g -C "$T/repo" add f
g -C "$T/repo" commit -q -m init
printf 'verksted/**/output/\noutput/\n' >> "$T/repo/.git/info/exclude"
WT=$T/repo/worktrees
mkdir -p "$WT"
for pair in 1-clean 2-dirty 3-ownerless 4-merged 6-nopr; do
  g -C "$T/repo" worktree add -q -b "issue/$pair" "$WT/issue-$pair"
done
g -C "$T/repo" worktree add -q --detach "$T/stray" HEAD
g -C "$T/repo" worktree add -q --detach "$WT/gone" HEAD
mkdir -p "$WT/issue-1-clean/output" "$WT/issue-2-dirty/output" "$WT/issue-4-merged/output" "$WT/issue-6-nopr/output"
printf 'path: x\nowner: agent-a\n' > "$WT/issue-1-clean/output/worktree.md"
printf 'owner: agent-b\n' > "$WT/issue-2-dirty/output/worktree.md"
printf 'owner: agent-d\n' > "$WT/issue-4-merged/output/worktree.md"
printf 'owner: agent-f\n' > "$WT/issue-6-nopr/output/worktree.md"
echo y > "$WT/issue-2-dirty/untracked.txt"
settle() { find "$1" -name .git -prune -o -type f -exec touch -d "@$2" {} +; }
settle "$WT/issue-1-clean" 1799999940
settle "$WT/issue-2-dirty" 1799985600
settle "$WT/issue-3-ownerless" 1799999940
settle "$WT/issue-4-merged" 1799999940
settle "$WT/issue-6-nopr" 1799999940
settle "$T/stray" 1799999940
rm -rf "$WT/gone"

before=$(g -C "$T/repo" worktree list | wc -l | tr -d ' ')
audit() { PATH=$T/bin:$PATH RCSR_ROOT=$T/repo RCSR_NOW=1800000000 bash "$TOOLS/wt-audit.sh" "$@" 2>&1; }
out=$(audit); rc=$?
check rc "$rc" 0
check rows "$(printf '%s\n' "$out" | wc -l | tr -d ' ')" 7
check main-not-listed "$(printf '%s\n' "$out" | grep -c "^$T/repo ")" 0
check clean "$(printf '%s\n' "$out" | grep "^$WT/issue-1-clean ")" "$WT/issue-1-clean branch=issue/1-clean age=1min state=clean owner=agent-a pr=OPEN flags=-"
check dirty "$(printf '%s\n' "$out" | grep "^$WT/issue-2-dirty ")" "$WT/issue-2-dirty branch=issue/2-dirty age=240min state=dirty owner=agent-b pr=OPEN flags=dirty-old"
check ownerless "$(printf '%s\n' "$out" | grep "^$WT/issue-3-ownerless ")" "$WT/issue-3-ownerless branch=issue/3-ownerless age=1min state=clean owner=none pr=OPEN flags=no-owner"
check merged "$(printf '%s\n' "$out" | grep "^$WT/issue-4-merged ")" "$WT/issue-4-merged branch=issue/4-merged age=1min state=clean owner=agent-d pr=MERGED flags=merged"
check pr-unknown "$(printf '%s\n' "$out" | grep "^$WT/issue-6-nopr ")" "$WT/issue-6-nopr branch=issue/6-nopr age=1min state=clean owner=agent-f pr=unknown flags=-"
check stray "$(printf '%s\n' "$out" | grep "^$T/stray ")" "$T/stray branch=detached age=1min state=clean owner=none pr=none flags=stray,no-owner"
check missing "$(printf '%s\n' "$out" | grep "^$WT/gone ")" "$WT/gone branch=? age=? state=? owner=none pr=unknown flags=missing"
check old-threshold "$(audit --old 300 | grep -c 'dirty-old')" 0
check nothing-removed "$(g -C "$T/repo" worktree list | wc -l | tr -d ' ')" "$before"
check branches-kept "$(g -C "$T/repo" branch --list 'issue/*' | wc -l | tr -d ' ')" 5
check dirty-file-kept "$(cat "$WT/issue-2-dirty/untracked.txt")" y

audit --old x > /dev/null; check bad-old $? 2
audit --old > /dev/null; check old-no-value $? 2
audit --bogus > /dev/null; check bad-flag $? 2
audit -h > /dev/null; check help $? 0

[ "$fails" -eq 0 ] && echo "test-wt-audit: ok"
exit "$fails"
