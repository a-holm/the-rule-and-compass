#!/usr/bin/env bash
# Usage: test-clerk-check.sh
# Hermetic test of clerk-check.sh against a temp bare origin and clone, with a fake gh.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
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
git_q() { git -c user.name=t -c user.email=t@t.invalid "$@"; }

git init -q --bare -b main "$T/origin.git"
git clone -q "$T/origin.git" "$T/clone" 2>/dev/null
git_q -C "$T/clone" checkout -q -b main
printf 'base\n' > "$T/clone/shared.txt"
git_q -C "$T/clone" add shared.txt
git_q -C "$T/clone" commit -q -m init
git_q -C "$T/clone" push -q origin main
git_q -C "$T/clone" checkout -q -b feat
printf 'feature\n' > "$T/clone/shared.txt"
mkdir -p "$T/clone/src"
printf 'new\n' > "$T/clone/src/a.rs"
git_q -C "$T/clone" add -A
git_q -C "$T/clone" commit -q -m feat
git_q -C "$T/clone" push -q origin feat
head=$(git -C "$T/clone" rev-parse HEAD)
git_q -C "$T/clone" checkout -q main
printf 'main change\n' > "$T/clone/shared.txt"
git_q -C "$T/clone" commit -q -am "diverge"
git_q -C "$T/clone" push -q origin main

mkdir -p "$T/bin" "$T/gates"
cat > "$T/bin/gh" <<'GH'
#!/bin/sh
n=0
[ -f "$FAKE_GH_COUNT" ] && n=$(cat "$FAKE_GH_COUNT")
echo $((n + 1)) > "$FAKE_GH_COUNT"
if [ "$n" -lt "${FAKE_GH_UNKNOWN:-0}" ]; then
  sed 's/"mergeable": *"[A-Z]*"/"mergeable":"UNKNOWN"/' "$FAKE_GH_JSON"
else
  cat "$FAKE_GH_JSON"
fi
GH
chmod +x "$T/bin/gh"
export PATH="$T/bin:$PATH" RCSR_ROOT="$T/clone" RCSR_GATES="$T/gates" RCSR_BASE=main
export FAKE_GH_COUNT=$T/count RCSR_RETRY_SLEEP=0

pr_json() {
  printf '{"state":"%s","baseRefName":"%s","headRefName":"feat","headRefOid":"%s","mergeable":"%s","files":[{"path":"shared.txt"},{"path":"src/a.rs"}]}\n' \
    "$1" "$2" "$head" "$3" > "$T/pr.json"
  export FAKE_GH_JSON=$T/pr.json
}
run() { bash "$TOOLS/clerk-check.sh" "$@" 2>&1; }

printf 'ok line\n' > "$T/log-a.log"
printf '%s\n' "unit|$T/log-a.log|ok [a-z]+" > "$T/claims-ok"
printf '%s\n' "unit|$T/log-a.log|ok [a-z]+" "gone|$T/missing.log|x" > "$T/claims-nolog"
printf '%s\n' "unit|$T/log-a.log|zzz" > "$T/claims-nomatch"

pr_json OPEN main MERGEABLE
out=$(run 7 --claims "$T/claims-ok" --scope 'shared.txt' --scope 'src/*'); rc=$?
check ok-rc "$rc" 0
check ok-state "$(grep -c "^state: OPEN  base: main  mergeable: MERGEABLE  head: $head" <<<"$out")" 1
check ok-found "$(grep -c '|found$' <<<"$out")" 1
check ok-merge "$(grep -c "^merge: gh pr merge 7 --squash" <<<"$out")" 1
check ok-stat "$(grep -c 'src/a.rs' <<<"$out")" 2
out=$(run 7 --claims "$T/claims-nolog"); rc=$?
check nolog-rc "$rc" 1
check nolog-msg "$(grep -c '|no log$' <<<"$out")" 1
out=$(run 7 --claims "$T/claims-nomatch"); rc=$?
check nomatch-rc "$rc" 1
check nomatch-msg "$(grep -c '|no match$' <<<"$out")" 1
out=$(run 7 --base other); rc=$?
check wrongbase-rc "$rc" 1
out=$(run 7 --scope 'src/*'); rc=$?
check scope-rc "$rc" 1
check scope-msg "$(grep -c '  shared.txt out of scope' <<<"$out")" 1
pr_json MERGED main MERGEABLE
out=$(run 7); rc=$?
check merged-rc "$rc" 1
pr_json OPEN main CONFLICTING
out=$(run 7); rc=$?
check conflict-rc "$rc" 1
check conflict-file "$(sed -n '/^conflicts:/,/^claims:/p' <<<"$out" | grep -c '  shared.txt')" 1
pr_json OPEN main MERGEABLE
rm -f "$FAKE_GH_COUNT"
out=$(FAKE_GH_UNKNOWN=2 run 7); rc=$?
check unknown-retry-rc "$rc" 0
check unknown-retry-count "$(cat "$FAKE_GH_COUNT")" 3
out=$(run); rc=$?
check usage-rc "$rc" 2
out=$(run abc); rc=$?
check usage-nonnumeric-rc "$rc" 2

# worktree section: gh pr merge --delete-branch removes the worktree that has the head branch
pr_json OPEN main MERGEABLE
out=$(run 7); check wt-none-msg "$(grep -c '^  none: gh pr merge --delete-branch has no worktree' <<<"$out")" 1
git_q -C "$T/clone" worktree add -q "$T/wt-feat" feat
out=$(run 7); rc=$?
check wt-clean-rc "$rc" 0
check wt-clean-msg "$(grep -c "^  $T/wt-feat clean at head" <<<"$out")" 1
printf 'x\n' > "$T/wt-feat/untracked.txt"
out=$(run 7); rc=$?
check wt-dirty-rc "$rc" 1
check wt-dirty-msg "$(grep -c "^  $T/wt-feat dirty" <<<"$out")" 1
rm "$T/wt-feat/untracked.txt"
git_q -C "$T/wt-feat" commit -q --allow-empty -m ahead
out=$(run 7); rc=$?
check wt-ahead-rc "$rc" 1
check wt-ahead-msg "$(grep -c "^  $T/wt-feat at " <<<"$out")" 1
git_q -C "$T/wt-feat" reset -q --hard "$head"
pr_json MERGED main MERGEABLE
out=$(run 7); check wt-merged-none "$(grep -c '^  none:' <<<"$out")" 1
pr_json OPEN main MERGEABLE
git_q -C "$T/clone" worktree remove "$T/wt-feat"

# read-only: the clone's refs and worktree are unchanged apart from the fetch
check readonly-status "$(git -C "$T/clone" status --porcelain)" ""
check readonly-no-worktrees "$(git -C "$T/clone" worktree list | wc -l)" 1

if [ "$fails" -ne 0 ]; then
  echo "test-clerk-check: $fails failed"
  exit 1
fi
echo "test-clerk-check: ok"
