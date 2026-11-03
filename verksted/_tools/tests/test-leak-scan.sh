#!/usr/bin/env bash
# Usage: test-leak-scan.sh
# Hermetic test of leak-scan.sh in scratch repositories with a made-up word.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCAN=$HERE/../leak-scan.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  local name=$1 got=$2 want=$3
  if [ "$got" != "$want" ]; then
    echo "FAIL $name: got '$got' want '$want'"
    fails=$((fails + 1))
  fi
}
git_q() { git -c user.name=t -c user.email=t@t.invalid "$@"; }
scan() { env -u RCSR_BANNED_WORDS RCSR_ROOT="$R" HOME="$T/home" bash "$SCAN" "$@" 2>&1; echo "rc=$?"; }

R=$T/repo
mkdir -p "$R" "$T/home"
git -C "$R" init -q
printf '# comment\n\nZorbulax\nSecondWord\n' > "$T/list"
printf 'clean line\n' > "$R/a.txt"
git_q -C "$R" add a.txt
git_q -C "$R" commit -q -m base
BASE=$(git -C "$R" rev-parse HEAD)

check clean "$(scan --list "$T/list")" "rc=0"

printf 'ok\nsee zORBULAX here\n' > "$R/b.txt"
git_q -C "$R" add b.txt
out=$(scan --list "$T/list")
check text-hit "$out" "$(printf 'b.txt:2: banned word 3\nrc=1')"
case "$out" in *[Zz]orbulax*) check word-not-printed "printed" "hidden" ;; esac

printf 'x\n' > "$R/secondword-notes.txt"
git_q -C "$R" add secondword-notes.txt
out=$(scan --list "$T/list")
check path-hit "$out" "$(printf 'b.txt:2: banned word 3\nsecondword-notes.txt:0: banned word 4\nrc=1')"

git_q -C "$R" rm -q -f b.txt secondword-notes.txt
printf 'Zorbulax\0\0bin\n' > "$R/bin.dat"
git_q -C "$R" add bin.dat
check binary-skipped "$(scan --list "$T/list")" "rc=0"

# no list
check no-list "$(scan --list "$T/missing")" "$(printf 'skip: no banned-word list at %s/missing\nrc=3' "$T")"
check no-list-default "$(scan)" "$(printf 'skip: no banned-word list at %s/home/.config/rcsr/banned-words\nrc=3' "$T")"
check env-list "$(env RCSR_BANNED_WORDS="$T/list" RCSR_ROOT="$R" bash "$SCAN" 2>&1; echo "rc=$?")" "rc=0"
check empty-list "$(printf '# only\n\n' > "$T/empty"; scan --list "$T/empty")" "$(printf 'skip: no banned-word list at %s/empty\nrc=3' "$T")"

# usage
check usage-bad-flag "$(scan --bogus | tail -1)" "rc=2"
check usage-missing-arg "$(scan --list | tail -1)" "rc=2"
check usage-bad-range "$(scan --list "$T/list" --diff nodots | tail -1)" "rc=2"

# --diff: added line and commit message
git_q -C "$R" commit -q -m "mention ZORBULAX in message"
printf 'second\nthird zorbulax line\n' >> "$R/a.txt"
git_q -C "$R" commit -q -am "plain message"
HEAD=$(git -C "$R" rev-parse HEAD)
out=$(scan --list "$T/list" --diff "$BASE..$HEAD")
check diff-line "$(echo "$out" | grep -c '^a.txt:3: banned word 3$')" "1"
check diff-message "$(echo "$out" | grep -c '^commit-[0-9a-f]*:0: banned word 3$')" "1"
check diff-rc "$(echo "$out" | tail -1)" "rc=1"
check diff-hidden "$(echo "$out" | grep -ci zorbulax)" "0"
check diff-clean "$(scan --list "$T/list" --diff "$HEAD..$HEAD")" "rc=0"
# a hit that is in the base but not in the added lines is not reported
check diff-ignores-old "$(scan --list "$T/list" --diff "$HEAD~1..$HEAD" | grep -c commit-)" "0"

if [ "$fails" -ne 0 ]; then
  echo "test-leak-scan: $fails failed"
  exit 1
fi
echo "test-leak-scan: ok"
