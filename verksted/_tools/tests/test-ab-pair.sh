#!/usr/bin/env bash
# Usage: test-ab-pair.sh
# Hermetic test of ab-pair.sh with a fake lock, fake binaries and a fake Xid command.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/fakes.sh"
AB=$HERE/../ab-pair.sh
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
mkdir "$T/gates" "$T/bin" "$T/fake"
fake_timeout "$T/fake"
echo 0 > "$T/xid"
printf '#!/bin/sh\nexec "$@"\n' > "$T/lock.sh"
cat > "$T/bin/good" <<'EOF'
#!/bin/sh
echo "args: $*"
i=0
while [ $i -lt 20000 ]; do i=$((i + 1)); done
EOF
printf '#!/bin/sh\necho "args: $*"\nexit 3\n' > "$T/bin/bad"
printf '#!/bin/sh\necho 5 > "%s/xid"\n' "$T" > "$T/bin/bump"
printf '#!/bin/sh\necho "hang ran" >&2\nexit 99\n' > "$T/bin/hang"
chmod +x "$T"/lock.sh "$T"/bin/*

ab() {
  PATH="$T/fake:$PATH" FAKE_TIMEOUT_LOG="$T/timeouts" RCSR_GATES=$T/gates RCSR_MACHINE_DIR=$T RCSR_LOCK_SCRIPT=$T/lock.sh RCSR_XID_CMD="cat $T/xid" bash "$AB" "$@" > "$T/out" 2>&1
}

# three seeds: order AB, BA, AB; {seed} replaced
ab --label ok --a "$T/bin/good" --b "$T/bin/good" -- --seed {seed} x; rc=$?
check ok-rc "$rc" "0"
d=$T/gates/ab-ok
check ok-rows "$(wc -l < "$d/runs.tsv")" "6"
check ok-order "$(awk -F'\t' '$1 == "A" { printf "%s ", $3 }' "$d/runs.tsv")" "AB BA AB "
check ok-first-arm "$(awk -F'\t' '{ printf "%s", $1 } END { print "" }' "$d/runs.tsv")" "ABBAAB"
check ok-seed-arg "$(cat "$d/B-s2.out")" "args: --seed 2 x"
check ok-cols "$(awk -F'\t' '{ print NF }' "$d/runs.tsv" | sort -u)" "8"
check ok-medians "$(grep -c 'median_user_s=' "$d/summary.txt")" "2"
check ok-ratio "$(grep -cE '^ratio_user b/a ' "$d/summary.txt")" "1"
check ok-sha "$(grep -c 'sha256=' "$d/summary.txt")" "2"
check ok-done "$(tail -n 1 "$T/gates/ab-ok.log" | cut -c1-12)" "RC_DONE rc=0"
check ok-limits "$(grep -c "^280 " "$T/timeouts")" "6"
check ok-kill-after "$(sort -u "$T/timeouts.kill" | tr "\n" " ")" "30 "
check ok-kill-after-count "$(wc -l < "$T/timeouts.kill" | tr -d " ")" "6"

# a failing arm: exit 1, pair continues
ab --label bad --a "$T/bin/good" --b "$T/bin/bad" --seeds 1,2 -- x; rc=$?
check bad-rc "$rc" "1"
check bad-rows "$(wc -l < "$T/gates/ab-bad/runs.tsv")" "4"
check bad-rc-col "$(awk -F'\t' '$8 == 3' "$T/gates/ab-bad/runs.tsv" | wc -l)" "2"

# one seed
ab --label one --a "$T/bin/good" --b "$T/bin/good" --seeds 7 -- x; rc=$?
check one-rc "$rc" "0"
check one-rows "$(wc -l < "$T/gates/ab-one/runs.tsv")" "2"

# a timeout records rc 124
ab --label slow --a "$T/bin/hang" --b "$T/bin/good" --seeds 1 --timeout 1 -- x; rc=$?
check slow-rc "$rc" "1"
check slow-124 "$(awk -F'\t' '$1 == "A" { print $8 }' "$T/gates/ab-slow/runs.tsv")" "124"
check slow-limit "$(grep -c "^1 $T/bin/hang x\$" "$T/timeouts")" "1"

# Xid rise
ab --label xid --a "$T/bin/bump" --b "$T/bin/good" --seeds 1,2,3 -- x; rc=$?
check xid-rc "$rc" "70"
check xid-stops "$(wc -l < "$T/gates/ab-xid/runs.tsv")" "2"
check xid-done "$(tail -n 1 "$T/gates/ab-xid.log" | cut -c1-13)" "RC_DONE rc=70"
echo 0 > "$T/xid"

# usage errors
ab --label t400 --a "$T/bin/good" --b "$T/bin/good" --timeout 400 -- x; rc=$?
check timeout-400 "$rc" "2"
check timeout-400-nodir "$([ -e "$T/gates/ab-t400" ] && echo yes || echo no)" "no"
ab --label ok --a "$T/bin/good" --b "$T/bin/good" -- x; rc=$?
check existing-runs "$rc" "2"
ab --label Bad_Label --a "$T/bin/good" --b "$T/bin/good" -- x; rc=$?
check label "$rc" "2"
ab --label nobin --a "$T/bin/missing" --b "$T/bin/good" -- x; rc=$?
check missing-bin "$rc" "2"
ab --label noseed --a "$T/bin/good" --b "$T/bin/good" --seeds a,b -- x; rc=$?
check bad-seeds "$rc" "2"
ab; rc=$?
check no-args "$rc" "2"
ab -h; rc=$?
check help "$rc" "0"
check help-text "$(head -n 1 "$T/out" | cut -c1-12)" "Usage: ab-pa"

if [ "$fails" -ne 0 ]; then
  echo "test-ab-pair: $fails failed"
  exit 1
fi
echo "test-ab-pair: ok"
