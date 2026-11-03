#!/usr/bin/env bash
# Usage: test-status.sh
# Hermetic test of status.sh with a fake gh, a fake nvidia-smi, a temp machine dir and a temp repo:
# lock holder, SILENT worktree, section headers, usage errors, no pgrep or pkill.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TOOLS=$HERE/..
T=$(mktemp -d)
bg=""
cleanup() { [ -z "$bg" ] || kill $bg 2> /dev/null; rm -rf "$T"; }
trap cleanup EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
g() { git -c user.name=t -c user.email=t@t.invalid "$@"; }

mkdir -p "$T/bin" "$T/oc" "$T/repo"
printf '#!/bin/sh\necho "#12 issue/12-x abc1234 2026-11-03T10:00:00Z"\n' > "$T/bin/gh"
printf '#!/bin/sh\necho "37 %%, 1024 MiB, 12288 MiB"\n' > "$T/bin/nvidia-smi"
printf '#!/bin/sh\nexit 1\n' > "$T/oc/desktop-busy.sh"
chmod +x "$T/bin/gh" "$T/bin/nvidia-smi" "$T/oc/desktop-busy.sh"
g -C "$T/repo" init -q -b main
g -C "$T/repo" commit -q --allow-empty -m "tip subject"
g -C "$T/repo" update-ref refs/remotes/origin/main HEAD
mkdir -p "$T/repo/worktrees"
g -C "$T/repo" worktree add -q -b issue/1-old "$T/repo/worktrees/old"
g -C "$T/repo" worktree add -q -b issue/2-new "$T/repo/worktrees/new"
g -C "$T/repo" worktree add -q -b issue/3-gone "$T/repo/worktrees/gone"
echo x > "$T/repo/worktrees/old/f"
find "$T/repo/worktrees/old" -name .git -prune -o -type f -exec touch -d '3 hours ago' {} +
touch "$T/repo/worktrees/new/f"
rm -rf "$T/repo/worktrees/gone"

( exec 9> "$T/oc/full-suite.lock"; flock 9; exec sleep 60 ) &
bg=$!
for _ in $(seq 1 50); do
  flock -n "$T/oc/full-suite.lock" true 2> /dev/null || break
  sleep 0.1
done

st() {
  PATH=$T/bin:$PATH RCSR_ROOT=$T/repo RCSR_BASE=main RCSR_MACHINE_DIR=$T/oc RCSR_XID_CMD='echo 7' \
    bash "$TOOLS/status.sh" --no-fetch "$@"
}
out=$(st); rc=$?
check rc "$rc" 0
for s in git agents machine; do check "header-$s" "$(echo "$out" | grep -c "^== $s\$")" 1; done
check tip "$(echo "$out" | grep -c '^origin/main: .* tip subject$')" 1
check pr "$(echo "$out" | grep -c '#12 issue/12-x abc1234')" 1
check holder "$(echo "$out" | grep -c "^lock: held by pid $bg ")" 1
check silent-old "$(echo "$out" | grep -c '^old: .*dirty=y.* SILENT$')" 1
check not-silent-new "$(echo "$out" | grep '^new: ' | grep -c SILENT)" 0
check missing "$(echo "$out" | grep -c '^gone: missing$')" 1
check gpu "$(echo "$out" | grep -c '^gpu: 37 %, 1024 MiB, 12288 MiB$')" 1
check xid "$(echo "$out" | grep -c '^xid: 7$')" 1
check busy "$(echo "$out" | grep -c '^desktop-busy: exit 1$')" 1
check flags "$(echo "$out" | grep -c '^flags: PAUSE=n STREAMING=n$')" 1

touch "$T/oc/PAUSE"
check pause "$(st --section machine | grep -c '^flags: PAUSE=y')" 1
only=$(st --section machine)
check section-only "$(echo "$only" | grep -c '^== ')" 1

kill $bg 2> /dev/null
wait $bg 2> /dev/null
bg=""
check lock-free "$(st --section machine | grep -c '^lock: free$')" 1

st --section bogus > /dev/null 2>&1
check usage-rc "$?" 2
check no-pgrep "$(grep -vE '^\s*#' "$TOOLS/status.sh" | grep -cE 'pgrep|pkill')" 0

if [ "$fails" -ne 0 ]; then
  echo "$fails failure(s)"
  exit 1
fi
echo "test-status: ok"
