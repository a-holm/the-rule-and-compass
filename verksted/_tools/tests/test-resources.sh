#!/usr/bin/env bash
# Usage: test-resources.sh
# Hermetic test of resources.sh with temp machine.md fixtures, a stubbed free-space command and a stubbed process list:
# key missing, machine.md missing, space short, ok, units, slot holders over the limit, a cargo process, usage.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/fakes.sh"
S=$HERE/../resources.sh
T=$(cd "$(mktemp -d)" && pwd -P)
cleanup() { [ -z "${HOLD_PID:-}" ] || release; rm -rf "$T"; }
trap cleanup EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}

mkdir -p "$T/oc/build-slots" "$T/local"
export RCSR_MACHINE_DIR=$T/oc RCSR_LOCAL=$T/local
printf '#!/bin/sh\necho "$1" > "%s/volume-arg"\necho "${FAKE_FREE:-0}"\n' "$T" > "$T/free"
chmod +x "$T/free"
export RCSR_FREE_CMD=$T/free RCSR_PS_CMD="printf 'bash\nrustc\n'"
profile() { printf -- '- BUILD_VOLUME: %s\n- DISK_MIN_FREE: %s\n- BUILD_SLOTS: %s\n' "$1" "$2" "$3" > "$T/local/machine.md"; }
run() { bash "$S" "$@" > "$T/out" 2> "$T/err"; echo $?; }

check no-machine-md-rc "$(run)" 2
check no-machine-md-names-key "$(grep -c 'DISK_MIN_FREE' "$T/err")" 1

printf -- '- BUILD_VOLUME: /vol\n- BUILD_SLOTS: 3\n' > "$T/local/machine.md"
check no-min-rc "$(run)" 2
check no-min-names-key "$(grep -c 'key DISK_MIN_FREE missing' "$T/err")" 1
printf -- '- DISK_MIN_FREE: 10 GB\n- BUILD_SLOTS: 3\n' > "$T/local/machine.md"
check no-volume-rc "$(run)" 2
check no-volume-names-key "$(grep -c 'key BUILD_VOLUME missing' "$T/err")" 1
printf -- '- BUILD_VOLUME: /vol\n- DISK_MIN_FREE: 10 GB\n' > "$T/local/machine.md"
check no-slots-rc "$(run)" 2
check no-slots-names-key "$(grep -c 'key BUILD_SLOTS missing' "$T/err")" 1
profile /vol lots 3
check bad-size-rc "$(run)" 2
check bad-size-names-key "$(grep -c 'key DISK_MIN_FREE is not a size' "$T/err")" 1

profile /vol '10 GB' 3
check short-rc "$(FAKE_FREE=9999999999 run)" 1
check short-measured "$(grep -c '^FAIL disk: free=9999999999 bytes on /vol, below DISK_MIN_FREE=10000000000 bytes$' "$T/out")" 1
check volume-passed "$(cat "$T/volume-arg")" /vol
check ok-rc "$(FAKE_FREE=10000000000 run)" 0
check ok-lines "$(grep -c '^ok ' "$T/out")" 3
profile /vol '2 GiB' 3
check gib-short-rc "$(FAKE_FREE=2147483647 run)" 1
check gib-ok-rc "$(FAKE_FREE=2147483648 run)" 0
profile /vol 500 3
check bytes-ok-rc "$(FAKE_FREE=500 run)" 0
profile /vol '1 TiB' 3
check tib-short-rc "$(FAKE_FREE=1099511627775 run)" 1
check tib-ok-rc "$(FAKE_FREE=1099511627776 run)" 0
check unreadable-free-rc "$(RCSR_FREE_CMD=true run)" 1
check unreadable-free-msg "$(grep -c '^FAIL disk: free space of /vol unreadable$' "$T/out")" 1

profile /vol '1 B' 1
hold "exec 9> $(printf %q "$T/oc/build-slots/slot1.lock") && flock 9"; check holder-ready $? 0
check one-holder-ok-rc "$(FAKE_FREE=9 run)" 0
check one-holder-line "$(grep -c '^ok builds: running=1, BUILD_SLOTS=1$' "$T/out")" 1
profile /vol '1 B' 0
check holders-over-rc "$(FAKE_FREE=9 run)" 1
check holders-over-line "$(grep -c '^FAIL builds: running=1, above BUILD_SLOTS=0$' "$T/out")" 1
release
profile /vol '1 B' 1
check no-holder-line "$(FAKE_FREE=9 run >/dev/null; grep -c '^ok builds: running=0, BUILD_SLOTS=1$' "$T/out")" 1

export RCSR_PS_CMD="printf 'bash\ncargo\ncargo\n'"
check cargo-rc "$(FAKE_FREE=9 run)" 1
check cargo-line "$(grep -c '^FAIL cargo: running=2, expected 0 before a build starts$' "$T/out")" 1
check cargo-other-lines-ok "$(grep -c '^ok ' "$T/out")" 2

check bogus-arg "$(run --bogus)" 2
check help "$(run -h)" 0

[ "$fails" -eq 0 ] && echo "test-resources: ok"
exit "$fails"
