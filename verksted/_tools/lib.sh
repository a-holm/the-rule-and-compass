#!/usr/bin/env bash
# Sourced by verksted/_tools scripts; not run.
# Writes: nothing.
# Defines the paths, the RCSR_* overrides and the helpers every script shares. Defaults are
# machine values; tests override them. No function creates $GATES.

VERKSTED=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RCSR_ROOT=${RCSR_ROOT:-$(git -C "$VERKSTED" rev-parse --show-toplevel)}
ROOT=$(dirname "$(git -C "$RCSR_ROOT" rev-parse --path-format=absolute --git-common-dir)")
WT=$ROOT/worktrees
BASE=${RCSR_BASE:-integration/gpu-hardening}
GATES=${RCSR_GATES:-/mnt/lager/tmp/rcsr-gates}
G=$GATES
OC=${RCSR_MACHINE_DIR:-/mnt/lager/tmp/opencode}
# /proc/*/fd links hold the physical path, so lock probes need $OC without symlinks.
OC=$(cd "$OC" 2> /dev/null && pwd -P || echo "$OC")
PAUSE=$OC/PAUSE
STREAMING=$OC/STREAMING
LOCKFILE=$OC/full-suite.lock
SLOTDIR=$OC/build-slots
B=${RCSR_SLOT_SCRIPT:-$OC/build-slot.sh}
L=${RCSR_LOCK_SCRIPT:-$OC/one-full-suite.sh}
SANITIZER=${RCSR_SANITIZER:-/usr/local/cuda-13.0/bin/compute-sanitizer}
CUDA_ARCH=sm_86
LOCK_HOLD_MAX=600
KILL_AFTER=${RCSR_KILL_AFTER:-30} # a positive number of seconds; anything else (0 disables the KILL) is 30
case $KILL_AFTER in
  ""|*[!0-9.]*|*.*.*) KILL_AFTER=30 ;;
  *[1-9]*) ;;
  *) KILL_AFTER=30 ;;
esac

need_gates() {
  if [ ! -d "$GATES" ]; then
    echo "gates dir $GATES missing; link it to the log directory or set RCSR_GATES" >&2
    exit 1
  fi
}

xid_count() {
  local n
  if [ -n "${RCSR_XID_CMD:-}" ]; then
    n=$(bash -c "$RCSR_XID_CMD")
  else
    n=$(journalctl -k 2>/dev/null | grep -c 'NVRM: Xid')
  fi
  echo "${n:-0}"
}

rc_done() {
  printf 'RC_DONE rc=%s at=%s\n' "$1" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

usage() {
  awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[1]}"
}

log_path() {
  case "$1" in
    ''|*[!0-9]*) echo "$GATES/$1-$2.log" ;;
    *) echo "$GATES/issue$1-$2.log" ;;
  esac
}
