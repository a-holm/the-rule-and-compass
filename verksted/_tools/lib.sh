#!/usr/bin/env bash
# Sourced by verksted/_tools scripts; not run.
# Writes: nothing.
# Defines the paths, the RCSR_* overrides and the helpers every script shares. GATES, MACHINE_SCRIPTS, SANITIZER
# and CUDA_ARCH come from $VLOCAL/machine.md (RCSR_GATES, RCSR_MACHINE_DIR and RCSR_SANITIZER override the first
# three); no tracked file holds a machine value. A script that sources this exits 2 when a key is missing; an interactive shell gets status 2 from the source and stays alive.
# local-init.sh sets LIB_NO_MACHINE=1 before it sources this, because it creates machine.md. No function creates $GATES.

VERKSTED=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RCSR_ROOT=${RCSR_ROOT:-$(git -C "$VERKSTED" rev-parse --show-toplevel)}
ROOT=$(dirname "$(git -C "$RCSR_ROOT" rev-parse --path-format=absolute --git-common-dir)")
# crate_dir [worktree]: the one first-level directory of the worktree (default $RCSR_ROOT) whose Cargo.toml
# declares package rule-and-compass; RCSR_CRATE names it instead. Prints the path; exit 1 with a message
# on stderr when none or several match.
crate_dir() {
  local wt=${1:-$RCSR_ROOT} f hits=()
  if [ -n "${RCSR_CRATE:-}" ]; then
    printf '%s\n' "$wt/$RCSR_CRATE"
    return 0
  fi
  for f in "$wt"/*/Cargo.toml; do
    [ -f "$f" ] || continue
    awk '/^\[/ { s = ($0 == "[package]") } s && /^name *= *"rule-and-compass" *$/ { ok = 1 } END { exit !ok }' "$f" &&
      hits+=("${f%/Cargo.toml}")
  done
  if [ "${#hits[@]}" -ne 1 ]; then
    echo "engine crate: ${#hits[@]} directories under $wt declare package rule-and-compass; set RCSR_CRATE" >&2
    return 1
  fi
  printf '%s\n' "${hits[0]}"
}
VLOCAL=${RCSR_LOCAL:-$ROOT/verksted/_local}
# machine_key <KEY>: prints the value of the line `- KEY: value` of $VLOCAL/machine.md (first match, trimmed).
# Status 2 with a message naming the key when machine.md or the line is missing, or the value is empty or `<set me>`:
# `return` in an interactive shell, `exit` otherwise, so a script never continues with the key unset.
# Inside $(...) only the subshell ends; write `X=$(machine_key KEY) || exit 2`.
machine_key() {
  local v=
  [ -f "$VLOCAL/machine.md" ] && v=$(sed -n "s/^- *$1: *//p" "$VLOCAL/machine.md" | head -n 1 | sed 's/[[:space:]]*$//')
  if [ -z "$v" ] || [ "$v" = "<set me>" ]; then
    echo "key $1 missing in $VLOCAL/machine.md (run local-init.sh, then set it; see _shared/local.md)" >&2
    [[ $- == *i* ]] && return 2
    exit 2
  fi
  printf '%s\n' "$v"
}
WT=$ROOT/worktrees
BASE=${RCSR_BASE:-integration/gpu-hardening}
if [ -z "${LIB_NO_MACHINE:-}" ]; then
  GATES=${RCSR_GATES:-}
  [ -n "$GATES" ] || GATES=$(machine_key GATES) || { [[ $- == *i* ]] && return 2; exit 2; }
  MACHINE_SCRIPTS=${RCSR_MACHINE_DIR:-}
  [ -n "$MACHINE_SCRIPTS" ] || MACHINE_SCRIPTS=$(machine_key MACHINE_SCRIPTS) || { [[ $- == *i* ]] && return 2; exit 2; }
  SANITIZER=${RCSR_SANITIZER:-}
  [ -n "$SANITIZER" ] || SANITIZER=$(machine_key SANITIZER) || { [[ $- == *i* ]] && return 2; exit 2; }
  CUDA_ARCH=$(machine_key CUDA_ARCH) || { [[ $- == *i* ]] && return 2; exit 2; }
  G=$GATES
  # /proc/*/fd links hold the physical path, so lock probes need $OC without symlinks.
  OC=$(cd "$MACHINE_SCRIPTS" 2> /dev/null && pwd -P || echo "$MACHINE_SCRIPTS")
  PAUSE=$OC/PAUSE
  STREAMING=$OC/STREAMING
  LOCKFILE=$OC/full-suite.lock
  SLOTDIR=$OC/build-slots
  B=${RCSR_SLOT_SCRIPT:-$OC/build-slot.sh}
  L=${RCSR_LOCK_SCRIPT:-$OC/one-full-suite.sh}
fi
LOCK_HOLD_MAX=600
KILL_AFTER=${RCSR_KILL_AFTER:-30} # a positive number of seconds; anything else (0 disables the KILL) is 30
case $KILL_AFTER in
  ""|*[!0-9.]*|*.*.*) KILL_AFTER=30 ;;
  *[1-9]*) ;;
  *) KILL_AFTER=30 ;;
esac

# tools_behind <worktree>: prints how many commits of origin/$BASE that touch verksted/ the worktree HEAD lacks.
# Status 1 (nothing printed) when origin/$BASE does not exist.
tools_behind() {
  git -C "$1" rev-parse --verify -q "refs/remotes/origin/$BASE^{commit}" > /dev/null || return 1
  git -C "$1" rev-list --count "HEAD..refs/remotes/origin/$BASE" -- verksted
}

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

# xid_start: prints the local start time of a hold, for xid_new_since.
xid_start() {
  date '+%Y-%m-%d %H:%M:%S'
}

# xid_new_since <time>: prints the number of `NVRM: Xid` kernel lines since <time>, never a journal total.
# Fails closed (returns 1, prints nothing) when <time> is empty or journalctl fails.
# RCSR_XID_CMD, when set, replaces journalctl and prints the new-line count itself.
xid_new_since() {
  local since=${1:-} out n
  [ -n "$since" ] || { echo "xid: start time missing" >&2; return 1; }
  if [ -n "${RCSR_XID_CMD:-}" ]; then
    n=$(bash -c "$RCSR_XID_CMD") || { echo "xid: check command failed" >&2; return 1; }
  else
    out=$(journalctl -k --since "$since" 2>/dev/null) || { echo "xid: journalctl failed" >&2; return 1; }
    n=$(printf '%s\n' "$out" | grep -c 'NVRM: Xid')
  fi
  case "$n" in ''|*[!0-9]*) echo "xid: unreadable count '$n'" >&2; return 1 ;; esac
  echo "$n"
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
