#!/usr/bin/env bash
# Sourced by runbooks; not run.
# Writes: $RUN/done/<step>.done, $RUN/run.log, $RUN/<name>.log
# Usage in a runbook (after `set -uo pipefail`):
#   RUN=<dir>; STEPS=(a b c); PROTECTED=(seal); step_a() { ...; }
#   source verksted/_tools/runlib.sh; runlib_main "$@"
# A step named x-y is the function step_x_y. Optional hooks and variables:
#   profile_<name>()  called when PROFILE=<name> (default real); an unknown PROFILE dies
#   RL_BIN, RL_BIN_SHA  pinned binary and its sha256 file (default $RL_BIN.sha256), checked before every hold
#   RL_TIME           the time command (default /usr/bin/time)
#   RL_LOCK_RUN       the reaper script (default lock-run.sh next to this file); rl_gpu runs every hold through it
#   RL_CHUNK_GAP_S    pause between chunks (default 0)
#   RL_GRACE_S        extra seconds of each rl_chunks hold beyond its budget (default 120)
# Commands (runlib_main):
#   (none)        run every step of STEPS without a done-file, in order; protected steps are skipped
#                 unless SEALED=1
#   list          print each step with its done time or `pending`
#   <step>...     run these steps; a done step is skipped, FORCE=1 reruns it; a protected step dies
#                 unless SEALED=1
# Holds: rl_gpu (test lock, Xid-checked, reaped by lock-run.sh), rl_cpu (build slot), rl_chunks (resumable search).
# After a rl_gpu command ends, lock-run.sh ends by pid every process it left holding the test lock; it acts only on a flock
# its own process holds. A runbook that takes the lock without runlib.sh runs `bash lock-run.sh "$LOCKFILE" <command...>` inside the lock script.
# The budget of rl_chunks counts the `[time] wall=` of each hold, never the wait for the lock.
# The command of rl_chunks runs with RL_CHUNK_S (its search budget) and RL_CHUNK_N exported; the hold
# timeout is RL_CHUNK_S + RL_GRACE_S, which must not exceed LOCK_HOLD_MAX.

: "${RUN:?runlib: set RUN before sourcing}"
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RL_TIME=${RL_TIME:-/usr/bin/time}
RL_LOCK_RUN=${RL_LOCK_RUN:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lock-run.sh}
RL_TIMEF='[time] wall=%e s user=%U s sys=%S s maxrss=%M KB rc=%x'
mkdir -p "$RUN/done"

rl_log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$RUN/run.log" >&2; }
rl_die() { rl_log "STOP: $*"; exit 1; }
rl_is_done() { [ -f "$RUN/done/$1.done" ] && [ "${FORCE:-0}" != 1 ]; }
rl_mark_done() { date -u +%FT%TZ > "$RUN/done/$1.done"; rl_log "done $1"; }

rl_is_protected() {
  local p
  for p in ${PROTECTED[@]+"${PROTECTED[@]}"}; do [ "$p" = "$1" ] && return 0; done
  return 1
}

rl_is_step() {
  local s
  for s in "${STEPS[@]}"; do [ "$s" = "$1" ] && return 0; done
  return 1
}

rl_check_bin() {
  [ -n "${RL_BIN:-}" ] || return 0
  local sha=${RL_BIN_SHA:-$RL_BIN.sha256}
  [ -x "$RL_BIN" ] || rl_die "no binary at $RL_BIN"
  [ -f "$sha" ] || rl_die "no sha256 file $sha"
  [ "$(sha256sum "$RL_BIN" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$sha")" ] || rl_die "$RL_BIN changed after it was pinned"
}

rl_hold() {
  local runner=$1 name=$2 to=$3; shift 3
  [ "${1:-}" = -- ] || rl_die "usage: rl_gpu|rl_cpu <name> <timeout> -- <cmd...>"
  shift
  [ "$#" -gt 0 ] || rl_die "$name: no command"
  rl_check_bin
  rl_log "$runner $name (timeout ${to}s): $*"
}

rl_gpu() {
  local name=$1 to=$2 x0 x1 rc
  case "$to" in ''|*[!0-9]*) rl_die "rl_gpu $name: timeout must be whole seconds, got '$to'" ;; esac
  [ "$to" -le "$LOCK_HOLD_MAX" ] || rl_die "refused: rl_gpu $name timeout ${to}s is above LOCK_HOLD_MAX=${LOCK_HOLD_MAX}s"
  rl_hold gpu "$@"
  shift 3
  x0=$(xid_count)
  "$L" bash "$RL_LOCK_RUN" "$LOCKFILE" "$RL_TIME" -f "$RL_TIMEF" timeout -k "$KILL_AFTER" "$to" "$@" >> "$RUN/$name.log" 2>&1
  rc=$?
  x1=$(xid_count)
  echo "[xid] before=$x0 after=$x1 rc=$rc" >> "$RUN/$name.log"
  [ "$x1" -le "$x0" ] || rl_die "NVRM Xid count rose $x0 -> $x1 during $name"
  return "$rc"
}

rl_cpu() {
  local name=$1 to=$2
  case "$to" in ''|*[!0-9]*) rl_die "rl_cpu $name: timeout must be whole seconds, got '$to'" ;; esac
  rl_hold cpu "$@"
  shift 3
  "$B" "$RL_TIME" -f "$RL_TIMEF" timeout -k "$KILL_AFTER" "$to" "$@" >> "$RUN/$name.log" 2>&1
}

# Whole seconds, rounded up, of the last `[time] wall=` line after the first $2 such lines.
rl_hold_wall() {
  local w
  w=$(grep -a '^\[time\] wall=' "$1" | sed -n "$(($2 + 1)),\$p" | tail -n 1 | sed -E 's/^\[time\] wall=([0-9.]+) s.*/\1/')
  [ -n "$w" ] || return 1
  awk -v w="$w" 'BEGIN { printf "%d", (w == int(w)) ? w : int(w) + 1 }'
}

# rl_chunks <name> <chunk_s> <budget_s> <stop-cmd> -- <cmd...>
# The command resumes from its own checkpoint when one exists. An empty stop-cmd output ends the loop.
rl_chunks() {
  local name=$1 chunk_s=$2 budget=$3 stopcmd=$4; shift 4
  [ "${1:-}" = -- ] || rl_die "usage: rl_chunks <name> <chunk_s> <budget_s> <stop-cmd> -- <cmd...>"
  shift
  local grace=${RL_GRACE_S:-120} spent=0 n=0 reason=none this left hw n0 rc
  [ "$((chunk_s + grace))" -le "$LOCK_HOLD_MAX" ] || rl_die "refused: rl_chunks $name chunk ${chunk_s}s + grace ${grace}s is above LOCK_HOLD_MAX=${LOCK_HOLD_MAX}s"
  while [ "$spent" -lt "$budget" ]; do
    n=$((n + 1))
    left=$((budget - spent))
    this=$chunk_s
    [ "$left" -lt "$this" ] && this=$left
    n0=$(grep -ac '^\[time\] wall=' "$RUN/$name.log" 2>/dev/null || true)
    export RL_CHUNK_S=$this RL_CHUNK_N=$n
    rl_gpu "$name" "$((this + grace))" -- "$@"
    rc=$?
    [ "$rc" -eq 0 ] || rl_die "$name chunk $n failed rc=$rc (log $RUN/$name.log)"
    hw=$(rl_hold_wall "$RUN/$name.log" "${n0:-0}") || rl_die "$name chunk $n: no [time] line in $RUN/$name.log"
    spent=$((spent + hw))
    reason=$(bash -c "$stopcmd" 2>/dev/null | tr -d '[:space:]')
    rl_log "[chunk] n=$n wall=$hw stop=${reason:-none} spent=$spent budget=$budget"
    echo "[chunk] n=$n wall=$hw stop=${reason:-none}" >> "$RUN/$name.log"
    [ "$reason" = time_budget ] || break
    [ "${RL_CHUNK_GAP_S:-0}" -gt 0 ] && sleep "$RL_CHUNK_GAP_S"
  done
  return 0
}

runlib_main() {
  local s fn
  if [ -n "${PROFILE:-}" ]; then
    if declare -F "profile_$PROFILE" > /dev/null; then "profile_$PROFILE"
    elif [ "$PROFILE" != real ]; then rl_die "unknown PROFILE '$PROFILE'"
    fi
  fi
  if [ "${1:-}" = list ]; then
    for s in "${STEPS[@]}"; do
      if [ -f "$RUN/done/$s.done" ]; then printf '%s\t%s\n' "$s" "$(cat "$RUN/done/$s.done")"
      else printf '%s\tpending\n' "$s"
      fi
    done
    return 0
  fi
  local todo=() explicit=0
  if [ "$#" -gt 0 ]; then
    explicit=1
    for s in "$@"; do
      rl_is_step "$s" || rl_die "unknown step '$s'"
      if rl_is_protected "$s" && [ "${SEALED:-0}" != 1 ]; then rl_die "step '$s' is protected; run it with SEALED=1"; fi
      todo+=("$s")
    done
  else
    todo=("${STEPS[@]}")
  fi
  for s in "${todo[@]}"; do
    if [ "$explicit" = 0 ] && rl_is_protected "$s" && [ "${SEALED:-0}" != 1 ]; then
      rl_log "skip $s (protected; needs SEALED=1)"
      continue
    fi
    if rl_is_done "$s"; then rl_log "skip $s (done)"; continue; fi
    fn=step_${s//-/_}
    declare -F "$fn" > /dev/null || rl_die "no function $fn for step '$s'"
    rl_log "step $s"
    "$fn" || rl_die "step $s failed rc=$?"
    rl_mark_done "$s"
  done
}
