#!/usr/bin/env bash
# Sourced by verksted/_tools/tests/test-*.sh; not run.
# Writes: fake commands into the directory each fake_* function gets; the clock files under $VCLOCK.
# Stand-ins for real timers, so a script test counts events instead of reading the wall clock.
# A script test never reaches the real machine directory: a test that needs one sets RCSR_MACHINE_DIR itself.
export RCSR_MACHINE_DIR=/nonexistent-machine-dir

# fake_timeout <dir>: <dir>/timeout appends "<limit> <cmd...>" to $FAKE_TIMEOUT_LOG and the -k value, when given, to
# $FAKE_TIMEOUT_LOG.kill. A command named hang stands for one that never ends: exit 124 without running it. Any other
# command runs in place of the real timeout.
fake_timeout() {
  cat > "$1/timeout" <<'FAKE'
#!/usr/bin/env bash
if [ "$1" = -k ]; then
  echo "$2" >> "${FAKE_TIMEOUT_LOG:?}.kill"
  shift 2
fi
echo "$*" >> "${FAKE_TIMEOUT_LOG:?}"
shift
[ "$(basename "$1")" = hang ] && exit 124
exec "$@"
FAKE
  chmod +x "$1/timeout"
}

# fake_clock <dir>: <dir>/sleep and <dir>/date share a virtual clock in the directory $VCLOCK. sleep adds its
# argument, rounded up and at least 1, to $VCLOCK/now, counts the call in $VCLOCK/polls, then runs once every line
# "<t> <command>" of $VCLOCK/events whose t has been reached. date +%s prints now; any other date call is the real one.
fake_clock() {
  cat > "$1/sleep" <<'FAKE'
#!/usr/bin/env bash
d=${VCLOCK:?}
s=${1%%.*}
[ "$1" = "$s" ] || s=$((${s:-0} + 1))
[ "${s:-0}" -ge 1 ] || s=1
now=$(( $(cat "$d/now") + s ))
echo "$now" > "$d/now"
echo $(( $(cat "$d/polls") + 1 )) > "$d/polls"
i=0
while IFS= read -r line; do
  i=$((i + 1))
  [ "${line%% *}" -le "$now" ] || continue
  grep -qx "$i" "$d/fired" && continue
  echo "$i" >> "$d/fired"
  bash -c "${line#* }"
done < "$d/events"
FAKE
  printf '#!/usr/bin/env bash\n[ "${1:-}" = +%%s ] && exec cat "${VCLOCK:?}/now"\nexec %q "$@"\n' "$(command -v date)" > "$1/date"
  chmod +x "$1/sleep" "$1/date"
}
vclock_reset() { mkdir -p "$VCLOCK"; echo 0 > "$VCLOCK/now"; echo 0 > "$VCLOCK/polls"; : > "$VCLOCK/events"; : > "$VCLOCK/fired"; }
vclock_at() { printf '%s %s\n' "$1" "$2" >> "$VCLOCK/events"; }
polls() { cat "$VCLOCK/polls"; }

# fake_slot_sleep <dir> <count-file>: <dir>/sleep counts its calls in <count-file>. The first call releases the lock the
# test holds on fd 9 (flock -u acts on the shared open file description). Past 20 calls it fails, which ends
# build-slot.sh under its set -e: a counted guard against a slot that never frees.
fake_slot_sleep() {
  cat > "$1/sleep" <<FAKE
#!/usr/bin/env bash
n=\$(( \$(cat "$2" 2>/dev/null || echo 0) + 1 ))
echo "\$n" > "$2"
[ "\$n" -le 20 ] || exit 1
[ "\$n" -ne 1 ] || flock -u 9
FAKE
  chmod +x "$1/sleep"
}

# hold <commands>: runs <commands> in a process substitution, then prints ready and execs the blocker there, so the
# holder keeps its cwd, its locks and its pid. Sets HOLD_PID and HOLD_FD. Returns 0 once the holder printed ready,
# 1 when it ended first (read sees end of file). No deadline either way.
hold() {
  exec {HOLD_FD}< <(eval "$1" && echo ready && exec sleep infinity)
  HOLD_PID=$!
  local line=
  read -r line <&"$HOLD_FD"
  [ "$line" = ready ]
}
release() {
  kill "$HOLD_PID" 2> /dev/null
  wait "$HOLD_PID" 2> /dev/null
  exec {HOLD_FD}<&-
  HOLD_PID=
}
