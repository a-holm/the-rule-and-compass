#!/usr/bin/env bash
# Usage: watch.sh [-m MINUTES] [-i SECONDS] [-1] <worktree>...
# Alerts when a worktree has had no file change for MINUTES (default 45) and no cargo, flock,
# one-full-suite, rustc, nvcc or rcsr process runs under it. Prints `SILENT <n>min: <path>`
# and exits 0 on the first alert. -1 does one pass and exits 0 (alert or not).
# The build directory and .git are ignored: transcripts buffer and do not show liveness.
set -u
minutes=45
interval=300
once=0
while getopts "m:i:1" opt; do
  case $opt in
    m) minutes=$OPTARG ;;
    i) interval=$OPTARG ;;
    1) once=1 ;;
    *) echo "usage: watch.sh [-m MINUTES] [-i SECONDS] [-1] <worktree>..." >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ $# -ge 1 ] || { echo "usage: watch.sh [-m MINUTES] [-i SECONDS] [-1] <worktree>..." >&2; exit 2; }
busy_under() {
  local w=$1 p cwd args
  for p in /proc/[0-9]*; do
    cwd=$(readlink "$p/cwd" 2>/dev/null) || continue
    args=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$cwd $args" in
      *"$w"*)
        case "$args" in
          *cargo*|*flock*|*one-full-suite*|*rustc*|*nvcc*|*rcsr*) return 0 ;;
        esac ;;
    esac
  done
  return 1
}
while true; do
  now=$(date +%s)
  for w in "$@"; do
    [ -d "$w" ] || continue
    newest=$(find "$w" -path "$w/my-version/target" -prune -o -name .git -prune -o -type f -printf '%T@\n' 2>/dev/null | sort -n | tail -n 1 | cut -d. -f1)
    age=$(( (now - ${newest:-0}) / 60 ))
    if [ "$age" -ge "$minutes" ] && ! busy_under "$w"; then
      echo "SILENT ${age}min: $w"
      exit 0
    fi
  done
  [ "$once" -eq 1 ] && exit 0
  sleep "$interval"
done
