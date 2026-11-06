#!/usr/bin/env bash
# Usage: wait-for.sh --log <file> [--pattern <ERE>] [--timeout S]
#        wait-for.sh --file <path> [--timeout S]
#        wait-for.sh --count <N> --dir <dir> [--glob <pattern>] [--timeout S]
#        wait-for.sh --pid <N> [--timeout S]
#        wait-for.sh --exe <absolute path> [--timeout S]
# One blocking wait inside this process, polling every 5 s (RCSR_POLL for tests): until the log has a line
# matching the pattern (default '^RC_DONE rc='), the file exists and is not empty, or the directory holds at
# least N entries matching the glob (default '*'), process <N> has ended (gone or a zombie), or no process runs the executable
# <path> (its /proc/<pid>/exe link equals the path exactly). Never reads a process command line, so a wait cannot match itself.
# With --log, on a match the matching line is printed, then test-summary.sh output when the log holds cargo test output.
# Default timeout 590 s, under the 600 s limit of one blocking tool call; use a larger one in a background shell.
# Writes: nothing.
# Exit: 0 matched; 124 timeout; 2 usage.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

die() {
  echo "wait-for.sh: $1" >&2
  usage >&2
  exit 2
}

log='' pattern='^RC_DONE rc=' file='' count='' dir='' glob='*' pid='' exe='' timeout=590
while [ $# -gt 0 ]; do
  case $1 in
    -h | --help) usage; exit 0 ;;
    --log | --pattern | --file | --count | --dir | --glob | --pid | --exe | --timeout)
      [ $# -ge 2 ] || die "$1 needs a value"
      case $1 in
        --log) log=$2 ;;
        --pattern) pattern=$2 ;;
        --file) file=$2 ;;
        --count) count=$2 ;;
        --dir) dir=$2 ;;
        --glob) glob=$2 ;;
        --pid) pid=$2 ;;
        --exe) exe=$2 ;;
        --timeout) timeout=$2 ;;
      esac
      shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

modes=0
[ -n "$log" ] && modes=$((modes + 1))
[ -n "$file" ] && modes=$((modes + 1))
[ -n "$count" ] && modes=$((modes + 1))
[ -n "$pid" ] && modes=$((modes + 1))
[ -n "$exe" ] && modes=$((modes + 1))
[ "$modes" -eq 1 ] || die "give exactly one of --log, --file, --count, --pid, --exe"
case $timeout in '' | *[!0-9]*) die "--timeout needs whole seconds" ;; esac
if [ -n "$count" ]; then
  case $count in '' | *[!0-9]*) die "--count needs a whole number" ;; esac
  [ -n "$dir" ] || die "--count needs --dir"
fi
if [ -n "$pid" ]; then
  case $pid in '' | *[!0-9]*) die "--pid needs a process id" ;; esac
  [ "$pid" -gt 0 ] || die "--pid needs a process id"
fi
if [ -n "$exe" ]; then
  case $exe in /*) ;; *) die "--exe needs an absolute path" ;; esac
  exe_glob=$(printf '%s' "$exe" | sed 's/[][*?\\]/\\&/g')
fi
POLL=${RCSR_POLL:-5}
case $POLL in '' | *[!0-9.]*) die "RCSR_POLL needs seconds" ;; esac

pid_ended() {
  local line rest
  { read -r line < "/proc/$pid/stat"; } 2> /dev/null || return 0
  rest=${line##*) }
  case ${rest%% *} in Z | X) return 0 ;; esac
  return 1
}

exe_ended() {
  local p self=$$
  while IFS= read -r p; do
    p=${p#/proc/}
    [ "$p" = "$self" ] || [ "$p" = "$BASHPID" ] || return 1
  done < <(find /proc/[0-9]*/exe -maxdepth 0 -lname "$exe_glob" -printf '%h\n' 2> /dev/null)
  return 0
}

matched() {
  if [ -n "$log" ]; then
    [ -f "$log" ] && grep -Eq -- "$pattern" "$log"
  elif [ -n "$file" ]; then
    [ -s "$file" ]
  elif [ -n "$pid" ]; then
    pid_ended
  elif [ -n "$exe" ]; then
    exe_ended
  else
    [ -d "$dir" ] && [ "$(find "$dir" -mindepth 1 -maxdepth 1 -name "$glob" 2>/dev/null | wc -l)" -ge "$count" ]
  fi
}

deadline=$(( $(date +%s) + timeout ))
while :; do
  matched && break
  [ "$(date +%s)" -ge "$deadline" ] && exit 124
  sleep "$POLL"
done

if [ -n "$log" ]; then
  grep -E -m 1 -- "$pattern" "$log"
  if grep -Eq '^test result:|^running [0-9]+ tests?$' "$log"; then
    "$(dirname "${BASH_SOURCE[0]}")/test-summary.sh" "$log" || true
  fi
fi
exit 0
