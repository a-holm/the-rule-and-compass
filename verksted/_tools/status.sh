#!/usr/bin/env bash
# Usage: status.sh [--no-fetch] [--section all|git|agents|machine]
# One read-only report for the orchestrator. Sections and their lines:
#   git:     clock (UTC and local); origin/$BASE tip (short sha, subject); open PRs: number, branch, head, updatedAt
#   agents:  one line per worktree under $WT: branch, dirty y/n, minutes since the newest file change outside
#            my-version/target and .git, SILENT when 45 or more and no cargo, rustc, nvcc, rcsr or flock process
#            has its cwd inside
#   machine: test-lock holder (pid and command, from /proc/*/fd links to $LOCKFILE), held build slots, PAUSE and
#            STREAMING present y/n, desktop-busy exit code, GPU utilisation and memory, Xid count, load average,
#            MemAvailable, and cargo, rustc, nvcc, rcsr and compute-sanitizer processes (ps -eo + awk, own pid excluded)
# Writes: nothing (git fetch -q unless --no-fetch).
# Exit: 0; 2 usage.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

fetch=1
section=all
while [ $# -gt 0 ]; do
  case $1 in
    --no-fetch) fetch=0 ;;
    --section)
      [ $# -ge 2 ] || { usage >&2; exit 2; }
      section=$2
      shift
      ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done
case $section in all | git | agents | machine) ;; *) usage >&2; exit 2 ;; esac
want() { [ "$section" = all ] || [ "$section" = "$1" ]; }

# pid and command of every process holding a file whose link target matches the glob.
fd_holders() {
  local pid
  for pid in $(find /proc/[0-9]*/fd -maxdepth 1 -lname "$1" -printf '%h\n' 2>/dev/null | cut -d/ -f3 | sort -un); do
    echo "$pid $(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-120)"
  done
}

busy_under() {
  local w=$1 p cwd comm
  for p in /proc/[0-9]*; do
    cwd=$(readlink "$p/cwd" 2>/dev/null) || continue
    case "$cwd/" in "$w"/*) ;; *) continue ;; esac
    comm=$(cat "$p/comm" 2>/dev/null) || continue
    case $comm in cargo | rustc | nvcc | rcsr | flock) return 0 ;; esac
  done
  return 1
}

if want git; then
  echo "== git"
  echo "clock: $(date -u +%Y-%m-%dT%H:%M:%SZ) UTC, $(date +%Y-%m-%dT%H:%M:%S%z) local"
  if [ "$fetch" -eq 1 ]; then git -C "$RCSR_ROOT" fetch -q origin 2>/dev/null || echo "fetch: failed"; fi
  tip=$(git -C "$RCSR_ROOT" log -1 --format='%h %s' "origin/$BASE" 2>/dev/null) || tip=unavailable
  echo "origin/$BASE: $tip"
  if command -v gh > /dev/null 2>&1; then
    prs=$(cd "$RCSR_ROOT" && gh pr list --base "$BASE" --state open --json number,headRefName,headRefOid,updatedAt \
      --jq '.[] | "#\(.number) \(.headRefName) \(.headRefOid[0:7]) \(.updatedAt)"' 2>/dev/null) || prs=unavailable
    echo "open PRs:"
    if [ -n "$prs" ]; then echo "$prs" | sed 's/^/  /'; else echo "  none"; fi
  else
    echo "open PRs: unavailable"
  fi
fi

if want agents; then
  echo "== agents"
  now=$(date +%s)
  n=0
  while IFS= read -r w; do
    [ -n "$w" ] || continue
    case $w in "$WT"/*) ;; *) continue ;; esac
    n=$((n + 1))
    if [ ! -d "$w" ]; then
      echo "$(basename "$w"): missing"
      continue
    fi
    branch=$(git -C "$w" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
    if [ -n "$(git -C "$w" status --porcelain 2>/dev/null | head -n 1)" ]; then dirty=y; else dirty=n; fi
    newest=$(find "$w" -path "$w/my-version/target" -prune -o -name .git -prune -o -type f -printf '%T@\n' 2>/dev/null |
      sort -n | tail -n 1 | cut -d. -f1)
    age=$(((now - ${newest:-0}) / 60))
    flag=""
    if [ "$age" -ge 45 ] && ! busy_under "$w"; then flag=" SILENT"; fi
    echo "$(basename "$w"): branch=$branch dirty=$dirty idle=${age}min$flag"
  done < <(git -C "$RCSR_ROOT" worktree list --porcelain 2>/dev/null | sed -n 's/^worktree //p')
  [ "$n" -gt 0 ] || echo "no worktrees under $WT"
fi

if want machine; then
  echo "== machine"
  h=$(fd_holders "$LOCKFILE")
  if [ -n "$h" ]; then echo "$h" | sed 's/^/lock: held by pid /'; else echo "lock: free"; fi
  s=$(fd_holders "$SLOTDIR/slot*.lock")
  if [ -n "$s" ]; then echo "$s" | sed 's/^/slot: held by pid /'; else echo "slot: none held"; fi
  yn() { if [ -e "$1" ]; then echo y; else echo n; fi; }
  echo "flags: PAUSE=$(yn "$PAUSE") STREAMING=$(yn "$STREAMING")"
  if [ -x "$OC/desktop-busy.sh" ]; then
    "$OC/desktop-busy.sh" > /dev/null 2>&1
    echo "desktop-busy: exit $?"
  else
    echo "desktop-busy: unavailable"
  fi
  if command -v nvidia-smi > /dev/null 2>&1; then
    echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo unavailable)"
  else
    echo "gpu: unavailable"
  fi
  echo "xid: $(xid_count)"
  echo "load: $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo unavailable)"
  echo "mem available: $(awk '/^MemAvailable:/{print $2 " kB"}' /proc/meminfo 2>/dev/null)"
  echo "processes:"
  ps -eo pid,etimes,pcpu,args 2>/dev/null | awk -v me="$$" -v parent="$PPID" '
    NR > 1 && $1 != me && $1 != parent {
      n = split($4, a, "/")
      if (a[n] ~ /^(cargo|rustc|nvcc|rcsr|compute-sanitizer)$/) { print "  " substr($0, 1, 150); found = 1 }
    }
    END { if (!found) print "  none" }'
fi
exit 0
