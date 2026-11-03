#!/usr/bin/env bash
# Usage: bin-pin.sh <worktree> <dest-dir> [<binary-name>...]   (default: rcsr)
# Copies <worktree>/my-version/target/release/<name> into <dest-dir>, writes <name>.sha256 (sha256sum line) and
# <name>.rev (git -C <worktree> rev-parse HEAD, with -dirty appended when `git status --porcelain` is not empty),
# then runs the copy (before it is moved into place) with --version under `env -i PATH=$PATH`: a binary that still needs its build directory fails here.
# Refuses an ELF binary whose RUNPATH or RPATH points into a build directory or whose NEEDED lists librcsrcuda.
# Refuses when <dest-dir>/<name> exists with a different sha256. Refuses the worktree named default-ab-run.
# Writes: <dest-dir>/<name>, <dest-dir>/<name>.sha256, <dest-dir>/<name>.rev; <dest-dir>/<name>.failed when the copy does not run.
# Exit: 0 pinned (or already pinned identically); 1 missing binary, copy fails to run, or a different pin exists; 2 usage.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

case "${1:-}" in -h|--help) usage; exit 0 ;; esac
if [ $# -lt 2 ]; then
  usage >&2
  exit 2
fi
wt=$1
dest=$2
shift 2
[ $# -gt 0 ] || set -- rcsr

[ -d "$wt" ] || { echo "bin-pin: no worktree $wt" >&2; exit 1; }
wt=$(cd "$wt" && pwd)
case "$(basename "$wt")" in
  default-ab-run) echo "bin-pin: refusing to pin from $wt" >&2; exit 1 ;;
esac
rel=$wt/my-version/target/release

rev=$(git -C "$wt" rev-parse HEAD) || { echo "bin-pin: $wt is not a git worktree" >&2; exit 1; }
[ -z "$(git -C "$wt" status --porcelain)" ] || rev=$rev-dirty

elf_refusal() {
  local dyn
  [ "$(head -c 4 "$1")" = $'\177ELF' ] || return 0
  dyn=$(readelf -d "$1" 2>/dev/null) || { echo "readelf failed"; return 0; }
  if grep -E 'RUNPATH|RPATH' <<<"$dyn" | grep -q '/build/'; then
    echo "RUNPATH points into a build dir"
  elif grep -q librcsrcuda <<<"$dyn"; then
    echo "NEEDED librcsrcuda"
  fi
}

rc=0
for name in "$@"; do
  src=$rel/$name
  if [ ! -x "$src" ]; then
    echo "bin-pin: missing or not executable: $src" >&2
    rc=1
    continue
  fi
  why=$(elf_refusal "$src")
  if [ -n "$why" ]; then
    echo "bin-pin: refusing $src: $why" >&2
    rc=1
    continue
  fi
  sum=$(sha256sum "$src" | awk '{print $1}')
  if [ -e "$dest/$name" ]; then
    have=$(sha256sum "$dest/$name" | awk '{print $1}')
    if [ "$have" != "$sum" ]; then
      echo "bin-pin: $dest/$name exists with a different sha256" >&2
      rc=1
      continue
    fi
  fi
  mkdir -p "$dest"
  tmp=$dest/.$name.$$
  cp "$src" "$tmp" && chmod +x "$tmp" || { echo "bin-pin: copy of $name failed" >&2; rc=1; continue; }
  if ! env -i PATH="$PATH" "$tmp" --version > /dev/null 2>&1; then
    mv -f "$tmp" "$dest/$name.failed"
    echo "bin-pin: the copy of $name fails to run with --version under env -i; kept as $dest/$name.failed" >&2
    rc=1
    continue
  fi
  mv -f "$tmp" "$dest/$name"
  printf '%s  %s\n' "$sum" "$name" > "$dest/$name.sha256"
  printf '%s\n' "$rev" > "$dest/$name.rev"
  echo "pinned $dest/$name sha256=$sum rev=$rev"
done
exit "$rc"
