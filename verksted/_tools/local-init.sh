#!/usr/bin/env bash
# Usage: local-init.sh [--dataset <id> [--card <file>] [--adapt <file>] [--manifest <file>]]
# Creates the local area $VLOCAL from the starters of _templates/local/. Never overwrites a file.
# --dataset creates datasets/<id>/ and copies each given file there as card.md, adapt.md, manifest.md.
# machine.md starts from the template with `<set me>` for every key; the tools exit 2 naming a key until it is set.
# Writes: $VLOCAL (RCSR_LOCAL overrides it).
set -uo pipefail
LIB_NO_MACHINE=1
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

dataset=
declare -A given=()
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --dataset|--card|--adapt|--manifest)
      [ $# -ge 2 ] || { usage >&2; exit 2; }
      if [ "$1" = --dataset ]; then dataset=$2; else given[${1#--}]=$2; fi
      shift 2 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if [ -z "$dataset" ] && [ ${#given[@]} -gt 0 ]; then
  echo "--card, --adapt and --manifest need --dataset" >&2
  exit 2
fi
case $dataset in
  */*|.|..) echo "invalid dataset id: $dataset" >&2; exit 2 ;;
esac

tpl=$VERKSTED/_templates/local

put() {
  local src=$1 dst=$2
  if [ -e "$dst" ]; then
    echo "kept $dst"
  else
    cp "$src" "$dst" || exit 1
    echo "created $dst"
  fi
}

mkdir -p "$VLOCAL"/{lessons,measurements,runs,datasets} || exit 1
for f in README decisions machine rules-local; do
  put "$tpl/$f.md" "$VLOCAL/$f.md"
done

if [ -n "$dataset" ]; then
  mkdir -p "$VLOCAL/datasets/$dataset" || exit 1
  for k in card adapt manifest; do
    [ -n "${given[$k]:-}" ] || continue
    [ -f "${given[$k]}" ] || { echo "no such file: ${given[$k]}" >&2; exit 1; }
    put "${given[$k]}" "$VLOCAL/datasets/$dataset/$k.md"
  done
fi
