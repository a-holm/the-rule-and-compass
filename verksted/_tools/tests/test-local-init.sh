#!/usr/bin/env bash
# Usage: test-local-init.sh
# Hermetic test of local-init.sh: layout, idempotence, kept edits, dataset option, argument errors.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INIT=$HERE/../local-init.sh
TPL=$HERE/../../_templates/local
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  local name=$1 got=$2 want=$3
  if [ "$got" != "$want" ]; then
    echo "FAIL $name: got '$got' want '$want'"
    fails=$((fails + 1))
  fi
}
init() { RCSR_LOCAL=$L bash "$INIT" "$@" 2>&1; echo "rc=$?"; }
tree() { (cd "$L" && find . | LC_ALL=C sort); }

# fresh directory (missing, with a space in the path)
L="$T/new area/local"
out=$(init)
check fresh-rc "$(tail -n 1 <<< "$out")" "rc=0"
check fresh-count "$(tree | wc -l)" "9"
want=$(printf '.\n./README.md\n./datasets\n./decisions.md\n./lessons\n./machine.md\n./measurements\n./rules-local.md\n./runs')
check fresh-tree "$(tree)" "$want"
for f in README decisions machine rules-local; do
  check "copy-$f" "$(cmp -s "$TPL/$f.md" "$L/$f.md" && echo same || echo differ)" "same"
done

# second run changes nothing; an edited file survives
echo mine >> "$L/decisions.md"
before=$(cd "$L" && find . -type f -exec cksum {} + | sort)
out=$(init)
check again-rc "$(tail -n 1 <<< "$out")" "rc=0"
check again-unchanged "$(cd "$L" && find . -type f -exec cksum {} + | sort)" "$before"
check again-edit "$(tail -n 1 "$L/decisions.md")" "mine"
check again-kept "$(grep -c '^kept ' <<< "$out")" "4"
check again-created "$(grep -c '^created ' <<< "$out")" "0"

# dataset option
out=$(init --dataset d1)
check ds-rc "$(tail -n 1 <<< "$out")" "rc=0"
check ds-dir "$([ -d "$L/datasets/d1" ] && echo yes || echo no)" "yes"
check ds-empty "$(find "$L/datasets/d1" -type f | wc -l)" "0"
echo card > "$T/c.txt"; echo adapt > "$T/a.txt"; echo man > "$T/m.txt"
out=$(init --dataset d2 --card "$T/c.txt" --adapt "$T/a.txt" --manifest "$T/m.txt")
check ds2-rc "$(tail -n 1 <<< "$out")" "rc=0"
check ds2-card "$(cat "$L/datasets/d2/card.md")" "card"
check ds2-adapt "$(cat "$L/datasets/d2/adapt.md")" "adapt"
check ds2-manifest "$(cat "$L/datasets/d2/manifest.md")" "man"
echo other > "$T/c2.txt"
out=$(init --dataset d2 --card "$T/c2.txt")
check ds2-keep "$(cat "$L/datasets/d2/card.md")" "card"
check ds2-keep-rc "$(tail -n 1 <<< "$out")" "rc=0"

# argument errors
check err-card-alone "$(init --card "$T/c.txt" | tail -n 1)" "rc=2"
check err-unknown "$(init --nope | tail -n 1)" "rc=2"
check err-missing-value "$(init --dataset | tail -n 1)" "rc=2"
check err-bad-id "$(init --dataset ../x | tail -n 1)" "rc=2"
check err-missing-file "$(init --dataset d3 --card "$T/none" | tail -n 1)" "rc=1"

# no-overwrite of a pre-existing local file in a fresh tree
L="$T/pre"
mkdir -p "$L"
echo keep > "$L/machine.md"
init > /dev/null
check pre-machine "$(cat "$L/machine.md")" "keep"
check pre-others "$([ -f "$L/rules-local.md" ] && echo yes || echo no)" "yes"

# machine.md: every profile key of _shared/local.md except TOOLS_WT starts as <set me>, and lib.sh refuses it
L="$T/keys"
init > /dev/null
keys=$(sed -n 's/^| `\([A-Z_]*\)` | .*/\1/p' "$HERE/../../_shared/local.md" | grep -vx TOOLS_WT | LC_ALL=C sort)
have=$(sed -n 's/^- \([A-Z_]*\): <set me>$/\1/p' "$L/machine.md" | LC_ALL=C sort)
check keys-in-template "$have" "$keys"
check keys-count "$(wc -l <<< "$have")" "17"
out=$(RCSR_LOCAL=$L bash -c "source '$HERE/../lib.sh'; echo reached" 2>&1); rc=$?
check seed-rc "$rc" "2"
check seed-names-first-key "$(grep -c 'key GATES missing' <<< "$out")" "1"
sed -i 's/^- \(GATES\|MACHINE_SCRIPTS\|SANITIZER\|CUDA_ARCH\): <set me>$/- \1: x/' "$L/machine.md"
check seeded-lib-rc "$(RCSR_LOCAL=$L bash -c "source '$HERE/../lib.sh'; echo \$CUDA_ARCH" 2>&1)" "x"

if [ "$fails" -ne 0 ]; then
  echo "test-local-init: $fails failed"
  exit 1
fi
echo "test-local-init: ok"
