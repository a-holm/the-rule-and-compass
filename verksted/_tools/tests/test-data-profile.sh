#!/usr/bin/env bash
# Usage: test-data-profile.sh
# Hermetic test of data-profile.sh on synthetic fixtures: counts, refusal of names at or after dev_end, no data row
# in the output, zip member names, units, exit codes; and the key lists of the type-stage contract and adapt.md.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/machine.sh"
PROF=$HERE/../data-profile.sh
V=$HERE/../..
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
key() { sed -n "s/^$1: //p" "$2" | head -n 1; }

D=$T/data
mkdir -p "$D/sub" "$D-sealed"
printf 'time,open,high,low,close,volume\n1577836800,111.25,2,1,1.5,0\n1577840400,1.5,2.5,1,2,11\n1577844000,2,3,1.5,2.5,12\n' > "$D/SYM-1h-2023-10.csv"
printf 'time,open,high,low,close,volume\n1700000000,987.654,2,1,1.5,5\n' > "$D/SYM-1h-2024-01.csv"
printf 'time,open,high,low,close,volume\n1700000000,777.123,2,1,1.5,5\n' > "$D/SYM-1h-2023-11.csv"
printf 'ts,a\n2020-01-02 00:00:00,5\n2020-01-02 01:00:00,6\n' > "$D/sub/other-20200102.csv"
printf 'note\n' > "$D/readme.json"
printf 'sealed-marker-555.5\n' > "$D-sealed/x-2020-01.csv"
mkdir -p "$D/zipsrc"
printf 'a,b\n1,2\n' > "$D/zipsrc/A-2020-01.csv"
printf 'a,b\n3,4\n' > "$D/zipsrc/A-2024-02.csv"
(cd "$D/zipsrc" && zip -q ../Z-2020-01.zip A-2020-01.csv A-2024-02.csv) && rm -r "$D/zipsrc"
printf 'zipname\n' > "$T/late.txt"
(cd "$T" && zip -q "$D/Z-2024-03.zip" late.txt)

out=$T/profile.md
bash "$PROF" --dir "$D" --dev-end 2023-11-15 --out "$out" > "$T/stdout" 2>&1
check rc "$?" 0
check refused "$(key files_at_or_after_dev_end "$out")" 4
check files "$(key files "$out")" 4
check rows "$(key rows "$out")" 5
check tables "$(key tables "$out")" 2
check first "$(key first_ts "$out")" 2020-01-01T00:00:00Z
check last "$(key last_ts "$out")" 2020-01-02T01:00:00Z
check step "$(key median_step_seconds "$out")" 3600
check unit "$(key time_unit "$out")" mixed:s+iso
check zv "$(key zero_volume_rows "$out")" 1
check zips "$(key zip_files "$out")" 1
check members "$(key zip_members "$out")" 1
check others "$(key other_files "$out")" 1
for marker in 987.654 777.123 111.25 sealed-marker 1700000000; do
  check "no-$marker" "$(cat "$out" "$T/stdout" | grep -c -F "$marker")" 0
done
check header "$(key header "$out")" "time,open,high,low,close,volume"

# a month before dev_end is kept, the month that holds dev_end is refused
out2=$T/p2.md
bash "$PROF" --dir "$D" --dev-end 2024-02-01 --out "$out2" > /dev/null 2>&1
check late-refused "$(key files_at_or_after_dev_end "$out2")" 2

# usage and missing directory
bash "$PROF" --dir "$T/none" --dev-end 2023-11-15 --out "$T/x.md" > /dev/null 2>&1
check missing-dir "$?" 1
bash "$PROF" --dir "$D" --out "$T/x.md" > /dev/null 2>&1
check usage "$?" 2
bash "$PROF" --dir "$D" --dev-end nonsense --out "$T/x.md" > /dev/null 2>&1
check bad-date "$?" 2
check no-output-on-error "$([ -e "$T/x.md" ] && echo yes || echo no)" no


# defect 1: a headerless file with a text first field prints no field of any row
H=$T/headerless
mkdir -p "$H"
printf 'SECRETSYM,1577836800,12.3456,99\nSECRETSYM,1577840400,12.4,98\n' > "$H/x.csv"
bash "$PROF" --dir "$H" --dev-end 2023-11-15 --out "$T/h.md" > "$T/h.out" 2>&1
check headerless-rc "$?" 0
check headerless-header "$(key header "$T/h.md")" none
check headerless-columns "$(key columns "$T/h.md")" 4
check headerless-rows "$(key rows "$T/h.md")" 2
for marker in SECRETSYM 12.3456 12.4; do
  check "headerless-no-$marker" "$(cat "$T/h.md" "$T/h.out" | grep -c -F "$marker")" 0
done

# defect 2: an undated file that spans dev_end contributes only its rows before it
S=$T/span
mkdir -p "$S"
printf 'time,open,volume\n1577836800,1.5,0\n1700000000,2.5,7\n1700003600,3.5,0\n1800000000,4.5,0\n' > "$S/all.csv"
bash "$PROF" --dir "$S" --dev-end 2023-11-15 --out "$T/s.md" > "$T/s.out" 2>&1
check span-rc "$?" 0
check span-rows "$(key rows "$T/s.md")" 3
check span-after "$(key rows_after_dev_end "$T/s.md")" 1
check span-last "$(key last_ts "$T/s.md")" 2023-11-14T23:13:20Z
check span-zero-volume "$(key zero_volume_rows "$T/s.md")" 2
check span-zero-counts "$(key zero_counts "$T/s.md")" "open=0,volume=2"
for marker in 1800000000 2027 2023-11-15 4.5; do
  check "span-no-$marker" "$(grep -v '^time_unit' "$T/s.md" "$T/s.out" | grep -c -F "$marker")" 0
done

# a file whose time stamps cannot be parsed is refused without partial output
U=$T/unparsable
mkdir -p "$U"
printf 'time,open\nsoon,1\nlater,2\n' > "$U/u.csv"
bash "$PROF" --dir "$U" --dev-end 2023-11-15 --out "$T/u.md" > /dev/null 2>&1
check unparsable-rc "$?" 3
check unparsable-no-output "$([ -e "$T/u.md" ] && echo yes || echo no)" no

# the contract table and the adapt template hold the same keys
contract=$(sed -n 's/^| `\([^|]*\)` | .*/\1/p' "$V/06_data/references/type-stage-contract.md" | tr -d '`' | tr ',' '\n' | tr -d ' ' | grep -v '^Key$' | LC_ALL=C sort)
template=$(sed -n 's/^\([a-z_]*\): .*/\1/p' "$V/_templates/adapt.md" | LC_ALL=C sort)
check contract-keys "$template" "$contract"
check contract-count "$(wc -l <<< "$template")" 21

# the register holds one ready row and six planned rows with issue numbers
reg=$V/06_data/references/data-types.md
check ready "$(grep -c '| ready |' "$reg")" 1
check planned "$(grep -cE '\| planned \(#[0-9]+\) \|' "$reg")" 6

if [ "$fails" -ne 0 ]; then echo "$fails failures"; exit 1; fi
echo "ok"
