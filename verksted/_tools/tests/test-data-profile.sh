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

# CRLF input: no carriage return in header or raw_columns, and a numeric last column (volume) is counted for zeros
C=$T/crlf
mkdir -p "$C"
printf 'time,open,high,low,close,volume\r\n1577836800,1.5,2,1,1.5,0\r\n1577840400,1.5,2.5,1,2,11\r\n1577844000,2,3,1.5,2.5,0\r\n' > "$C/x.csv"
bash "$PROF" --dir "$C" --dev-end 2023-11-15 --out "$T/c.md" > "$T/c.out" 2>&1
check crlf-rc "$?" 0
check crlf-header "$(key header "$T/c.md")" "time,open,high,low,close,volume"
check crlf-raw "$(key raw_columns "$T/c.md")" "open,high,low,close,volume"
check crlf-zero-volume "$(key zero_volume_rows "$T/c.md")" 2
check crlf-zero-counts "$(key zero_counts "$T/c.md")" "open=0,high=0,low=0,close=0,volume=2"
check crlf-no-cr "$(grep -c $'\r' "$T/c.md")" 0
check crlf-rows "$(key rows "$T/c.md")" 3
for marker in 1577836800 2.5; do
  check "crlf-no-$marker" "$(grep -v '^time_unit' "$T/c.md" "$T/c.out" | grep -c -F "$marker")" 0
done

# CRLF with a span across dev_end keeps the leak guard: no row at or after dev_end is counted or printed
printf 'time,open,volume\r\n1577836800,1.5,0\r\n1800000000,4.5,0\r\n' > "$C/y.csv"
rm "$C/x.csv"
bash "$PROF" --dir "$C" --dev-end 2023-11-15 --out "$T/c2.md" > "$T/c2.out" 2>&1
check crlf-span-rows "$(key rows "$T/c2.md")" 1
check crlf-span-after "$(key rows_after_dev_end "$T/c2.md")" 1
check crlf-span-zero-volume "$(key zero_volume_rows "$T/c2.md")" 1
check crlf-span-last "$(key last_ts "$T/c2.md")" 2020-01-01T00:00:00Z
check crlf-span-no-late "$(grep -v '^time_unit' "$T/c2.md" "$T/c2.out" | grep -c -F 1800000000)" 0

# a file whose time stamps cannot be parsed is refused without partial output
U=$T/unparsable
mkdir -p "$U"
printf 'time,open\nsoon,1\nlater,2\n' > "$U/u.csv"
bash "$PROF" --dir "$U" --dev-end 2023-11-15 --out "$T/u.md" > /dev/null 2>&1
check unparsable-rc "$?" 3
check unparsable-no-output "$([ -e "$T/u.md" ] && echo yes || echo no)" no

# type_stage_suggestion reads the header or the column count only
sugg() {
  local d=$T/sg-$1
  mkdir -p "$d"
  printf '%b' "$2" > "$d/x.csv"
  bash "$PROF" --dir "$d" --dev-end 2023-11-15 --out "$d.md" > "$d.out" 2>&1
  check "sugg-$1-rc" "$?" 0
  check "sugg-$1" "$(key type_stage_suggestion "$d.md")" "$3"
  for marker in 4242.5 7777.5 1577836800 SUGSYM; do
    check "sugg-$1-no-$marker" "$(cat "$d.md" "$d.out" | grep -c -F "$marker")" 0
  done
}
sugg ohlc 'time,open,high,low,close\n1577836800,4242.5,2,1,1.5\n1577840400,7777.5,3,1,2\n' 04_ohlcv
sugg ohlcv 'Time,Open,High,Low,Close,Volume\n1577836800,4242.5,2,1,1.5,3\n1577840400,7777.5,3,1,2,4\n' 04_ohlcv
sugg alias 'ts,o,h,l,c\n1577836800,4242.5,2,1,1.5\n1577840400,7777.5,3,1,2\n' 04_ohlcv
sugg no-ohlc 'time,price,amount\n1577836800,4242.5,2\n1577840400,7777.5,3\n' 04_new_type
sugg partial 'time,open,close,volume\n1577836800,4242.5,2,1\n1577840400,7777.5,3,1\n' 04_new_type
sugg headerless12 'SUGSYM,1577836800,4242.5,2,1,1.5,3,1577840399,4,5,6,7\nSUGSYM,1577840400,7777.5,3,1,2,4,1577843999,4,5,6,7\n' 04_ohlcv
sugg headerless7 '1577836800,4242.5,2,1,1.5,3,9\n1577840400,7777.5,3,1,2,4,9\n' 04_ohlcv
sugg headerless4 'SUGSYM,1577836800,4242.5,99\nSUGSYM,1577840400,7777.5,98\n' 04_new_type

# lock boundary: a time stamp with an offset or Z is converted to UTC before it meets dev_end
HDR='time,open,high,low,close,volume'
O=$T/offset
mkdir -p "$O"
printf '%s\n' "$HDR" '2024-06-30T10:00:00-05:00,4242.5,2,1,1.5,3' '2024-07-01T01:00:00+02:00,4243.5,2,1,1.5,3' '2024-06-30T22:00:00-05:00,7777.5,2,1,1.5,3' '2024-07-01T00:00:00Z,7778.5,2,1,1.5,3' > "$O/x.csv"
bash "$PROF" --dir "$O" --dev-end 2024-07-01 --out "$T/o.md" > "$T/o.out" 2>&1
check offset-rc "$?" 0
check offset-rows "$(key rows "$T/o.md")" 2
check offset-after "$(key rows_after_dev_end "$T/o.md")" 2
check offset-first "$(key first_ts "$T/o.md")" 2024-06-30T15:00:00Z
check offset-last "$(key last_ts "$T/o.md")" 2024-06-30T23:00:00Z
check offset-step "$(key median_step_seconds "$T/o.md")" 28800
for marker in 4242.5 4243.5 7777.5 7778.5 22:00:00 2024-07-01T00; do
  check "offset-no-$marker" "$(cat "$T/o.md" "$T/o.out" | grep -c -F "$marker")" 0
done

# CRLF with offsets
printf '%s\r\n' "$HDR" '2024-06-30T10:00:00-05:00,4242.5,2,1,1.5,3' '2024-06-30T22:00:00-05:00,7777.5,2,1,1.5,3' > "$O/x.csv"
bash "$PROF" --dir "$O" --dev-end 2024-07-01 --out "$T/oc.md" > "$T/oc.out" 2>&1
check crlf-offset-rc "$?" 0
check crlf-offset-rows "$(key rows "$T/oc.md")" 1
check crlf-offset-after "$(key rows_after_dev_end "$T/oc.md")" 1
check crlf-offset-last "$(key last_ts "$T/oc.md")" 2024-06-30T15:00:00Z
check crlf-offset-header "$(key header "$T/oc.md")" "$HDR"

# headerless with offsets
printf '%s\n' '2024-06-30T10:00:00-05:00,4242.5,2,1,1.5,3' '2024-06-30T22:00:00-05:00,7777.5,2,1,1.5,3' > "$O/x.csv"
bash "$PROF" --dir "$O" --dev-end 2024-07-01 --out "$T/oh.md" > "$T/oh.out" 2>&1
check headerless-offset-rc "$?" 0
check headerless-offset-header "$(key header "$T/oh.md")" none
check headerless-offset-rows "$(key rows "$T/oh.md")" 1
check headerless-offset-after "$(key rows_after_dev_end "$T/oh.md")" 1
check headerless-offset-last "$(key last_ts "$T/oh.md")" 2024-06-30T15:00:00Z
check headerless-offset-type "$(key type_stage_suggestion "$T/oh.md")" 04_ohlcv
for marker in 4242.5 7777.5; do
  check "headerless-offset-no-$marker" "$(cat "$T/oh.md" "$T/oh.out" | grep -c -F "$marker")" 0
done

# a zone-less time of day needs --tz
printf '%s\n' "$HDR" '2024-07-01 01:00:00,4242.5,2,1,1.5,3' > "$O/x.csv"
rm -f "$T/z.md"
bash "$PROF" --dir "$O" --dev-end 2024-07-01 --out "$T/z.md" > /dev/null 2>&1
check zoneless-rc "$?" 3
check zoneless-no-output "$([ -e "$T/z.md" ] && echo yes || echo no)" no
bash "$PROF" --dir "$O" --dev-end 2024-07-01 --tz +02:00 --out "$T/z.md" > /dev/null 2>&1
check tz-plus-rc "$?" 0
check tz-plus-rows "$(key rows "$T/z.md")" 1
bash "$PROF" --dir "$O" --dev-end 2024-07-01 --tz UTC --out "$T/z2.md" > /dev/null 2>&1
check tz-utc-after "$(key rows_after_dev_end "$T/z2.md")" 1
bash "$PROF" --dir "$O" --dev-end 2024-07-01 --tz nonsense --out "$T/z3.md" > /dev/null 2>&1
check tz-bad-rc "$?" 2

# dev_end with an offset is converted too
printf '%s\n' "$HDR" '2024-06-30T23:30:00Z,4242.5,2,1,1.5,3' '2024-07-01T00:30:00Z,7777.5,2,1,1.5,3' > "$O/x.csv"
bash "$PROF" --dir "$O" --dev-end 2024-07-01T02:00:00+02:00 --out "$T/de.md" > /dev/null 2>&1
check dev-end-offset-rows "$(key rows "$T/de.md")" 1
check dev-end-offset-after "$(key rows_after_dev_end "$T/de.md")" 1

# zip-only directory, the documented main layout: monthly zips, streamed, nothing extracted
Z=$T/ziponly
mkdir -p "$Z" "$T/zsrc"
printf '%s\r\n' "$HDR" '1577836800,4242.5,2,1,1.5,0' '1577840400,4243.5,2,1,1.5,3' > "$T/zsrc/SYM-2020-01.csv"
printf '%s\n' "$HDR" '1706745600,7777.5,2,1,1.5,3' > "$T/zsrc/SYM-2024-02.csv"
printf '%s\n' "$HDR" '1577844000,4244.5,2,1,1.5,3' '1800000000,7778.5,2,1,1.5,3' > "$T/zsrc/undated.csv"
(cd "$T/zsrc" && zip -q "$Z/SYM-2020-01.zip" SYM-2020-01.csv SYM-2024-02.csv undated.csv)
before=$(cd "$Z" && ls -laR)
bash "$PROF" --dir "$Z" --dev-end 2023-11-15 --out "$T/zo.md" > "$T/zo.out" 2>&1
check zip-only-rc "$?" 0
check zip-only-tables "$(key tables "$T/zo.md")" 2
check zip-only-type "$(key type_stage_suggestion "$T/zo.md")" 04_ohlcv
check zip-only-rows "$(key rows "$T/zo.md")" 3
check zip-only-after "$(key rows_after_dev_end "$T/zo.md")" 1
check zip-only-refused "$(key files_at_or_after_dev_end "$T/zo.md")" 1
check zip-only-members "$(key zip_members "$T/zo.md")" 2
check zip-only-last "$(key last_ts "$T/zo.md")" 2020-01-01T02:00:00Z
check zip-only-header "$(key header "$T/zo.md")" "$HDR"
check zip-only-zv "$(key zero_volume_rows "$T/zo.md")" 1
after=$(cd "$Z" && ls -laR)
check zip-only-no-extract "$after" "$before"
for marker in 4242.5 4243.5 4244.5 7777.5 7778.5 1800000000 1706745600; do
  check "zip-only-no-$marker" "$(cat "$T/zo.md" "$T/zo.out" | grep -c -F "$marker")" 0
done

# gzip table with CRLF and offsets goes through the same filter
G=$T/gz
mkdir -p "$G"
printf '%s\r\n' "$HDR" '2024-06-30T10:00:00-05:00,4242.5,2,1,1.5,3' '2024-06-30T22:00:00-05:00,7777.5,2,1,1.5,3' | gzip -c > "$G/SYM.csv.gz"
bash "$PROF" --dir "$G" --dev-end 2024-07-01 --out "$T/g.md" > "$T/g.out" 2>&1
check gzip-rc "$?" 0
check gzip-tables "$(key tables "$T/g.md")" 1
check gzip-rows "$(key rows "$T/g.md")" 1
check gzip-after "$(key rows_after_dev_end "$T/g.md")" 1
check gzip-last "$(key last_ts "$T/g.md")" 2024-06-30T15:00:00Z
check gzip-others "$(key other_files "$T/g.md")" 0
for marker in 4242.5 7777.5; do
  check "gzip-no-$marker" "$(cat "$T/g.md" "$T/g.out" | grep -c -F "$marker")" 0
done

# a newest-first table: ordered first and last, positive step
N=$T/desc
mkdir -p "$N"
printf '%s\n' "$HDR" '1577844000,4242.5,2,1,1.5,3' '1577840400,4243.5,2,1,1.5,3' '1577836800,4244.5,2,1,1.5,3' > "$N/x.csv"
bash "$PROF" --dir "$N" --dev-end 2023-11-15 --out "$T/d.md" > "$T/d.out" 2>&1
check desc-rc "$?" 0
check desc-first "$(key first_ts "$T/d.md")" 2020-01-01T00:00:00Z
check desc-last "$(key last_ts "$T/d.md")" 2020-01-01T02:00:00Z
check desc-step "$(key median_step_seconds "$T/d.md")" 3600
check desc-order "$(key order "$T/d.md")" descending
check asc-order "$(key order "$T/s.md")" ascending

# an unsorted table is refused
printf '%s\n' "$HDR" '1577840400,4242.5,2,1,1.5,3' '1577836800,4243.5,2,1,1.5,3' '1577844000,4244.5,2,1,1.5,3' > "$N/x.csv"
rm -f "$T/us.md"
bash "$PROF" --dir "$N" --dev-end 2023-11-15 --out "$T/us.md" > "$T/us.out" 2>&1
check unsorted-rc "$?" 3
check unsorted-no-output "$([ -e "$T/us.md" ] && echo yes || echo no)" no
check unsorted-message "$(grep -c 'not sorted' "$T/us.out")" 1
check unsorted-no-value "$(grep -c -F 4242.5 "$T/us.out")" 0

# a banner line is skipped with its own message and never printed
B=$T/banner
mkdir -p "$B"
printf '# BANNERTEXT vendor, v2\n%s\n1577836800,4242.5,2,1,1.5,3\n1577840400,4243.5,2,1,1.5,3\n' "$HDR" > "$B/x.csv"
bash "$PROF" --dir "$B" --dev-end 2023-11-15 --out "$T/b.md" > "$T/b.out" 2>&1
check banner-rc "$?" 0
check banner-key "$(key banner_lines "$T/b.md")" 1
check banner-rows "$(key rows "$T/b.md")" 2
check banner-header "$(key header "$T/b.md")" "$HDR"
check banner-message "$(grep -c 'banner' "$T/b.out")" 1
for marker in BANNERTEXT 4242.5; do
  check "banner-no-$marker" "$(cat "$T/b.md" "$T/b.out" | grep -c -F "$marker")" 0
done

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
