#!/usr/bin/env bash
# Usage: data-profile.sh --dir <dir> --dev-end <ISO date or time> --out <file>
# Profiles the development files below <dir> and writes `key: value` lines to <out>: files, name_pattern, header,
# columns, raw_columns, type_stage_suggestion, rows, first_ts, last_ts, median_step_seconds, time_unit, zero_volume_rows, zero_counts,
# zip_files, zip_members, other_files, files_at_or_after_dev_end. A file or zip member whose name carries a date
# (YYYY-MM, YYYY-MM-DD or YYYYMMDD) at or after --dev-end is counted and never read or named. A directory whose
# name ends in -sealed is never entered. Zip member names come from `unzip -Z1`; nothing is extracted. A table
# file (.csv, .tsv, .txt) is read for its header, its row count, its first and last time stamp, the median step of
# its first 10000 rows and the zero count per numeric column; a trailing carriage return is stripped from every line. No data row is printed.
# type_stage_suggestion is 04_ohlcv when the header names open, high, low and close (aliases o, h, l, c), or when a headerless
# first table has 6, 7 or 12 columns (the documented bar layouts), else 04_new_type; it reads no row value.
# Writes: <out> only. No network.
# A row stamped at or after --dev-end counts in rows_after_dev_end and enters no other statistic. A table whose
# time stamps cannot be parsed is refused and nothing is written.
# Exit: 0 written; 1 <dir> missing or not a directory; 2 usage; 3 unparseable table.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

dir='' dev_end='' out=''
while [ $# -gt 0 ]; do
  case $1 in
    -h | --help) usage; exit 0 ;;
    --dir | --dev-end | --out)
      [ $# -ge 2 ] || { usage >&2; exit 2; }
      case $1 in --dir) dir=$2 ;; --dev-end) dev_end=$2 ;; --out) out=$2 ;; esac
      shift 2 ;;
    *) usage >&2; exit 2 ;;
  esac
done
{ [ -n "$dir" ] && [ -n "$dev_end" ] && [ -n "$out" ]; } || { usage >&2; exit 2; }
[[ $dev_end =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}([T\ ].*)?$ ]] || { echo "data-profile.sh: --dev-end is not an ISO date: $dev_end" >&2; exit 2; }
[ -d "$dir" ] || { echo "data-profile.sh: $dir is not a directory" >&2; exit 1; }

end_day=${dev_end:0:10}
end_compact=${end_day//-/}
end_sec=$(date -u -d "${dev_end:0:10}T${dev_end:11:8}" +%s 2> /dev/null) || end_sec=$(date -u -d "${end_day}T00:00:00" +%s)

# name_stamp <name>: prints the first date stamp of a file name, or nothing.
name_stamp() {
  local name=$1
  if [[ $name =~ ([0-9]{4}-[0-9]{2}(-[0-9]{2})?) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
  elif [[ $name =~ (^|[^0-9])((19|20)[0-9]{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01]))([^0-9]|$) ]]; then
    printf '%s\n' "${BASH_REMATCH[2]}"
  fi
}

# at_or_after <name>: exit 0 when the name carries a stamp at or after dev_end. A month holding dev_end counts.
at_or_after() {
  local stamp
  stamp=$(name_stamp "$1")
  [ -n "$stamp" ] || return 1
  if [[ $stamp == *-* ]]; then
    [[ ! $stamp < ${end_day:0:${#stamp}} ]]
  else
    [[ ! $stamp < $end_compact ]]
  fi
}

# pattern_of <basename>: the name with its date and its leading upper-case symbol masked.
pattern_of() {
  printf '%s\n' "$1" | sed -E 's/[0-9]{4}-[0-9]{2}-[0-9]{2}/YYYY-MM-DD/; t d; s/[0-9]{4}-[0-9]{2}/YYYY-MM/; t d; s/(^|[^0-9])(19|20)[0-9]{6}([^0-9]|$)/\1YYYYMMDD\3/; :d; s/^[A-Z0-9]{2,}([-_.])/<SYMBOL>\1/'
}

# one table file -> one `|` separated line: unit, first, last, median step, rows, columns, volume zeros, zero counts, rows after dev_end, header; or BAD
read -r -d '' PROFILE_AWK << 'AWK'
function isnum(s) { return s ~ /^[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$/ }
function days(y, m, d,   a, yy, mm, era, yoe, doy, doe) {
  yy = y - (m <= 2); era = int(yy / 400); yoe = yy - era * 400
  mm = m + (m > 2 ? -3 : 9); doy = int((153 * mm + 2) / 5) + d - 1
  doe = yoe * 365 + int(yoe / 4) - int(yoe / 100) + doy
  return era * 146097 + doe - 719468
}
function stamp(s,   ip, n, y, m, d, hh, mi, ss) {
  gsub(/^"|"$/, "", s)
  if (isnum(s)) {
    ip = s; sub(/\..*/, "", ip); n = length(ip)
    if (n <= 11) { unit = "s"; return s + 0 }
    if (n <= 14) { unit = "ms"; return s / 1000 }
    if (n <= 17) { unit = "us"; return s / 1000000 }
    unit = "ns"; return s / 1000000000
  }
  if (match(s, /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/)) {
    y = substr(s, 1, 4) + 0; m = substr(s, 6, 2) + 0; d = substr(s, 9, 2) + 0
    hh = 0; mi = 0; ss = 0
    if (substr(s, 11, 1) ~ /[T ]/ && substr(s, 12, 8) ~ /^[0-9][0-9]:[0-9][0-9]:[0-9][0-9]/) {
      hh = substr(s, 12, 2) + 0; mi = substr(s, 15, 2) + 0; ss = substr(s, 18, 2) + 0
    }
    unit = "iso"; return days(y, m, d) * 86400 + hh * 3600 + mi * 60 + ss
  }
  unit = ""; return -1
}
BEGIN { FS = sep; nrows = 0; after = 0; bad = 0 }
{ sub(/\r$/, "") }
function isname(s) { return s != "" && !isnum(s) && s !~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]/ }
NR == 1 {
  ncol = NF
  hasheader = 1
  for (j = 1; j <= NF; j++) {
    name[j] = $j; gsub(/^"|"$/, "", name[j])
    if (!isname(name[j])) hasheader = 0
  }
  tcol = 0
  if (hasheader) {
    for (j = 1; j <= NF; j++) if (tolower(name[j]) ~ /^(time|ts|timestamp|date|datetime|open_time|opentime|open_ts)$/) { tcol = j; break }
    if (tcol == 0) tcol = 1
    hdr = $0
  } else {
    for (j = 1; j <= NF; j++) {
      v = $j; gsub(/^"|"$/, "", v); name[j] = "column" j
      if (tcol == 0 && ((isnum(v) && length(v) >= 9 && v !~ /\./) || v ~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/)) tcol = j
    }
    if (tcol == 0) { bad = 1; exit }
    hdr = "none"
  }
  if (hasheader) next
}
{
  t = stamp($tcol)
  if (unit == "") { bad = 1; exit }
  if (t >= endsec) { after++; next }
  nrows++
  if (nrows == 1) { firstt = t; funit = unit }
  lastt = t
  if (nrows > 1 && nrows <= 10001) { st = t - prev; hist[st]++; nst++ }
  prev = t
  for (j = 1; j <= ncol; j++) {
    if (j == tcol) continue
    if (isnum($j)) { isn[j] = 1; if ($j + 0 == 0) zc[j]++ }
  }
}
END {
  if (bad) { print "BAD"; exit }
  med = "none"
  if (nst > 0) {
    k = 0
    for (v in hist) keys[++k] = v + 0
    for (a = 2; a <= k; a++) { x = keys[a]; b = a - 1; while (b >= 1 && keys[b] > x) { keys[b + 1] = keys[b]; b-- } keys[b + 1] = x }
    half = (nst + 1) / 2; acc = 0
    for (a = 1; a <= k; a++) { acc += hist[keys[a] ""]; if (acc >= half) { med = keys[a]; break } }
    if (med != "none") { for (v in hist) if ((v + 0) == med) { med = v + 0; break } }
  }
  zlist = ""; zv = 0
  for (j = 1; j <= ncol; j++) {
    if (j == tcol || !isn[j]) continue
    zlist = zlist (zlist == "" ? "" : ",") name[j] "=" (zc[j] + 0)
    if (tolower(name[j]) == "volume") zv = zc[j] + 0
  }
  printf "%s|%s|%s|%s|%d|%d|%d|%s|%d|%s\n", (funit == "" ? "none" : funit), (firstt == "" ? "none" : firstt), (lastt == "" ? "none" : lastt), med, nrows, ncol, zv, (zlist == "" ? "none" : zlist), after, hdr
}
AWK

tmp=$(mktemp -d) || exit 1
trap 'rm -rf "$tmp"' EXIT
: > "$tmp/tables"

files=0 zips=0 members=0 others=0 refused=0
declare -A patterns=()
while IFS= read -r path; do
  base=${path##*/}
  if at_or_after "$base"; then
    refused=$((refused + 1))
    continue
  fi
  files=$((files + 1))
  p=$(pattern_of "$base")
  patterns[$p]=$(( ${patterns[$p]:-0} + 1 ))
  lower=${base,,}
  case $lower in
    *.zip)
      zips=$((zips + 1))
      while IFS= read -r member; do
        if at_or_after "${member##*/}"; then refused=$((refused + 1)); else members=$((members + 1)); fi
      done < <(unzip -Z1 "$path" 2> /dev/null)
      ;;
    *.csv | *.tsv | *.txt)
      sep=,
      first=$(head -n 1 "$path" | tr -d '\r')
      case $first in *$'\t'*) sep=$'\t' ;; esac
      line=$(awk -v sep="$sep" -v endsec="$end_sec" "$PROFILE_AWK" "$path")
      if [ -z "$line" ] || [ "$line" = BAD ]; then
        echo "data-profile.sh: $base: no parseable time column; refused, nothing written" >&2
        exit 3
      fi
      printf '%s\n' "$line" >> "$tmp/tables"
      ;;
    *) others=$((others + 1)) ;;
  esac
done < <(find "$dir" \( -name '*-sealed' -prune \) -o -type f -print | LC_ALL=C sort)

name_pattern=none npat=0
for p in "${!patterns[@]}"; do
  npat=$((npat + 1))
  if [ "${patterns[$p]}" -gt "${best:-0}" ] || { [ "${patterns[$p]}" -eq "${best:-0}" ] && [[ $p < $name_pattern ]]; }; then
    best=${patterns[$p]}
    name_pattern=$p
  fi
done

summary=$(awk -F'|' '
function iso(sec,   d, s, z, era, doe, yoe, y, doy, mp, m, dd) {
  d = int(sec / 86400); s = sec - d * 86400
  z = d + 719468; era = int(z / 146097); doe = z - era * 146097
  yoe = int((doe - int(doe / 1460) + int(doe / 36524) - int(doe / 146096)) / 365)
  y = yoe + era * 400; doy = doe - (365 * yoe + int(yoe / 4) - int(yoe / 100))
  mp = int((5 * doy + 2) / 153); dd = doy - int((153 * mp + 2) / 5) + 1
  m = mp + (mp < 10 ? 3 : -9); if (m <= 2) y++
  return sprintf("%04d-%02d-%02dT%02d:%02d:%02dZ", y, m, dd, int(s / 3600), int((s % 3600) / 60), int(s % 60))
}
{
  n++; units[$1] = 1; rows += $5; zv += $7; after += $9
  if ($2 != "none" && (first == "" || $2 + 0 < first + 0)) first = $2
  if ($3 != "none" && (last == "" || $3 + 0 > last + 0)) last = $3
  if ($4 != "none") meds[++nm] = $4 + 0
  hdrs[$10] = 1
  if (n == 1) { cols = $6; zc = $8; hdr = $10 }
}
END {
  nu = 0; ulist = ""
  split("s ms us ns iso", order, " ")
  for (i = 1; i <= 5; i++) { u = order[i]; if (u in units) { nu++; ulist = ulist (ulist == "" ? "" : "+") u } }
  if (nu == 0) ulist = "none"; else if (nu > 1) ulist = "mixed:" ulist
  med = "none"
  if (nm > 0) {
    for (a = 2; a <= nm; a++) { x = meds[a]; b = a - 1; while (b >= 1 && meds[b] > x) { meds[b + 1] = meds[b]; b-- } meds[b + 1] = x }
    med = meds[int((nm + 1) / 2)]
  }
  nh = 0; for (h in hdrs) nh++
  raw = "none"
  if (n > 0 && hdr != "none") raw = hdr
  printf "tables=%d\nheader=%s\ncolumns=%s\nraw=%s\nrows=%d\nafter=%d\nfirst=%s\nlast=%s\nmed=%s\nunit=%s\nzv=%d\nzc=%s\nvariants=%d\n", n, (n > 0 ? hdr : "none"), (n > 0 ? cols : 0), raw, rows, after, (first == "" ? "none" : iso(first + 0)), (last == "" ? "none" : iso(last + 0)), med, ulist, zv, (n > 0 ? zc : "none"), nh
}' "$tmp/tables")

get() { sed -n "s/^$1=//p" <<< "$summary" | head -n 1; }
header=$(get header)
raw=$(get raw)
if [ "$raw" != none ]; then
  raw=$(awk -v h="$raw" 'BEGIN { n = split(h, a, /[,\t]/); s = ""; for (i = 1; i <= n; i++) { gsub(/^"|"$/, "", a[i]); if (tolower(a[i]) ~ /^(time|ts|timestamp|date|datetime|open_time|opentime|open_ts)$/) continue; s = s (s == "" ? "" : ",") a[i] } print (s == "" ? "none" : s) }')
fi
header_display=$header
[ "$header" = none ] || header_display=$(printf '%s' "$header" | tr '\t' ',' | tr -d '"')

type_stage_suggestion=04_new_type
if [ "$header" != none ]; then
  has_ohlc=$(printf '%s\n' "$header_display" | awk -F, '{ for (i = 1; i <= NF; i++) { n = tolower($i); gsub(/^ +| +$/, "", n); if (n == "open" || n == "o") a = 1; else if (n == "high" || n == "h") b = 1; else if (n == "low" || n == "l") c = 1; else if (n == "close" || n == "c") d = 1 } print (a && b && c && d) ? 1 : 0 }')
  [ "$has_ohlc" = 1 ] && type_stage_suggestion=04_ohlcv
else
  case $(get columns) in 6 | 7 | 12) type_stage_suggestion=04_ohlcv ;; esac
fi

mkdir -p "$(dirname "$out")" || exit 1
{
  echo "files: $files"
  echo "name_pattern: $name_pattern"
  echo "name_patterns: $npat"
  echo "tables: $(get tables)"
  echo "header: $header_display"
  echo "header_variants: $(get variants)"
  echo "columns: $(get columns)"
  echo "raw_columns: $raw"
  echo "type_stage_suggestion: $type_stage_suggestion"
  echo "rows: $(get rows)"
  echo "rows_after_dev_end: $(get after)"
  echo "first_ts: $(get first)"
  echo "last_ts: $(get last)"
  echo "median_step_seconds: $(get med)"
  echo "time_unit: $(get unit)"
  echo "zero_volume_rows: $(get zv)"
  echo "zero_counts: $(get zc)"
  echo "zip_files: $zips"
  echo "zip_members: $members"
  echo "other_files: $others"
  echo "files_at_or_after_dev_end: $refused"
} > "$out"
echo "wrote $out"
