#!/usr/bin/env bash
# Usage: data-profile.sh --dir <dir> --dev-end <ISO date or time> [--tz <offset>] --out <file>
# Profiles the development files below <dir> and writes `key: value` lines to <out>: files, name_pattern, header,
# columns, raw_columns, type_stage_suggestion, rows, first_ts, last_ts, median_step_seconds, time_unit, zero_volume_rows, zero_counts,
# order, banner_lines, zip_files, zip_members, other_files, files_at_or_after_dev_end. A file or zip member whose name carries a date
# (YYYY-MM, YYYY-MM-DD or YYYYMMDD) at or after --dev-end is counted and never read or named. A directory whose
# name ends in -sealed is never entered. Zip member names come from `unzip -Z1`; nothing is extracted. A table
# file (.csv, .tsv, .txt, also gzip-compressed or as a zip member) is streamed through one filter and read for its header, its row count, its first and last time stamp, the median step of
# its first 10000 rows and the zero count per numeric column; a trailing carriage return is stripped from every line. No data row is printed.
# type_stage_suggestion is 04_ohlcv when the header names open, high, low and close (aliases o, h, l, c), or when a headerless
# first table has 6, 7 or 12 columns (the documented bar layouts), else 04_new_type; it reads no row value.
# Writes: <out> only. No network.
# Every time stamp is converted to UTC first: an offset or Z in the stamp is applied, and --dev-end may carry one. A
# zone-less time of day needs --tz (Z, UTC, +hh:mm, -hh:mm, +hhmm or +hh) and is refused without it; epoch stamps and
# date-only stamps need none. A row at or after --dev-end counts in rows_after_dev_end and enters no other statistic.
# A table sorted newest first is profiled with order: descending (first_ts is the earliest stamp, the step is
# positive); an unsorted table is refused. A first line with another field count than the second (or starting with #)
# is a banner: it is skipped, counted in banner_lines and never printed. A table whose time stamps cannot be
# parsed is refused and nothing is written.
# Exit: 0 written; 1 <dir> missing or not a directory; 2 usage; 3 unparseable, zone-less without --tz, or unsorted table.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

dir='' dev_end='' out='' tz=''
while [ $# -gt 0 ]; do
  case $1 in
    -h | --help) usage; exit 0 ;;
    --dir | --dev-end | --out | --tz)
      [ $# -ge 2 ] || { usage >&2; exit 2; }
      case $1 in --dir) dir=$2 ;; --dev-end) dev_end=$2 ;; --out) out=$2 ;; --tz) tz=$2 ;; esac
      shift 2 ;;
    *) usage >&2; exit 2 ;;
  esac
done
{ [ -n "$dir" ] && [ -n "$dev_end" ] && [ -n "$out" ]; } || { usage >&2; exit 2; }
[[ $dev_end =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}([T\ ].*)?$ ]] || { echo "data-profile.sh: --dev-end is not an ISO date: $dev_end" >&2; exit 2; }
[ -d "$dir" ] || { echo "data-profile.sh: $dir is not a directory" >&2; exit 1; }

end_sec=$(date -u -d "$dev_end" +%s 2> /dev/null) || end_sec=$(date -u -d "${dev_end:0:10}T${dev_end:11:8}" +%s 2> /dev/null) || end_sec=$(date -u -d "${dev_end:0:10}T00:00:00" +%s) || exit 2
end_day=$(date -u -d "@$end_sec" +%F)
end_compact=${end_day//-/}

# tz_seconds <offset>: prints the offset in seconds east of UTC, or fails.
tz_seconds() {
  local s=$1 sign hh mm
  case ${s^^} in Z | UTC) echo 0; return 0 ;; esac
  [[ $s =~ ^([-+])([0-9]{2})(:?([0-9]{2}))?$ ]] || return 1
  sign=${BASH_REMATCH[1]} hh=${BASH_REMATCH[2]} mm=${BASH_REMATCH[4]:-00}
  [ "$((10#$hh))" -le 14 ] && [ "$((10#$mm))" -le 59 ] || return 1
  echo "$(( ${sign}1 * (10#$hh * 3600 + 10#$mm * 60) ))"
}
tz_sec=''
if [ -n "$tz" ]; then
  tz_sec=$(tz_seconds "$tz") || { echo "data-profile.sh: --tz is not Z, UTC or an offset like +02:00: $tz" >&2; exit 2; }
fi

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

# one table stream on stdin -> one `|` separated line: unit, first, last, median step, rows, columns, volume zeros, zero counts, rows after dev_end, order, banner, header; or BAD, ZONELESS, UNSORTED
read -r -d '' PROFILE_AWK << 'AWK'
function isnum(s) { return s ~ /^[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$/ }
function days(y, m, d,   a, yy, mm, era, yoe, doy, doe) {
  yy = y - (m <= 2); era = int(yy / 400); yoe = yy - era * 400
  mm = m + (m > 2 ? -3 : 9); doy = int((153 * mm + 2) / 5) + d - 1
  doe = yoe * 365 + int(yoe / 4) - int(yoe / 100) + doy
  return era * 146097 + doe - 719468
}
function stamp(s,   ip, n, y, m, d, hh, mi, ss, rest, off, sign, mm2) {
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
    hh = 0; mi = 0; ss = 0; off = 0
    if (length(s) > 10) {
      if (substr(s, 11, 1) !~ /[T ]/) { unit = ""; return -1 }
      rest = substr(s, 12)
      if (rest ~ /^[0-9][0-9]:[0-9][0-9]:[0-9][0-9]/) { hh = substr(rest, 1, 2) + 0; mi = substr(rest, 4, 2) + 0; ss = substr(rest, 7, 2) + 0; rest = substr(rest, 9) }
      else if (rest ~ /^[0-9][0-9]:[0-9][0-9]/) { hh = substr(rest, 1, 2) + 0; mi = substr(rest, 4, 2) + 0; rest = substr(rest, 6) }
      else { unit = ""; return -1 }
      sub(/^[.,][0-9]+/, "", rest); sub(/^ +/, "", rest)
      if (rest == "") {
        if (!(tzset + 0)) { unit = "zoneless"; return -1 }
        off = tzsec + 0
      } else if (rest ~ /^(Z|z|UTC)$/) off = 0
      else if (rest ~ /^[-+][0-9][0-9](:?[0-9][0-9])?$/) {
        sign = substr(rest, 1, 1) == "-" ? -1 : 1
        mm2 = length(rest) >= 5 ? substr(rest, length(rest) - 1, 2) + 0 : 0
        off = sign * (substr(rest, 2, 2) * 3600 + mm2 * 60)
      } else { unit = ""; return -1 }
    }
    unit = "iso"; return days(y, m, d) * 86400 + hh * 3600 + mi * 60 + ss - off
  }
  unit = ""; return -1
}
function isname(s) { return s != "" && !isnum(s) && s !~ /^[0-9][0-9][0-9][0-9]-[0-9][0-9]/ }
function feed(line,   j, v, t) {
  $0 = line
  fed++
  if (fed == 1) {
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
      if (tcol == 0) { bad = 1; return }
      hdr = "none"
    }
    if (hasheader) return
  }
  t = stamp($tcol)
  if (unit == "zoneless") { bad = 2; return }
  if (unit == "") { bad = 1; return }
  if (t >= endsec) { after++; return }
  nrows++
  if (nrows == 1) { mint = t; maxt = t; funit = unit }
  else {
    if (t < prev) asc = 0
    if (t > prev) desc = 0
    if (!asc && !desc) { bad = 3; return }
    if (t < mint) mint = t
    if (t > maxt) maxt = t
    if (nrows <= 10001) { st = t - prev; if (st < 0) st = -st; hist[st]++; nst++ }
  }
  prev = t
  for (j = 1; j <= ncol; j++) {
    if (j == tcol) continue
    if (isnum($j)) { isn[j] = 1; if ($j + 0 == 0) zc[j]++ }
  }
}
BEGIN { FS = ","; nrows = 0; after = 0; bad = 0; asc = 1; desc = 1; banner = 0; fed = 0 }
{ sub(/\r$/, "") }
NR == 1 { held = $0; next }
NR == 2 {
  FS = index($0, "\t") ? "\t" : ","
  n1 = split(held, parts, FS)
  line2 = $0
  $0 = line2
  if (held ~ /^#/ || n1 != NF) banner = 1
  else { feed(held); if (bad) exit }
  feed(line2)
  if (bad) exit
  next
}
{ feed($0); if (bad) exit }
END {
  if (NR == 1 && !bad) { FS = index(held, "\t") ? "\t" : ","; feed(held) }
  if (bad == 1) { print "BAD"; exit }
  if (bad == 2) { print "ZONELESS"; exit }
  if (bad == 3) { print "UNSORTED"; exit }
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
  printf "%s|%s|%s|%s|%d|%d|%d|%s|%d|%s|%d|%s\n", (funit == "" ? "none" : funit), (nrows == 0 ? "none" : mint), (nrows == 0 ? "none" : maxt), med, nrows, ncol, zv, (zlist == "" ? "none" : zlist), after, (asc ? "ascending" : "descending"), banner, hdr
}
AWK

# profile_stream <label>: profiles the table on stdin; appends its line to $tmp/tables or exits 3 with a message
profile_stream() {
  local label=$1 line
  line=$(awk -v tzset="${tz_sec:+1}" -v tzsec="${tz_sec:-0}" -v endsec="$end_sec" "$PROFILE_AWK")
  case $line in
    ZONELESS) echo "data-profile.sh: $label: time stamps carry no zone; pass --tz; refused, nothing written" >&2; exit 3 ;;
    UNSORTED) echo "data-profile.sh: $label: time stamps are not sorted; refused, nothing written" >&2; exit 3 ;;
    '' | BAD) echo "data-profile.sh: $label: no parseable time column; refused, nothing written" >&2; exit 3 ;;
  esac
  local banner
  banner=$(cut -d'|' -f11 <<< "$line")
  [ "$banner" != 1 ] || echo "data-profile.sh: $label: first line is a banner; skipped" >&2
  printf '%s\n' "$line" >> "$tmp/tables"
}

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
        if at_or_after "${member##*/}"; then refused=$((refused + 1)); continue; fi
        members=$((members + 1))
        case ${member,,} in
          *.csv | *.tsv | *.txt)
            esc=$(printf "%s" "$member" | sed "s/[][*?\\\\]/\\\\&/g")
            profile_stream "${member##*/}" < <(unzip -p "$path" "$esc" 2> /dev/null)
            ;;
        esac
      done < <(unzip -Z1 "$path" 2> /dev/null)
      ;;
    *.csv | *.tsv | *.txt) profile_stream "$base" < "$path" ;;
    *.csv.gz | *.tsv.gz | *.txt.gz) profile_stream "$base" < <(gzip -dc -- "$path" 2> /dev/null) ;;
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
  hdrs[$12] = 1; ords[$10] = 1; banners += $11
  if (n == 1) { cols = $6; zc = $8; hdr = $12 }
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
  no = 0; ord = "none"; for (o in ords) { no++; ord = o }
  if (no > 1) ord = "mixed"
  raw = "none"
  if (n > 0 && hdr != "none") raw = hdr
  printf "order=%s\nbanners=%d\ntables=%d\nheader=%s\ncolumns=%s\nraw=%s\nrows=%d\nafter=%d\nfirst=%s\nlast=%s\nmed=%s\nunit=%s\nzv=%d\nzc=%s\nvariants=%d\n", ord, banners + 0, n, (n > 0 ? hdr : "none"), (n > 0 ? cols : 0), raw, rows, after, (first == "" ? "none" : iso(first + 0)), (last == "" ? "none" : iso(last + 0)), med, ulist, zv, (n > 0 ? zc : "none"), nh
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
  echo "order: $(get order)"
  echo "median_step_seconds: $(get med)"
  echo "time_unit: $(get unit)"
  echo "zero_volume_rows: $(get zv)"
  echo "zero_counts: $(get zc)"
  echo "banner_lines: $(get banners)"
  echo "zip_files: $zips"
  echo "zip_members: $members"
  echo "other_files: $others"
  echo "files_at_or_after_dev_end: $refused"
} > "$out"
echo "wrote $out"
