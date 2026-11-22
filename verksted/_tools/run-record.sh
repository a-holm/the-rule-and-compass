#!/usr/bin/env bash
# Usage: run-record.sh new --id <YYYYMMDD-slug> --kind store-build|measurement|search|walkforward|gate|campaign-unit
#                          --bin <path> --worktree <path> [--store <path>]... [--cmd-file <path>]
#                          [--features <text>] [--mask <cpu list>] [--split <id>] [--seed <n>]
#        run-record.sh close --id <id> --log <log>... [--metrics <file>] [--after-seal <verdict file>]
#                            [--artefact <path>]... [--deviation <text>]...
#        run-record.sh latest [--kind <kind>] [--slug <slug>]
# Writes: only under $VLOCAL/runs/ (RCSR_LOCAL overrides $VLOCAL).
# Exit: 0 ok; 1 refused or failed; 2 usage.
# The run registry: one card per run at $VLOCAL/runs/<id>/run.md (template _templates/local/run.md) and the generated runs/_index.md.
# new: fills start (UTC), binary sha256 (sha256sum), commit (git -C <worktree> rev-parse HEAD, -dirty when the tree is not clean) and the
#   stores. A store digest comes from the digest cache <store>.sha256.json when its size, mtime and inode still match the file; else the file
#   is hashed when under 2 GiB; else the manifest value (<store>.manifest.json, data.sha256 or sha256) is recorded as such, or the card says unhashed.
#   --cmd-file records the path and sha256 of the script that holds the command.
# close: sums `[time] wall= user= rc=` lines of the logs, counts lock-hours (wall of logs with an `xid new=` line), sums the `xid new=` lines,
#   sets end, load and status (ok, failed or unknown without any rc line), then regenerates runs/_index.md.
#   A --metrics JSON file that reports a locked read is refused (exit 1) unless --after-seal names an existing, non-empty verdict file.
#   Locked-read keys (the engine's own detector): protocol.test_read, protocol.lockbox.evaluated, protocol.lockbox.rows > 0,
#   protocol.lockbox.read_count > 0, protocol.lockbox.read_marker, lockbox.evaluated, lockbox.read_count > 0, locked_reads.
#   Key metrics keep only the finite numbers and booleans at the top level of a development report; no raw data is stored.
# latest: prints the path of the newest card (by start, then id), of one kind with --kind, of an id ending in -<slug> with --slug. Exit 1 when none.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TWO_GIB=2147483648
RUNS=$VLOCAL/runs
KINDS="store-build measurement search walkforward gate campaign-unit"

die() { echo "run-record: $1" >&2; exit 1; }
bad() { echo "run-record: $1" >&2; usage >&2; exit 2; }

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac
[ $# -ge 1 ] || { usage >&2; exit 2; }
cmd=$1
shift

# set_key <file> <key> <value>: replaces the `<key>:` line of the card; the value is passed through the environment.
set_key() {
  local f=$1 k=$2 tmp
  tmp=$(mktemp "$(dirname "$f")/.card.XXXXXX") || die "cannot write under $(dirname "$f")"
  KEY=$k VAL=$3 awk 'BEGIN { k = ENVIRON["KEY"] ":"; v = ENVIRON["VAL"] }
    $0 == k || index($0, k " ") == 1 { print k (v == "" ? "" : " " v); done = 1; next } { print }
    END { if (!done) exit 3 }' "$f" > "$tmp" || { rm -f "$tmp"; die "card $f has no key $2"; }
  mv "$tmp" "$f"
}

# set_list <file> <key> <lines>: replaces the `<key>:` line and the `- ` lines under it.
set_list() {
  local f=$1 k=$2 tmp
  tmp=$(mktemp "$(dirname "$f")/.card.XXXXXX") || die "cannot write under $(dirname "$f")"
  KEY=$k LIST=$3 awk 'BEGIN { k = ENVIRON["KEY"] ":" }
    skip && /^- / { next }
    { skip = 0 }
    $0 == k { print; print ENVIRON["LIST"]; skip = 1; done = 1; next }
    { print }
    END { if (!done) exit 3 }' "$f" > "$tmp" || { rm -f "$tmp"; die "card $f has no key $2"; }
  mv "$tmp" "$f"
}

card_value() { sed -n "s/^$2: *//p" "$1" | head -n 1; }

valid_id() { [[ $1 =~ ^[0-9]{8}-[a-z0-9][a-z0-9-]*$ ]]; }

store_line() {
  local s=$1 side=$1.sha256.json size mt ino sha man
  [ -f "$s" ] || die "no such store file: $s"
  size=$(stat -c %s "$s")
  if [ -f "$side" ]; then
    mt=$(stat -c %.9Y "$s"); mt=${mt/./}
    ino=$(stat -c %i "$s")
    sha=$(jq -r --arg size "$size" --arg mt "$mt" --arg ino "$ino" \
      'select((.size|tostring) == $size and (.mtime_ns|tostring) == $mt and (.inode|tostring) == $ino) | .sha256' "$side" 2> /dev/null)
    if [[ $sha =~ ^[0-9a-f]{64}$ ]]; then
      echo "- $s sha256=$sha (digest cache)"
      return
    fi
  fi
  if [ "$size" -lt "$TWO_GIB" ]; then
    echo "- $s sha256=$(sha256sum "$s" | cut -d' ' -f1) (hashed)"
    return
  fi
  man=$s.manifest.json
  sha=
  [ -f "$man" ] && sha=$(jq -r '.data.sha256 // .sha256 // empty' "$man" 2> /dev/null)
  if [[ $sha =~ ^[0-9a-f]{64}$ ]]; then
    echo "- $s sha256=$sha (manifest value; file over 2 GiB, not hashed here)"
  else
    echo "- $s sha256=unhashed (file over 2 GiB, no valid digest cache or manifest value)"
  fi
}

cmd_new() {
  local id= kind= bin= wt= cmdfile= features= mask= split= seed= stores=() k ok=0
  while [ $# -gt 0 ]; do
    [ $# -ge 2 ] || bad "$1 needs a value"
    case $1 in
      --id) id=$2 ;; --kind) kind=$2 ;; --bin) bin=$2 ;; --worktree) wt=$2 ;; --cmd-file) cmdfile=$2 ;;
      --store) stores+=("$2") ;; --features) features=$2 ;; --mask) mask=$2 ;; --split) split=$2 ;; --seed) seed=$2 ;;
      *) bad "unexpected argument '$1'" ;;
    esac
    shift 2
  done
  [ -n "$id" ] && [ -n "$kind" ] && [ -n "$bin" ] && [ -n "$wt" ] || bad "new needs --id, --kind, --bin and --worktree"
  valid_id "$id" || bad "--id must be YYYYMMDD-slug (lowercase letters, digits, hyphens): $id"
  for k in $KINDS; do [ "$k" = "$kind" ] && ok=1; done
  [ "$ok" -eq 1 ] || bad "--kind must be one of: $KINDS"
  [ -f "$bin" ] || die "no such binary: $bin"
  [ -d "$wt" ] || die "no such worktree: $wt"
  [ -z "$cmdfile" ] || [ -f "$cmdfile" ] || die "no such command file: $cmdfile"
  [ ! -e "$RUNS/$id" ] || die "run $id exists: $RUNS/$id"

  local commit sha sl list=
  commit=$(git -C "$wt" rev-parse HEAD) || die "$wt is not a git worktree"
  [ -z "$(git -C "$wt" status --porcelain)" ] || commit=$commit-dirty
  sha=$(sha256sum "$bin" | cut -d' ' -f1)
  for k in "${stores[@]}"; do
    sl=$(store_line "$k") || exit 1
    list+=$sl$'\n'
  done
  list=${list%$'\n'}
  [ -n "$list" ] || list="- none"

  mkdir -p "$RUNS/$id" || die "cannot create $RUNS/$id"
  local f=$RUNS/$id/run.md
  sed "s/^# Run card: .*/# Run card: $id/" "$VERKSTED/_templates/local/run.md" > "$f" || die "cannot write $f"
  set_key "$f" id "$id"
  set_key "$f" kind "$kind"
  set_key "$f" start "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set_key "$f" status running
  if [ -n "$cmdfile" ]; then
    set_key "$f" command "$cmdfile sha256=$(sha256sum "$cmdfile" | cut -d' ' -f1)"
  else
    set_key "$f" command unrecorded
  fi
  set_key "$f" binary "$bin"
  set_key "$f" binary_sha256 "$sha"
  set_key "$f" commit "$commit"
  set_key "$f" features "$features"
  set_list "$f" stores "$list"
  set_key "$f" split_id "$split"
  set_key "$f" seed "$seed"
  set_key "$f" cpu_mask "$mask"
  regen_index
  echo "$f"
}

# locked_reason <json file>: prints why the report reads locked data; returns 1 when it does not.
locked_reason() {
  jq -r '
    (.protocol? | objects // {}) as $p
    | ($p.lockbox? | objects // {}) as $pl
    | (.lockbox? | objects // {}) as $l
    | [ (if $p.test_read == true then "protocol.test_read is true" else empty end),
        (if $pl.evaluated == true then "protocol.lockbox.evaluated is true" else empty end),
        (if (($pl.rows // 0) | numbers) > 0 then "protocol.lockbox.rows is above 0" else empty end),
        (if (($pl.read_count // 0) | numbers) > 0 then "protocol.lockbox.read_count is above 0" else empty end),
        (if ($pl.read_marker // null) != null then "protocol.lockbox.read_marker is set" else empty end),
        (if $l.evaluated == true then "lockbox.evaluated is true" else empty end),
        (if (($l.read_count // 0) | numbers) > 0 then "lockbox.read_count is above 0" else empty end),
        (if (type == "object" and has("locked_reads")) then "locked_reads is present" else empty end)
      ] | join("; ")' "$1" 2> /dev/null | { read -r r; [ -n "$r" ] && echo "$r"; }
}

regen_index() {
  local tmp d f
  mkdir -p "$RUNS" || die "cannot create $RUNS"
  tmp=$(mktemp "$RUNS/.index.XXXXXX") || die "cannot write under $RUNS"
  {
    echo "# Run index"
    echo
    echo "Generated by run-record.sh, newest first. One line per run card."
    echo
    for f in "$RUNS"/*/run.md; do
      [ -f "$f" ] || continue
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(card_value "$f" start)" "$(card_value "$f" id)" "$(card_value "$f" kind)" \
        "$(card_value "$f" status)" "$(card_value "$f" wall_s)" "$(card_value "$f" commit)"
    done | LC_ALL=C sort -t$'\t' -k1,1r -k2,2r | awk -F'\t' '{
      c = substr($6, 1, 12)
      printf "- %s | %s | %s | start %s | wall %s s | %s\n", $2, $3, $4, $1, ($5 == "" ? "-" : $5), (c == "" ? "-" : c) }'
  } > "$tmp"
  mv "$tmp" "$RUNS/_index.md"
}

cmd_close() {
  local id= metrics= seal= logs=() arts=() devs=() f l
  while [ $# -gt 0 ]; do
    [ $# -ge 2 ] || bad "$1 needs a value"
    case $1 in
      --id) id=$2 ;; --log) logs+=("$2") ;; --metrics) metrics=$2 ;; --after-seal) seal=$2 ;;
      --artefact) arts+=("$2") ;; --deviation) devs+=("$2") ;;
      *) bad "unexpected argument '$1'" ;;
    esac
    shift 2
  done
  [ -n "$id" ] && [ ${#logs[@]} -gt 0 ] || bad "close needs --id and at least one --log"
  valid_id "$id" || bad "--id must be YYYYMMDD-slug: $id"
  f=$RUNS/$id/run.md
  [ -f "$f" ] || die "no run card $f"
  for l in "${logs[@]}"; do [ -f "$l" ] || die "no such log: $l"; done

  local mline= lockednote=
  if [ -n "$metrics" ]; then
    [ -f "$metrics" ] || die "no such metrics file: $metrics"
    jq -e . "$metrics" > /dev/null 2>&1 || die "metrics file is not JSON: $metrics"
    if reason=$(locked_reason "$metrics"); then
      if [ -z "$seal" ]; then
        die "refused: $metrics reports a locked read ($reason); a locked-window result is recorded only after the seal, with --after-seal <verdict file>"
      fi
      [ -s "$seal" ] || die "refused: --after-seal file missing or empty: $seal"
      lockednote="after seal, verdict $seal sha256=$(sha256sum "$seal" | cut -d' ' -f1) ($reason)"
    fi
    mline="file $metrics sha256=$(sha256sum "$metrics" | cut -d' ' -f1)"
    local nums
    nums=$(jq -r '[to_entries[] | select((.value | type) == "number" or (.value | type) == "boolean") | "\(.key)=\(.value)"] | .[0:24] | join(" ")' "$metrics")
    [ -z "$nums" ] || mline="$mline; $nums"
  elif [ -n "$seal" ]; then
    bad "--after-seal needs --metrics"
  fi

  local wall user lockwall xid=0 xcheck=0 rcs=0 bad_rc=0 n
  wall=0 user=0 lockwall=0
  for l in "${logs[@]}"; do
    read -r w u c < <(awk '/^\[time\] wall=/ {
        for (i = 1; i <= NF; i++) { split($i, kv, "="); if (kv[1] == "wall") w += kv[2]; if (kv[1] == "user") u += kv[2]; if (kv[1] == "rc" && kv[2] != 0) bad++ }
        n++ }
      /^RC_DONE rc=/ { split($2, kv, "="); n++; if (kv[2] != 0) bad++ }
      END { printf "%.3f %.3f %d:%d\n", w, u, n, bad }' "$l")
    wall=$(awk -v a="$wall" -v b="$w" 'BEGIN { printf "%.3f", a + b }')
    user=$(awk -v a="$user" -v b="$u" 'BEGIN { printf "%.3f", a + b }')
    rcs=$((rcs + ${c%%:*}))
    bad_rc=$((bad_rc + ${c##*:}))
    if grep -q '^xid new=' "$l"; then
      lockwall=$(awk -v a="$lockwall" -v b="$w" 'BEGIN { printf "%.3f", a + b }')
      n=$(sed -n 's/^xid new=\([0-9][0-9]*\) since=.*/\1/p' "$l" | awk '{ s += $1 } END { print s + 0 }')
      xid=$((xid + n))
    fi
    ! grep -q '^STOP: Xid check failed' "$l" || xcheck=1
  done
  local status=ok xidv=$xid
  [ "$rcs" -gt 0 ] || status=unknown
  if [ "$bad_rc" -gt 0 ] || [ "$xid" -gt 0 ] || [ "$xcheck" -eq 1 ]; then status=failed; fi
  [ "$xcheck" -eq 0 ] || xidv=check-failed

  set_key "$f" end "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  set_key "$f" status "$status"
  set_key "$f" wall_s "$(awk -v w="$wall" 'BEGIN { printf "%.2f", w }')"
  set_key "$f" user_s "$(awk -v w="$user" 'BEGIN { printf "%.2f", w }')"
  set_key "$f" lock_hours "$(awk -v w="$lockwall" 'BEGIN { printf "%.3f", w / 3600 }')"
  set_key "$f" load "$(cut -d' ' -f1-3 /proc/loadavg 2> /dev/null)"
  set_key "$f" xid_new "$xidv"
  [ -z "$mline" ] || set_key "$f" key_metrics "$mline"
  [ -z "$lockednote" ] || set_key "$f" locked_read "$lockednote"
  if [ ${#arts[@]} -gt 0 ]; then
    local a list=
    for a in "${arts[@]}"; do list+="$a; "; done
    set_key "$f" artefacts "${list%; }"
  fi
  if [ "$status" = unknown ]; then devs+=("no rc line in the logs"); fi
  if [ ${#devs[@]} -gt 0 ]; then
    local d list=
    for d in "${devs[@]}"; do list+="$d; "; done
    set_key "$f" deviations "${list%; }"
  fi
  regen_index
  echo "$f"
  [ "$status" = ok ] || { echo "run-record: status $status" >&2; return 0; }
}

cmd_latest() {
  local kind= slug= f line
  while [ $# -gt 0 ]; do
    [ $# -ge 2 ] || bad "$1 needs a value"
    case $1 in --kind) kind=$2 ;; --slug) slug=$2 ;; *) bad "unexpected argument '$1'" ;; esac
    shift 2
  done
  for f in "$RUNS"/*/run.md; do
    [ -f "$f" ] || continue
    [ -z "$kind" ] || [ "$(card_value "$f" kind)" = "$kind" ] || continue
    [ -z "$slug" ] || [[ $(card_value "$f" id) == *-"$slug" ]] || continue
    printf '%s\t%s\t%s\n' "$(card_value "$f" start)" "$(card_value "$f" id)" "$f"
  done | LC_ALL=C sort -t$'\t' -k1,1r -k2,2r | head -n 1 | cut -f3 | { read -r line; [ -n "$line" ] && echo "$line"; } || die "no run card${kind:+ of kind $kind}${slug:+ for slug $slug}"
}

case $cmd in
  new) cmd_new "$@" ;;
  close) cmd_close "$@" ;;
  latest) cmd_latest "$@" ;;
  *) bad "unknown command '$cmd'" ;;
esac
