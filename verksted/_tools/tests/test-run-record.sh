#!/usr/bin/env bash
# Usage: test-run-record.sh
# Hermetic test of run-record.sh: new, close, index, store digests, locked-read refusal, latest, writes only under RCSR_LOCAL.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RR=$HERE/../run-record.sh
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
L=$T/local
rr() { RCSR_LOCAL=$L bash "$RR" "$@" 2>&1; echo "rc=$?"; }
rc_of() { tail -n 1 <<< "$1"; }
key() { sed -n "s/^$2: *//p" "$1" | head -n 1; }

# fixtures: a git worktree with one commit, a fake binary, stores, logs
W=$T/wt
mkdir -p "$W" && git -C "$W" init -q && git -C "$W" -c user.name=t -c user.email=t@t commit -q --allow-empty -m c
commit=$(git -C "$W" rev-parse HEAD)
printf 'fake binary\n' > "$T/rcsr"
printf 'a,b\n1,2\n' > "$T/store1.csv"
printf 'cmd\n' > "$T/cmd.sh"
sha1=$(sha256sum "$T/store1.csv" | cut -d' ' -f1)
printf 'x,y\n' > "$T/store2.csv"
cached=$(printf 'c%.0s' $(seq 64))
mt=$(stat -c %.9Y "$T/store2.csv"); mt=${mt/./}
printf '{"path":"p","size":%s,"mtime_ns":%s,"inode":%s,"sha256":"%s"}\n' \
  "$(stat -c %s "$T/store2.csv")" "$mt" "$(stat -c %i "$T/store2.csv")" "$cached" > "$T/store2.csv.sha256.json"
printf 'z\n' > "$T/store3.csv"
printf '{"path":"p","size":1,"mtime_ns":1,"inode":1,"sha256":"%s"}\n' "$cached" > "$T/store3.csv.sha256.json"
sha3=$(sha256sum "$T/store3.csv" | cut -d' ' -f1)
cat > "$T/a.log" <<'LOG'
[time] wall=10.50 s user=20.25 s sys=1.00 s maxrss=100 KB rc=0
xid new=0 since=2026-01-01 00:00:00
RC_DONE rc=0 at=2026-01-01T00:00:00Z
LOG
cat > "$T/b.log" <<'LOG'
[time] wall=3600.00 s user=7000.00 s sys=1.00 s maxrss=100 KB rc=0
xid new=0 since=2026-01-01 01:00:00
RC_DONE rc=0 at=2026-01-01T02:00:00Z
LOG
printf '{"best":{"x":1},"stop_reason":"time","converged_early":false,"rows":120,"lockbox":{"rows":0,"evaluated":false},"protocol":{"test_read":false,"lockbox":{"rows":0,"read_count":0}}}\n' > "$T/dev.json"
printf '{"protocol":{"test_read":true,"lockbox":{"rows":9}}}\n' > "$T/locked.json"
printf '{"lockbox":{"evaluated":true}}\n' > "$T/locked2.json"
printf '{"locked_reads":[1]}\n' > "$T/locked3.json"
printf '{"protocol":{"lockbox":{"read_marker":{"event":"test_read"}}}}\n' > "$T/locked4.json"
printf 'verdict good\n' > "$T/verdict.md"

cat > "$T/c.log" <<'LOG'
[time] wall=1.00 s user=1.00 s sys=0.00 s maxrss=1 KB rc=1
xid new=2 since=2026-01-01 03:00:00
STOP: 2 new NVRM Xid line(s) since 2026-01-01 03:00:00
RC_DONE rc=70 at=2026-01-01T03:00:01Z
LOG
printf 'STOP: Xid check failed (start=missing); failing closed\nplain line\n' > "$T/d.log"
printf 'plain line\n' > "$T/e.log"
: > "$T/empty.md"
printf 'not json\n' > "$T/bad.json"
before=$(cd "$T" && find . -path ./local -prune -o -type f -exec cksum {} + | sort)

# new
out=$(rr new --id 20260101-first --kind search --bin "$T/rcsr" --worktree "$W" --store "$T/store1.csv" --store "$T/store2.csv" \
  --store "$T/store3.csv" --cmd-file "$T/cmd.sh" --features cpu --mask maskval --split s1 --seed 7)
check new-rc "$(rc_of "$out")" "rc=0"
C=$L/runs/20260101-first/run.md
check new-card "$([ -f "$C" ] && echo yes || echo no)" "yes"
check new-title "$(head -n 1 "$C")" "# Run card: 20260101-first"
check new-hash "$(key "$C" binary_sha256 | grep -cE '^[0-9a-f]{64}$')" "1"
check new-hash-value "$(key "$C" binary_sha256)" "$(sha256sum "$T/rcsr" | cut -d' ' -f1)"
check new-commit "$(key "$C" commit)" "$commit"
check new-kind "$(key "$C" kind)" "search"
check new-status "$(key "$C" status)" "running"
check new-start "$(key "$C" start | grep -cE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9:]{8}Z$')" "1"
check new-mask "$(key "$C" cpu_mask)" "maskval"
check new-seed "$(key "$C" seed)" "7"
check new-cmd "$(key "$C" command | grep -c "cmd.sh sha256=")" "1"
check store-hashed "$(grep -c "store1.csv sha256=$sha1 (hashed)" "$C")" "1"
check store-cache "$(grep -c "store2.csv sha256=$cached (digest cache)" "$C")" "1"
check store-stale-cache "$(grep -c "store3.csv sha256=$sha3 (hashed)" "$C")" "1"
check new-no-placeholder "$(grep -c '^- none' "$C")" "0"
check new-dup "$(rc_of "$(rr new --id 20260101-first --kind search --bin "$T/rcsr" --worktree "$W")")" "rc=1"
check new-bad-kind "$(rc_of "$(rr new --id 20260101-x --kind nope --bin "$T/rcsr" --worktree "$W")")" "rc=2"
check new-bad-id "$(rc_of "$(rr new --id first --kind gate --bin "$T/rcsr" --worktree "$W")")" "rc=2"
check new-no-bin "$(rc_of "$(rr new --id 20260101-y --kind gate --bin "$T/none" --worktree "$W")")" "rc=1"
check new-no-store "$(rc_of "$(rr new --id 20260101-y --kind gate --bin "$T/rcsr" --worktree "$W" --store "$T/none")")" "rc=1"
check new-no-leftover "$([ -e "$L/runs/20260101-y" ] && echo yes || echo no)" "no"
rm -rf "$L/runs/20260101-y"

# close
out=$(rr close --id 20260101-first --log "$T/a.log" --log "$T/b.log" --metrics "$T/dev.json" --artefact "$T/out1" --deviation "slow disk")
check close-rc "$(rc_of "$out")" "rc=0"
check close-wall "$(key "$C" wall_s)" "3610.50"
check close-user "$(key "$C" user_s)" "7020.25"
check close-lock-hours "$(key "$C" lock_hours)" "1.003"
check close-xid "$(key "$C" xid_new)" "0"
check close-status "$(key "$C" status)" "ok"
check close-end "$(key "$C" end | grep -cE '^[0-9]{4}-')" "1"
check close-load "$([ -n "$(key "$C" load)" ] && echo yes || echo no)" "yes"
check close-metrics "$(key "$C" key_metrics | grep -c 'rows=120.*converged_early=false\|converged_early=false.*rows=120')" "1"
check close-metrics-file "$(key "$C" key_metrics | grep -c "file $T/dev.json sha256=")" "1"
check close-artefacts "$(key "$C" artefacts)" "$T/out1"
check close-deviation "$(key "$C" deviations)" "slow disk"
check close-locked-empty "$(key "$C" locked_read)" ""
check index-lines "$(grep -c '^- ' "$L/runs/_index.md")" "1"
check index-line "$(grep '^- ' "$L/runs/_index.md" | cut -d'|' -f1-3)" "- 20260101-first | search | ok "

# a second run, a failed one; index newest first; latest
out=$(rr new --id 20260101-second --kind gate --bin "$T/rcsr" --worktree "$W")
check second-rc "$(rc_of "$out")" "rc=0"
out=$(rr close --id 20260101-second --log "$T/c.log")
check failed-rc "$(rc_of "$out")" "rc=0"
C2=$L/runs/20260101-second/run.md
check failed-status "$(key "$C2" status)" "failed"
check failed-xid "$(key "$C2" xid_new)" "2"
check index-two "$(grep -c '^- ' "$L/runs/_index.md")" "2"
check index-order "$(grep '^- ' "$L/runs/_index.md" | head -n 1 | cut -d' ' -f2)" "20260101-second"
check latest "$(rc_of "$(rr latest)" )" "rc=0"
check latest-path "$(rr latest | head -n 1)" "$C2"
check latest-kind "$(rr latest --kind search | head -n 1)" "$C"
check latest-slug "$(rr latest --slug second | head -n 1)" "$C2"
check latest-slug-none "$(rc_of "$(rr latest --slug nope)")" "rc=1"
check latest-none "$(rc_of "$(rr latest --kind walkforward)")" "rc=1"

# xid check failure and no rc lines
rr new --id 20260101-third --kind measurement --bin "$T/rcsr" --worktree "$W" > /dev/null
rr close --id 20260101-third --log "$T/d.log" > /dev/null
C3=$L/runs/20260101-third/run.md
check xid-check-failed "$(key "$C3" xid_new)" "check-failed"
check xid-check-status "$(key "$C3" status)" "failed"
rr new --id 20260101-fourth --kind measurement --bin "$T/rcsr" --worktree "$W" > /dev/null
rr close --id 20260101-fourth --log "$T/e.log" > /dev/null
C4=$L/runs/20260101-fourth/run.md
check norc-status "$(key "$C4" status)" "unknown"
check norc-deviation "$(key "$C4" deviations)" "no rc line in the logs"

# locked reads
rr new --id 20260101-locked --kind walkforward --bin "$T/rcsr" --worktree "$W" > /dev/null
CL=$L/runs/20260101-locked/run.md
snap=$(cksum < "$CL")
for m in locked locked2 locked3 locked4; do
  out=$(rr close --id 20260101-locked --log "$T/a.log" --metrics "$T/$m.json")
  check "refuse-$m" "$(rc_of "$out")" "rc=1"
  check "refuse-$m-msg" "$(grep -c 'locked read' <<< "$out")" "1"
done
check refuse-card-unchanged "$(cksum < "$CL")" "$snap"
check refuse-seal-missing "$(rc_of "$(rr close --id 20260101-locked --log "$T/a.log" --metrics "$T/locked.json" --after-seal "$T/none.md")")" "rc=1"
check refuse-seal-empty "$(rc_of "$(rr close --id 20260101-locked --log "$T/a.log" --metrics "$T/locked.json" --after-seal "$T/empty.md")")" "rc=1"
check refuse-card-still "$(cksum < "$CL")" "$snap"
out=$(rr close --id 20260101-locked --log "$T/a.log" --metrics "$T/locked.json" --after-seal "$T/verdict.md")
check seal-rc "$(rc_of "$out")" "rc=0"
check seal-note "$(key "$CL" locked_read | grep -c "^after seal, verdict $T/verdict.md sha256=")" "1"
check seal-alone "$(rc_of "$(rr close --id 20260101-locked --log "$T/a.log" --after-seal "$T/verdict.md")")" "rc=2"
check metrics-not-json "$(rc_of "$(rr close --id 20260101-locked --log "$T/a.log" --metrics "$T/bad.json")")" "rc=1"

# argument errors
check close-no-card "$(rc_of "$(rr close --id 20260101-none --log "$T/a.log")")" "rc=1"
check close-no-log "$(rc_of "$(rr close --id 20260101-first)")" "rc=2"
check close-missing-log "$(rc_of "$(rr close --id 20260101-first --log "$T/none.log")")" "rc=1"
check unknown-cmd "$(rc_of "$(rr nope)")" "rc=2"
check no-args "$(rc_of "$(RCSR_LOCAL=$L bash "$RR" 2>&1; echo "rc=$?")")" "rc=2"
check help "$(RCSR_LOCAL=$L bash "$RR" --help | head -n 1 | cut -c1-20)" "Usage: run-record.sh"

# nothing outside the local area changed; nothing stray inside runs/
check outside-unchanged "$(cd "$T" && find . -path ./local -prune -o -type f -exec cksum {} + | sort)" "$before"
check no-temp-files "$(find "$L" -name '.card.*' -o -name '.index.*' | wc -l)" "0"
check only-runs "$(ls "$L")" "runs"

[ "$fails" -eq 0 ] && echo "test-run-record: ok" || { echo "test-run-record: $fails failure(s)"; exit 1; }
