#!/usr/bin/env bash
# Usage: test-seven-checks.sh
# Hermetic test of the Checks of 01_wave_plan, 06_data, 02_close, 04_new_type, 05_panel_pit, 06_merge_cleanup, 01_preconditions
# and the empty-file cases of 03_dispatch and 05_receive_and_verify. Each real contract is copied into a fixture run directory and run
# through check.sh --stage: a pass fixture and a fail fixture per Check. gh is a stub that serves state files.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/machine.sh"
REPO=$(cd "$HERE/../../.." && pwd)
CHECK=$REPO/verksted/_tools/check.sh
T=$(cd "$(mktemp -d)" && pwd -P)
pids=()
cleanup() {
  local p
  for p in "${pids[@]}"; do kill "$p" 2> /dev/null; done
  rm -rf "$T"
}
trap cleanup EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
git_q() { git -c user.name=t -c user.email=t@t.invalid "$@"; }

V=$T/v
mkdir -p "$T/bin" "$T/gh" "$T/gates" "$V/_tools"
for s in 03_orchestrate/stages/01_wave_plan 03_orchestrate/stages/03_dispatch 03_orchestrate/stages/05_receive_and_verify \
  03_orchestrate/stages/06_merge_cleanup 06_data 06_data/stages/04_new_type 06_data/stages/05_panel_pit \
  07_longrun/stages/01_preconditions 07_longrun/stages/02_launch 08_close/stages/02_close; do
  mkdir -p "$V/$s"
  cp "$REPO/verksted/$s/CONTEXT.md" "$V/$s/CONTEXT.md"
done
printf '#!/usr/bin/env bash\nexit 0\n' > "$V/_tools/run-record.sh"
printf '#!/usr/bin/env bash\necho "$1/crate"\n' > "$V/_tools/crate-dir.sh"
chmod +x "$V/_tools/run-record.sh" "$V/_tools/crate-dir.sh"

cat > "$T/bin/gh" <<'GH'
#!/usr/bin/env bash
g=$FAKE_GH
case "$*" in
  "pr view "*"--json state"*) cat "$g/pr-$3" 2> /dev/null || echo MERGED ;;
  "pr view "*"--json headRefOid"*) cat "$g/head-$3" ;;
  "issue view "*"--json state,stateReason"*) cat "$g/full-$3" 2> /dev/null || echo "CLOSED COMPLETED" ;;
  "issue view "*"--json state"*) cat "$g/issue-$3" 2> /dev/null || echo OPEN ;;
  *) echo "fake gh: unexpected $*" >&2; exit 1 ;;
esac
GH
chmod +x "$T/bin/gh"

# root: fixture repository with origin/integ; RUN: a clone at the same commit
git init -q --bare "$T/origin.git"
git init -q "$T/root"
git_q -C "$T/root" checkout -q -b integ
git_q -C "$T/root" commit -q --allow-empty -m init
git_q -C "$T/root" remote add origin "$T/origin.git"
git_q -C "$T/root" push -q origin integ
git clone -q -b integ "$T/origin.git" "$T/run" 2> /dev/null

# st <pipeline/stage> [VAR=value...]: check.sh --stage from the fixture repo with the fixture variables; prints nothing, returns the exit code
st() {
  local s=${1/\/stages\//\/}
  shift
  (cd "$T/root" && env V="$V" N=5 PR=9 OUT="$T/out" RUN="$T/run" LABEL=lab RCSR_BASE=integ RCSR_ROOT="$T/root" RCSR_GATES="$T/gates" \
    STORE="$T/store" PANEL="$T/panel" MEMBERSHIP="$T/members.csv" PATH="$T/bin:$PATH" FAKE_GH="$T/gh" "$@" "$CHECK" --stage "$s" > /dev/null 2> "$T/err")
}
head1() { head -1 "$V/$1/output/check.md"; }

# 01_wave_plan
WP=03_orchestrate/stages/01_wave_plan
mkdir -p "$V/$WP/output"
w=$V/$WP/output/wave.md
rows() { # rows <count> [lock-wait count]: disjoint rows, the last <lock-wait count> rows marked
  local i
  for i in $(seq 1 "$1"); do
    if [ "$i" -gt $(($1 - ${2:-0})) ]; then
      printf '| #%s | f%s.rs | sonnet | lock-wait |\n' "$i" "$i"
    else
      printf '| #%s | f%s.rs | sonnet |\n' "$i" "$i"
    fi
  done
}
rows 3 > "$w"; st $WP; check wave-three-pass "$?" 0
rows 5 > "$w"; st $WP; check wave-five-no-exception-fail "$?" 1
check wave-five-fail-head "$(head1 $WP | cut -c1-4)" fail
rows 5 2 > "$w"; st $WP; check wave-five-two-waiting-pass "$?" 0
rows 5 1 > "$w"; st $WP; check wave-five-one-waiting-fail "$?" 1
{ rows 5; echo 'exception: user, 2026-11-23'; } > "$w"; st $WP; check wave-five-exception-pass "$?" 0
rows 11 8 > "$w"; st $WP; check wave-eleven-fail "$?" 1
: > "$w"; st $WP; check wave-empty-fail "$?" 1
rows 2 > "$w"; echo '| #9 | f1.rs,f9.rs | sonnet |' >> "$w"; st $WP; check wave-shared-file-fail "$?" 1

# 06_data pipeline
for s in 01_intake 02_acquire 03_profile 04_ohlcv 05_panel_pit 06_enrich 07_audit_lock 08_verify_manifest 09_lessons; do
  mkdir -p "$V/06_data/stages/$s/output"
  echo pass > "$V/06_data/stages/$s/output/check.md"
done
mkdir -p "$V/06_data/output"
st 06_data; check data-pipeline-pass "$?" 0
rm "$V/06_data/stages/07_audit_lock/output/check.md"; st 06_data; check data-pipeline-no-audit-fail "$?" 1
echo pass > "$V/06_data/stages/07_audit_lock/output/check.md"
mkdir -p "$V/06_data/stages/04_new_type/output"; echo pass > "$V/06_data/stages/04_new_type/output/check.md"
st 06_data; check data-pipeline-two-type-stages-fail "$?" 1
rm -r "$V/06_data/stages/04_new_type/output"
echo 'fail: x' > "$V/06_data/stages/06_enrich/output/check.md"; st 06_data; check data-pipeline-red-stage-fail "$?" 1
echo pass > "$V/06_data/stages/06_enrich/output/check.md"
rm "$V/06_data/stages/04_ohlcv/output/check.md"; st 06_data; check data-pipeline-no-type-stage-fail "$?" 1
echo pass > "$V/06_data/stages/04_ohlcv/output/check.md"

# 02_close: closed, kept open, empty
CL=08_close/stages/02_close
mkdir -p "$V/$CL/output" "$T/out"
ver=$T/out/verification.md
closed=$V/$CL/output/closed.md
printf '| AC1 | first | log a | met |\n| AC2 | second | later | not run (#7) |\n\nVerdict: CLOSE\n' > "$ver"
echo '{"state":"CLOSED"}' > "$closed"
st $CL; check close-pass "$?" 0
echo CLOSED > "$T/gh/issue-7"; st $CL; check close-owner-closed-fail "$?" 1
rm "$T/gh/issue-7"
: > "$closed"; st $CL; check close-empty-outcome-fail "$?" 1
rm "$closed"; st $CL; check close-missing-outcome-fail "$?" 1
echo '{"state":"CLOSED"}' > "$closed"
echo "OPEN" > "$T/gh/full-5"; st $CL; check close-not-closed-fail "$?" 1
rm "$T/gh/full-5"
printf '| AC1 | first | log a | unmet |\n\nVerdict: KEEP OPEN\n' > "$ver"
echo 'kept open: AC1' > "$closed"
echo OPEN > "$T/gh/issue-5"; st $CL; check close-keep-open-pass "$?" 0
echo CLOSED > "$T/gh/issue-5"; st $CL; check close-keep-open-but-closed-fail "$?" 1
echo OPEN > "$T/gh/issue-5"
echo '{"state":"OPEN"}' > "$closed"; st $CL; check close-keep-open-no-line-fail "$?" 1
: > "$closed"; st $CL; check close-keep-open-empty-fail "$?" 1
printf '| AC1 | first | log a | unmet |\n\nVerdict: CLOSE\n' > "$ver"
echo 'kept open: AC1' > "$closed"; st $CL; check close-unmet-with-close-verdict-fail "$?" 1
rm "$T/gh/issue-5"

# 04_new_type: a normal run, a reclassified run, the pipeline after the move
NT=06_data/stages/04_new_type
PF=$V/06_data/stages/03_profile/output/profile.md
mkdir -p "$V/06_data/stages/03_profile/output" "$V/$NT/output"
printf 'type_stage: 04_new_type\nfiles_at_or_after_dev_end: 0\n' > "$PF"
echo x > "$T/store"
a=$V/$NT/output/adapt.md
: > "$a"
for k in type variant layout bar_seconds source_bar_seconds first_bar store_first_bar approved_start dev_end locked_test series symbols raw_columns \
  zero_volume_rows zero_volume_policy feature_groups view_excludes labels periods_per_year deviations; do echo "$k: v" >> "$a"; done
{ echo 'type: 04_new_type'; echo "store: $T/store"; echo 'first_bar: 1'; echo 'store_first_bar: 1'; } >> "$a"
sed -i '/^type: v$/d; /^first_bar: v$/d; /^store_first_bar: v$/d' "$a"
st $NT; check newtype-own-pass "$?" 0
sed -i '/^deviations:/d' "$a"; st $NT; check newtype-missing-key-fail "$?" 1
printf 'type_stage: 04_ohlcv\nfiles_at_or_after_dev_end: 0\n' > "$PF"
st $NT; check newtype-reclassified-unrecorded-fail "$?" 1
echo 'reclassified: 04_ohlcv' > "$V/$NT/output/reclassified.md"
st $NT; check newtype-reclassified-pass "$?" 0
echo 'reclassified: 04_other' > "$V/$NT/output/reclassified.md"
st $NT; check newtype-reclassified-wrong-stage-fail "$?" 1
echo 'reclassified: 04_ohlcv' > "$V/$NT/output/reclassified.md"
st $NT
mkdir -p "$V/06_data/stages/04_new_type/output" "$V/06_data/stages/04_ohlcv/output"
echo pass > "$V/06_data/stages/04_ohlcv/output/check.md"
mv "$V/06_data/stages/04_new_type/output" "$V/06_data/stages/04_new_type/reclassified"
check newtype-one-check-md "$(ls -d "$V"/06_data/stages/04_*/output/check.md | wc -l)" 1
check newtype-moved-record-kept "$(head -1 "$V/06_data/stages/04_new_type/reclassified/check.md")" pass
st 06_data; check newtype-reclassified-pipeline-pass "$?" 0
rm -r "$V/06_data/stages/04_new_type/reclassified"

# 05_panel_pit: zero ended symbols, one ended symbol, a wrong count, single series
PP=06_data/stages/05_panel_pit
mkdir -p "$V/$PP/output" "$T/run/crate/target/release"
printf '#!/usr/bin/env bash\nexit 0\n' > "$T/run/crate/target/release/rcsr"
chmod +x "$T/run/crate/target/release/rcsr"
printf 'series: panel\n' > "$V/06_data/stages/04_ohlcv/output/adapt.md"
p=$V/$PP/output/panel.md
printf 'as_of,symbol,member,adv_usd,n_bars\n2020-01-01,A,1,5,9\n2020-01-01,B,1,5,9\n2020-02-01,A,1,5,9\n2020-02-01,B,1,5,9\n' > "$T/members.csv"
printf 'commands\nended: 0\n' > "$p"
st $PP; check panel-zero-ended-pass "$?" 0
printf 'commands\nended: 1\n' > "$p"
st $PP; check panel-zero-ended-wrong-count-fail "$?" 1
printf 'commands\n' > "$p"
st $PP; check panel-count-missing-fail "$?" 1
printf 'as_of,symbol,member,adv_usd,n_bars\n2020-01-01,A,1,5,9\n2020-01-01,B,1,5,9\n2020-02-01,A,1,5,9\n' > "$T/members.csv"
printf 'commands\nended: 1\n' > "$p"
st $PP; check panel-one-ended-pass "$?" 0
printf 'commands\nended: 0\n' > "$p"
st $PP; check panel-one-ended-count-zero-fail "$?" 1
printf 'series: single\n' > "$V/06_data/stages/04_ohlcv/output/adapt.md"
echo 'panel: not applicable (single series)' > "$p"
st $PP; check panel-single-pass "$?" 0
echo 'panel: x' > "$p"
st $PP; check panel-single-fail "$?" 1

# 06_merge_cleanup: merged and clean, an open PR that is listed, an open PR that is not, empty files
MC=03_orchestrate/stages/06_merge_cleanup
RV=03_orchestrate/stages/05_receive_and_verify
mkdir -p "$V/$MC/output" "$V/$RV/output"
vf=$V/$RV/output/verify.md
rf=$V/$MC/output/report.md
echo '| #5 | 9 | abc | abc |' > "$vf"
echo 'kept open: none' > "$rf"
st $MC; check merge-clean-pass "$?" 0
echo OPEN > "$T/gh/pr-9"
st $MC; check merge-open-pr-unlisted-fail "$?" 1
echo 'kept open: #5' > "$rf"
st $MC; check merge-open-pr-listed-pass "$?" 0
echo 'kept open: #50' > "$rf"
st $MC; check merge-listed-other-number-fail "$?" 1
rm "$T/gh/pr-9"
echo 'kept open: none' > "$rf"
git_q -C "$T/root" branch issue/5-x
st $MC; check merge-branch-left-fail "$?" 1
echo 'kept open: #5' > "$rf"
st $MC; check merge-branch-left-listed-pass "$?" 0
git_q -C "$T/root" branch -q -D issue/5-x
: > "$vf"; st $MC; check merge-empty-verify-fail "$?" 1
echo '| #5 | 9 | abc | abc |' > "$vf"
: > "$rf"; st $MC; check merge-empty-report-fail "$?" 1
echo 'kept open: none' > "$rf"

# 01_preconditions: blocking list, this run's pid file, other rcsr processes
PC=07_longrun/stages/01_preconditions
mkdir -p "$V/$PC/output" "$T/run/bin"
printf '#!/usr/bin/env bash\nexit 0\n' > "$T/run/bin/rcsr"
chmod +x "$T/run/bin/rcsr"
(cd "$T/run/bin" && sha256sum rcsr > rcsr.sha256)
pc=$V/$PC/output/precond.md
echo 'blocking: none' > "$pc"
st $PC; check precond-pass "$?" 0
cp "$(command -v tail)" "$T/bin/rcsr"
"$T/bin/rcsr" -f /dev/null &
other=$!
pids+=("$other")
check precond-other-rcsr-running "$(test "$(ps -eo comm= | grep -cx rcsr)" -ge 1 && echo yes)" yes
st $PC; check precond-other-rcsr-process-pass "$?" 0
echo 'blocking: #7' > "$pc"
echo CLOSED > "$T/gh/issue-7"
st $PC; check precond-blocker-closed-pass "$?" 0
echo OPEN > "$T/gh/issue-7"
st $PC; check precond-blocker-open-fail "$?" 1
rm "$T/gh/issue-7"
echo 'parked: #3' > "$pc"
st $PC; check precond-no-blocking-line-fail "$?" 1
echo 'blocking: none' > "$pc"
echo "$other" > "$T/run/run.pid"
st $PC; check precond-run-process-alive-fail "$?" 1
kill "$other"; wait "$other" 2> /dev/null
st $PC; check precond-run-process-gone-pass "$?" 0
rm "$T/run/run.pid"
git_q -C "$T/run" commit -q --allow-empty -m ahead
st $PC; check precond-binary-not-at-base-fail "$?" 1
git_q -C "$T/run" reset -q --hard origin/integ
echo bad > "$T/run/bin/rcsr.sha256"
st $PC; check precond-bad-pin-fail "$?" 1

# 03_dispatch and 05_receive_and_verify: an empty file fails
DP=03_orchestrate/stages/03_dispatch
mkdir -p "$V/$DP/output"
d=$V/$DP/output/dispatch.md
rows 2 > "$w"
printf '| #1 | agent-a | sonnet |\n| #2 | agent-b | opus |\n' > "$d"
st $DP; check dispatch-pass "$?" 0
: > "$d"; st $DP; check dispatch-empty-fail "$?" 1
printf '| #1 | agent-a | sonnet |\n' > "$d"; st $DP; check dispatch-missing-row-fail "$?" 1
printf '| #1 | agent-a | sonnet |\n| #2 | agent-b | opus |\n' > "$d"
: > "$w"; st $DP; check dispatch-empty-wave-fail "$?" 1
echo ghsha > "$T/gh/head-9"
echo '| #5 | 9 | ghsha | ghsha |' > "$vf"
st $RV; check receive-pass "$?" 0
echo '| #5 | 9 | ghsha | other |' > "$vf"
st $RV; check receive-mismatch-fail "$?" 1
: > "$vf"; st $RV; check receive-empty-fail "$?" 1

# 02_launch: the units of this run come from run.pid; another rcsr process does not count
LA=07_longrun/stages/02_launch
mkdir -p "$V/$LA/output"
printf -- '--cpu --run-manifest --json-out --progress --dev-log\n' > "$V/$LA/output/dry.txt"
echo 'windows: 2' > "$V/$LA/output/launch.md"
"$T/bin/rcsr" -f /dev/null &
u1=$!
"$T/bin/rcsr" -f /dev/null &
u2=$!
"$T/bin/rcsr" -f /dev/null &
u3=$!
pids+=("$u1" "$u2" "$u3")
printf '%s\n%s\n' "$u1" "$u2" > "$T/run/run.pid"
st $LA; check launch-other-rcsr-process-pass "$?" 0
printf '%s\n' "$u1" > "$T/run/run.pid"
st $LA; check launch-one-of-two-fail "$?" 1
printf '%s\n%s\n%s\n' "$u1" "$u2" "$u3" > "$T/run/run.pid"
st $LA; check launch-three-live-pids-fail "$?" 1
rm "$T/run/run.pid"
st $LA; check launch-no-pid-file-fail "$?" 1
printf '%s\n%s\n' "$u1" "$u2" > "$T/run/run.pid"
echo '--cpu' > "$V/$LA/output/dry.txt"
st $LA; check launch-dry-run-flags-missing-fail "$?" 1
kill "$u1" "$u2" "$u3"; wait "$u1" "$u2" "$u3" 2> /dev/null
rm -f "$T/run/run.pid"

# the process step of 04_tested_vs_merged names origin/$BASE
TV=$REPO/verksted/04_gate/stages/04_tested_vs_merged/CONTEXT.md
check tested-step-no-local-base "$(grep -c 'git diff --stat <sha> "\$BASE"' "$TV")" 0
check tested-step-origin-base "$(grep -c 'git diff --stat <sha> "origin/\$BASE"' "$TV")" 1

if [ "$fails" -ne 0 ]; then
  echo "test-seven-checks: $fails failed"
  exit 1
fi
echo "test-seven-checks: ok"
