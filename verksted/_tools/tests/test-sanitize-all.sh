#!/usr/bin/env bash
# Usage: test-sanitize-all.sh
# Hermetic test of sanitize-all.sh with a stub slot, lock, cargo, sanitizer and Xid command.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/fakes.sh"
ALL=$HERE/../sanitize-all.sh
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
has() {
  local name=$1 file=$2 pattern=$3
  if ! grep -qE -- "$pattern" "$file"; then
    echo "FAIL $name: '$pattern' not in $file"
    fails=$((fails + 1))
  fi
}
hasnt() {
  local name=$1 file=$2 pattern=$3
  if grep -qE -- "$pattern" "$file"; then
    echo "FAIL $name: '$pattern' found in $file"
    fails=$((fails + 1))
  fi
}

mkdir -p "$T/gates" "$T/wt/my-version" "$T/bin" "$T/cuda-13.0/bin"
touch "$T/wt/my-version/Cargo.toml"
cat > "$T/plan" <<'PLAN'
# comment line
gpu_parity      *            skip           device fault under memcheck
gpu_big         slow_case    skip           too slow under memcheck
gpu_memory_plan refused_alloc api-errors-no  provokes an out-of-memory on purpose
PLAN
echo 4 > "$T/xid"
for kind in slot lock; do
  bump=; [ "$kind" = lock ] && bump="case \"\$*\" in *\"\${FAKE_XID_BUMP_ON:-@none@}\"*) echo 5 > $T/xid ;; esac\n"
  printf '#!/usr/bin/env bash\necho "%s $*" >> "%s/calls"\n%bexec "$@"\n' "$kind" "$T" "$bump" > "$T/$kind.sh"
  chmod +x "$T/$kind.sh"
done
cat > "$T/bin/cargo" <<'STUB'
#!/usr/bin/env bash
echo "cargo $*" >> "$FAKE_DIR/cargo-calls"
art() { echo "{\"reason\":\"compiler-artifact\",\"target\":{\"name\":\"$1\",\"kind\":[\"$2\"]},\"profile\":{\"test\":$3},\"executable\":$4}"; }
art rcsr lib false null
art rcsr lib true '"/fake/deps/rcsr-aaa"'
art rcsr bin true '"/fake/deps/rcsr_bin-bbb"'
art gpu_parity test true '"/fake/deps/gpu_parity-ccc"'
art gpu_big test true '"/fake/deps/gpu_big-ddd"'
art gpu_memory_plan test true '"/fake/deps/gpu_memory_plan-eee"'
art gpu_fit test true '"/fake/deps/gpu_fit-fff"'
echo '{"reason":"build-finished","success":true}'
exit "${FAKE_BUILD_RC:-0}"
STUB
chmod +x "$T/bin/cargo"
fake_timeout "$T/bin"
cat > "$T/cuda-13.0/bin/compute-sanitizer" <<'STUB'
#!/usr/bin/env bash
echo "san under=${RCSR_UNDER_SANITIZER:-} $*" >> "$FAKE_DIR/san-calls"
case "$*" in
  *"${FAKE_NC_ON:-@none@}"*)
    case "${FAKE_NC_MODE:-ok}" in
      failed) echo "running 17 tests"; echo "test result: FAILED. 16 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out;" ;;
      notr) echo "running 17 tests" ;;
      ignored) echo "running 0 tests"; echo "test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out;" ;;
      *) echo "running 17 tests"; echo "test result: ok. 17 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;" ;;
    esac
    [ "${FAKE_NC_MODE:-ok}" = nomsg ] || echo "========= Error: Target application terminated before first instrumented API call"
    exit "${FAKE_NC_RC:-255}"
    ;;
  *"${FAKE_ZERO_ON:-@none@}"*)
    case "${FAKE_ZERO_MODE:-plan}" in
      run1) echo "running 1 test"; echo "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;" ;;
      nofilter) echo "running 0 tests"; echo "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;" ;;
      *) echo "running 0 tests"; echo "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out;" ;;
    esac
    [ "${FAKE_ZERO_MODE:-plan}" = nomsg ] || echo "========= Error: Target application terminated before first instrumented API call"
    exit "${FAKE_ZERO_RC:-255}"
    ;;
esac
echo "running 1 test"
echo "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;"
rc=0
summary="ERROR SUMMARY: 0 errors"
case "$*" in
  *"${FAKE_FAIL_ON:-@none@}"*) rc=${FAKE_SAN_RC:-99}; summary="${FAKE_SUMMARY:-ERROR SUMMARY: 2 errors}" ;;
esac
echo "========= $summary"
exit "$rc"
STUB
chmod +x "$T/cuda-13.0/bin/compute-sanitizer"

run() {
  : > "$T/calls"; : > "$T/san-calls"; : > "$T/cargo-calls"; : > "$T/timeouts"
  env FAKE_DIR="$T" RCSR_GATES="$T/gates" RCSR_SLOT_SCRIPT="$T/slot.sh" RCSR_LOCK_SCRIPT="$T/lock.sh" \
    RCSR_XID_CMD="cat '$T/xid'" RCSR_SANITIZER="$T/cuda-13.0/bin/compute-sanitizer" RCSR_SANITIZER_PLAN="${PLAN:-$T/plan}" \
    FAKE_TIMEOUT_LOG="$T/timeouts" PATH="$T/bin:$PATH" bash "$ALL" "$@"
}
sum=$T/gates/issue9-san-summary.log

# full pass: skips, a named skip, the api-errors-no second run, the summary format
run "$T/wt" 9 > "$T/out" 2>&1; r=$?
check full-rc "$r" "0"
has full-cargo "$T/cargo-calls" 'cargo test --release --features cuda --no-run --message-format=json'
has full-star-skip "$sum" '^gpu_parity skip \| device fault under memcheck$'
has full-named-skip "$sum" '^gpu_big:slow_case skip \| too slow under memcheck$'
hasnt full-star-not-run "$T/san-calls" 'gpu_parity-ccc'
has full-named-skip-arg "$T/san-calls" 'san under=memcheck --tool memcheck --leak-check no --error-exitcode 99 /fake/deps/gpu_big-ddd --skip slow_case --exact$'
has full-api-skip-arg "$T/san-calls" 'error-exitcode 99 /fake/deps/gpu_memory_plan-eee --skip refused_alloc --exact$'
has full-api-run "$T/san-calls" 'error-exitcode 99 --report-api-errors no /fake/deps/gpu_memory_plan-eee refused_alloc --exact$'
has full-plain "$T/san-calls" 'error-exitcode 99 /fake/deps/gpu_fit-fff$'
has full-lib "$T/san-calls" 'error-exitcode 99 /fake/deps/rcsr-aaa$'
has full-bin "$T/san-calls" 'error-exitcode 99 /fake/deps/rcsr_bin-bbb$'
has full-lock "$T/calls" '^lock .*timeout 580 '
has full-line "$sum" '^gpu_fit rc=0 xid=4->4 secs=[0-9]+ \| test result: ok\. 1 passed.* \| =+ ERROR SUMMARY: 0 errors$'
has full-api-line "$sum" '^gpu_memory_plan:refused_alloc rc=0 xid=4->4 secs='
check full-run-lines "$(grep -c ' rc=' "$sum")" "6"
check full-limits "$(grep -c "^580 .*compute-sanitizer" "$T/timeouts")" "$(wc -l < "$T/san-calls" | tr -d " ")"
check full-limits-all "$(grep -c compute-sanitizer "$T/timeouts")" "$(wc -l < "$T/san-calls" | tr -d " ")"

# --only: two binaries, four-style runs for the api-errors-no binary
run "$T/wt" 9 --only gpu_memory_plan --only gpu_fit > /dev/null 2>&1; r=$?
check only-rc "$r" "0"
check only-san-calls "$(wc -l < "$T/san-calls" | tr -d ' ')" "3"
hasnt only-others "$T/san-calls" 'rcsr-aaa|gpu_big'
hasnt only-no-skip-lines "$sum" 'skip \|'

# an error summary or nonzero rc is a failure, other binaries still run
FAKE_FAIL_ON=gpu_fit-fff run "$T/wt" 9 > /dev/null 2>&1; r=$?
check err-rc "$r" "1"
has err-line "$sum" '^gpu_fit rc=99 .*ERROR SUMMARY: 2 errors$'
has err-others-ran "$sum" '^gpu_memory_plan rc=0'

# a timeout is a budget failure
FAKE_FAIL_ON=gpu_big-ddd FAKE_SAN_RC=124 FAKE_SUMMARY='no summary' run "$T/wt" 9 > /dev/null 2>&1; r=$?
check budget-rc "$r" "1"
has budget-line "$sum" '^gpu_big rc=124 .* \| budget$'

# a binary whose tests are all filtered out by the plan makes the sanitizer exit 255: skipped by plan, not a failure
FAKE_ZERO_ON=gpu_fit-fff run "$T/wt" 9 > /dev/null 2>&1; r=$?
check zero-plan-rc "$r" "0"
has zero-plan-line "$sum" '^gpu_fit skipped by plan \| 0 tests ran, all filtered out'
hasnt zero-plan-not-run-line "$sum" '^gpu_fit rc='
has zero-plan-others-ran "$sum" '^gpu_memory_plan rc=0'
# real failures with the same exit code stay failures
for mode in run1 nofilter nomsg; do
  FAKE_ZERO_MODE=$mode FAKE_ZERO_ON=gpu_fit-fff run "$T/wt" 9 > /dev/null 2>&1; r=$?
  check "zero-$mode-rc" "$r" "1"
  case $mode in
    nomsg) has "zero-$mode-line" "$sum" '^gpu_fit rc=255 ' ;;
    *) has "zero-$mode-line" "$sum" '^gpu_fit unexpected no CUDA call: add a plan line or fix the test' ;;
  esac
done
FAKE_ZERO_RC=99 FAKE_ZERO_ON=gpu_fit-fff run "$T/wt" 9 > /dev/null 2>&1; r=$?
check zero-rc99-rc "$r" "1"

# no CUDA call: rc 255 with the sanitizer message and passing tests is a pass only for a binary listed `* no-cuda`
cp "$T/plan" "$T/plan-nc"
echo 'gpu_fit * no-cuda pure CPU: no CUDA call by design' >> "$T/plan-nc"
FAKE_NC_ON=gpu_fit-fff run "$T/wt" 9 > /dev/null 2>&1; r=$?
check nc-unlisted-rc "$r" "1"
has nc-unlisted-line "$sum" '^gpu_fit unexpected no CUDA call: add a plan line or fix the test \| test result: ok\. 17 passed'
hasnt nc-unlisted-not-listed "$sum" 'no CUDA call \(listed\)'
has nc-unlisted-count "$sum" '^unexpected no-cuda: 1$'
has nc-unlisted-others-ran "$sum" '^gpu_memory_plan rc=0'
PLAN=$T/plan-nc FAKE_NC_ON=gpu_fit-fff run "$T/wt" 9 > "$T/out" 2>&1; r=$?
check nc-listed-rc "$r" "0"
has nc-listed-line "$sum" '^gpu_fit no CUDA call \(listed\) \| test result: ok\. 17 passed; 0 failed'
hasnt nc-listed-no-rc-line "$sum" '^gpu_fit rc='
has nc-listed-count "$sum" '^no-cuda binaries: 1$'
has nc-listed-unexpected-count "$sum" '^unexpected no-cuda: 0$'
has nc-listed-called-count "$sum" '^listed no-cuda but called CUDA: 0$'
has nc-listed-stdout "$T/out" '^no-cuda binaries: 1$'
has nc-listed-runs-whole-binary "$T/san-calls" 'error-exitcode 99 /fake/deps/gpu_fit-fff$'
# a listed binary whose only test is ignored ran no test and made no CUDA call
PLAN=$T/plan-nc FAKE_NC_MODE=ignored FAKE_NC_ON=gpu_fit-fff run "$T/wt" 9 > /dev/null 2>&1; r=$?
check nc-ignored-rc "$r" "0"
has nc-ignored-line "$sum" '^gpu_fit no CUDA call \(listed\) \| test result: ok\. 0 passed; 0 failed; 1 ignored'
# a listed binary that calls CUDA gets the memcheck verdict and the summary says so
PLAN=$T/plan-nc run "$T/wt" 9 > /dev/null 2>&1; r=$?
check nc-called-rc "$r" "0"
has nc-called-line "$sum" '^gpu_fit rc=0 .*ERROR SUMMARY: 0 errors \| listed no-cuda but called CUDA$'
has nc-called-count "$sum" '^listed no-cuda but called CUDA: 1$'
has nc-called-zero-listed "$sum" '^no-cuda binaries: 0$'
PLAN=$T/plan-nc FAKE_FAIL_ON=gpu_fit-fff run "$T/wt" 9 > /dev/null 2>&1; r=$?
check nc-called-errors-rc "$r" "1"
has nc-called-errors-line "$sum" '^gpu_fit rc=99 .*ERROR SUMMARY: 2 errors \| listed no-cuda but called CUDA$'
# everything else stays a failure for a listed binary
for mode in failed nomsg notr; do
  PLAN=$T/plan-nc FAKE_NC_MODE=$mode FAKE_NC_ON=gpu_fit-fff run "$T/wt" 9 > /dev/null 2>&1; r=$?
  check "nc-$mode-rc" "$r" "1"
  has "nc-$mode-line" "$sum" '^gpu_fit rc=255 '
  hasnt "nc-$mode-not-listed-pass" "$sum" 'no CUDA call \(listed\)'
done
PLAN=$T/plan-nc FAKE_NC_RC=99 FAKE_NC_ON=gpu_fit-fff run "$T/wt" 9 > /dev/null 2>&1; r=$?
check nc-rc99-rc "$r" "1"
PLAN=$T/plan-nc FAKE_NC_RC=139 FAKE_NC_ON=gpu_fit-fff run "$T/wt" 9 > /dev/null 2>&1; r=$?
check nc-rc139-rc "$r" "1"
has nc-rc139-line "$sum" '^gpu_fit rc=139 '
PLAN=$T/plan-nc FAKE_NC_RC=124 FAKE_NC_ON=gpu_fit-fff run "$T/wt" 9 > /dev/null 2>&1; r=$?
check nc-budget-rc "$r" "1"
has nc-budget-line "$sum" '^gpu_fit rc=124 .* \| budget$'
# the directive takes only `*`; a named test with no-cuda is a plan error
printf 'gpu_fit some_test no-cuda reason\n' > "$T/plan-bad"
PLAN=$T/plan-bad run "$T/wt" 9 > "$T/out" 2>&1; r=$?
check nc-named-rc "$r" "2"
has nc-named-msg "$T/out" "no-cuda needs '\*'"

# a rising Xid count stops at once with 70
echo 4 > "$T/xid"
FAKE_XID_BUMP_ON=gpu_big-ddd run "$T/wt" 9 > /dev/null 2>&1; r=$?
check xid-rc "$r" "70"
has xid-line "$sum" '^gpu_big rc=70 xid=4->5 '
hasnt xid-stopped "$sum" '^gpu_memory_plan rc='
echo 4 > "$T/xid"

# --dry-run builds and runs nothing and lists skips and commands
run "$T/wt" 9 --dry-run > "$T/out" 2>&1; r=$?
check dry-rc "$r" "0"
check dry-no-cargo "$(wc -c < "$T/cargo-calls" | tr -d ' ')" "0"
check dry-no-san "$(wc -c < "$T/san-calls" | tr -d ' ')" "0"
has dry-skip "$T/out" '^gpu_parity skip \| device fault under memcheck$'
has dry-api "$T/out" 'run gpu_memory_plan:refused_alloc: .*--report-api-errors no .*refused_alloc --exact$'

# a missing binary named by --only fails; a failed build gives 1
run "$T/wt" 9 --only nothing_here > /dev/null 2>&1; r=$?
check missing-only-rc "$r" "1"
FAKE_BUILD_RC=101 run "$T/wt" 9 > /dev/null 2>&1; r=$?
check build-fail-rc "$r" "1"

# usage errors give 2; a sanitizer outside CUDA 13 gives 3
run "$T/wt" > /dev/null 2>&1; r=$?
check usage-rc "$r" "2"
run "$T/wt" 9 --bogus > /dev/null 2>&1; r=$?
check bogus-rc "$r" "2"
RCSR_SANITIZER_PLAN="$T/plan" RCSR_SANITIZER=/nonexistent/compute-sanitizer bash "$ALL" "$T/wt" 9 > /dev/null 2>&1; r=$?
check san-rc "$r" "3"

check help-rc "$(bash "$ALL" -h > "$T/h"; echo $?)" "0"
has help-usage "$T/h" '^Usage: sanitize-all.sh <worktree>'

if [ "$fails" -ne 0 ]; then
  echo "test-sanitize-all: $fails failed"
  exit 1
fi
echo "test-sanitize-all: ok"
