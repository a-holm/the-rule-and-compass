#!/usr/bin/env bash
# Usage: test-sanitize.sh
# Hermetic test of sanitize.sh with a stub slot, lock, cargo, sanitizer and Xid command.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/fakes.sh"
SAN=$HERE/../sanitize.sh
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

mkdir -p "$T/gates" "$T/wt/my-version" "$T/bin" "$T/cuda-13.0/bin" "$T/cuda-12.0/bin"
touch "$T/wt/my-version/Cargo.toml"
echo 4 > "$T/xid"
for kind in slot lock; do
  bump=; [ "$kind" = lock ] && bump="[ -z \"\${FAKE_XID_BUMP:-}\" ] || echo 5 > $T/xid\n"
  printf '#!/usr/bin/env bash\necho "%s $*" >> "%s/calls"\n%bexec "$@"\n' "$kind" "$T" "$bump" > "$T/$kind.sh"
  chmod +x "$T/$kind.sh"
done
# The fake cargo prints a compiler line for the lib unit tests, one for a test file, and a non-test build.
cat > "$T/bin/cargo" <<'STUB'
#!/usr/bin/env bash
echo "cargo $*" >> "$FAKE_DIR/cargo-calls"
echo "   Compiling rcsr v0.1.0" >&2
echo '{"reason":"compiler-artifact","target":{"name":"rcsr","kind":["lib"]},"profile":{"test":false},"executable":null}'
echo '{"reason":"compiler-artifact","target":{"name":"rcsr","kind":["lib"]},"profile":{"test":true},"executable":"/fake/deps/rcsr-aaa"}'
if [ -z "${FAKE_NO_ARTIFACT:-}" ]; then
  echo '{"reason":"compiler-artifact","target":{"name":"gpu_fit","kind":["test"]},"profile":{"test":true},"executable":"/fake/deps/gpu_fit-bbb"}'
fi
if [ -n "${FAKE_TWO_ARTIFACTS:-}" ]; then
  echo '{"reason":"compiler-artifact","target":{"name":"gpu_fit","kind":["test"]},"profile":{"test":true},"executable":"/fake/deps/gpu_fit-ccc"}'
fi
echo '{"reason":"build-finished","success":true}'
exit "${FAKE_BUILD_RC:-0}"
STUB
chmod +x "$T/bin/cargo"
fake_timeout "$T/bin"
for v in 13.0 12.0; do
  cat > "$T/cuda-$v/bin/compute-sanitizer" <<'STUB'
#!/usr/bin/env bash
echo "san under=${RCSR_UNDER_SANITIZER:-} $*" >> "$FAKE_DIR/san-calls"
echo "req=${RCSR_REQUIRE_DEVICE:-}" >> "$FAKE_DIR/san-req"
[ -z "${FAKE_BANNER:-}" ] || echo "$FAKE_BANNER"
echo "${FAKE_SUMMARY:-ERROR SUMMARY: 0 errors}"
exit "${FAKE_SAN_RC:-0}"
STUB
  chmod +x "$T/cuda-$v/bin/compute-sanitizer"
done

run() {
  env FAKE_DIR="$T" RCSR_MACHINE_DIR="$T" RCSR_GATES="$T/gates" RCSR_SLOT_SCRIPT="$T/slot.sh" RCSR_LOCK_SCRIPT="$T/lock.sh" \
    RCSR_XID_CMD="cat '$T/xid'" RCSR_SANITIZER="${SANITIZER_PATH:-$T/cuda-13.0/bin/compute-sanitizer}" \
    FAKE_TIMEOUT_LOG="$T/timeouts" PATH="$T/bin:$PATH" bash "$SAN" "$@"
}
blog=$T/gates/issue7-s-build.log
rlog=$T/gates/issue7-s.log

# clean run: binary path from the JSON line, flags, filters, build in the slot, run in the lock
: > "$T/calls"; : > "$T/san-calls"; : > "$T/timeouts"
run "$T/wt" 7 s --test gpu_fit -- some_test --exact > /dev/null 2>&1; r=$?
check clean-rc "$r" "0"
has clean-slot "$T/calls" '^slot .*cargo test'
has clean-lock "$T/calls" '^lock .*compute-sanitizer'
has clean-san-args "$T/san-calls" 'san under=memcheck --tool memcheck --leak-check no --error-exitcode 99 /fake/deps/gpu_fit-bbb some_test --exact'
check clean-require-device "$(sort -u "$T/san-req" | tr "\n" " ")" "req=1 "
has clean-cargo-args "$T/cargo-calls" 'cargo test --release --features cuda --no-run --message-format=json --test gpu_fit'
has clean-summary "$rlog" '^ERROR SUMMARY: 0 errors$'
has clean-xid "$rlog" '^xid before=4 after=4$'
has clean-done "$rlog" '^RC_DONE rc=0 '
check clean-limits "$(grep -c "^580 .*compute-sanitizer" "$T/timeouts")" "$(wc -l < "$T/san-calls" | tr -d " ")"
has clean-build-done "$blog" '^RC_DONE rc=0 '

# --api-errors maps to --report-api-errors
: > "$T/san-calls"
run "$T/wt" 7 s --test gpu_fit --api-errors no -- t --exact > /dev/null 2>&1; r=$?
check api-errors-rc "$r" "0"
has api-errors-args "$T/san-calls" 'error-exitcode 99 --report-api-errors no /fake/deps/gpu_fit-bbb t --exact'
run "$T/wt" 7 s --test gpu_fit --api-errors maybe > /dev/null 2>&1; r=$?
check api-errors-bad-rc "$r" "2"

# --lib picks the lib unit-test artifact
: > "$T/san-calls"
run "$T/wt" 7 s --lib > /dev/null 2>&1; r=$?
check lib-rc "$r" "0"
has lib-bin "$T/san-calls" '/fake/deps/rcsr-aaa'
has lib-cargo "$T/cargo-calls" 'no-run --message-format=json --lib'

# errors in the sanitizer output give 1
FAKE_SUMMARY='ERROR SUMMARY: 2 errors' run "$T/wt" 7 s --test gpu_fit > /dev/null 2>&1; r=$?
check errors-rc "$r" "1"
# a nonzero sanitizer exit with a clean summary gives 1
FAKE_SAN_RC=99 run "$T/wt" 7 s --test gpu_fit > /dev/null 2>&1; r=$?
check san-rc99 "$r" "1"
# a missing summary gives 1
FAKE_SUMMARY='no summary here' run "$T/wt" 7 s --test gpu_fit > /dev/null 2>&1; r=$?
check no-summary "$r" "1"

# a zero-test run passes with a warning
out=$(FAKE_BANNER='running 0 tests' run "$T/wt" 7 s --test gpu_fit -- nomatch 2>&1); r=$?
check zero-tests-rc "$r" "0"
case "$out" in *'tests run: 0'*) ;; *) echo "FAIL zero-tests-warning: '$out'"; fails=$((fails + 1)) ;; esac

# a rising Xid count gives 70
echo 4 > "$T/xid"
FAKE_XID_BUMP=1 run "$T/wt" 7 s --test gpu_fit > /dev/null 2>&1; r=$?
check xid-rc "$r" "70"
echo 4 > "$T/xid"

# sanitizer path without /cuda-13, or missing, gives 3 and never builds
: > "$T/cargo-calls"
SANITIZER_PATH=$T/cuda-12.0/bin/compute-sanitizer run "$T/wt" 7 s --test gpu_fit > /dev/null 2>&1; r=$?
check cuda12-rc "$r" "3"
SANITIZER_PATH=$T/none/compute-sanitizer run "$T/wt" 7 s --test gpu_fit > /dev/null 2>&1; r=$?
check missing-san-rc "$r" "3"
check no-build-on-3 "$(wc -c < "$T/cargo-calls" | tr -d ' ')" "0"

# no matching artifact, two artifacts, and a failed build give 1
FAKE_NO_ARTIFACT=1 run "$T/wt" 7 s --test gpu_fit > /dev/null 2>&1; r=$?
check no-artifact-rc "$r" "1"
FAKE_TWO_ARTIFACTS=1 run "$T/wt" 7 s --test gpu_fit > /dev/null 2>&1; r=$?
check two-artifacts-rc "$r" "1"
FAKE_BUILD_RC=101 run "$T/wt" 7 s --test gpu_fit > /dev/null 2>&1; r=$?
check build-fail-rc "$r" "1"

# usage errors give 2
run "$T/wt" 7 s > /dev/null 2>&1; r=$?
check no-target-rc "$r" "2"
run "$T/wt" 7 s --test gpu_fit --timeout 900 > /dev/null 2>&1; r=$?
check timeout-rc "$r" "2"
run "$T/wt" 7 s --test a --lib > /dev/null 2>&1; r=$?
check both-targets-rc "$r" "2"

# -h prints the header
check help-rc "$(bash "$SAN" -h > "$T/h"; echo $?)" "0"
has help-usage "$T/h" '^Usage: sanitize.sh <worktree>'

if [ "$fails" -ne 0 ]; then
  echo "test-sanitize: $fails failed"
  exit 1
fi
echo "test-sanitize: ok"
