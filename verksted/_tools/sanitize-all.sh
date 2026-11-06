#!/usr/bin/env bash
# Usage: sanitize-all.sh <worktree> <id> [--only <binary>]... [--dry-run]
# The whole memcheck pass from <worktree>/my-version/tests/sanitizer.plan (override: RCSR_SANITIZER_PLAN).
# Builds once in the build slot (cargo test --release --features cuda --no-run --message-format=json), lists every test
# executable (lib, bin-<name>, or the --test target name), then runs each one under the CUDA 13.0 memcheck through the
# test lock, one lock hold per run, rc-run.sh run --cuda --timeout 580:
#   env RCSR_UNDER_SANITIZER=memcheck RCSR_REQUIRE_DEVICE=1 $SANITIZER --tool memcheck --leak-check no --error-exitcode 99 <bin> --skip <test> ... --exact
# Plan modes: `<binary> * skip` runs nothing; `<binary> <test> skip` adds --skip <test>; `<binary> <test> api-errors-no`
# is skipped in the binary run and gets its own run with --report-api-errors no and `--exact <test>`.
# --only limits the pass to the named binaries. --dry-run prints the planned commands, builds and runs nothing.
# A run that hits the 580 s timeout is a failure labelled `budget`. The timeout is never raised: size the fixture or add a plan line.
# Writes: one log per run (<id>-san-<binary>[-<test>]), the build log <id>-san-build, and the summary <id>-san-summary
# with one line per run: <binary>[:<test>] rc=<n> xid=<a>-><b> secs=<s> | <test result line> | <ERROR SUMMARY line>,
# plus `<binary>[:<test>] skip | <reason>` lines.
# Plan mode `<binary> * no-cuda <reason>` lists a binary that makes no CUDA call by design. It still runs under memcheck.
# Rc 255 with `terminated before first instrumented API call` and `test result: ok` (0 failed) is a pass only for a listed binary
# (`<binary> no CUDA call (listed)`); for an unlisted binary it is a failure (`<binary> unexpected no CUDA call: add a plan line or fix the test`).
# Plan mode `<binary> * no-cuda-app-only <reason>` is `no-cuda` plus `--target-processes application-only`: only for a binary whose child processes make no CUDA call either (children are then not traced). With it the sanitizer message `No attachable process found` (launch timeout) also counts as no CUDA call.
# A listed binary whose run printed an ERROR SUMMARY called CUDA: it gets the normal verdict and the line ends `listed no-cuda but called CUDA`.
# The summary ends with the counts `no-cuda binaries`, `unexpected no-cuda` and `listed no-cuda but called CUDA`.
# A run with rc 255, `running 0 tests`, a non-zero filtered-out count and the sanitizer message `terminated before first instrumented API call` (the plan filtered out every test) is written as `<binary> skipped by plan | ...` and is not a failure.
# Exit: 0 when every run has rc 0 and `ERROR SUMMARY: 0 errors`; 70 at once when the Xid count rose; 1 otherwise; 3 when $SANITIZER is missing or not CUDA 13; 2 usage.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RC_RUN=$(dirname "${BASH_SOURCE[0]}")/rc-run.sh

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac
[ $# -ge 2 ] || { usage >&2; exit 2; }
wt=$1 id=$2
shift 2
only=()
dry=0
while [ $# -gt 0 ]; do
  case "$1" in
    --only) [ $# -ge 2 ] || { usage >&2; exit 2; }; only+=("$2"); shift 2 ;;
    --dry-run) dry=1; shift ;;
    *) usage >&2; exit 2 ;;
  esac
done
plan=${RCSR_SANITIZER_PLAN:-$wt/my-version/tests/sanitizer.plan}
[ -f "$plan" ] || { echo "sanitize-all: plan $plan missing" >&2; exit 2; }

san_real=$(realpath "$SANITIZER" 2>/dev/null || true)
case "$san_real" in
  */cuda-13*/*) ;;
  *) echo "sanitize-all: $SANITIZER is not a CUDA 13 compute-sanitizer" >&2; exit 3 ;;
esac
[ -x "$san_real" ] && [ -f "$san_real" ] || { echo "sanitize-all: $SANITIZER is not executable" >&2; exit 3; }

selected() {
  [ ${#only[@]} -eq 0 ] && return 0
  local o
  for o in "${only[@]}"; do [ "$o" = "$1" ] && return 0; done
  return 1
}

# plan_lines: binary<TAB>test<TAB>mode<TAB>reason
plan_lines=$(awk '/^[[:space:]]*(#|$)/ { next } {
  reason = ""; for (i = 4; i <= NF; i++) reason = reason (i > 4 ? " " : "") $i
  printf "%s\t%s\t%s\t%s\n", $1, $2, $3, reason }' "$plan")
while IFS=$'\t' read -r b t m r; do
  [ -n "$b" ] || continue
  case "$m" in skip|api-errors-no) ;; no-cuda|no-cuda-app-only) [ "$t" = "*" ] || { echo "sanitize-all: $m needs '*' as the test, got '$t' for $b in $plan" >&2; exit 2; } ;; *) echo "sanitize-all: bad mode '$m' in $plan" >&2; exit 2 ;; esac
  [ -n "$r" ] || { echo "sanitize-all: empty reason for $b $t in $plan" >&2; exit 2; }
done <<< "$plan_lines"

summary=$(log_path "$id" san-summary)
fails=0
ok_re='^test result: ok\. [0-9]+ passed; 0 failed;'
listed=0
target_args=()
n_nocuda=0
n_unexpected=0
n_called=0
emit() {
  echo "$1"
  [ "$dry" -eq 1 ] || echo "$1" >> "$summary"
}

san_cmd() { # <api-errors or empty> <bin> args...
  local api=$1 bin=$2
  shift 2
  printf 'env RCSR_UNDER_SANITIZER=memcheck RCSR_REQUIRE_DEVICE=1 %s --tool memcheck' "$SANITIZER"
  [ ${#target_args[@]} -eq 0 ] || printf ' %s' "${target_args[@]}"
  printf ' --leak-check no --error-exitcode 99'
  [ -z "$api" ] || printf ' --report-api-errors %s' "$api"
  printf ' %s' "$bin"
  [ $# -eq 0 ] || printf ' %s' "$@"
  printf '\n'
}

plan_skips_for() { # binary -> named skip + api-errors-no test names
  printf '%s\n' "$plan_lines" | awk -F'\t' -v b="$1" '$1 == b && $2 != "*" { print $2 }'
}
plan_star_mode() { # binary -> mode of its * line, or empty
  printf '%s\n' "$plan_lines" | awk -F'\t' -v b="$1" '$1 == b && $2 == "*" { print $3 }'
}

run_one() { # label tag api bin args...
  local label=$1 tag=$2 api=$3 bin=$4
  shift 4
  local log t0 t1 rc
  local argv=(env RCSR_UNDER_SANITIZER=memcheck RCSR_REQUIRE_DEVICE=1 "$SANITIZER" --tool memcheck ${target_args[@]+"${target_args[@]}"} --leak-check no --error-exitcode 99)
  [ -z "$api" ] || argv+=(--report-api-errors "$api")
  argv+=("$bin" "$@")
  if [ "$dry" -eq 1 ]; then
    echo "run $label: $(san_cmd "$api" "$bin" "$@")"
    return 0
  fi
  t0=$(date +%s)
  "$RC_RUN" run "$wt" "$id" "$tag" --cuda --timeout 580 -- "${argv[@]}" > /dev/null
  rc=$?
  t1=$(date +%s)
  log=$(log_path "$id" "$tag")
  local xid res err extra= by_plan=0 no_cuda=0
  xid=$(sed -n 's/^xid before=\([0-9]*\) after=\([0-9]*\)$/\1->\2/p' "$log" | tail -1)
  res=$(grep -a '^test result:' "$log" | tail -1)
  err=$(grep -a 'ERROR SUMMARY:' "$log" | tail -1)
  if [ "$rc" -eq 124 ]; then
    extra=" | budget"
  fi
  # The sanitizer exits 255 when the target makes no CUDA call, which happens when the plan filters out every test.
  if [ "$rc" -eq 255 ] && grep -aq '^running 0 tests' "$log" && grep -aEq '[1-9][0-9]* filtered out' "$log" &&
    grep -aq 'terminated before first instrumented API call' "$log"; then
    by_plan=1
    emit "$label skipped by plan | 0 tests ran, all filtered out | xid=${xid:-?}"
  elif [ "$rc" -eq 255 ] && [ -z "$err" ] && [[ $res =~ $ok_re ]] &&
    { grep -aq 'terminated before first instrumented API call' "$log" ||
      { [ ${#target_args[@]} -gt 0 ] && grep -aq 'No attachable process found' "$log"; }; }; then
    no_cuda=1
    if [ "$listed" -eq 1 ]; then
      n_nocuda=$((n_nocuda + 1))
      emit "$label no CUDA call (listed) | $res | xid=${xid:-?}"
    else
      n_unexpected=$((n_unexpected + 1))
      emit "$label unexpected no CUDA call: add a plan line or fix the test | $res | xid=${xid:-?}"
    fi
  else
    if [ "$listed" -eq 1 ] && [ -n "$err" ]; then
      extra="$extra | listed no-cuda but called CUDA"
      n_called=$((n_called + 1))
    fi
    emit "$label rc=$rc xid=${xid:-?} secs=$((t1 - t0)) | $res | $err$extra"
  fi
  [ "$rc" -ne 70 ] || { echo "sanitize-all: Xid count rose during $label; stopped" >&2; exit 70; }
  [ "$by_plan" -eq 0 ] || return 0
  if [ "$no_cuda" -eq 1 ]; then
    [ "$listed" -eq 1 ] && return 0
    fails=$((fails + 1))
    return 0
  fi
  if [ "$rc" -eq 0 ] && grep -aq 'ERROR SUMMARY: 0 errors' "$log"; then
    return 0
  fi
  fails=$((fails + 1))
}

run_binary() { # name bin
  local name=$1 bin=$2 star args=() t mode
  star=$(plan_star_mode "$name")
  if [ "$star" = skip ]; then
    return 0
  fi
  listed=0
  target_args=()
  case "$star" in
    no-cuda) listed=1 ;;
    no-cuda-app-only) listed=1; target_args=(--target-processes application-only) ;;
  esac
  if [ "$star" = api-errors-no ]; then
    run_one "$name" "san-$name" no "$bin"
    return 0
  fi
  while IFS=$'\t' read -r b t mode r; do
    [ "$b" = "$name" ] && [ "$mode" != no-cuda ] && [ "$mode" != no-cuda-app-only ] || continue
    args+=(--skip "$t")
  done <<< "$plan_lines"
  [ ${#args[@]} -eq 0 ] || args+=(--exact)
  run_one "$name" "san-$name" "" "$bin" "${args[@]}"
  while IFS=$'\t' read -r b t mode r; do
    [ "$b" = "$name" ] && [ "$mode" = api-errors-no ] || continue
    run_one "$name:$t" "san-$name-$t" no "$bin" "$t" --exact
  done <<< "$plan_lines"
}

if [ "$dry" -eq 0 ]; then
  need_gates
  : > "$summary"
fi

# skip lines first, straight from the plan
while IFS=$'\t' read -r b t m r; do
  [ -n "$b" ] && [ "$m" = skip ] && selected "$b" || continue
  if [ "$t" = "*" ]; then emit "$b skip | $r"; else emit "$b:$t skip | $r"; fi
done <<< "$plan_lines"

if [ "$dry" -eq 1 ]; then
  echo "build: cargo test --release --features cuda --no-run --message-format=json"
  names=$(printf '%s\n' "$plan_lines" | awk -F'\t' '$1 != "" { print $1 }' | sort -u)
  for n in $names; do
    selected "$n" || continue
    [ "$(plan_star_mode "$n")" != skip ] || continue
    run_binary "$n" "<$n test binary>"
  done
  echo "every other test binary: $(san_cmd "" "<binary>")"
  exit 0
fi

build_log_tag=san-build
"$RC_RUN" build "$wt" "$id" "$build_log_tag" --cuda -- \
  cargo test --release --features cuda --no-run --message-format=json > /dev/null
brc=$?
build_log=$(log_path "$id" "$build_log_tag")
if [ "$brc" -ne 0 ]; then
  echo "sanitize-all: build failed (rc=$brc); see $build_log" >&2
  exit 1
fi

bins=$(grep -a '^{' "$build_log" | jq -r '
  select(.reason == "compiler-artifact" and .profile.test == true and .executable != null)
  | (if (.target.kind | index("lib")) then "lib"
     elif (.target.kind | index("bin")) then "bin-" + .target.name
     elif (.target.kind | index("test")) then .target.name
     else empty end) + "\t" + .executable' 2>/dev/null | sort -u)
[ -n "$bins" ] || { echo "sanitize-all: no test executables in $build_log" >&2; exit 1; }

for o in ${only[@]+"${only[@]}"}; do
  if ! printf '%s\n' "$bins" | cut -f1 | grep -qx -- "$o"; then
    case "$(plan_star_mode "$o")" in skip) continue ;; esac
    emit "$o missing | no test executable with this name"
    fails=$((fails + 1))
  fi
done

while IFS=$'\t' read -r name bin; do
  [ -n "$name" ] && selected "$name" || continue
  run_binary "$name" "$bin"
done <<< "$bins"

emit "no-cuda binaries: $n_nocuda"
emit "unexpected no-cuda: $n_unexpected"
emit "listed no-cuda but called CUDA: $n_called"
echo "summary: $summary"
[ "$fails" -eq 0 ] || exit 1
exit 0
