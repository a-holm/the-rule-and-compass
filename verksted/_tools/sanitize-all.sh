#!/usr/bin/env bash
# Usage: sanitize-all.sh <worktree> <id> [--only <binary>]... [--dry-run]
# The whole memcheck pass from <worktree>/<crate dir>/tests/sanitizer.plan (override: RCSR_SANITIZER_PLAN).
# Builds once in the build slot (cargo test --release --features cuda --no-run --message-format=json), lists every test
# executable (lib, bin-<name>, or the --test target name), then runs each one under the CUDA 13.0 memcheck through the
# test lock, one lock hold per run, rc-run.sh run --cuda --timeout 580:
#   env RCSR_UNDER_SANITIZER=memcheck RCSR_REQUIRE_DEVICE=1 $SANITIZER --tool memcheck --leak-check no --error-exitcode 99 <bin> --skip <test> ... --exact
# Plan modes: `<binary> * skip` runs nothing; `<binary> <test> skip` adds --skip <test>; `<binary> <test> api-errors-no`
# is skipped in the binary run and gets its own run with --report-api-errors no and `--exact <test>`.
# --only limits the pass to the named binaries. --dry-run prints the planned commands, builds and runs nothing.
# A run that hits the 580 s timeout is a failure labelled `budget`. The timeout is never raised: size the fixture or add a plan line.
# A run that ends with rc 139 (compute-sanitizer itself died with SIGSEGV) is a failure labelled `tool crash`, written without a test verdict and never retried.
# Writes: one log per run (<id>-san-<binary>[-<test>]), the build log <id>-san-build, and the summary <id>-san-summary
# with one line per run: <binary>[:<test>] rc=<n> xid=<new lines> secs=<s> | <test result line> | <ERROR SUMMARY line>,
# plus `<binary>[:<test>] skip | <reason>` lines.
# Plan mode `<cargo bin name> * run-bin <arguments>` (for example `rcsr-gpu-bench * run-bin --batch 64 --reps 1 --depth 3`) runs the release binary itself under memcheck, not its unit tests: it builds `cargo build --release --features cuda --bin <name>` and runs that executable with the arguments in a lock hold of its own, label `<name>`. `--only <name>` runs just that line and skips the test-binary build. The unit-test executable of the same binary (`bin-<name>`) has its own `skip` line. A run-bin run uses `--target-processes application-only` and must start no CUDA-using child. A name may have several run-bin lines: the first logs as `<name>`, later ones as `<name>-<n>`.
# Plan mode `<binary> * no-cuda <reason>` lists a binary that makes no CUDA call by design. It still runs under memcheck.
# Rc 255 with `terminated before first instrumented API call` and `test result: ok` (0 failed) is a pass only for a listed binary
# (`<binary> no CUDA call (listed)`); for an unlisted binary it is a failure (`<binary> unexpected no CUDA call: add a plan line or fix the test`).
# Plan mode `<binary> * no-cuda-app-only <reason>` is `no-cuda` plus `--target-processes application-only`: only for a binary whose child processes make no CUDA call either (children are then not traced). With it the sanitizer message `No attachable process found` (launch timeout) also counts as no CUDA call.
# Plan mode `<binary> * app-only <reason>` runs the test process under memcheck with `--target-processes application-only` and gives the normal verdict (rc 0 and `ERROR SUMMARY: 0 errors`): for a binary whose test process makes CUDA calls but whose children must not be traced (the sanitizer crashes on them or they are not meant to run on the device). The reason names what stays unsanitized.
# Plan mode `<binary> * device <reason>` lists a binary that calls CUDA by design; the runner treats it exactly like an unlisted binary (default --target-processes all, normal verdict, not counted as no-cuda). It may be combined with named `skip` and `api-errors-no` lines. tests/sanitizer_plan.rs requires exactly one `*` line per test executable.
# A listed binary whose run printed an ERROR SUMMARY called CUDA: it gets the normal verdict and the line ends `listed no-cuda but called CUDA`.
# Plan mode `<binary> shard=<name> shard <prefix>[,<prefix>...]|rest <reason>` (lib and bin-<name> only, next to the binary's one `*` line) splits one binary into several runs, each its own lock hold with the log <id>-san-<binary>-shard-<name>. The runner lists the binary with --list, a shard takes every test whose full path starts with one of its prefixes (a prefix ends in ::), `rest` takes every test no shard claims, and each run passes the exact names with --exact. A bare test name on a skip or api-errors-no line resolves to the one full path that ends in ::<name>; no match or two matches fail the pass. A shard with no test, or a test claimed by two shards, fails the pass.
# The summary ends with the counts `no-cuda binaries`, `unexpected no-cuda` and `listed no-cuda but called CUDA`.
# A run with rc 255, `running 0 tests`, a non-zero filtered-out count and the sanitizer message `terminated before first instrumented API call` (the plan filtered out every test) is written as `<binary> skipped by plan | ...` and is not a failure.
# Exit: 0 when every run has rc 0 and `ERROR SUMMARY: 0 errors`; 70 at once when a new Xid line appeared or the Xid check failed; 1 otherwise; 3 when $SANITIZER is missing or not CUDA 13; 2 usage.
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
C=$(crate_dir "$wt") || { echo "sanitize-all: no single engine crate under $wt" >&2; exit 2; }
plan=${RCSR_SANITIZER_PLAN:-$C/tests/sanitizer.plan}
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
  case "$m" in skip|api-errors-no) ;; shard) case "$t" in shard=?*) ;; *) echo "sanitize-all: shard needs shard=<name> as the test, got '$t' for $b in $plan" >&2; exit 2 ;; esac ;; no-cuda|no-cuda-app-only|app-only|device|run-bin) [ "$t" = "*" ] || { echo "sanitize-all: $m needs '*' as the test, got '$t' for $b in $plan" >&2; exit 2; } ;; *) echo "sanitize-all: bad mode '$m' in $plan" >&2; exit 2 ;; esac
  [ -n "$r" ] || { echo "sanitize-all: empty reason for $b $t in $plan" >&2; exit 2; }
done <<< "$plan_lines"
run_bin_names=$(printf '%s\n' "$plan_lines" | awk -F'\t' '$3 == "run-bin" && !seen[$1]++ { print $1 }')

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
  xid=$(sed -n 's/^xid new=\([0-9]*\) since=.*$/\1/p' "$log" | tail -1)
  res=$(grep -a '^test result:' "$log" | tail -1)
  err=$(grep -a 'ERROR SUMMARY:' "$log" | tail -1)
  if [ "$rc" -eq 124 ]; then
    extra=" | budget"
  fi
  if [ "$rc" -eq 139 ]; then
    emit "$label tool crash rc=139 xid=${xid:-?} secs=$((t1 - t0)) | compute-sanitizer died with SIGSEGV; not retried | ${res:-no test result}"
    fails=$((fails + 1))
    return 0
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
  [ "$rc" -ne 70 ] || { echo "sanitize-all: new Xid line during $label (or Xid check failed); stopped" >&2; exit 70; }
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

plan_shards_for() { # binary -> shard names of its `shard=` lines
  printf '%s\n' "$plan_lines" | awk -F'\t' -v b="$1" '$1 == b && $3 == "shard" { sub(/^shard=/, "", $2); print $2 }'
}
plan_shard_spec() { # binary shard -> the comma list of prefixes (or `rest`), the first word of the reason
  printf '%s\n' "$plan_lines" | awk -F'\t' -v b="$1" -v s="shard=$2" '$1 == b && $2 == s && $3 == "shard" { split($4, a, " "); print a[1] }'
}
list_tests() { # test binary -> one full libtest name per line
  "$1" --list 2> /dev/null | sed -n 's/: test$//p'
}
resolve_test() { # list file, test -> the one full libtest name that is the test or ends in ::<test>; empty unless exactly one
  awk -v t="$2" '{ n = length($0); k = length(t) } $0 == t || (n > k + 2 && substr($0, n - k - 1) == "::" t) { print; c++ } END { if (c != 1) exit 1 }' "$1"
}
shard_select() { # list file, prefix comma list or `rest`, every shard prefix list, excluded names file -> selected names
  awk -v spec="$2" -v all="$3" -v exf="$4" '
    BEGIN { while ((getline l < exf) > 0) ex[l] = 1; np = split(all, a, ","); ns = split(spec, s, ",") }
    function claimed(name,   i) { for (i = 1; i <= np; i++) if (a[i] != "rest" && index(name, a[i]) == 1) return 1; return 0 }
    function mine(name,   i) { for (i = 1; i <= ns; i++) if (index(name, s[i]) == 1) return 1; return 0 }
    !($0 in ex) { if (spec == "rest" ? !claimed($0) : mine($0)) print }' "$1"
}
shard_overlaps() { # list file, one shard prefix list per line -> names claimed by more than one shard
  awk -v all="$2" 'BEGIN { ns = split(all, sh, ";") } { c = 0; for (j = 1; j <= ns; j++) { np = split(sh[j], p, ","); for (i = 1; i <= np; i++) if (p[i] != "rest" && index($0, p[i]) == 1) { c++; break } } if (c > 1) print }' "$1"
}

run_sharded() { # name bin (target_args and listed are set by run_binary)
  local name=$1 bin=$2 s spec all= prefixes= names=() ex sel n t full apifull=() list= mode r
  while IFS= read -r s; do
    spec=$(plan_shard_spec "$name" "$s")
    all="$all${all:+,}$spec"
    [ "$spec" = rest ] || prefixes="$prefixes${prefixes:+;}$spec"
  done <<< "$(plan_shards_for "$name")"
  if [ "$dry" -eq 1 ]; then
    while IFS= read -r s; do
      run_one "$name:shard=$s" "san-$name-shard-$s" "" "$bin" "<tests of shard $s: $(plan_shard_spec "$name" "$s")>" --exact
    done <<< "$(plan_shards_for "$name")"
    return 0
  fi
  list=$(mktemp)
  ex=$(mktemp)
  list_tests "$bin" > "$list"
  if [ ! -s "$list" ]; then
    emit "$name missing | $bin --list printed no tests"
    fails=$((fails + 1))
    rm -f "$list" "$ex"
    return 0
  fi
  while IFS=$'\t' read -r b t mode r; do
    [ "$b" = "$name" ] && { [ "$mode" = skip ] || [ "$mode" = api-errors-no ]; } && [ "$t" != "*" ] || continue
    if full=$(resolve_test "$list" "$t"); then
      echo "$full" >> "$ex"
      [ "$mode" != api-errors-no ] || apifull+=("$full")
    else
      emit "$name:$t missing | no single test named or ending in ::$t in $bin --list"
      fails=$((fails + 1))
    fi
  done <<< "$plan_lines"
  n=$(shard_overlaps "$list" "$prefixes" | wc -l)
  if [ "$n" -ne 0 ]; then
    emit "$name overlap | $n tests match the prefixes of more than one shard"
    fails=$((fails + 1))
  fi
  while IFS= read -r s; do
    spec=$(plan_shard_spec "$name" "$s")
    mapfile -t names < <(shard_select "$list" "$spec" "$all" "$ex")
    if [ ${#names[@]} -eq 0 ]; then
      emit "$name:shard=$s empty | no test matches $spec in $bin --list"
      fails=$((fails + 1))
      continue
    fi
    run_one "$name:shard=$s" "san-$name-shard-$s" "" "$bin" "${names[@]}" --exact
  done <<< "$(plan_shards_for "$name")"
  for full in ${apifull[@]+"${apifull[@]}"}; do
    run_one "$name:$full" "san-$name-${full//::/-}" no "$bin" "$full" --exact
  done
  rm -f "$list" "$ex"
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
    app-only) target_args=(--target-processes application-only) ;;
    device) ;;
  esac
  if [ "$star" = api-errors-no ]; then
    run_one "$name" "san-$name" no "$bin"
    return 0
  fi
  if [ -n "$(plan_shards_for "$name")" ]; then
    run_sharded "$name" "$bin"
    return 0
  fi
  while IFS=$'\t' read -r b t mode r; do
    [ "$b" = "$name" ] && [ "$mode" != no-cuda ] && [ "$mode" != no-cuda-app-only ] && [ "$mode" != app-only ] && [ "$mode" != device ] || continue
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

is_run_bin() { printf '%s\n' "$run_bin_names" | grep -qx -- "$1"; }
run_bin_args() { printf '%s\n' "$plan_lines" | awk -F'\t' -v b="$1" '$1 == b && $3 == "run-bin" { print $4 }'; }
run_bin_tag() { if [ "$2" -eq 1 ]; then printf '%s' "$1"; else printf '%s-%s' "$1" "$2"; fi; }

# only_run_bins: --only names nothing but run-bin lines, so the test binaries are not built
only_run_bins=0
if [ ${#only[@]} -gt 0 ]; then
  only_run_bins=1
  for o in "${only[@]}"; do is_run_bin "$o" || only_run_bins=0; done
fi

if [ "$dry" -eq 1 ]; then
  if [ "$only_run_bins" -eq 0 ]; then
    echo "build: cargo test --release --features cuda --no-run --message-format=json"
    names=$(printf '%s\n' "$plan_lines" | awk -F'\t' '$1 != "" && $3 != "run-bin" { print $1 }' | sort -u)
    for n in $names; do
      selected "$n" || continue
      [ "$(plan_star_mode "$n")" != skip ] || continue
      run_binary "$n" "<$n test binary>"
    done
  fi
  for n in $run_bin_names; do
    selected "$n" || continue
    echo "build: cargo build --release --features cuda --bin $n --message-format=json"
    k=0
    while IFS= read -r line; do
      k=$((k + 1))
      read -r -a ra <<< "$line"
      listed=0
      target_args=(--target-processes application-only)
      run_one "$(run_bin_tag "$n" "$k")" "san-$(run_bin_tag "$n" "$k")" "" "<$n release binary>" "${ra[@]}"
    done <<< "$(run_bin_args "$n")"
  done
  [ "$only_run_bins" -eq 1 ] || echo "every other test binary: $(san_cmd "" "<binary>")"
  exit 0
fi

if [ "$only_run_bins" -eq 0 ]; then
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
    is_run_bin "$o" && continue
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
fi

# run-bin lines: the release binary itself runs under memcheck with the planned arguments
for n in $run_bin_names; do
  selected "$n" || continue
  bt=san-build-$n
  "$RC_RUN" build "$wt" "$id" "$bt" --cuda -- \
    cargo build --release --features cuda --bin "$n" --message-format=json > /dev/null
  brc=$?
  if [ "$brc" -ne 0 ]; then
    emit "$n build failed | rc=$brc | see $(log_path "$id" "$bt")"
    fails=$((fails + 1))
    continue
  fi
  exe=$(grep -a '^{' "$(log_path "$id" "$bt")" | jq -r --arg n "$n" '
    select(.reason == "compiler-artifact" and .profile.test == false and (.target.kind | index("bin")) and .target.name == $n and .executable != null)
    | .executable' 2>/dev/null | sort -u)
  if [ "$(printf '%s' "$exe" | grep -c .)" -ne 1 ]; then
    emit "$n missing | expected one release executable, found: $(printf '%s' "$exe" | tr '\n' ' ')"
    fails=$((fails + 1))
    continue
  fi
  k=0
  while IFS= read -r line; do
    k=$((k + 1))
    read -r -a ra <<< "$line"
    listed=0
    target_args=(--target-processes application-only)
    run_one "$(run_bin_tag "$n" "$k")" "san-$(run_bin_tag "$n" "$k")" "" "$exe" "${ra[@]}"
  done <<< "$(run_bin_args "$n")"
done

emit "no-cuda binaries: $n_nocuda"
emit "unexpected no-cuda: $n_unexpected"
emit "listed no-cuda but called CUDA: $n_called"
echo "summary: $summary"
[ "$fails" -eq 0 ] || exit 1
exit 0
