#!/usr/bin/env bash
# Usage: ab-pair.sh --label <name> --a <bin> --b <bin> [--seeds 1,2,3] [--timeout S] -- <args, {seed} replaced per run>
# Paired A/B. For each seed, one test-lock hold runs both arms back to back: A first for the 1st, 3rd, ... seed,
# B first for the 2nd, 4th, ... Each run: /usr/bin/time -f '%e %U %S %M %x' and an outer timeout S (default 280).
# The Xid count is read before and after each hold; a rise stops the run.
# Results in $G/ab-<label>/: runs.tsv (arm seed order wall_s user_s sys_s maxrss_kb rc), <arm>-s<seed>.out and .err,
# summary.txt (per arm: runs, median user_s, median wall_s, max maxrss_kb; then `ratio_user b/a <x>`).
# Log $G/ab-<label>.log ends with the RC_DONE line.
# Compare user CPU time inside these pairs only, never with older runs.
# Writes: $G/ab-<label>/ and $G/ab-<label>.log only.
# Exit: 0 every run rc 0; 1 any run failed; 70 Xid rose; 2 usage (2 x timeout must stay under 600 s; --label kebab-case).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [ "${1:-}" = __pair ]; then
  # Runs inside the lock: __pair <dir> <seed> <order> <timeout> <bin-a> <bin-b> -- <args>
  dir=$2 seed=$3 order=$4 tmo=$5 bin_a=$6 bin_b=$7
  shift 8
  args=()
  for x in "$@"; do args+=("${x//\{seed\}/$seed}"); done
  before=$(xid_count)
  for i in 0 1; do
    arm=${order:$i:1}
    if [ "$arm" = A ]; then bin=$bin_a; else bin=$bin_b; fi
    base=$dir/$arm-s$seed
    /usr/bin/time -f '%e %U %S %M %x' -o "$base.time" timeout "$tmo" "$bin" "${args[@]}" > "$base.out" 2> "$base.err"
    read -r wall user sys rss rc < <(tail -n 1 "$base.time")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$seed" "$order" "$wall" "$user" "$sys" "$rss" "$rc" >> "$dir/runs.tsv"
  done
  after=$(xid_count)
  if [ "$after" -gt "$before" ]; then
    echo "ab-pair: Xid count rose from $before to $after in the hold for seed $seed" >&2
    exit 70
  fi
  exit 0
fi

label=
bin_a=
bin_b=
seeds=1,2,3
tmo=280
while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --label) label=${2:-}; shift 2 || break ;;
    --a) bin_a=${2:-}; shift 2 || break ;;
    --b) bin_b=${2:-}; shift 2 || break ;;
    --seeds) seeds=${2:-}; shift 2 || break ;;
    --timeout) tmo=${2:-}; shift 2 || break ;;
    --) shift; break ;;
    *) echo "ab-pair: unknown argument $1" >&2; usage >&2; exit 2 ;;
  esac
done
die() { echo "ab-pair: $*" >&2; exit 2; }
if [ -z "$label" ] || [ -z "$bin_a" ] || [ -z "$bin_b" ]; then
  usage >&2
  exit 2
fi
[[ $label =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] || die "--label must be kebab-case: $label"
[[ $tmo =~ ^[1-9][0-9]*$ ]] || die "--timeout must be a positive integer: $tmo"
[ $((2 * tmo)) -lt "$LOCK_HOLD_MAX" ] || die "2 x timeout must stay under $LOCK_HOLD_MAX s: $tmo"
[[ $seeds =~ ^[0-9]+(,[0-9]+)*$ ]] || die "--seeds must be a comma list of integers: $seeds"
for b in "$bin_a" "$bin_b"; do
  [ -f "$b" ] && [ -x "$b" ] || die "not an executable file: $b"
  case $b in */target/*) echo "ab-pair: warning: $b is under a target/ dir; pin it with bin-pin.sh first" >&2 ;; esac
done
need_gates
dir=$G/ab-$label
log=$G/ab-$label.log
[ ! -e "$dir/runs.tsv" ] || die "$dir/runs.tsv exists; pick another label"
mkdir -p "$dir"
bin_a=$(cd "$(dirname "$bin_a")" && pwd)/$(basename "$bin_a")
bin_b=$(cd "$(dirname "$bin_b")" && pwd)/$(basename "$bin_b")

say() { printf '%s\n' "$*" | tee -a "$log"; }
median() {
  sort -n | awk '{ v[NR] = $1 } END { if (NR == 0) print "n/a"; else if (NR % 2) print v[(NR + 1) / 2]; else print (v[NR / 2] + v[NR / 2 + 1]) / 2 }'
}
column() { awk -F'\t' -v arm="$1" -v c="$2" '$1 == arm { print $c }' "$dir/runs.tsv"; }

summarize() {
  local arm bin
  {
    echo "label $label"
    echo "timeout_s $tmo"
    for arm in A B; do
      if [ $arm = A ]; then bin=$bin_a; else bin=$bin_b; fi
      echo "$arm bin=$bin sha256=$(sha256sum "$bin" | awk '{print $1}')"
      echo "$arm runs=$(column $arm 2 | wc -l) median_user_s=$(column $arm 5 | median) median_wall_s=$(column $arm 4 | median) max_maxrss_kb=$(column $arm 7 | sort -n | tail -n 1)"
    done
    awk -v a="$(column A 5 | median)" -v b="$(column B 5 | median)" \
      'BEGIN { if (a + 0 > 0 && b != "n/a") printf "ratio_user b/a %.4f\n", b / a; else print "ratio_user b/a n/a" }'
  } > "$dir/summary.txt"
}

say "ab-pair $label: A=$bin_a B=$bin_b seeds=$seeds timeout=$tmo"
rc=0
n=0
IFS=, read -r -a seed_list <<< "$seeds"
for seed in "${seed_list[@]}"; do
  n=$((n + 1))
  if [ $((n % 2)) -eq 1 ]; then order=AB; else order=BA; fi
  "$L" bash "$0" __pair "$dir" "$seed" "$order" "$tmo" "$bin_a" "$bin_b" -- "$@" >> "$log" 2>&1
  hold=$?
  say "seed $seed order $order hold exit $hold"
  if [ "$hold" -ne 0 ]; then
    rc=$hold
    break
  fi
done
[ -s "$dir/runs.tsv" ] && summarize
if [ "$rc" -eq 0 ] && awk -F'\t' '$8 != 0 { f = 1 } END { exit !f }' "$dir/runs.tsv"; then
  say "ab-pair $label: at least one run failed; see runs.tsv"
  rc=1
fi
[ -f "$dir/summary.txt" ] && tee -a "$log" < "$dir/summary.txt"
rc_done "$rc" | tee -a "$log"
exit "$rc"
