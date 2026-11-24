#!/usr/bin/env bash
# The h1-mini run fixture (#1444): a seeded synthetic hourly panel (40,000 rows, 8 groups) and one pinned h1-like run.
# Usage:
#   h1-mini.sh prep <rcsr> <dir>                          panel, import and enrich into <dir> (CPU, no data file in git)
#   h1-mini.sh run  <rcsr> <dir> <seed> <chunks> <out>    one arm; <chunks> is 1 (one process) or 3 (up to three processes, see below)
# Chunked arm: the first two processes run with --time-budget $CHUNK_SECONDS --interim-chunk and stop on the clock, as a pilot unit does;
# the last resumes and runs to --generations. The generation each chunk stopped at goes to <out>/chunks.txt.
# Environment: H1_MINI_BACKEND gpu (default) or cpu. A gpu arm holds the device: start it through the test lock with an outer timeout.
# Writes under <dir> (prep) and <out> (run): unit.ckpt, run.json, dev.ndjson, chunk<k>.log, chunks.txt. Prints only status lines.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VARS=r1,r3,r6,r12,r24,r48,vol20,vol60,ma20,ma50,rsi14,range1,clv1,body1,don20,vz96
OPS=add,sub,mul,div,abs,exp,square,tanh,logistic,gauss,min,max,greater,if
FEATURES=returns,ma,vol,rsi,range,candle,donchian,vz
GENERATIONS=${H1_MINI_GENERATIONS:-30}
MIGRATION_INTERVAL=${H1_MINI_MIGRATION_INTERVAL:-25}
MIN_TRADES=${H1_MINI_MIN_TRADES:-10}
FIT_ROWS=${H1_MINI_FIT_ROWS:-1024}
CHUNK_SECONDS=${H1_MINI_CHUNK_SECONDS:-10}
PANEL_SEED=${H1_MINI_PANEL_SEED:-7}

die() { echo "h1-mini: $*" >&2; exit 2; }

prep() {
  local bin=$1 dir=$2
  [ -x "$bin" ] || die "no binary $bin"
  mkdir -p "$dir"
  nice -n 19 rustc -O "$here/panel.rs" -o "$dir/panel-gen" 2> "$dir/panel-gen.log" || die "panel generator build failed, see $dir/panel-gen.log"
  "$dir/panel-gen" "$PANEL_SEED" > "$dir/panel.csv"
  RCSR_GPU=0 "$bin" import "$dir/panel.csv" "$dir/panel.rcsb" --time ts --quiet
  RCSR_GPU=0 "$bin" enrich --data "$dir/panel.rcsb" --out "$dir/feat.rcsb" --features "$FEATURES" \
    --label long:1,name=h1,fee=10 --cpu > "$dir/enrich.log" 2>&1
  rm -f "$dir/panel.csv" "$dir/panel.rcsb"
  echo "prep ok: $dir/feat.rcsb"
}

run() {
  local bin=$1 dir=$2 seed=$3 chunks=$4 out=$5 backend=${H1_MINI_BACKEND:-gpu} k rc max
  [ -x "$bin" ] || die "no binary $bin"
  [ -s "$dir/feat.rcsb" ] || die "no store in $dir: run prep first"
  case $chunks in 1) max=1 ;; 3) max=3 ;; *) die "chunks is 1 or 3" ;; esac
  local dev=(--cpu); [ "$backend" = gpu ] && dev=(--gpu --fit-strategy gpu --fit-max-iters 12 --gpu-memory 6GiB --pipeline-depth 0)
  mkdir -p "$out"; rm -f "$out"/unit.ckpt* "$out"/run.json "$out"/dev.ndjson "$out"/chunk* "$out"/chunks.txt
  for k in $(seq "$max"); do
    local res=() budget=()
    [ -f "$out/unit.ckpt" ] && res=(--resume "$out/unit.ckpt")
    [ "$chunks" = 3 ] && [ "$k" -lt 3 ] && budget=(--time-budget "$CHUNK_SECONDS" --interim-chunk)
    set +e
    "$bin" --data "$dir/feat.rcsb" --target lbl_h1_y --target-spec "$dir/feat.targets.json" --vars "$VARS" \
      --time ts --group-by symbol --split temporal --validation-frac 0.2 --test-frac 0.2 \
      --metric squared --objective sharpe --cost-bps 10 --fill drop --require-clean-inputs --calibration anchors --max-memory 16GiB \
      --ops "$OPS" --max-complexity 12 --pop 32 --islands 4 --explore-islands 1 --migration-policy ring-tournament \
      --qd-policy --qd-descriptor book --adaptive-mix --patience 0 --fit-accept fitness --fit-thresholds --fit-iters 3 --fit-rows "$FIT_ROWS" \
      --fitness pnl --pnl-blocks 8 --pnl-block-lambda 1 --position long_only --threshold 1e-30 \
      --min-trades "$MIN_TRADES" --band 0 --min-holding 12 --migration-interval "$MIGRATION_INTERVAL" \
      "${dev[@]}" --seed "$seed" --generations "$GENERATIONS" "${res[@]}" "${budget[@]}" \
      --save "$out/unit.ckpt" --save-every 10 --json-out "$out/run.json" --dev-log "$out/dev.ndjson" \
      > "$out/chunk$k.out" 2> "$out/chunk$k.log"
    rc=$?
    set -e
    [ $rc = 0 ] || { echo "h1-mini: chunk $k exit $rc, see $out/chunk$k.log" >&2; return $rc; }
    printf 'chunk %s generations %s\n' "$k" "$(sed -n 's/^generations //p' "$out/unit.ckpt")" >> "$out/chunks.txt"
    [ "$(jq -r '.interim // false' "$out/run.json")" = true ] || break
  done
  [ "$(jq -r '.interim // false' "$out/run.json")" = false ] || { echo "h1-mini: run is still interim after $max chunks" >&2; return 3; }
  echo "run ok: $out"
}

case ${1:-} in
  prep) [ $# = 3 ] || die "usage: prep <rcsr> <dir>"; prep "$2" "$3" ;;
  run) [ $# = 6 ] || die "usage: run <rcsr> <dir> <seed> <chunks> <out>"; run "$2" "$3" "$4" "$5" "$6" ;;
  *) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
