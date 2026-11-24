#!/usr/bin/env bash
# The h1-mini hash set H (#1444). Reads grep, jq and sha256sum only; prints hash lines and never a run value.
# Usage:
#   hash.sh <out>          nine lines `<name> <sha256>` for the run directory <out> (unit.ckpt, run.json)
#   hash.sh keys <a> <b>   the key paths whose value differs between the run.json of two run directories (names only)
# Strip list S: strip.jq next to this script; H1_MINI_STRIP names another jq file (strip-ticket.jq: the ticket list, for comparison).
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
strip=${H1_MINI_STRIP:-$here/strip.jq}
sha() { sha256sum | cut -d' ' -f1; }

case ${1:-} in
  keys)
    [ $# = 3 ] || { echo "usage: hash.sh keys <a> <b>" >&2; exit 2; }
    flat() { jq -S -c -f "$strip" "$1/run.json" | jq -c '[paths(scalars) as $p | {k: ($p | map(tostring) | join(".")), v: getpath($p)}]'; }
    jq -n -r --slurpfile a <(flat "$2") --slurpfile b <(flat "$3") '
      ($a[0] | map({(.k): .v}) | add) as $x | ($b[0] | map({(.k): .v}) | add) as $y
      | (($x | keys) + ($y | keys) | unique)[] | select($x[.] != $y[.])'
    ;;
  "")
    echo "usage: hash.sh <out> | hash.sh keys <a> <b>" >&2; exit 2 ;;
  *)
    out=$1
    [ -s "$out/unit.ckpt" ] && [ -s "$out/run.json" ] || { echo "hash.sh: no unit.ckpt and run.json in $out" >&2; exit 2; }
    ck=$out/unit.ckpt
    printf 'ckpt_front %s\n' "$(grep -E '^(archive|archive_hash|final|sol|fsol) ' "$ck" | sha)"
    printf 'ckpt_ind %s\n' "$(grep -E '^ind ' "$ck" | sha)"
    printf 'ckpt_rng_mix %s\n' "$(grep -E '^(rng|mix) ' "$ck" | sha)"
    printf 'ckpt_qd %s\n' "$(grep -E '^qd_' "$ck" | sha)"
    printf 'run_pareto %s\n' "$(jq -S -c '.pareto' "$out/run.json" | sha)"
    printf 'run_pareto_expressions %s\n' "$(jq -c '[.pareto[].expression_exact]' "$out/run.json" | sha)"
    printf 'run_finalist %s\n' "$(jq -r '.objective.finalist.expression_exact' "$out/run.json" | sha)"
    printf 'run_stripped %s\n' "$(jq -S -c -f "$strip" "$out/run.json" | sha)"
    printf 'run_stripped_chunked %s\n' "$(jq -S -c -f "$here/strip-chunk.jq" "$out/run.json" | sha)"
    ;;
esac
