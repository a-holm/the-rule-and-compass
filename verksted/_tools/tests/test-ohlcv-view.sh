#!/usr/bin/env bash
# Usage: test-ohlcv-view.sh
# Hermetic test of ohlcv-view.sh and of the 04_ohlcv stage files: the three variants for single and panel series, the
# refusal exit, the --names filter, the --derive list against a synthetic schema, volume-names.txt against the view table,
# the lessons file, and a dry walk of two synthetic CSVs (OHLC only, OHLCV) through data-profile.sh, ohlcv-view.sh and the
# stage check with a hand-filled adapt.md, including the red case (ohlc without flow in view_excludes).
# When RCSR_BIN names an executable rcsr, its `schema` output is also compared with volume-names.txt.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/machine.sh"
VIEW=$HERE/../ohlcv-view.sh
PROF=$HERE/../data-profile.sh
CHECK=$HERE/../check.sh
V=$HERE/../..
STAGE=$V/06_data/stages/04_ohlcv
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
key() { sed -n "s/^$1: //p" "$2" | head -n 1; }

# the three variants, single and panel
for v in ohlc ohlcv ohlcv_flow; do
  for s in single panel; do
    bash "$VIEW" --variant $v --series $s > "$T/$v-$s" 2>&1
    check "$v-$s-rc" "$?" 0
    check "$v-$s-lines" "$(wc -l < "$T/$v-$s")" 2
  done
done
check ohlc-excludes-flow "$(key view_excludes "$T/ohlc-single" | tr ',' '\n' | grep -cx flow)" 1
check ohlc-excludes-vz96 "$(key view_excludes "$T/ohlc-single" | tr ',' '\n' | grep -cx vz96)" 1
check ohlc-groups-no-volume "$(key feature_groups "$T/ohlc-panel" | tr ',' '\n' | grep -cxE 'flow|vz96|cmf20|mfi14|vol20_ratio|ta:core|ta:full|cross|obv_z96|vwap20_dev')" 0
check ohlcv-keeps-volume-names "$(key feature_groups "$T/ohlcv-single" | tr ',' '\n' | grep -cxE 'vz96|cmf20|mfi14|vol20_ratio|obv_z96')" 5
check ohlcv-excludes-quote "$(key view_excludes "$T/ohlcv-single" | tr ',' '\n' | grep -cxE 'vwap|taker_imb16|amihud96|avg_trade_size96')" 4
check ohlcv-not-exclude-volume-name "$(key view_excludes "$T/ohlcv-single" | tr ',' '\n' | grep -cxE 'vz96|cmf20|obv_z96')" 0
check flow-excludes "$(key view_excludes "$T/ohlcv_flow-single")" none
check flow-has-flow "$(key feature_groups "$T/ohlcv_flow-single" | tr ',' '\n' | grep -cx flow)" 1
for v in ohlc ohlcv ohlcv_flow; do
  check "$v-single-no-cross" "$(key feature_groups "$T/$v-single" | tr ',' '\n' | grep -cE '^(cross|xs|btc)')" 0
  check "$v-panel-has-cross" "$(key feature_groups "$T/$v-panel" | tr ',' '\n' | grep -cE '^(cross|xs|btc)')" "$([ $v = ohlcv_flow ] && echo 2 || echo 3)"
done

# --names: refusal on ohlc, acceptance on flow, partial on ohlcv
bash "$VIEW" --variant ohlc --series single --names r96,vz96 > "$T/n1" 2>&1
check names-ohlc-rc "$?" 1
check names-ohlc-refused "$(grep -c '^refused: vz96 ' "$T/n1")" 1
check names-ohlc-kept "$(grep '^names:' "$T/n1")" "names: r96"
bash "$VIEW" --variant ohlcv_flow --series single --names taker_imb16 > "$T/n2" 2>&1
check names-flow-rc "$?" 0
check names-flow-out "$(cat "$T/n2")" "names: taker_imb16"
bash "$VIEW" --variant ohlcv --series single --names vz96,taker_imb16 > "$T/n3" 2>&1
check names-ohlcv-rc "$?" 1
check names-ohlcv-refused "$(grep -c '^refused: taker_imb16 (needs taker)' "$T/n3")" 1
check names-ohlcv-kept "$(grep '^names:' "$T/n3")" "names: vz96"
bash "$VIEW" --variant ohlc --series single --names volume > /dev/null 2>&1
check names-raw-volume "$?" 1

# usage
bash "$VIEW" --variant nope --series single > /dev/null 2>&1
check bad-variant "$?" 2
bash "$VIEW" --variant ohlc > /dev/null 2>&1
check no-series "$?" 2

# --derive and --schema on a synthetic schema
cat > "$T/schema.json" << 'JSON'
{"feature_registry":[
{"name":"ret1","family":"returns","available":true,"formula":"ln(C_t/C_{t-1})"},
{"name":"vz96","family":"vol","available":true,"formula":"(ln(1+V_t) - mean96(ln(1+V))) / sd96(ln(1+V))"},
{"name":"vwap","family":"flow","available":true,"formula":"quote_volume / volume"},
{"name":"taker_imb16","family":"flow","available":true,"formula":"(2 mean(tb,16)-mean(V,16))/mean(V,16)"},
{"name":"mfi14","family":"rsi","available":true,"formula":"MFI14/50 - 1"},
{"name":"vol20","family":"vol","available":true,"formula":"population sd of 1-bar log returns over w contiguous bars"},
{"name":"mkt_r16","family":"cross","available":true,"formula":"equal-weight mean of PIT member r16 at t"},
{"name":"btc_volume_share","family":"cross","available":true,"formula":"BTC volume / total PIT member volume at t"},
{"name":"xs_rank_amihud96","family":"cross","available":true,"formula":"2*(rank-1)/(n-1)-1 per timestamp"},
{"name":"funding_rate","family":"cross","available":false,"formula":"reserved"}
]}
JSON
check derive "$(bash "$VIEW" --derive "$T/schema.json" | tr '\n' ' ')" "btc_volume_share mfi14 taker_imb16 vwap vz96 "
check derive-missing "$(bash "$VIEW" --derive "$T/none.json" > /dev/null 2>&1; echo $?)" 1
bash "$VIEW" --variant ohlc --series panel --schema "$T/schema.json" > "$T/sp-ohlc" 2>&1
check schema-ohlc-members "$(key feature_groups "$T/sp-ohlc" | tr ',' '\n' | grep -cx mkt_r16)" 1
check schema-ohlc-no-share "$(key feature_groups "$T/sp-ohlc" | tr ',' '\n' | grep -cxE 'btc_volume_share|xs_rank_amihud96|funding_rate')" 0
bash "$VIEW" --variant ohlcv --series panel --schema "$T/schema.json" > "$T/sp-ohlcv" 2>&1
check schema-ohlcv-share "$(key feature_groups "$T/sp-ohlcv" | tr ',' '\n' | grep -cx btc_volume_share)" 1
check schema-ohlcv-no-rank "$(key feature_groups "$T/sp-ohlcv" | tr ',' '\n' | grep -cx xs_rank_amihud96)" 0

# volume-names.txt: bare names, sorted, unique, and each has a field entry in the tool
names_file=$STAGE/references/volume-names.txt
check names-bare "$(grep -cvE '^[a-z][a-z0-9_]*$' "$names_file")" 0
check names-sorted "$(LC_ALL=C sort -u "$names_file" | cmp - "$names_file" > /dev/null 2>&1; echo $?)" 0
for n in $(cat "$names_file"); do
  bash "$VIEW" --variant ohlc --series single --names "$n" > /dev/null 2>&1
  check "names-refused-on-ohlc-$n" "$?" 1
done
for n in vz96 taker_imb16 mfi14 vwap20_dev xs_rank_amihud96 btc_volume_share vol_z96; do
  check "names-listed-$n" "$(grep -cx "$n" "$names_file")" 1
done

# optional: against the schema of a real binary
if [ -n "${RCSR_BIN:-}" ] && [ -x "$RCSR_BIN" ]; then
  "$RCSR_BIN" schema > "$T/real-schema.json" 2> /dev/null
  derived=$(bash "$VIEW" --derive "$T/real-schema.json")
  want=$({ printf '%s\n' "$derived"; echo xs_rank_amihud96; } | LC_ALL=C sort -u)
  check names-equal-schema "$(cat "$names_file")" "$want"
  jq -e '.feature_registry | length > 0' "$T/real-schema.json" > /dev/null 2>&1
  check real-schema-parsed "$?" 0
fi

# stage files
for f in formats-and-venues columns-and-views calendar-and-gaps lessons; do
  check "frontmatter-$f" "$(head -1 "$STAGE/references/$f.md")" "---"
  check "frontmatter-type-$f" "$(sed -n '2p' "$STAGE/references/$f.md" | cut -c1-5)" "type:"
  check "frontmatter-desc-$f" "$(sed -n '3p' "$STAGE/references/$f.md" | cut -c1-12)" "description:"
done
check lessons-count "$(grep -cE '^[0-9]+\. ' "$STAGE/references/lessons.md")" 14
check lessons-source "$(grep -E '^[0-9]+\. ' "$STAGE/references/lessons.md" | grep -c 'Source: [0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}')" 14
check lessons-no-path "$(grep -cE '(^|[ (`])/[A-Za-z]+/' "$STAGE/references/lessons.md")" 0

# dry walk of two synthetic CSVs
sh_block() { awk '/^## Check/ { c = 1 } c && /^```sh$/ { b = 1; next } b && /^```$/ { exit } b { print }' "$STAGE/CONTEXT.md"; }
sh_block > "$T/check-cmd"
check check-cmd-found "$([ -s "$T/check-cmd" ] && echo yes || echo no)" yes
CMD=$(cat "$T/check-cmd")

walk() { # <name> <variant> <csv-body-file> <extra view_excludes override or ''>
  local name=$1 variant=$2 dir=$T/$1
  mkdir -p "$dir/data" "$dir/out"
  cp "$3" "$dir/data/SYM-1h-2020-01.csv"
  bash "$PROF" --dir "$dir/data" --dev-end 2023-11-15 --out "$dir/profile.md" > /dev/null 2>&1
  check "$name-profile-rc" "$?" 0
  bash "$VIEW" --variant "$variant" --series single > "$dir/view.txt" 2>&1
  check "$name-view-rc" "$?" 0
  echo store > "$dir/store.rcsb"
  echo '{}' > "$dir/store.rcsb.manifest.json"
  {
    echo "type: 04_ohlcv"
    echo "variant: $variant"
    echo "layout: ohlcv"
    echo "bar_seconds: 3600"
    echo "source_bar_seconds: $(key median_step_seconds "$dir/profile.md")"
    echo "first_bar: $(key first_ts "$dir/profile.md")"
    echo "store_first_bar: $(key first_ts "$dir/profile.md")"
    echo "approved_start: none"
    echo "dev_end: 2023-11-15T00:00:00Z"
    echo "locked_test: 2023-11-15,2026-11-15"
    echo "series: single"
    echo "symbols: 1"
    echo "raw_columns: $(key raw_columns "$dir/profile.md")"
    echo "zero_volume_rows: $(key zero_volume_rows "$dir/profile.md")"
    echo "zero_volume_policy: keep, prices repeat the previous bar"
    cat "$dir/view.txt"
    echo "labels: long:24,fee=10,stride=24"
    echo "periods_per_year: 365"
    echo "store: $dir/store.rcsb"
    echo "deviations: none"
  } > "$dir/out/adapt.md"
  V=$STAGE STORE=$dir/store.rcsb bash "$CHECK" "$dir/out" "$(sed "s|\$V/06_data/stages/04_ohlcv/output/adapt.md|$dir/out/adapt.md|" <<< "$CMD")" > /dev/null 2>&1
}

printf 'time,open,high,low,close\n1577836800,1,2,0.5,1.5\n1577840400,1.5,2.5,1,2\n1577844000,2,3,1.5,2.5\n' > "$T/ohlc.csv"
printf 'time,open,high,low,close,volume\r\n1577836800,1,2,0.5,1.5,0\r\n1577840400,1.5,2.5,1,2,11\r\n1577844000,2,3,1.5,2.5,12\r\n' > "$T/ohlcv.csv"
walk dry-ohlc ohlc "$T/ohlc.csv"
check dry-ohlc-raw "$(key raw_columns "$T/dry-ohlc/profile.md")" "open,high,low,close"
check dry-ohlc-check "$(head -1 "$T/dry-ohlc/out/check.md")" pass
walk dry-ohlcv ohlcv "$T/ohlcv.csv"
check dry-ohlcv-raw "$(key raw_columns "$T/dry-ohlcv/profile.md")" "open,high,low,close,volume"
check dry-ohlcv-zero "$(key zero_volume_rows "$T/dry-ohlcv/profile.md")" 1
check dry-ohlcv-check "$(head -1 "$T/dry-ohlcv/out/check.md")" pass

# red: ohlc whose view_excludes lacks flow fails the check; so does a missing key and a moved start
sed -i 's/^view_excludes: flow,/view_excludes: /' "$T/dry-ohlc/out/adapt.md"
V=$STAGE STORE=$T/dry-ohlc/store.rcsb bash "$CHECK" "$T/dry-ohlc/out" "$(sed "s|\$V/06_data/stages/04_ohlcv/output/adapt.md|$T/dry-ohlc/out/adapt.md|" <<< "$CMD")" > /dev/null 2>&1
check red-no-flow "$(head -1 "$T/dry-ohlc/out/check.md" | cut -c1-5)" 'fail:'
sed -i '/^deviations: /d' "$T/dry-ohlcv/out/adapt.md"
V=$STAGE STORE=$T/dry-ohlcv/store.rcsb bash "$CHECK" "$T/dry-ohlcv/out" "$(sed "s|\$V/06_data/stages/04_ohlcv/output/adapt.md|$T/dry-ohlcv/out/adapt.md|" <<< "$CMD")" > /dev/null 2>&1
check red-missing-key "$(head -1 "$T/dry-ohlcv/out/check.md" | cut -c1-5)" 'fail:'

if [ "$fails" -ne 0 ]; then echo "$fails failures"; exit 1; fi
echo "ok"
