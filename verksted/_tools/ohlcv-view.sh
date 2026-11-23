#!/usr/bin/env bash
# Usage: ohlcv-view.sh --variant ohlc|ohlcv|ohlcv_flow --series single|panel [--schema <schema.json>] [--names <a,b,c>]
#        ohlcv-view.sh --derive <schema.json>
# Prints `feature_groups:` (the --features value of stage 06) and `view_excludes:` of one bar variant, from
# 06_data/stages/04_ohlcv/references/columns-and-views.md. ohlc has no volume field, ohlcv has volume only, ohlcv_flow
# has volume, quote volume, trade count and taker buy base. --series panel adds the cross-section groups; for ohlc and ohlcv the
# safe groups cross:btc, xs and btc only, and with --schema (the output of `rcsr schema`) the group `cross` is replaced by
# its available members whose raw fields the variant holds.
# --names prints `names: <the survivors>` and exits 1 with `refused: <name> (needs <fields>)` lines when a requested name
# reads a field the variant lacks. The volume names are the lines of references/volume-names.txt, each with the fields it reads.
# --derive prints the registry names (from the `feature_registry` of a schema file) whose formula text reads volume, V, V_t,
# quote_volume, trades, tb, taker, MFI or OBV, one per line, sorted. volume-names.txt holds these plus xs_rank_amihud96,
# which ranks amihud96.
# Writes: nothing.
# Exit: 0; 1 a requested name is refused or a schema is unreadable; 2 usage.
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAMES_FILE=$VERKSTED/06_data/stages/04_ohlcv/references/volume-names.txt
# name -> fields it reads
FIELDS='amihud96 quote_volume
avg_trade_size96 quote_volume,trades
btc_volume_share volume
cmf20 volume
cvd_proxy volume,taker
mfi14 volume
obv_z96 volume
taker_imb16 volume,taker
taker_ratio96 volume,taker
trade_intensity96 volume,trades
vol20_ratio volume
vol_z96 volume
vwap volume,quote_volume
vwap20_dev volume,quote_volume
vwap_dev96 volume,quote_volume
vz96 volume
xs_rank_amihud96 quote_volume
volume volume
quote_volume quote_volume
trades trades
taker_buy_base taker
taker_buy_quote taker'

variant='' series='' names='' schema='' derive=''
while [ $# -gt 0 ]; do
  case $1 in
    -h | --help) usage; exit 0 ;;
    --variant | --series | --names | --schema | --derive)
      [ $# -ge 2 ] || { usage >&2; exit 2; }
      case $1 in --variant) variant=$2 ;; --series) series=$2 ;; --names) names=$2 ;; --schema) schema=$2 ;; --derive) derive=$2 ;; esac
      shift 2 ;;
    *) usage >&2; exit 2 ;;
  esac
done

if [ -n "$derive" ]; then
  [ -r "$derive" ] || { echo "ohlcv-view.sh: cannot read $derive" >&2; exit 1; }
  jq -r '.feature_registry[] | select(.formula | test("volume|trades|taker|(^|[^A-Za-z_0-9])(V|V_t|tb|MFI[0-9]*|OBV)([^A-Za-z_]|$)")) | .name' "$derive" | LC_ALL=C sort
  exit $?
fi

case $variant in ohlc | ohlcv | ohlcv_flow) ;; *) usage >&2; exit 2 ;; esac
case $series in single | panel) ;; *) usage >&2; exit 2 ;; esac
[ -r "$NAMES_FILE" ] || { echo "ohlcv-view.sh: cannot read $NAMES_FILE" >&2; exit 1; }

# have <field>: the variant holds the raw field
have() {
  case $variant in
    ohlc) return 1 ;;
    ohlcv) [ "$1" = volume ] ;;
    ohlcv_flow) return 0 ;;
  esac
}

# missing <name>: prints the fields of the name that the variant lacks, comma separated; nothing when none
missing() {
  local f out=''
  local fields
  fields=$(awk -v n="$1" '$1 == n { print $2 }' <<< "$FIELDS")
  [ -n "$fields" ] || return 0
  for f in ${fields//,/ }; do
    have "$f" || out="$out${out:+,}$f"
  done
  printf '%s\n' "$out"
}

if [ -n "$names" ]; then
  rc=0 kept=''
  for n in ${names//,/ }; do
    m=$(missing "$n")
    if [ -n "$m" ]; then
      echo "refused: $n (needs $m)"
      rc=1
    else
      kept="$kept${kept:+,}$n"
    fi
  done
  echo "names: $kept"
  exit $rc
fi

join() { local IFS=,; printf '%s' "$*"; }

case $variant in
  ohlc)
    groups='ret1,ret5,ma20_ratio,ma60_ratio,bb20_z,atr14,rsi14,natr14,adx14,di_plus,di_minus,exotic:cheap,candle:geometry,mtf,price_action,trend'
    cross_ok='cross:btc,xs,btc' ;;
  ohlcv)
    groups='ret1,ret5,ma20_ratio,ma60_ratio,bb20_z,atr14,rsi14,vol20_ratio,natr14,adx14,di_plus,di_minus,cmf20,mfi14,vz96,obv_z96,exotic:cheap,candle:geometry,mtf,price_action,trend'
    cross_ok='cross:btc,xs,btc' ;;
  ohlcv_flow)
    groups='ta:full,exotic:cheap,candle:geometry,mtf,price_action,trend,flow,vz96'
    cross_ok='cross,cross:btc' ;;
esac

if [ "$series" = panel ]; then
  if [ "$variant" != ohlcv_flow ] && [ -n "$schema" ]; then
    [ -r "$schema" ] || { echo "ohlcv-view.sh: cannot read $schema" >&2; exit 1; }
    members=
    while read -r n; do
      [ -z "$(missing "$n")" ] && members="$members${members:+,}$n"
    done < <(jq -r '.feature_registry[] | select(.family == "cross" and .available == true) | .name' "$schema")
    [ -n "$members" ] || { echo "ohlcv-view.sh: no cross members in $schema" >&2; exit 1; }
    groups="$groups,cross:btc,$members"
  else
    groups="$groups,$cross_ok"
  fi
fi

case $variant in
  ohlc)
    excl="flow,$(paste -sd, - < <(grep . "$NAMES_FILE")),quote_volume,trades,taker_buy_base,taker_buy_quote" ;;
  ohlcv)
    list=''
    while read -r n _; do
      [ -n "$(missing "$n")" ] && list="$list${list:+,}$n"
    done < <(grep . "$NAMES_FILE" | sed 's/$/ x/')
    excl="$list,quote_volume,trades,taker_buy_base,taker_buy_quote" ;;
  ohlcv_flow) excl=none ;;
esac

echo "feature_groups: $groups"
echo "view_excludes: $excl"
