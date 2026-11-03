# 03_panel_pit: membership decided before the bar

One job: build point-in-time membership and the panel, keeping delisted coins.

## Inputs

**This run:**
- `../02_import_prep/output/prep.md`
- `../02_import_prep/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../../docs/panel.md`
- `../../CONTEXT.md`

Do NOT load: ranking by today's liquidity.

## Process

1. If an input is missing or empty, stop and say which.
2. `rcsr panel membership --data daily.csv --top 100 --lookback 90 --min-history 90 --rebalance 30 --out "$MEMBERSHIP"`, then `rcsr panel build --data "$STORE" --membership "$MEMBERSHIP" --out panel.rcsb`.
3. A delisted symbol stops at its last bar. Nothing is forward-filled or interpolated.
4. Write `panel.md`: commands and the count of symbols that end before the last rebalance.

## Outputs

- `panel.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
m=$MEMBERSHIP; head -1 "$m" | grep -qx 'as_of,symbol,member,adv_usd,n_bars' && awk -F, 'NR>1{if($1>max)max=$1; if($1>last[$2])last[$2]=$1} END{n=0; for(s in last) if(last[s]<max) n++; exit !(n>=1)}' "$m"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
