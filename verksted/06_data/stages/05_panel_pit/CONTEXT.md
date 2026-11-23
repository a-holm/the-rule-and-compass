# 05_panel_pit: membership decided before the bar

One job: build point-in-time membership and the panel, keeping delisted symbols, or record that the dataset is one series.

## Inputs

**This run:**
- the `adapt.md` of the type stage that ran
- the `check.md` of the type stage that ran (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../../docs/panel.md`
- `../../references/type-stage-contract.md`
- `../../CONTEXT.md`

Do NOT load: ranking by today's liquidity, any file of a type stage that did not run.

## Process

1. If an input is missing or empty, stop and say which. The inputs are `../04_*/output/adapt.md` and `../04_*/output/check.md` of the one type stage that has an `output/`.
2. Branch on `series:` of `adapt.md`.
3. `single`: write `panel.md` with the line `panel: not applicable (single series)`.
4. `panel`: run `rcsr panel membership --data "$STORE" --start <first_bar> --end <dev_end> --top N --lookback D --min-history D --min-bars M --rebalance D --out "$MEMBERSHIP"`, then `rcsr panel build --data "$STORE" --membership "$MEMBERSHIP" --snapshot <snapshot.json> --bar-seconds <bar_seconds> --relist <SYMBOL@DATE,...> --jump-factor 5 --locked-test <locked_test> --lock-record <record> --out "$PANEL"`, then `rcsr panel validate --data "$PANEL" --base "$STORE" --membership "$MEMBERSHIP"`. Omit `--locked-test` and `--lock-record` when `locked_test` is `none`.
5. If membership refuses the start, record the message and use the first start it accepts; no other start without approval.
6. A jump failure names a symbol and time: add it to the relist list and rerun; nothing is bridged.
7. A delisted symbol stops at its last bar. Nothing is forward-filled or interpolated.
8. Write `panel.md`: commands, exit codes and the count of symbols that end before the last rebalance.

## Outputs

- `panel.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
p="$V/06_data/stages/05_panel_pit/output/panel.md"; a=$(ls "$V"/06_data/stages/04_*/output/adapt.md); if grep -qx 'series: single' "$a"; then grep -qx 'panel: not applicable (single series)' "$p"; else m=$MEMBERSHIP; head -1 "$m" | grep -qx 'as_of,symbol,member,adv_usd,n_bars' && awk -F, 'NR>1{if($1>max)max=$1; if($1>last[$2])last[$2]=$1} END{n=0; for(s in last) if(last[s]<max) n++; exit !(n>=1)}' "$m" && "$(verksted/_tools/crate-dir.sh "$RUN")/target/release/rcsr" panel validate --data "$PANEL" --base "$STORE" --membership "$m"; fi
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
