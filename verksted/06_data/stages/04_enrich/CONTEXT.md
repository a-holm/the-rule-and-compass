# 04_enrich: append features, CPU equals GPU

One job: append features to the store and prove the device path matches the CPU.

## Inputs

**This run:**
- `../03_panel_pit/output/panel.md`
- `../03_panel_pit/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../../docs/ohlcv.md`
- `../../CONTEXT.md`

Do NOT load: rebuilding the store, new feature design.

## Process

1. If an input is missing or empty, stop and say which.
2. `rcsr enrich --data "$STORE" --out "$ENRICHED" --features all --gpu on`. The default financial feature set is the full expanded set.
3. Run the parity test through the lock: `--features cuda --test enrich_gpu`, log `$G/issue$N-enrich.log`.
4. Write `enrich.md`: command, report path `<out>.enrich.json`, parity log.

## Outputs

- `enrich.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
test -s "$ENRICHED.enrich.json" && grep -q 'test result: ok' "$G/issue$N-enrich.log" && ! grep -q 'test result: FAILED' "$G/issue$N-enrich.log"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
