# 06_enrich: append the feature view and the labels

One job: append the features and labels that the adapt recipe names to the store, and prove the view holds no column the source cannot support.

## Inputs

**This run:**
- `../05_panel_pit/output/panel.md`
- `../05_panel_pit/output/check.md` (must be `pass`)
- the `adapt.md` of the type stage that ran

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../../docs/ohlcv.md`
- `../../references/time-rules.md`
- `../../CONTEXT.md`

Do NOT load: rebuilding the store, new feature design, any file of a type stage that did not run except the `columns-and-views.md` of the one that ran.

## Process

1. If an input is missing or empty, stop and say which. The recipe is `../04_*/output/adapt.md`.
2. Read `feature_groups`, `view_excludes`, `labels`, `locked_test` and `periods_per_year` from `adapt.md`. Never use `--features all`: the recipe names the view.
3. Run `rcsr enrich --data "$STORE" --out "$ENRICHED" --features <feature_groups> --label <labels> --calibration anchors --locked-test <locked_test> --lock-record <record> --json --dry-run` first, then the same call without `--dry-run`. Omit `--locked-test` and `--lock-record` when `locked_test` is `none`. Add `--allow-overlap` only for strided labels, with the reason recorded in `enrich.md`.
4. Panel: set the floors and make the two-stage call as the `columns-and-views.md` of the type stage describes, then `rcsr panel append --data "$PANEL" --family "$ENRICHED" --columns @all --locked-test <locked_test> --lock-record <record> --out "$PANEL_ENRICHED"` and `rcsr panel schema --data "$PANEL_ENRICHED" --verify --lock-record <record>`.
5. GPU: device steps run through `verksted/_tools/rc-run.sh` or `runlib.sh` holds, after `verksted/_tools/sanitize.sh` is clean for changed device code, and the log shows zero new Xid lines. A CPU build (`--gpu off`) needs no lock.
6. Write `enrich.md`: the commands, the report path `<out>.enrich.json`, `registry_version:` (`feature_registry_version` of `rcsr schema`), `cross_floor:` (from the report), `columns_missing_from_view:` (names in `feature_groups` that the store lacks, or `none`) and `columns:` (the comma-separated column names of the report).

## Outputs

- `enrich.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
e="$V/06_data/stages/06_enrich/output/enrich.md"; a=$(ls "$V"/06_data/stages/04_*/output/adapt.md); t=$(sed -n 's/^type: //p' "$a"); test -s "$ENRICHED.enrich.json" && grep -qx 'columns_missing_from_view: none' "$e" && if grep -qx 'variant: ohlc' "$a"; then ! sed -n 's/^columns: //p' "$e" | tr -d ' ' | tr ',' '\n' | grep -qxFf <(grep . "$V/06_data/stages/$t/references/volume-names.txt"); fi
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
