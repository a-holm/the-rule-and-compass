# Leakage detectors

Statistical gates (DSR, PBO, SPA) cannot detect leakage: a leaked P&L passes
all of them. Leakage has to be caught differentially — run a pipeline twice on
inputs that differ only where the answer must not change. The detectors live in
`my-version/src/leakage.rs`, are exercised by `my-version/tests/leakage.rs` and
`my-version/tests/leakage_audit.rs`, and run in the `rcsr audit` subcommand.

## The suite

| detector | check | fails when |
|---|---|---|
| D0 spec | every required key is resolved, no duplicate or unresolved value | a config is inherited by accident or a default is missing |
| D1 time | normalise `s`/`ms`/`us`/`ns`, check monotonicity, duplicate instants and spacing | a unit error, a backwards print or a repeated instant |
| D2 resample | `1m->1h` must equal `1m->15m->1h`; OHLC first/max/min/last, volume sum, VWAP volume-weighted, session folds | a cascade changes a bar or uses a different aggregation |
| D3 as-of | `JoinRule::Timestamp` (a `ts <= t` higher-timeframe join) is refused; `JoinRule::BarClose` (`close = open + timeframe`) is accepted | a bar is used before it completes |
| D3c canary | column names containing `future`/`canary`/`lookahead`, plus the D5 probe | a known canary reaches the pipeline; i.i.d. noise must not fire |
| D4 replay | run the pipeline at ~20 cut points and on the full panel; every output row with `close <= T` must be **bit-identical** | a row at or before the cut changes (whole-panel means, global scalers, future joins) |
| D5 differential | fit a candidate future column in-sample and at prediction time; a leak improves only in-sample | the future column is mistaken for a usable feature |
| D6 overlap | `effective_n(n, horizon)` reports `n / horizon` | — (reported, not enforced) |
| D7 gap | a wall-time `--gap 90m` is `ceil(seconds / median_dt)`, never silently zero | the embargo is dropped under explicit row selection (a warning is emitted) |
| D8 universe | rows must lie inside a symbol's point-in-time listing interval, including renames | survivorship or a pre-listing/delisted symbol in the panel |
| D9 corporate actions | a split/dividend is applied only from the instant it is known; adjusted **and** unadjusted series are returned | a factor learned after a cut rewrites an earlier row |
| D10 join | right-side keys must be unique and the result may not multiply rows | a duplicate key fans out a join |
| D11 determinism | FNV-1a hash of the resolved spec (including the tzdb version) | two collection paths disagree |
| D13a-c target | a declared target (`--target-spec`): the future return or target as an input (a), a scale window that ends after the decision time (b), a structural term whose zero-return prediction depends on a column (c) | the label can be scored without forecasting a return; D13c has the override `--allow-target-structure` |
| D12 fixtures | clean and leaky synthetic panels test the tests | the suite no longer catches the planted leak |

`Report::assert_clean()` **panics** on a leak and prints the detector, column
and row. `Report::result()` returns a `LeakageFailure` for callers that map it
onto their own exit code. Neither path logs and continues.

## `rcsr audit` (exit code 5)

```sh
rcsr audit --data panel.csv --target y --time ts --horizon 24 --json
rcsr audit --data panel.csv --target y --time ts --join-on symbol --report audit.json
rcsr audit --data bars1m.csv --target y --time ts --bars bars1m.csv \
           --resample-small 15 --resample-large 60
rcsr audit --data panel.csv --target y --time ts --symbol symbol \
           --universe listings.csv --price close --corporate-actions actions.csv
```

The command runs the D0–D6/D11/D12 panel suite (the optional `--bars`,
`--universe` and `--corporate-actions` tables add D2, D8 and D9), writes a machine-readable report
(`--json` or `--report <path>`) and exits:

| code | meaning |
|---|---|
| 0 | clean |
| 2 | usage error |
| 3 | input error |
| 5 | **leakage detected** |

`rcsr audit --target-spec FILE` adds D13a-c for the target against `--vars` (all
numeric columns except the target and time when `--vars` is absent). The same
check runs before every search with a spec and exits 5 on a finding. See
`docs/time-series.md` §2b.

The JSON report contains `clean`, `checks`, `exit_code`, `spec_hash`,
`tzdb_version`, `rows`, `horizon`, `overlap_rows`, `effective_n`, a
`time_audit` object and a `leaks` array of
`{detector,column,row,message}` objects. Exit 5 is stable and is also listed in
`rcsr schema`.

## Running the suite on your own panel

The detectors take a `Panel` (a time column plus named value columns) and a
pipeline `Fn(&Panel) -> Panel` that adds the features your search will see. The
pipeline must return one row per input row, in order.

```rust
use rcsr::leakage::{cut_points, truncation_replay, canary_leakage, Panel};

let panel = Panel::from_csv("panel.csv", "ts", &["open", "close", "volume", "y"])?;

// D4: your feature builder must not read beyond the cut.
let replay = truncation_replay(&panel, &|p| my_features(p), &cut_points(&panel, 20));
replay.assert_clean();                       // panic = real leak

// D3c/D5: no canary of the future target may survive preparation.
canary_leakage(&panel, "y", 24).assert_clean();
```

Put that in a `#[test]` in your own crate (or in a `tests/` file pointing at
your fixture). Because the assertion panics, a leak fails CI rather than
printing a warning. If a detector reports a leak, the message names the output
column and (for replay) the first diverging row; the usual causes are a
whole-panel scaler, a forward fill that reads the next row, a `ts <= t` join,
or a feature built from a future bar.

## Tier-A callers

The auditors that need more than a wide panel are library calls:

```rust
use rcsr::leakage::{audit_universe, adjust_corporate_actions, ...};
```

- `time_normalizer(times, TimeUnit::Millis)` -> `(seconds, TimeAudit, Report)`;
- `reconcile_resample(&bars, 15, 60, session_gap)` for the D2 cascade;
- `audit_universe(&rows, &events)` for listings, delistings and renames;
- `adjust_corporate_actions(&times, &prices, &actions, "AAA")` returns
  `AdjustedSeries { unadjusted, adjusted, split_factor, dividends, actions_known }`;
- `corporate_action_replay(...)` checks that a cut cannot learn a future factor;
- `audit_join_cardinality(&left_keys, &right_keys)` rejects row multiplication;
- `audit_spec` / `audit_determinism` / `spec_hash` implement D0 and D11.
