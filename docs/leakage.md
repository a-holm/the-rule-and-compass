# Leakage detectors

Statistical gates (DSR, PBO, SPA) cannot detect leakage: a leaked P&L passes
all of them. Leakage has to be caught differentially — run a pipeline twice on
inputs that differ only where the answer must not change. The detectors live in
`my-version/src/leakage.rs` and are exercised by `my-version/tests/leakage.rs`
(`cargo test --release --test leakage`).

## What runs

| detector | check | fails when |
|---|---|---|
| D4 truncation replay | run the pipeline at ~20 cut points and on the full panel; every output row with `close <= T` must be **bit-identical** | a row at or before the cut changes (whole-panel means, global scalers, future joins) |
| D5 future shift | flag a column that correlates `>= 0.995` with `target[t + horizon]` and clearly more than with `target[t]` | a deliberately injected `future_*` column reaches the pipeline |
| D3c canary | flag column names containing `future`/`canary`/`lookahead`, plus the D5 probe | a known canary is present; i.i.d. noise is *not* flagged |
| D3 as-of | `JoinRule::Timestamp` (a `ts <= t` higher-timeframe join) is refused; `JoinRule::BarClose` (`close = open + timeframe`) is accepted | a bar is used before it completes |
| D7 gap | `wall_time_gap(seconds, median_dt)` = `ceil(h / median_dt)` | never silently rounds a wall-time embargo down to zero rows |
| D6 overlap | `effective_n(n, horizon)` reports `n / horizon` | — (reported, not enforced) |
| D12 fixtures | clean and leaky synthetic panels test the tests | the suite no longer catches the planted leak |

`Report::assert_clean()` **panics** on a leak and prints the detector, column
and row. `Report::result()` returns a `LeakageFailure` for callers that want to
map it onto their own exit code. Neither path logs and continues.

## Running the suite on your own panel

The detectors take a `Panel` (a time column plus named value columns) and a
pipeline `Fn(&Panel) -> Panel` that adds the features your search will see.
The pipeline must return one row per input row, in order.

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

## Not yet wired

The Tier-A preparation auditors (D0, D1, D2, D8–D11), the differential learner
for D5 (this module uses the future-correlation probe), and a `rcsr audit`
subcommand with exit code 5 are follow-up work. The detectors here run in
`cargo test` today and are callable from any binary.
