# Time-series and panel checklist

A symbolic-regression fit on ordered data is easy to make look excellent and
useless in production. This page is the audit you run **before** believing a
time-series or panel result. Every item names the flag or behaviour that
matters, what goes wrong when it is ignored, and a command or test that catches
it.

Run this checklist in order. If an item cannot be checked, record why in the
report.

---

## 0. Write the question first

Decide these three things before touching `rcsr`:

1. **What is one prediction?** The row, the timestamp, the entity, and the
   target horizon `h` (how far ahead the target looks).
2. **What is the validation period?** A contiguous block after the training
   period, never a random sample of it.
3. **What is the lockbox?** A final block that is not looked at until the
   expression and its constants are frozen. If there is no lockbox, say so.

A result without (1)–(3) is a fit, not a forecast.

---

## 1. Split chronologically, never randomly

**Check.** A time-ordered table must use:

```sh
rcsr --data prices.csv --target y --vars x1,x2 \
     --split temporal --time date --test-frac 0.25 --gap 5
```

or, for panel data:

```sh
rcsr --data panel.csv --target next_close --vars close,volume \
     --split temporal-group --time date --group-by symbol --test-frac 0.25 --gap 5
```

**What fails.** `--split random` (and therefore a bare `--test-frac`) shuffles
rows. For almost every training row there is a test row from the next day, so an
interpolating model looks accurate and cannot be traded. The engine warns when
it detects a time-like or group-like column, but it does **not** silently change
the split you asked for.

**Verify.**

* The text header says `split=temporal` or `split=temporal-group`; in `--json`,
  `split.mode` is the same (not `random`).
* The test rows are a contiguous suffix (per series), not a hash sample.
* The warning path is pinned by
  `tests/split_warning.rs::the_random_warning_does_not_claim_a_split_it_did_not_use`.
* `tests/split.rs::temporal_split_is_chronological_and_honours_the_gap`.

**`--metric slope`.** Slope differences each row with the previous row of the
same group in the data order (`--time`, then `--group-by`). A gap in the time
index is not detected: two rows on either side of a missing day are one pair,
and their difference is a slope over the gap. The GPU scores and fits slope
only when the training pairs are its consecutive rows: a single series under
`--split temporal`. A panel with `--group-by` or a shuffled split keeps
the CPU with the reason `slope pairs are not the consecutive rows of the
training sample` (#236).

---

## 2. Purge the label horizon and embargo the boundary

**Check.** `--gap N` drops `N` rows immediately after the train/test boundary
(per series in `temporal-group`). Set it from the target, not from habit:

```text
N >= label horizon h
```

If the last training label is built from the next `h` bars, those bars must not
be test rows. For a 5-bar forward return, `--gap 5` is the minimum. Add any
additional look-ahead introduced by preparation (for example the right half of
an opt-in `--smooth-centred` window; see §4) to `N`.

If features only look **backwards** (`--lags`), no extra gap is needed for the
lookback itself: a lag taken from before the boundary is information the
forecaster would have had. The leak is the **label** crossing the boundary.
When in doubt, use the conservative `h + lookback` and check whether the extra
rows cost you the result.

With a declared target spec (§2b) the engine derives the purge from the label
window instead: `required_purge_seconds = entry_offset_seconds + delay_seconds +
horizon_seconds`. After the split and the validation carve it drops every
training row (fit and validation) whose stamp is less than that many seconds
from any test stamp, then every fit row that close to a validation stamp. The
time axis is global, so cross-symbol overlap on a panel and misaligned
`temporal-group` series are covered, and the rule holds for every split mode.
`--gap` still applies on top. It needs a `--time` column holding epoch seconds
(exit 2 otherwise). `split` in `run.json` records `required_purge_seconds`,
`purge_seconds` (the smallest stamp distance between retained training rows and
test rows, also recorded without a spec), `validation_purge_seconds`,
`purged_rows.test_boundary`, `purged_rows.validation_boundary` and
`purge_source`. It also records `train_time` and `validation_time` as `[first, last]` raw time keys (`validation_time` is `null` without a validation slice; both are absent without `--time`) and `search.candidates_evaluated`, the offspring scored over the run. `rcsr walkforward --front` reads them. The text header prints the same facts on a `# purge:` line.

The 2026H2 lockbox check covers the fit rows, the validation slice and the test
blocks alike: a validation slice that overlaps the lockbox is refused with the
message shape and exit code of a test block (`validation split: N row(s) ...
overlap the lockbox`), unless `--allow-lockbox` declares a final test.

The protocol locked test is a second, separate box: declare it with the data
(`rcsr panel build --locked-test START,END`) and the search, `walkforward`,
`ladder` and `ablation score` refuse every fit, validation or test row inside
it (exit 3), and every row whose label window reaches it: a row within the target's `required_purge_seconds` before the range (the protocol default of 219600 s without a target spec) is refused too, and the refusal names the exact `--exclude 'ts >= START - purge'`. `walkforward`, `ladder` and `ablation score` refuse a window that ends after `START - purge`. `--allow-lockbox` does not open it. Only the sealed
`--locked-test` path of `walkforward` and `ladder` reads it, once. A pilot
split such as `--split temporal --validation-frac 0.20` on a panel that ends
inside the locked range puts its test block there; the refusal tells you to
`--exclude` the rows.

**What fails.** Without the purge, the last training row and first test row
share most of their label window. The test error is then a near-copy of the
training error, and `--folds` blocks look suspiciously similar at the boundary.

**Verify.**

* The split note and `--json` `split.gap` equal the requested value.
* `tests/split.rs::temporal_split_is_chronological_and_honours_the_gap`.
* For cross-validation, `tests/cv.rs::purged_embargo_removes_training_rows`
  proves `purged` removes `--gap` rows either side of each test block.

### `--gap` and explicit row selection

**Check.** If either `--train-select` or `--test-select` is present, row
selection replaces the positional mode. `--gap` then does **not** apply; the
report says so:

```text
# note: --gap is dropped because explicit --train-select/--test-select replace
# positional splitting; add the embargo to the selection expressions instead
```

**What fails.** A user writes `--train-select "date < 2020" --gap 5` and believes
an embargo was applied. It was not; the boundary is wherever the expression
puts it. Add the purge rows to the expressions yourself, or use a positional
split.

**Verify.** `tests/selection.rs::selection_overrides_a_temporal_spec_and_says_so`.

---

## 2b. Target structure: the label must not encode a column

**Check.** Declare how the label was built in a target spec
(`--target-spec FILE`, or `<data stem>.targets.json` beside the data; contract
`rcsr-target-spec/1`, see `rcsr schema`). For a forward return `r` the label is

```text
T = clip((r - c) / (m * g), lo, hi)
```

with offset `c`, multiplier `m`, a scale `g` (a constant or a column such as a
volatility estimate) and an optional clip. The prediction at zero return,
`m0 = T(r = 0) = clip(-c / (m * g), lo, hi)`, is a function of `g` alone. When
`g` is a column and `c != 0` (or the clip is asymmetric), a model scores by
recovering `g` from volatility proxies and never forecasts a return. The engine
calls this a structural term and refuses the run with exit 5 (`D13c`), also when
the scale column is left out of `--vars`: other volatility columns recover most
of it.

| measured on the compact pinned cell | share of the rung's gain recovered without the scale column in `--vars` |
|---|---|
| `parkinson96` | 92 % (Spearman 0.981 with `vol96`) |
| `natr14` | 69 % |
| `vol16` | 54 % |
| `bb_width96` | 50 % |
| `vol_z2880` | 49 % |

Preferred target: the raw log return (`offset 0`, constant scale 1, no clip),
cost charged outside the label. Allowed: a gross vol-scaled return with a
symmetric clip and a scale that ends at or before the decision time
(`scale.ends_at_offset_seconds <= entry_offset_seconds + delay_seconds`).

| detector | refuses | override |
|---|---|---|
| `D13a` | `return_column`, the target or any label-family column (`lbl_*`) in `--vars` | none |
| `D13b` | a scale window that ends after the decision time | none |
| `D13c` | a structural term | `--allow-target-structure` (diagnostic replays) |

The spec sha256 is written into every checkpoint. `--resume` under a different
spec, or under none, is refused with exit 2 and a message naming both hashes,
even when the purge removes the same rows and the data fingerprints agree. A
spec's `note` (for example a survivorship or point-in-time statement) is echoed
in `run.json` as `target_spec.note`; the engine adds no statement of its own.

A construction column in `--vars` without a structural term runs and is recorded
in `target_spec.check.construction_in_vars`. Under the override the run prints a
`# WARNING` line, records `check.status = "overridden"` and a proxy table
(Spearman correlation of each input with the scale column on a 65,536-row
stride sample of training rows, top five).

Every run with a spec prints and records the zero-parameter rung
`baselines.target_structure`: the error of `m0` on the training rows, the
validation slice and the test blocks, in the run metric. For the raw return this
is the zero forecast, the out-of-sample benchmark of Gu, Kelly and Xiu (RFS
2020, eq. 19). `TargetSpec::return_from_target` maps a label-space value back to
a return for tools that report return-space metrics.

With a structural term the run also fits the affine rung `a + b*S`, the 20-bin
`E[y|S]` and the OLS rung on the scale column, all on fit-train only, and reports
all four in `baselines.target_structure.rungs` with train, validation and test
errors (`docs/overfitting.md` section 3.10b). A spec with a constant scale and a
zero offset has no structure: the reference is the mean. Per-member flags
`beats_zero_param_test` and `beats_zero_param_validation` say whether a member
beats the zero-parameter rung beyond noise.

**Verify.** `cargo test --release target::`, `tests/target_spec.rs`; the golden
cell test `tests/target_golden.rs` (ignored; needs `RCSR_COMPACT_CELL`).

---

## 3. Decide overlapping labels once

**Check.** If labels overlap (a 5-bar return sampled every bar, or a rolling
maximum), choose one policy and apply it everywhere:

* **purge** `h` rows at every boundary and between CV folds, or
* **sample** the target every `h` bars so labels do not overlap.

Do not mix: a purged train/test split with overlapping folds in `--cv` still
leaks inside cross-validation.

**What fails.** Overlapping labels make neighbouring rows almost duplicates.
A random or unpurged CV estimate reports tiny variance and an optimistic mean;
the model is not better, the rows are not independent.

**Verify.**

* Label horizon written down next to the command.
* `--gap` (and `--cv purged --cv-k K --gap h`) reflect it.
* `tests/cv.rs::purged_embargo_removes_training_rows` and
  `tests/cv.rs::rolling_folds_expand_forwards`.

---

## 3b. Long-only labels and their purge

**Check.** `rcsr::labels` builds the labels of a long-only book in exact CPU f64. For a row at bar `t`, horizon `h` bars and delay `d` bars:

```text
P0       = open[t+1+d]                       entry
fwd      = ln(close[t+h+d] / P0)             long gross log return
sigma_H  = max(s_W, floor) * sqrt(h)         s_W: population sd of the last W one-bar log returns (the vol<W> column)
y        = clip(fwd / sigma_H, -3, +3)       gross regression target
long_net = fwd - c                           c = 2*fee + spread + slippage (+ funding paid by a held long perp)
action   = 1 if fwd > c else 0               {0, +1}; a negative return is "stay flat", never -1
```

The triple barrier is long side only: profit `P0*exp(+u*sigma_H)`, stop `P0*exp(-d*sigma_H)` (exit to flat), vertical bar `t+h+d`. A bar that touches both barriers exits at the stop. Barrier levels are the fill prices. `t1` is the end of the exit bar and never exceeds the end of the vertical bar. `m_net = 1[exit return > c]`, `m_barrier = 1[profit first]`, and the meta label `1[exit return - c > 0]` exists on primary-long rows only. Funding is charged only on a held long position and only when a funding column is supplied; a missing funding value makes the row `NaN`. There is no short label, barrier or borrow term: `--side short` and `--allow-short` are refused.

A row is `NaN` when any bar from `t` to `t+h+d` is missing, partial or has a non-positive price. No label bridges a gap and nothing after `t+h+d` is read.

**Purge.** `purge_seconds = entry_offset + h + delay`, derived from the spec and never from a name: a 60 minute label on 15 minute bars purges 4500 s, a 60 hour label on hourly bars 219600 s. `LongLabelSpec::target_spec` writes the matching `rcsr-target-spec/1` entry (`long_only: true`, `return_column`, scale column `vol<W>`), so the split purges `required_purge_seconds` around every validation and test row. `--gap` stays row based and is ignored with explicit `--train-select`/`--test-select`; `rcsr protocol selectors` prints selectors that carry the purge, and `rcsr walkforward` refuses a window whose gap is shorter than `--entry-offset + --horizon + --delay`.

**Overlap.** Uniqueness weights `u_i = mean over the label's bars of 1/c_t` and `w_i = u_i*n/sum(u)` are stored with the labels (`c_t` is the number of labels covering bar `t`). The report records `effective_n = sum(u)` and `effective_n_hours = effective_n * horizon hours`. A stride below the horizon is refused for horizons of 4 h and longer unless `--allow-overlap` is given (`rcsr enrich`, `rcsr prep`); stride at `h` is the primary fix.

**Report.** The label report lists the share of rows clipped at the bound, the sign shares of `fwd`, its mean and the 1, 5, 25, 50, 75, 95 and 99 % quantiles, the cost-to-move ratio (round trip over mean absolute `fwd`), the exit counts, the purge and the label hash (SHA-256 of the definition version, spec and every column). It also lists the 10 largest absolute forward moves with symbol, decision timestamp and value, the replacement for the retired jumps audit. A panel report merges all symbols; its label hash is the SHA-256 of the spec and each symbol's series hash in store order.

**Columns.** `rcsr enrich --label long:<tf|bars>[,name=<id>][,fee=<bp>][,spread=<bp>][,slip=<bp>][,vol=<bars>][,floor=<bp>][,clip=<c>][,up=<u>][,down=<d>][,delay=<bars>][,stride=<n>]` appends the scale column `vol<W>` when the store lacks it, and `lbl_<name>_fwd`, `_y`, `_long_net`, `_action`, `_t1`, `_exit_ret`, `_m_net`, `_m_barrier` and `_weight` (default name `h<bars>`). They are registered in family `label`: the barrier-scaled kinds (`_y`, `_exit_ret`, `_m_net`, `_m_barrier`) are `NaN` for the first `W` bars of a symbol, the PIT rule is "reads bars t+1 through t+h+d". Next to the store it writes `<out>.labels.json` (the report per target column) and a `y` entry in `<out>.targets.json`, so a run on `lbl_<name>_y` loads the spec and echoes the report in `run.json` as `target_spec.label_report`. A label-family column is refused in `--vars` (D13a, exit 5) and is left out of the default variable list. `rcsr prep` keeps the legacy `fwd:` label only.

**Verify.** `tests/long_labels.rs`: `horizon_windows_enter_next_open_and_exit_at_close_exactly`, `action_is_zero_or_plus_one_and_follows_the_cost`, `barrier_hits_profit_stop_and_vertical_in_the_expected_order`, `both_barriers_inside_one_bar_assume_the_stop`, `missing_partial_and_gap_crossing_windows_are_nan`, `purge_is_derived_from_the_spec_not_from_a_name`, `t1_and_the_derived_spec_drive_the_split_purge`, `overlap_is_guarded_and_uniqueness_weights_are_stored`, `spec_parses_and_refuses_what_it_cannot_label`, `top_moves_are_the_largest_absolute_forward_moves_with_stamps`, `accumulator_merges_symbols_and_hash_follows_every_series`, `label_family_columns_are_refused_as_inputs_by_the_d13a_canary`; `tests/enrich.rs`: `long_label_family_is_stored_registered_and_reported`, `legacy_fwd_label_is_unchanged_next_to_a_long_label`, `long_label_dry_run_writes_no_sidecar_and_bad_specs_fail`.

---

## 4. Audit preparation for look-ahead

These flags are correct for i.i.d. tables and dangerous for ordered rows.

| Flag | Why it can look ahead | Safe choice for forecasting |
|---|---|---|
| `--smooth k` | **trailing** mean: row `i` reads only rows `i-k+1..i` and shrinks at the left edge, so the default does not read future bars | leave the default; it is already causal |
| `--smooth-centred k` | **centred** mean: row `i` uses `i+1`; the right half of the window is future data. It is the explicit opt-in and is reported as look-ahead | leave it off, or accept that the model is not a real-time forecast and say so in the report |
| `--fill interpolate` | linear interpolation uses the **next** non-missing value, which can be a future observation; on time-ordered data the engine refuses it without `--allow-interpolate` | `--fill previous` or `--fill drop`; use `interpolate` only with `--allow-interpolate` when the gap is genuinely ignorable and the fill is not a predictor |
| `--outlier-iqr` | bounds are training-only, so this is safe across the split; it can still delete the rare events you are trying to predict | keep `k` large, or exclude outlier-sensitive rows explicitly |
| `--history-frac` | reserves leading rows for lags and drops them from scoring; it does not read ahead | use it with `--lags`; check the reported history rows |
| `--lags K` | takes the previous row **inside the same series**; a gap or group boundary stops the chain | pair with `--gap-break` and `--group-by` |
| `--center` / `--scale` | statistics come from training rows only, then are applied to test | leave the defaults (mean/std) or use robust training statistics |

**Check.** `--smooth` is the *trailing* moving average (row `i` reads only
itself and earlier rows). The old centred window still exists, but only behind
`--smooth-centred k`, which is opt-in and reported as look-ahead:

```sh
# Causal default: row i uses rows i-2, i-1, i (and shrinks at the left edge).
rcsr ... --smooth 3

# Look-ahead opt-in: row i also uses i+1. Only choose this deliberately.
rcsr ... --smooth-centred 3
```

If the target is not observable at prediction time, do not smooth it with
`--smooth-centred`. If inputs are noisy sensors and you need smoothing, the
trailing `--smooth` is safe for a real-time forecast; `--lags` gives a longer
causal memory without averaging away the last bar.

**What fails.** The model learns a target that contains its own future
neighbours; the reported test error is better than any live forecast can be.

**Verify.**

* `tests/prep.rs::smoothing_is_trailing_by_default_and_shrinks_at_the_edges`
  pins the causal default.
* `tests/prep.rs::centred_smoothing_requires_the_explicit_flag_and_is_reported`
  pins the opt-in and its `look-ahead` report.
* `tests/ts_hazards.rs` covers the interpolate refusal and
  `--allow-interpolate` override.
* `src/data.rs` unit test `interpolate_uses_neighbours_and_drops_edges`.
* The report's preparation block says which rows/cells were filled or smoothed
  and names `smooth_mode=trailing|centred`; `--json` has
  `preparation.rows_filled`, `preparation.smoothed` and
  `preparation.smooth_mode`.

### Multi-timeframe features

**Check.** A higher-timeframe value is usable only after its bar has closed.
`rcsr enrich --features mtf` produces the 36 `mtf_<1h|4h|1d>_<leaf>` columns
with the rule `higher.time + higher_tf.seconds() <= base.time`; the partial
current higher bar is never visible. `bar_age` reports the exact seconds since
the last closed higher bar.

* A window counts **higher-timeframe** bars. A missing higher bucket is `NaN`
  and no mean/ATR/RSI window bridges it; Wilder state restarts after the gap.
* If the expected last-closed higher bucket is absent, every value leaf is
  `NaN`; `bar_age` remains finite and tells you how stale the snapshot is.
* The join is by timestamp, not row index, and a prefix truncation must not
  change an earlier value.

**What fails.** Joining the current, still-forming hourly/daily bar gives the
model the future close. Shifting a higher column one bucket earlier is a leak
that can turn a flat strategy into a spectacular one.

**Verify.** `tests/enrich.rs` pins hand-built higher-bar values for all 36
columns, the NaN/gap cases, `bar_age` (0 at the close and mid-bucket values), the
future-shift failure through `ohlcv::lookahead_violations`, and prefix replay.
`tests/panel_pit.rs` and `tests/panel_gpu.rs` repeat the same checks for the
PIT membership as-of join on the CPU reference and the device merge:
`<=` boundaries at `as_of`, chain resets at `member=0`, listing, delisting and migration, prefix
replay at 20 cuts, and a future-shifted column that fails `lookahead_violations`.
`docs/mtf-inventory.json` is the 36-row inventory; `rcsr schema` lists every
column with `family=mtf`, timeframe and PIT rule.

`docs/default_financial_features.json` is the inventory of the whole default financial feature set. Each row names its PIT rule and the tests that prove it: the unit or parity test, the NaN/gap test, and for cross-asset columns the timestamp-join test and for higher-timeframe columns the future-shift test. The compact 52 are a marked subset (`compact:true`), used as the regression fixture.

---

## 5. Select on validation, keep a lockbox

**Check.** Use the train/test split as **validation** while choosing the metric,
operators, complexity cap and feature set. Do not keep re-reading the same
`test_error` hundreds of times. Freeze the expression, then score it once on a
lockbox that was excluded from every choice.

On a time-ordered run the engine designates a selected member only from a
validation slice (`--validation-frac`). Without one the verdict is `unvalidated`,
`selected_rank` is null and the run prints one warning: the front is a menu.
`--require-validation` refuses such a run with exit 2; campaign recipes pass it.
The `--select-by` order is display only. The selected member comes from the
validation gates: a member is supported when it improves on the last supported
one by more than `--equivalence-z` (default 1) Driscoll-Kraay standard errors of
the paired loss differential, and the verdict is `signal` only when the top
supported member beats the reference (the affine rung with a declared
structure, else the mean) by a one-sided Bonferroni threshold at 0.05. Test
errors are scoring-only. For overlapping labels set `--noise-lag` to at least
the horizon in steps minus one; with a target spec the horizon is read from it.

Example with a numeric year column:

```sh
# Development: everything before the lockbox.
rcsr --data d.csv --target y --vars x1,x2 --split temporal --time t \
     --exclude "year >= 2022" --folds 5 --gap 5

# Freeze the expression, then score it once on 2022+ with no search.
rcsr --data d.csv --target y --vars x1,x2 --eval-expr 'the_chosen_expression' \
     --train-select "year < 2022" --test-select "year >= 2022"
```

`--eval-expr` does not search; it parses the formula, scores it on the selected
test rows, and reports train/test error. With a single test block the test error
is the error on that block (`fold_errors` has one entry); `--json-out FILE`
writes the result object to the file. With a target spec the result also carries
`target_spec` and `target_structure`.

**What fails.** Tuning until `test_error` is small and then quoting that same
number is selection on the test set. The lockbox is the only defence left; if
it is also used for choices, there is no holdout.

**Verify.**

* The chosen expression is written down before the lockbox run.
* `--eval-expr` output names the metric and reports `train_error` / `test_error`
  separately; the test side is the lockbox rows.
* `tests/reports.rs::json_stays_valid_when_a_report_is_requested` and the
  report tests cover the non-search evaluation path.

### Exact threshold and the shared mask (walk-forward)

**Check.** When a walk-forward threshold is reported, it must be the exact
type-7 quantile of the train/validation scores, not a sketch. `rcsr walkforward`
selects the two ranks around `h = (n - 1) q` by order-preserving radix count,
canonicalises `-0` to `+0`, drops NaN/`±inf`, and interpolates in `f64`. The
window's `threshold_source` is `train_type7`, `validation_type7` or `fixed`
(a pre-registered `--threshold-fixed` value).

Positions use the shared valid-row mask: a row is usable only when **both** its
score and target are finite. That matters at the boundary — in the pinned eval
cell six rows have a finite score and a `NaN` target. Keeping them would change
exposure and every pooled return even though `position_for` alone would score
the row. `score == theta` is flat (`strict >`).

**What fails.** A histogram/subsample quantile places reported positions, so the
reported P&L depends on the sketch; or the score-only and score-and-target
masks disagree, so the P&L, risk and baseline tables use different
denominators.

**Verify.**

* `tests/pnl_gpu.rs` compares exact radix selection with a CPU sort on 1e6
  scores containing duplicates, NaNs and `±0` (bit-identical `theta` and
  positions), reproduces the six-row NaN-target difference, and checks batch
  and thread-count invariance.
* `--threshold-fixed` is reported as `threshold_source = "fixed"`; a
  screening-only threshold is refused by the position vector.
* `docs/walkforward.md` § Exact threshold carries the contract.

---

## 6. Panel data: groups are not one-hot inputs

**Check.** For a panel (many assets, subjects or sensors), use:

```sh
rcsr --data panel.csv --target next_close --vars close,volume \
     --split temporal-group --time date --group-by symbol \
     --lags 1 --history-frac 0.05 --gap-break 3 --gap 1
```

* `--group-by symbol` keeps each series intact for lags and the split.
* `--split temporal-group` trains on the past of every series and tests on its
  future; `--split group` instead holds out **whole** symbols, which answers a
  different question ("a new asset").
* `--vars` should list **continuous** market/geometry columns. Do **not** add
  the symbol as a categorical input:

```sh
# Wrong for generalisation: one-hot asset identity lets the model memorise
# per-symbol levels instead of learning a transferable relation.
--dtype symbol=categorical --vars close,volume,symbol
```

If asset identity genuinely matters, model it through group-level summaries,
interactions with continuous variables, or a separate model per group. A
one-hot identifier is a lookup table, not a law.

**What fails.** A one-hot symbol turns "which stock is this?" into a free
parameter. The model can special-case the training symbols and has nothing to
say about a new one; a whole-group split exposes it immediately.

**Verify.**

* `tests/split.rs::temporal_group_tests_every_series_and_uses_whole_series_history`.
* `tests/split.rs::group_split_never_puts_a_group_on_both_sides`.
* `tests/dtypes.rs::categorical_expands_to_sorted_one_hot_indicators` (shows what
  the flag actually creates, so its use is deliberate).

### Canonical training order

A per-row P&L book needs each group's rows contiguous and time-ordered. `data::SplitSpec::canonical_train_order` sorts the training rows by (group id, time key, table row) after the centre and scale statistics are computed, so `means` and `stds` do not change. Every train-aligned output (`train`, `keys.train`, `target_scale.train`, `slope_pairs.train`, `trading.train`, `train_indices`) follows the sorted order. A temporal split returns training rows in time order, so a store of any layout needs the flag; a split that keeps table order on a symbol-major store is unchanged by it. `data::TradingColumns::train` builds `TradingSplit.train`; `TradingRows::is_canonical` checks the order and `TradingRows::seam_flags` marks the first and last row of each group. The flag is library-only: no CLI option sets it yet.

### Per-group reporting

The engine reports global test error and the number of groups in train/test, not
one error per group. Check it yourself before accepting an aggregate:

```sh
# Evaluate the frozen law on each symbol separately.
for s in $(tail -n +2 panel.csv | cut -d, -f3 | sort -u); do
  awk -F, -v s="$s" 'NR==1 || $3==s' panel.csv > "/tmp/panel-$s.csv"
  rcsr --data "/tmp/panel-$s.csv" --target next_close --vars close,volume \
       --eval-expr 'the_chosen_expression' --split temporal --time date --gap 1 \
       --json | grep -o '"test_error":[^,}]*'
done
```

**What fails.** One profitable symbol can carry the mean while the model loses
on every other symbol. The per-group spread, not the pooled mean, is the result.

### Cross-sectional context is a PIT join, not a survivor list

`rcsr enrich --features cross` produces equal-weight market state, breadth,
dispersion, the average-correlation proxy and `[-1,1]` ranks. Treat them as
time-clustered panel columns:

* the statistic at `t` uses only rows live at `t` (`member_asof <= t` when the
  store carries it); a pre-listing or delisted row is excluded and its own
  cross-sectional output is `NaN`;
* `N < 20` valid members makes the timestamp `NaN`, never a rank over a
  partial cross-section;
* benchmark joins (`btc_*`, `eth_*`) use the exact timestamp; a missing
  benchmark bucket is `NaN`, never forward-filled;
* a cross-sectional rank is a property of the timestamp, so effective `N` for
  inference is the number of timestamps, not the number of asset rows. Block
  bootstrap by time (and, when appropriate, by symbol cluster), not i.i.d. rows.

The tests
`tests/enrich.rs::cross_family_is_exactly_the_registry_and_market_state_is_ex_self_pit`,
`a_partial_cross_section_is_nan_and_a_prefix_is_causal` and
`btc_and_eth_joins_use_the_same_timestamp_and_a_shift_fails` pin these rules.

---

## 7. Count every trial, then deflate

**Check.** The effective number of trials is not the number of reported rows.
Record it:

```text
N_trials = islands × pop × generations × seeds
         × CV folds actually inspected
         × metrics/splits/feature sets tried
```

At a minimum, print the search budget and every rerun:

```sh
rcsr ... --generations 200 --islands 4 --pop 200 --seed 0 --json
echo $((4 * 200 * 200))          # one run, before seeds/metrics/splits
```

If you tried five feature sets and three metrics, multiply by 15. If you ran
`--cv 5`, multiply by the folds you actually read. Include `--stability` and
`--null-runs` runs in the count too.

The engine does **not** implement deflated Sharpe (DSR), probability of
backtest overfitting (PBO) or the SPA test. Compute them from the trial return
series and report the inputs:

* **DSR** — deflate the observed Sharpe ratio by the expected maximum of
  `N_trials` independent trials, using the return skewness and kurtosis; report
  `N_trials`, the raw Sharpe, and the deflated probability.
* **PBO** — combinatorial symmetric cross-validation (CSCV): split the return
  matrix into `S` blocks, rank the best in-sample strategy out-of-sample, and
  report the fraction of splits where it lands below the median.
* **SPA** — bootstrap the maximum excess return over the benchmark across all
  trials; report the p-value after the same trial count.

If those numbers are not in the report, the result is a candidate, not a
validated strategy. The in-engine controls are useful but not substitutes:

```sh
# Is the fit better than the identical search on shuffled targets?
rcsr ... --null-runs 19
# Does the same structural form survive resampling?
rcsr ... --stability 50
```

**What fails.** A Sharpe of 2 looked up after 10 000 candidate expressions is
not the same statistic as a Sharpe of 2 from one pre-registered model. Quoting
it without `N_trials` is the multiple-testing error.

**Verify.**

* Null control: `diagnostics::tests::permutation_p_value_is_small_when_observed_wins`.
* Stability: `tests/stability.rs::a_true_term_is_stable_and_a_spurious_one_is_not`.
* Record the trial count and the DSR/PBO/SPA inputs in the run report; a missing
  trial count fails the audit even if the p-value looks good.

---

### Baselines and ablation folds

`rcsr ladder` and `rcsr ablation score` take their folds from the protocol windows table. A fold fits on `[train_start, validation_start - purge)`, selects on `[validation_start, train_end)` and tests on `[test_start, test_end)`; `train_end` is already purged from the test start, and `--purge-seconds` purges the fit range from the validation slice. Every block starts flat and pays its final exit, so no position carries across a fold. A placebo column keeps the marginal distribution and the lag-1 autocorrelation of its template and nothing else; a permuted column is a per-symbol circular-block shuffle whose columns share one permutation. Both are generated with the seed in the arm's manifest, never from the test range. See [`overfitting.md`](overfitting.md#313-baseline-ladder-and-add-one-family-ablation-implemented).

## 8. Costs and horizon: the break-even hit rate

**Check.** A directional strategy must beat its round-trip cost before any
statistical claim matters. The roadmap measurement recorded in issue #115 for a
**10 bp round trip** gives these break-even hit rates:

| Bar interval | Break-even hit rate |
|---|---:|
| 1 minute | **113%** |
| 15 minutes | **70%** |
| 60 minutes | **60%** |
| Daily | **52%** |

The 1-minute row exceeds 100%, i.e. at a 10 bp round trip the required hit rate
is unattainable: do not trade that horizon at that cost. The point is not the
exact number for your venue; it is that **costs are a first-order term**, not a
rounding error.

**What fails.** A model with 53% directional accuracy and a positive gross
backtest is assumed to be profitable. At 1-minute frequency, 53% is far below
the break-even. Always subtract fees, spread and slippage before comparing
Sharpe ratios or hit rates.

**Verify.**

* Put the assumed round-trip cost and bar interval next to every hit-rate or
  Sharpe number.
* Recompute the break-even with your own fee model; if the cost assumption
  changes, the table changes.
* The engine's `test_error` is a statistical error, not a P&L; do not compare it
  directly to a trading return.

---

## 9. Feature engineering: geometry, not named patterns

**Check.** Prefer a small set of continuous, scale-aware features:

* returns and log-returns over several horizons;
* volatility (rolling standard deviation) and range;
* volume/participation ratios;
* distances from moving averages or bands, in units of volatility.

`--lags K` materialises past values; `--ops` can restrict the search to smooth
functions. A named candlestick pattern ("morning star", "engulfing") is a
discrete, hand-labelled feature: it adds parameters, is unstable across assets,
and is rarely the primitive that generalises.

Redundancy control matters more than indicator count:

```sh
# Start with a handful of geometry columns and one lag.
rcsr --data d.csv --target y --vars ret_1,vol_20,range_20,volume_ratio \
     --lags 1 --split temporal --time date --gap 5 \
     --max-complexity 20 --metric absolute
```

**What fails.** Twenty correlated indicators let the search choose whichever
one fits the validation window; the expression is long, unstable and no better
out of sample. The stability report shows the chosen form changing across
resamples.

**Verify.**

* `--stability 50` — a true term survives; a redundant one does not
  (`tests/stability.rs::a_true_term_is_stable_and_a_spurious_one_is_not`).
* AICc/BIC and the complexity column: a larger expression must earn its extra
  constants.
* Drop one of two highly correlated inputs and rerun; if the result collapses,
  the signal was redundancy, not geometry.

---

## Copy-paste audit

```sh
# 1. Structure and split
rcsr --data d.csv --target y --vars x1,x2 \
     --split temporal --time date --test-frac 0.25 --gap 5 \
     --folds 5 --json > run.json
grep -o '"mode":"[^"]*"' run.json                  # must not be random
grep -o '"gap":[0-9]*' run.json                    # must be >= label horizon
grep -i 'gap.*ignored' run.json                    # must be absent

# 2. Honest evaluation
rcsr ... --cv purged --cv-k 5 --gap 5
rcsr ... --null-runs 19
rcsr ... --stability 50

# 3. Freeze, then lockbox
rcsr ... --eval-expr 'frozen_expression' \
     --train-select "year < 2022" --test-select "year >= 2022"

# 4. Count trials and add costs
echo $((islands * pop * generations * seeds * folds))
# report DSR, PBO and SPA with that trial count and the round-trip cost
```

If every item above has a number, a command or a written justification, the
result is reviewable. If any item is missing, the default conclusion is "not
yet a forecast".

---

## Long-only evaluation protocol

For the hourly panel the windows, purge, locked test and uncertainty are fixed in `protocol/windows.csv` and `docs/walkforward.md` (Long-only evaluation protocol). The checklist for a long-only book:

* **Purge in time.** `purge_seconds >= entry_offset + h + delay` for every symbol; a late listing or a gap in one symbol never puts a training label inside a test window. Verify: `tests/eval_protocol.rs::no_training_label_reaches_a_test_window_on_a_ragged_panel` and `tests/protocol.rs::generated_selectors_leave_the_required_purge_on_a_ragged_panel`.
* **Row counts are not sample sizes.** 4.78 M rows hide at most about 78 k hourly time points and about 1300 non-overlapping 60 h labels. Report `effective_n` from the block bootstrap design effect, in hours.
* **Block, not row, uncertainty.** A circular block bootstrap with `L >= max(h, holding period)`. Verify: `tests/inference.rs::iid_over_rejects_on_ar1_and_the_block_does_not`.
* **Benchmarks beside the book.** Cash, buy-and-hold, equal-weight universe and the best long/flat baseline, with exposure and beta, so market drift is visible.
* **One read of the locked test**, after `locked_candidates.json` is sealed. A tweak and rerun is a new trial.
