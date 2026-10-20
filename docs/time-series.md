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

---

## 2. Purge the label horizon and embargo the boundary

**Check.** `--gap N` drops `N` rows immediately after the train/test boundary
(per series in `temporal-group`). Set it from the target, not from habit:

```text
N >= label horizon h
```

If the last training label is built from the next `h` bars, those bars must not
be test rows. For a 5-bar forward return, `--gap 5` is the minimum. Add any
additional look-ahead introduced by preparation (for example the right half of a
centred smoothing window; see §4) to `N`.

If features only look **backwards** (`--lags`), no extra gap is needed for the
lookback itself: a lag taken from before the boundary is information the
forecaster would have had. The leak is the **label** crossing the boundary.
When in doubt, use the conservative `h + lookback` and check whether the extra
rows cost you the result.

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

## 4. Audit preparation for look-ahead

These flags are correct for i.i.d. tables and dangerous for ordered rows.

| Flag | Why it can look ahead | Safe choice for forecasting |
|---|---|---|
| `--smooth k` | a **centred** mean: row `i` uses `i+1`; the right half of the window is future data | leave smoothing off, or smooth the *input* file causally before loading (only past rows), or accept that the model is not a real-time forecast |
| `--fill interpolate` | linear interpolation uses the **next** non-missing value, which can be a future observation | `--fill previous` or `--fill drop`; use `interpolate` only when the gap is genuinely ignorable and the fill is not a predictor |
| `--outlier-iqr` | bounds are training-only, so this is safe across the split; it can still delete the rare events you are trying to predict | keep `k` large, or exclude outlier-sensitive rows explicitly |
| `--history-frac` | reserves leading rows for lags and drops them from scoring; it does not read ahead | use it with `--lags`; check the reported history rows |
| `--lags K` | takes the previous row **inside the same series**; a gap or group boundary stops the chain | pair with `--gap-break` and `--group-by` |
| `--center` / `--scale` | statistics come from training rows only, then are applied to test | leave the defaults (mean/std) or use robust training statistics |

**Check.** `--smooth`:

```sh
# smooth is centred: row i uses i-1, i, i+1 (and shrinks at the edge)
rcsr ... --smooth 3
```

If the target is not observable at prediction time, do not smooth it. If inputs
are noisy sensors and you need smoothing, build a causal moving average outside
`rcsr` or use `--lags`.

**What fails.** The model learns a target that contains its own future
neighbours; the reported test error is better than any live forecast can be.

**Verify.**

* `tests/prep.rs::smoothing_uses_a_centred_window_and_shrinks_at_the_edges`.
* `src/data.rs` unit test `interpolate_uses_neighbours_and_drops_edges`.
* The report's preparation block says which rows/cells were filled or smoothed;
  `--json` has `preparation.rows_filled` and `preparation.smoothed`.

---

## 5. Select on validation, keep a lockbox

**Check.** Use the train/test split as **validation** while choosing the metric,
operators, complexity cap and feature set. Do not keep re-reading the same
`test_error` hundreds of times. Freeze the expression, then score it once on a
lockbox that was excluded from every choice.

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
test rows, and reports train/test error.

**What fails.** Tuning until `test_error` is small and then quoting that same
number is selection on the test set. The lockbox is the only defence left; if
it is also used for choices, there is no holdout.

**Verify.**

* The chosen expression is written down before the lockbox run.
* `--eval-expr` output names the metric and reports `train_error` / `test_error`
  separately; the test side is the lockbox rows.
* `tests/reports.rs::json_stays_valid_when_a_report_is_requested` and the
  report tests cover the non-search evaluation path.

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
