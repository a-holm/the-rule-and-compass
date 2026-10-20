# Walk-forward evaluation (`rcsr walkforward`)

`rcsr walkforward` evaluates one expression (or selects one of several on a
validation slice) on a sequence of time windows, applies a long/flat position
rule and reports trading metrics. It is the Rust
replacement for the pilot's Python backtest: it reads the `.rcsb` column store
directly, uses the engine's own `expr::eval` (same f64 semantics as
`--eval-expr`), and never shells out to Python or evaluates a code string.

```sh
rcsr walkforward \
  --data panel_daily_long.rcsb --target fwd1 \
  --expr 'logistic(r60) - tanh(r120)' \
  --vars r60,r120 --time ts --group-by symbol \
  --horizon 172800 \
  --window '2019,0,1546128000,1546300800,1577836800' \
  --cost-bps 10 --threshold-quantile 0.7 \
  --periods-per-year 365 --json
```

## Windows and leakage

Each `--window` (or row of a `--windows` CSV with columns
`label,train_start,train_end,test_start,test_end`) defines two half-open time
ranges: `[train_start, train_end)` and `[test_start, test_end)`. Windows are
reported and pooled in test-time order; overlapping test blocks are refused.

The purge gap is `test_start - train_end`. It must be at least both `--horizon`
(the forward-label horizon) and `--embargo` (default: the horizon). Rows in the
gap are never used. The threshold is computed only from training rows, or from
the trailing `--validation-frac` of the training time range when that option is
set; the test block is never consulted. Quantiles use linear interpolation
(numpy's default, type 7), so a reference run that uses `np.quantile(x, q)`
gets the same threshold bit for bit.

There is intentionally no arbitrary test-block selector: the test block is
defined by the window's time bounds.

## Selecting among candidates

`--expr` may be repeated. With more than one expression, every window selects
the candidate with the lowest mean squared error against the target on the
trailing `--validation-frac` slice; the threshold then uses that candidate's
validation scores. The choice is frozen before the test block is read, and the
report records `selection.selected_on = "validation"` and
`selection.test_used_for_selection = false`. Multiple candidates without a
validation slice are refused, because the only remaining alternative would be
selection on test. A single `--expr` keeps the original behaviour.

## Position rule, costs and aggregation

* `position = 1` if `score > threshold` and the score is finite, otherwise `0`.
  There is no short leg and no leverage; `rcsr::walkforward::position_for`
  exposes the rule.
* `--cost-bps` is charged **per side** on `|position - previous position|`,
  separately inside each `--group-by` series. Per-window metrics start every
  series flat, so an entry costs one side and its exit costs the other: a round
  trip is `2 * cost_bps`.
* **Pooled P&L carries the seam.** The pooled path is the concatenation of the
  window paths, so a group that ends window `k` long and starts window `k+1`
  long is not charged again; a change is charged once at the first row of that
  group in the next window. Total variation is therefore
  `TV(p·q) = TV(p) + TV(q) + |p_end - q_start|`, and the JSON
  `pooled_seam.seam_cost`/`seam_trades` report the extra boundary term.
  `--pool-reset-flat` restores the old behaviour (every boundary starts flat),
  which is useful for reproducing pre-seam reports but is not the real path.
* Rows sharing a timestamp are equal-weighted, exactly as in the pilot: the
  period return is the mean net return over the groups present at that
  timestamp. With no `--group-by` the whole table is one series. The internal
  `CrossSection` carries `(sum, count)` per timestamp and can be merged across
  symbol shards, so a symbol-parallel evaluation gives exactly the sequential
  series even when a symbol is missing at some timestamps.
* Costs and positions are per group. Counting position changes on a flattened
  timestamp/symbol row order would count spurious changes at every symbol
  boundary; this implementation does not.

## Metrics

Per window and pooled:

| metric | definition |
|---|---|
| Sharpe | `mean / std(ddof=1) * sqrt(periods_per_year)`; `n/a` when the series is too short or has zero variance |
| CAGR | `equity_end^(periods_per_year / periods) - 1` |
| max drawdown | most negative `equity / running_peak - 1` (starts from equity 1) |
| turnover | `sum |position change| / years`, where `years = periods / periods_per_year` |
| exposure | long rows / test rows |
| hit rate | long rows with `target > 0` / long rows |
| break-even hit rate | `(average loss + realised cost per long row) / (average win + average loss)` |
| trades | position changes (an entry and its exit count as two) |
| entries | flat-to-long transitions |
| rows / periods | test rows and distinct timestamps in the return series |
| total cost / total variation | charged cost and `sum |position change|`; pooled values include the carried seam |

`--periods-per-year` should be given when the time unit is not epoch seconds.
Without it, the evaluator infers the period from the median positive time step
of the test rows (`365*86400 / dt`), which gives 365 for daily crypto bars and
8760 for hourly bars. The report always states the value it used.

## Fold sign test

The report also carries a one-sided binomial sign test over the per-window mean
net returns, `P(X >= positive)` under `Bin(total, 0.5)`. It is only computed
when the folds are independent: if any training interval overlaps another
fold's test block, the test is reported with `applicable: false`, `p_value:
null` and the reason. Expanding walk-forward always trips this because
`test_k` sits inside `train_{k+1}`; use disjoint rolling folds or a block
bootstrap over the return series instead of pretending the folds are a product.

## Output

`--json` prints the complete report to stdout: expression, variables, window
definition, purge/embargo, cost model, position rule, threshold rule,
annualisation, candidate/selection provenance, pooled seam accounting, fold
sign test, per-window metrics and pooled metrics. `--out file` writes the
same JSON, `--markdown file` writes a Markdown report and `--markdown -` prints
it to stdout. Without those flags a compact text table is printed.

## Store handling and memory

A `.rcsb` path is opened with `store::FileStore` and scanned one row group at a
time; the resident state is one group plus the selected test rows and one score
per training row per window. A CSV path is read as a resident `data::Table` and
accumulated with the same rule. The JSON `memory_bound` field states the bound
for the run.

## Reference agreement

On the saved 2019 window of `panel_daily_long` (expression from `wf/2019.json`,
10 bp per side, q70, 365 periods/year), the Rust evaluator and the pilot's
Python P&L agree exactly on the threshold (`-0.017029354682699175`), test rows
(3702), exposure, Sharpe, CAGR, max drawdown, turnover and mean return (to
machine precision). The Python script counts trades on the flattened
timestamp/symbol order, which adds cross-symbol boundary changes (456 vs the
correct 352 per-group changes); that is why its trade count and its cost-based
break-even hit rate differ. The per-group turnover and every P&L number match.

## Performance and the 10x requirement

Measured with the release binary on an Intel i7-3770K (8 threads), 31 GiB,
Linux, using the 3,943,856 x 56 hourly store
(`/mnt/lager/tmp/opencode/panel_hourly_v2.rcsb`, 2024 test window, 850,584 test
rows, target `y60`, expression `r16`, streamed one 65,536-row group at a time):

| run | wall | peak RSS |
|---|---:|---:|
| `rcsr walkforward ... --group-by symbol --horizon 172800` | 1.49 s | 110 MB |

A direct 10x comparison against the pilot Python path on this 4M-row panel is
**not runnable from this checkout**, for two concrete reasons:

1. The Python reference (`pilot/evaluate_walkforward.py`) reads
   `pilot/panel_daily_long.csv` (154,674 rows), not the hourly store. There is
   no CSV projection of the 3.9M-row hourly panel in the repository or in the
   benchmark workspace, and the source 25 GB CSV the issue mentions is not
   available either. Rebuilding it would create a multi-gigabyte text artifact
   just to time a path the Rust tool no longer uses.
2. The 60-second benchmark cap applies, while the Python script loads every
   column into per-row Python objects (`csv.DictReader`) before evaluating any
   expression; its 154k-row run already takes 2.48 s just to reach the fourth
   window, and it aborts there because `pilot/wf/2022.json` is empty. Extrapolating
   the row-count factor alone puts the 3.9M-row Python path far above the cap,
   and the memory footprint above the 31 GiB host.

The honest claim is therefore: the Rust evaluator reads the store directly and
completes the 4M-row window in 1.49 s with 110 MB RSS. The 10x box stays open
until the missing hourly CSV (or the Python reference's store reader) and its
exact artifact are available; no number is fabricated from a different fixture.
