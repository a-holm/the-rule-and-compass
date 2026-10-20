# Walk-forward evaluation (`rcsr walkforward`)

`rcsr walkforward` evaluates one expression on a sequence of time windows,
applies a long/flat position rule and reports trading metrics. It is the Rust
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

## Position rule, costs and aggregation

* `position = 1` if `score > threshold` and the score is finite, otherwise `0`.
  There is no short leg and no leverage; `rcsr::walkforward::position_for`
  exposes the rule.
* `--cost-bps` is charged **per side** on `|position - previous position|`,
  separately inside each `--group-by` series. Every series starts flat, so an
  entry costs one side and its exit costs the other: a round trip is
  `2 * cost_bps`.
* Rows sharing a timestamp are equal-weighted, exactly as in the pilot: the
  period return is the mean net return over the groups present at that
  timestamp. With no `--group-by` the whole table is one series.
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

`--periods-per-year` should be given when the time unit is not epoch seconds.
Without it, the evaluator infers the period from the median positive time step
of the test rows (`365*86400 / dt`), which gives 365 for daily crypto bars and
8760 for hourly bars. The report always states the value it used.

## Output

`--json` prints the complete report to stdout: expression, variables, window
definition, purge/embargo, cost model, position rule, threshold rule,
annualisation, per-window metrics and pooled metrics. `--out file` writes the
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
