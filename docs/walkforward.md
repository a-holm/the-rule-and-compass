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
  separately inside each `--group-by` series. Every series starts flat, so an
  entry costs one side and its exit (including the **final** exit of every
  group) costs the other: a round trip is `2 * cost_bps`. #145 changed the
  final exit from free to charged; a missing exit would flatter the strategy
  by one side.
* Rows sharing a timestamp are equal-weighted, exactly as in the pilot: the
  period return is the mean net return over the groups present at that
  timestamp. With no `--group-by` the whole table is one series.
* Every window is checked against the 2026H2 lockbox
  `[2026-07-01, 2027-01-01)`, on **both** sides: a training range inside the
  holdout is refused even when the test block is in 2027. `--allow-lockbox` is
  the only override (declared final test only) and is echoed as
  `"lockbox": {"permitted": true, ...}` in the JSON report. The same
  `rcsr::lockbox::Guard` is used by `rcsr run` (the default search),
  `rcsr panel build --start/--end` and a labelled `rcsr prep`.
* Costs and positions are per group. Counting position changes on a flattened
  timestamp/symbol row order would count spurious changes at every symbol
  boundary; this implementation does not.

## Metrics

Per window and pooled:

| metric | definition |
|---|---|
| Sharpe | `mean / std(ddof=1) * sqrt(periods_per_year)`; `n/a` when the series is too short or has zero variance |
| CAGR | `equity_end^(periods_per_year / periods) - 1` |
| max drawdown | most negative `equity / running_peak - 1` (starts from equity 1); `-1` on ruin |
| Sortino | `(mean - MAR) / DD * sqrt(periods_per_year)`, where `DD = sqrt(sum(min(r-MAR,0)^2)/n)` (population, `ddof=0`) and `MAR` is 0 by default |
| log-growth | mean of `log1p(r)` per period; `-inf` when any `r <= -1` (ruin) |
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
8760 for hourly bars. The report always states the value it used and, since
#307, also records it in `risk_model` with the annualisation mode.

**Per-period versus daily.** Every risk statistic in this report is computed on
the per-period series and annualised with `sqrt(periods_per_year)` (Sharpe,
Sortino) or `periods_per_year` (alpha). That is a *per-period* statistic: hourly
returns with `periods_per_year = 8760` are not the same as a daily Sharpe. For a
daily Sharpe, compound the hourly returns into daily returns first
(`risk::compound_blocks(&returns, 24)`) and then annualise the daily series with
`periods_per_year = 365`. Cash (zero variance) reports `NaN` Sharpe and `NaN`
Sortino, never `0`.

## Risk monoids

`rcsr::risk` exposes the exact `f64` chunk-merge carries behind these numbers
(#307): Welford/Chan `(n, mean, M2)` for mean/variance/Sharpe, `(n,
sum(min(r-MAR,0)^2))` for Sortino, `(n, sum(log1p(r)), ruined)` for log-growth,
`(n,total,min,max,dd,ruined)` on log equity for drawdown, and `(n, sum x, sum y,
sum xy, sum x^2)` for beta/alpha. Each carries a fixed canonical merge order;
`RiskAccumulator::push_chunk` buffers a trailing partial so arbitrary chunk
sizes give identical bits, and `merge` appends canonical shards in ascending
order. The serial `walkforward::sharpe` and `equity_metrics` remain the
reference; the risk module is tested to `1e-12` against them on 100 k returns.
There is no short or borrow kernel: the input is the long-only period series.

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
(`arbeidsmappe/crypto-rcsr/pilot/panel_hourly_v2.rcsb`, 2024 test window,
target `y60`, expression `r16`, streamed one group at a time). The purge-matched
window is train `[2021-01-01, 2024-01-01)`, test `[2024-01-03, 2025-01-01)`,
horizon 172,800 s, q70, 10 bp/side, 8,760 periods/year.

| side | wall | peak RSS | q70 threshold |
|---|---:|---:|---:|
| pinned Python row-wise reference (`benchmarks/eval_python_reference.py`) | 81.4 s | 104.9 MiB | `0.007034873088967409` |
| `rcsr walkforward --group-by symbol --horizon 172800` | 4.06 s | 107.7 MiB | `0.007034873088967409` |

The committed warm-cache report measures **20.1x**, above the 10x target. The
feasibility pass measured a colder cache and reported **51x–75x** for the two
pilot Python variants (125.5 s and 185.3 s against 2.46 s); the range is
cache/load-sensitive but every measurement clears 10x. The threshold is
bit-identical in both. Test rows differ by 6 out of 845,928 because the Rust
evaluator drops non-finite score/target rows while the Python reference still
charges their cost; see the report's `test_row_delta`.

The full protocol — corpus hashes, window, command, why the reference is
row-wise and not a vectorised rewrite, and the generated evidence — is in
[`eval-benchmark.md`](eval-benchmark.md) and
[`../benchmarks/results/issue-124-eval-10x.md`](../benchmarks/results/issue-124-eval-10x.md).
Run it with `python3 benchmarks/eval_10x.py --repeats 2` or the ignored test
`walkforward_bench`. The 10x box is evidenced by the ratio gate in that report,
not by a fixed wall time on a different host.

