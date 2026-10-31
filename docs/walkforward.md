# Walk-forward evaluation (`rcsr walkforward`)

`rcsr walkforward` evaluates one expression (or selects one of several on a
validation slice) on a sequence of time windows, applies a long/flat position
rule and reports trading metrics. It reads the `.rcsb` column store directly, uses the engine's own `expr::eval` (same f64 semantics as
`--eval-expr`), and never shells out or evaluates a code string.

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

### Exact threshold (#305)

The threshold is a **discrete input**: a one-ulp difference flips every row
whose score sits on the boundary. `rcsr` therefore selects the type-7 quantile
exactly instead of sorting or sketching:

* each finite score is mapped to an order-preserving unsigned key; negative
  zero is canonicalised to positive zero and NaN/`±inf` are excluded, exactly
  as the reference `quantile` filters them;
* the two ranks bracketing `h = (n - 1) * q` are selected by a 256-bin radix
  count, and the interpolation `lo + (h - lo) * (hi - lo)` is evaluated in
  `f64` with the same expression as numpy's type 7;
* the radix count is integer-only, so chunk boundaries, batch composition and
  thread count cannot change `theta`.

`--threshold-fixed <theta>` supplies a pre-registered constant instead. The
window report names the provenance in `threshold_source`:
`train_type7`, `validation_type7` or `fixed`. An approximate/sketch threshold
may be used to screen candidates, but it is labelled screening-only and cannot
place a reported position.

### Shared valid-row mask (#305)

A row enters the P&L denominator only when **both** its score and its target
are finite (`trading_gpu::ValidMask`). This is deliberately stricter than
`position_for`, which only checks the score: in the pinned cell six rows have a
finite score and a `NaN` target, and keeping them would change exposure, hit
rate and every pooled return. `position = 1` iff the mask says the row is valid
and `score > theta`; `score == theta` is flat.

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

### Banded finalist selection (#308)

`--select-band` replaces validation MSE as the selection statistic with the
long-only P&L band. It is for the case where the search ranks candidates on a
fast surrogate and the report must return the exact winner: a near tie on the
screen must not hide a different candidate on the full rows.

For each window, after the exact threshold is frozen on the validation slice:

1. score every candidate on the validation rows with the fused long-only
   reduction (`trading_gpu::reduce_device` when the CUDA feature and a device
   are present, otherwise the identical canonical CPU segments). The value is a
   **screening** score only; it never appears as a reported P&L;
2. with `P` candidates set `K = min(P, max(8, P/16))` and
   `epsilon = max(2e-3, 0.005 * |J_K|)`, where `J_K` is the screening score at
   rank `K`;
3. admit every candidate with a finite screening score at least
   `J_K - epsilon`; non-finite screens are never admitted;
4. rescore every admitted candidate on the full validation rows with the exact
   CPU `f64` path (`selection::rescore_exact`), using the same threshold, the
   shared valid-row mask and the canonical `(group, time, row)` order;
5. order the admitted set by `(higher net, lower turnover, lower complexity,
   lower index)`. The comparison is a total order: net `+inf` is best, net
   `-inf`/`NaN` is worst, turnover `-inf` is best and turnover
   `+inf`/`NaN` is worst, with `f64::total_cmp` inside each class.

The function also rescores the out-of-band candidates as an audit and reports
`recall_band`. If the exhaustive exact winner is outside the band the run fails
loudly instead of reporting the wrong finalist. The JSON block carries
`surrogate_rank`, `exact_rank`, `band` (cut rank, cut score, epsilon, lower
bound, `finalists_considered`), `recall_band` and `scored_by: "cpu_f64"`.
`--select-band` requires more than one `--expr` and `--validation-frac > 0`;
the test block is never read before the candidate is frozen.

`epsilon` is measured, not assumed. The 200-candidate fixture in
`tests/pnl_gpu.rs` plants the exact CPU winner `0.0005` below the screening cut;
with `P = 200`, `K = 12`, `J_K = 0.4989` and
`epsilon = max(0.002, 0.005 * |J_K|) = 0.0024945`, so the admitted band starts
at `0.4964055`. The test then requires recall 1.0 and equality of the reported
finalist with exhaustive CPU selection. A wider band costs only exact CPU
rescores; a narrower one fails the recall audit rather than silently dropping
the winner.

## Position rule, costs and aggregation

* `position = 1` if `score > threshold`, the score is finite and the shared
  valid-row mask keeps the row (target finite too), otherwise `0`. There is no
  short leg and no leverage; `rcsr::walkforward::position_for` exposes the
  score-only comparison and `rcsr::trading_gpu::long_positions` the masked
  vector.
* `--policy threshold` (default) is the rule above.
* `--policy band` is a long-only no-trade band. With threshold `θ` and
  half-width `β = --band`, an entry needs `score > θ + β`; a long book stays
  long while `score > θ − β` and exits when the score is non-finite or at/below
  `θ − β`. `--min-hold K` (default 1) delays an exit until the book has been
  long for `K` consecutive rows, counting the entry row. `β = 0` with
  `--min-hold 1` reproduces the threshold rule **bit for bit**. The band is a
  deterministic `{0,+1}` finite-state automaton; it never opens a short.
* `--policy dp` keeps the reported book on the threshold rule and adds the
  exact hindsight-optimal DP as a diagnostic `hindsight_bound`. The DP is the
  best path with hindsight, so it always flatters the strategy and is **never**
  reported as the strategy's P&L. The report labels it `"label":
  "hindsight_bound"`, `"is_strategy": false`, and carries the threshold book's
  net, the gap and `rule_le_bound`.
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

## Associative carries and chunk invariance

The band and the hindsight DP are implemented as associative carries in
`rcsr::trading_gpu`, with the serial CPU reference kept as the authority.

* **Band (`Carry2`).** A row block induces, for each start state
  `s ∈ {0, 1}`, an end state `e(s)` and a reward `w(s)`. The carry is
  `(e0, e1, w0, w1)`, identity `(0, 1, 0, 0)`, and blocks compose
  left-then-right:
  `e(s) = right.e(left.e(s))`, `w(s) = left.w(s) + right.w(left.e(s))`.
  The final exit is charged on the last row of each group; the group seam resets
  the carry. `--min-hold K > 1` uses the `K+1`-state `CooldownCarry` variant.
* **Hindsight DP (`DpCarry`).** `M_t = [[0, −s], [r_t − s, r_t]]` (destination
  rows, source columns) is combined with the associative `(max, +)` product;
  identity `[[0, −∞], [−∞, 0]]`. The answer is
  `max(V_T(0), V_T(1) − s)`, four max-adds per row, `O(2n)` for the whole
  scan. Ties use a fixed destination/source order, so the result is
  deterministic. The DP resets at every group seam and its per-group bounds add.
* Chunks are merged in ascending order. A chunk that ends inside a group
  continues the same carry, so **alone, 1-row, 4 k and 64 k chunks produce
  identical carry bits and bounds**. Nothing in these scans depends on a
  `P×N` allocation.
* The carries are associative as algebra, but their rewards and DP values are
  `f64` sums, and `f64` addition is not associative: rewards
  `[1, 2^-53, 2^-53]` fold to `1.0` left to right and to `1 + 2^-52` when the
  last two are added first. The CPU reference is the left fold, so a parallel
  prefix over carries cannot reproduce its bits
  (`f64_carry_rewards_do_not_reassociate_but_end_states_do` in
  `tests/pnl_gpu.rs`). Only the end-state map, an integer function
  composition, reassociates exactly.

### Device scan (`--gpu`)

`--gpu` runs the band/cooldown scan (`--policy band`) and the hindsight DP
(`--policy dp`) on the CUDA device (`src/cuda/pnl.cu`). The device result is
bit-identical to the CPU scan: same positions, same carry bits, same net and
bound.

* A **chain** is one `(group, start state)` fold: `K+1` chains per group for
  the band, two (one per source state) for the DP. One device thread folds one
  chain over the group's rows in ascending order, with the same `f64`
  operations in the same order as the CPU. The kernels are built with
  `-fmad=false`, so no product is fused into an add.
* The block size (1 up to the device limit) only changes which thread owns a
  chain, never the order of its sums. The scan is launched once per row chunk;
  a chain whose group continues from an earlier chunk resumes its state and
  value from device memory.
* The per-group carries come back to the host, and the net and the bound are
  summed there by the same code as the CPU path.
* Scores stay `f64` on the device, so a score on `θ ± β` compares exactly as on
  the CPU. Candidates stream through the declared 8-candidate tile; there is no
  persistent `P×N` buffer.
* When the device refuses (non-CUDA build, no device, a refused allocation, a
  launch or copy error) the CPU scan runs instead. The `band` and
  `hindsight_bound` JSON objects carry `backend` (`cuda` or `cpu`) and
  `fallback_reason` (`null` unless `--gpu` fell back), the report's `policy`
  object carries `gpu`, and the CLI prints one `note:` line per fallback on
  stderr. Invalid input fails with the CPU's own error.
* Without `--gpu` the walk-forward runs the CPU scans exactly as before.

```sh
# no-trade band, beta 0.1, minimum hold 3 rows
rcsr walkforward ... --policy band --band 0.1 --min-hold 3

# threshold strategy plus the exact hindsight bound (never a strategy)
rcsr walkforward ... --policy dp

# the same scans on the device, bit-identical to the CPU
rcsr walkforward ... --policy band --band 0.1 --gpu
```

## Fused long-only P&L (`--device-pnl`)

`--device-pnl` recomputes each test block's long-only book through the fused
reduction in `src/trading_gpu.rs` (device kernel `src/cuda/pnl.cu` when built
with `--features cuda`, otherwise the identical canonical CPU segments). It is
a cross-check beside the scalar evaluator; the scalar evaluator and its metrics
remain the reported authority.

The reduction is the same rule, with `f64` accumulation for every term:

```
p_i     = 1 if isfinite(score_i) and score_i > threshold else 0
d_i     = p_i - p_{i-1},              p at a group start = 0
gross_i = p_i * r_i
cost_i  = s * |d_i| + [i is the group's last row] * s * p_i
net_i   = gross_i - cost_i
```

* Positions are `{0, 1}`. A negative position is a hard error; there is no short
  kernel.
* A round trip costs `2s`; the final exit is charged exactly once per group, so
  a book ending long pays one extra side. Flat is free.
* `gross`, `cost`, `turnover` (`sum |d|`), `net`, exposure, entries, exits,
  trades, hits and final exits are summed per canonical 8,192-row segment over
  32 fixed virtual lanes (row offset modulo 32), merged by a fixed pairwise
  lane tree and then in ascending segment order.
* The device path never allocates `P x N`: candidates stream through a declared
  tile of at most 8 candidates. Alone, padded, 4 k and 64 k chunks and
  128/256/512 threads produce identical bits.
* `cash` (`p = 0`, zero net and cost by default) and `buy-and-hold`
  (`p = 1`, `net = sum(r) - 2s * groups`) are reported with the strategy block.

The JSON gains a `device_pnl` object per window and a report-level summary with
`backend`, `segment_rows`, `tile_candidates`, `rows` and
`max_abs_net_delta`. `backend` is `cuda` when the kernel ran and
`cpu-canonical` on a non-CUDA build or when no device is available.

## Long-only controls

Every report lists four controls per window, in `controls` (JSON) and the
`## Controls` table (Markdown). They use the window's valid test rows, the same
per-side cost and the same equal-weight period aggregation as the strategy.
Each control starts flat and pays its final exit, like the fused reduction.
Positions are `{0, 1}`; there is no short or dollar-neutral control.

| control | positions |
|---|---|
| `cash` | `p = 0`: net 0, cost 0, Sharpe `NaN` (zero variance) |
| `buy_and_hold` | `p = 1`: one entry and one charged exit per group, `net = sum(r) - 2s * groups` |
| `constant` | one fixed side for the block: long iff the mean finite training target is strictly positive, else flat |
| `matched_exposure_random_long` | the strategy's own long and flat runs, with the order of the long runs and of the flat runs shuffled inside each group |

The matched control preserves long rows, entries, exits and final exits exactly
for every group. Its shuffle is a Fisher-Yates driven by a counter-based RNG,
`draw(seed, group, stream, index)`, so a group's result does not depend on the
other groups. `--control-seed <n>` (default 0) is recorded as `controls.seed`;
the same seed gives the same report.

Beside each control: `exposure`, `turnover` (`sum |position change|`), `beta`
(on the buy-and-hold period series, from the #307 beta/alpha monoid),
`excess_over_cash` (`net - cash net`), annualised `sharpe` and `mean_return`.
Values are CPU `f64`. With `--device-pnl` the four position vectors are also
priced in one device batch (the canonical CPU reduction on non-CUDA builds) and
`controls.device` records `backend`, `max_abs_delta` and `counts_equal`. The
matched permutation itself is generated on the host.

The full CPU ladder harness (mean, OLS, kNN, top-tercile) is #266 and is not
part of this report.

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

## Performance

Measured with the release binary on an Intel i7-3770K (8 threads), 31 GiB,
Linux, using the 3,943,856 x 56 hourly store, streamed one group at a time:
`rcsr walkforward --group-by symbol --horizon 172800` takes 4.06 s with a peak
RSS of 107.7 MiB. The corpus, window, command and measurement protocol are in
[`eval-benchmark.md`](eval-benchmark.md).
