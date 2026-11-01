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

The purge gap is `test_start - train_end`. It must be at least `--entry-offset + --horizon + --delay` (the forward-label horizon and the seconds before and after it) and at least `--embargo` (default: the horizon). Rows in the
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

## Candidates from a run's front (`--front`)

`rcsr walkforward --front run.json` takes every `pareto[k].expression` of a `run.json` as one candidate, labelled `front_rank_<k+1>` in archive order. It replaces `--expr` (passing both exits 2). With one member the walk-forward is the single-expression case. An empty front, a run without `split.train_time` (no time column) and an expression that uses a column missing from `--data` exit 3.

Defaults taken from the run, each overridden by its explicit flag:

| setting | run.json field | flag |
|---|---|---|
| variables | `variables` | `--vars` |
| return column | `target_spec.return_column` | `--target` |
| trials of the deflated Sharpe | `search.candidates_evaluated` (offspring scored over the run, summed over islands and generations, kept across `--resume`) | `--trials` (without either, exit 2) |
| position rule | `objective.spec.threshold`, used as a fixed threshold | `--threshold-fixed`, `--threshold-quantile` |
| per-side cost | `objective.spec.per_side_bps` | `--cost-bps` |

A run with an `--objective` selects under `1[pred > threshold]`. Without the carried threshold the walk-forward would use its 0.70 training quantile and score the front under a different book than the one it was selected on.

In-sample guard: `run.json` records `split.train_time` and `split.validation_time` as `[first, last]` raw time keys. With `seen = max(train_time[1], validation_time[1])` and `purge = split.required_purge_seconds` (else `split.purge_seconds`), a window with `test_start < seen + purge` exits 2:

```
error: window 'w1' tests rows the run trained or selected on (test_start 150 < last seen 179 + purge 5)
```

`--allow-in-sample-windows` downgrades the refusal: the window is scored and its JSON carries `"in_sample":true`. Use it for diagnostics only.

The report's `selection` block gains `source: "front"`, `run`, `run_sha256` (SHA-256 of the file) and `candidate_labels`. `--select-band` and `--batch-pnl` work unchanged on the front's candidates.

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
* `--cost-bps` (default 10; a `--target-spec` cost wins, see "One book from search to evaluation") is charged **per side** on `|position - previous position|`,
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
* The protocol locked test is checked on every run, with or without
  `--ledger`: a window whose train or test range touches it exits 3. The range
  is `--lockbox START,END`, else the one declared with the data
  (`rcsr panel build --locked-test`), else the protocol default
  2025-07-01..2026-07-01. `--lockbox none` is the only opt-out. The JSON
  `protocol.lockbox` object always carries `declared`, `source` (`flag`,
  `store`, `protocol_default`, `none`), `declared_in`, `rows_in_scope` and
  `evaluated`. See [`panel.md`](panel.md#the-declared-locked-test).
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
  value from device memory. The walk-forward uses one chunk for all rows, so
  every group's chains run in the same launch.
* The device scan is for exactness, not speed: on a 3.9 M-row panel it is
  level with the CPU, and a single long series is 10 to 30 times slower (one
  thread per chain). The measurements are in
  [`docs/gpu-backend.md`](gpu-backend.md#band-cooldown-and-hindsight-dp-scan-issue-306).
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

## Batch scoring (`--batch-pnl`)

`--batch-pnl` scores every candidate on every test window in one batch
(`src/wf_batch.rs`, device kernel `src/cuda/wf_batch.cu` when built with
`--features cuda`, otherwise the identical host loop). It consumes the windows,
thresholds and scores the protocol already produced and does not redefine them.
The reported finalist values stay the scalar evaluator's exact CPU `f64`. Every
batch value carries a `value_label`:

| label | meaning |
|---|---|
| `gpu_proxy` | computed by the device kernel |
| `cpu_batch` | computed by the host mirror of the kernel |
| `exact_cpu_f64` | computed by the scalar evaluator (finalist check, scalar fallback, ladder) |

The rows of all windows are laid out once in chain order. A chain is a run of
rows that share one position state, ascending in `(window, time)` and sorted by
`(group, time)` inside a window whatever the input order, so a flattened
`(time, group)` scan invents no trades.

* `pooled`: one chain per group. The position carries across the window seam, so
  a position held over the seam pays no second entry. Every window block still
  pays its own final exit, like the scalar evaluator with a shared `previous`
  map.
* `per_window`: one chain per `(window, group)`. Every window starts flat.

One device thread folds one `(candidate, chain)` in ascending row order, so a
block's `f64` sums do not depend on the block size or the tile. Blocks merge
into window cells in ascending block order. A cell equals the scalar evaluator
to `1e-9` relative; the integer counters are equal.

Candidates stream through a declared tile of at most 64 candidates (default
byte budget 256 MiB, f64 scores). The population is never held as `P x N`. A
candidate with a non-finite score on a finite-target row is computed by the
scalar evaluator, because the evaluator drops such rows per candidate; its cells
are labelled `exact_cpu_f64` and counted in `fallback_candidates`.

The baseline ladder is scored in every batch. Per window, `cash`,
`buy_and_hold` and `constant` are closed forms; `matched_exposure_random_long`
is priced per candidate from the candidate's own positions. All start flat and
are exact CPU `f64`.

The `batch_pnl` JSON block holds `reported_values`, `batch_values`,
`max_abs_net_delta` (selected candidate, scalar evaluator versus batch cell),
`finalists`, and for each seam the `cells`, the `ladder` and `stats`:
`candidates_per_s`, `row_evals_per_s`, `device_fraction`, `device_bytes`,
`host_tile_bytes`, `device_tiles`, `host_tiles` and `fallback_candidates`.
`backend` is `cuda` when the kernel ran and `cpu` otherwise, with a
`fallback_reason` when a tile ran on the host.

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

## Exact long/flat book (`--objective`)

`src/trading.rs` scores one prediction vector as a long/flat book in exact CPU `f64`. It is the evaluator behind `rcsr --objective pnl|sharpe|log_growth`.

```text
p_t     = 1[pred_t > theta]  or  clip((pred_t - theta) / tau, 0, 1)    (times max_position)
gross_t = p_t * r_t + (1 - p_t) * cash_return - p_t * funding_t
cost_t  = c_side * |p_t - p_{t-1}|        (each group starts flat)
net_t   = gross_t - cost_t                (the last row of a group also pays c_side * p_end)
```

* A position is never negative. `evaluate_positions` rejects a negative or non-finite position and a position above `max_position` as an error. There is no short leg, borrow or funding term.
* Cost is charged per side on `|delta p|`, including the final exit. A 17-row hold pays one entry and one exit, not a charge per row.
* Under the plain threshold rule with no cash return and no funding, the period metric block (`period` in the Rust API) equals the scalar walk-forward evaluator bit for bit. The strict `>` comparison and the shared score-and-return finite mask are the same.
* Rows are equal-weighted per timestamp. The Sharpe is daily: sub-daily periods are compounded per calendar day and annualised with `sqrt(365)`, ddof 1. Daily or slower periods use their own annualisation. A series with no variance has a `NaN` Sharpe, which is unusable and never zero.
* The book reports gross and net P&L, total return, log growth (`sum ln(1 + R)`), Sharpe, Sortino, maximum drawdown, Calmar, turnover, trades, entries, exits, trades per year, exposure (time in market), hit rate on long rows and the break-even cost per side. The break-even cost is `gross P&L / sum |delta p|` including exits; at that cost the net P&L is zero.
* Every book also carries a cash book (flat the whole time, earning `cash_return`), a buy-and-hold book (long from the first row to the end, paying the entry and the exit) and a cost ladder that re-prices the same positions at 0, 5, 10, 20 and 50 bp per side.

A `.rcsb` store read with `--vars` and `--time` is projected to the columns the run names. With a trading objective the projection also keeps the target spec's `return_column` and the `--funding` column. A run without `--objective` keeps its projected column list. A `return_column` that is not in the store is an input error (exit 3) that names the target spec.

Finalist selection with `--objective` scores every Pareto member this way on the validation slice only. The winner has the lowest objective value (`-net P&L`, `-Sharpe` or `-log growth`; ties keep the archive order, so the smaller expression wins). Candidates below `--min-trades` or with a non-finite value are unusable. The test block is scored once, for the winner.

## One book from search to evaluation (#487)

The search objective, `rcsr walkforward`, `rcsr ablation score` and `rcsr ladder` price one long/flat book with one cost, one return unit and one position rule. The declarations live in the target spec (`<data>.targets.json`) that `rcsr enrich --label long:<h>,fee=<bp>` writes, and in the `run.json` of the search.

* **Cost.** `rcsr enrich --label long:<h>` needs `fee=` (exit 2; `fee=0` is allowed). The spec carries `cost_bps` (round trip, `2 * fee + spread + slip`) and `cost_per_side_bps = cost_bps / 2`. Every evaluator charges `cost_bps / 2` per side. `--cost-bps` that differs from the spec is refused (exit 2); `--cost-override <bp>` prices a different cost on purpose. Without a spec cost the flag or the default applies (walk-forward 10 bp, ladder and ablation 12 bp per side; the search needs the flag). A spec whose label is already net of cost (`cost_in_label`) declares no per-side cost.
* **Units.** A spec with `return_units: "log"` makes the walk-forward, the ladder and `ablation score` convert the return column with `exp_m1` before any P&L, as the objective does. Without a spec the column is read as a simple return.
* **Position rule.** `rcsr walkforward --from-run run.json` and `rcsr ablation score --from-run run.json` take the expression, the fixed threshold `objective.spec.threshold`, the per-side cost `objective.spec.per_side_bps` and the target spec (return column, units, horizon, entry offset, delay) from the run. A run without an `objective` block is refused (exit 3). `--expr`, `--threshold-quantile`, `--threshold-fixed` and `--target-spec` with `--from-run` are usage errors (exit 2). The quantile rule of a plain `walkforward --expr` is a different book: for a constant finalist every quantile threshold is flat, and the command warns.
* **Target spec in walk-forward.** `--target-spec <file>` fills `--horizon`, `--entry-offset`, `--delay`, the return column, the units and the cost from the entry whose `column`, `name` or `return_column` is `--target`. A flag that disagrees is refused like the cost.
* **Ladder grid.** With a spec cost the default `--cost-grid` is 1x, 2x and 3.33x the per-side cost. With a flag or the default cost it stays `12,20,40`. An explicit `--cost-grid` is absolute.
* **Reports.** The `objective` block of `run.json`, the walk-forward report, the ladder report and the ablation result carry `cost: {per_side_bps, source, return_units}` with `source` one of `target_spec`, `flag` or `default`. A `flag` source priced against a spec adds `spec_per_side_bps`. `objective.finalist.constant` is true when the finalist reads no variable.

Verify: `tests/cost_contract.rs` (`one_cost_one_book_across_evaluators`, `walkforward_from_run_prices_the_selected_book`, `a_log_return_column_is_converted_with_exp_m1_in_every_evaluator`, `the_cost_flag_must_agree_with_the_target_spec`, `the_cost_source_is_reported`, `from_run_refuses_a_run_without_an_objective_and_competing_rules`, `a_long_label_needs_an_explicit_fee`) and the `resolve_cost_*` unit tests in `src/target.rs`.

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

## Long-only evaluation protocol

The primary book is long or flat against cash. Short exposure, borrow and funding on shorts are out of scope. A long-only result can be market drift, so every report carries exposure, beta and excess over cash beside the book.

### Four tiers and the windows table

Training fits, validation selects, development out-of-sample evaluates the procedure, and a locked test is read once per sealed candidate. [`protocol/windows.csv`](../protocol/windows.csv) holds the UTC boundaries (`rcsr protocol windows` regenerates it, and a test pins the file to the generator):

| range | UTC |
|---|---|
| warm-up (indicators only, never trained on) | 2021 |
| development | 2022-01-01 to 2025-07-01 |
| expanding folds | quarterly tests from 2023-01-01, train from 2022-01-01 |
| rolling folds | quarterly tests from 2024-01-01, train on the previous 24 months |
| locked test | 2025-07-01 to 2026-07-01 |
| reserved | 2026H2 (the engine lockbox) |

Each row has `train_end = test_start - purge_seconds` (219600 s: 1 h entry plus a 60 h label) and `validation_start` six calendar months before the test start. `--windows protocol/windows.csv --windows-scheme expanding` loads one scheme; the table's `validation_start` replaces `--validation-frac`. The validation slice sets the threshold and selects among `--expr` candidates. Select the procedure, not the lucky candidate: pool the development series of the chosen procedure.

### Purge

The required purge is `--entry-offset + --horizon + --delay` seconds. A window whose `test_start - train_end` is shorter is refused, and the report carries `split.required_purge_seconds`, `split.purge_seconds` (the smallest gap) and per window `train_last_time`: the stamp of the last training row, whose label ends at most `required_purge_seconds` later, before `test_start`. The purge is in time and applies to every symbol, so a late listing or a hole in one symbol cannot bring a training label into a test window.

`--gap` counts rows and does not apply to explicit `--train-select`/`--test-select`. `rcsr protocol selectors --windows protocol/windows.csv --label expanding-2023Q1 --time-col ts` prints selectors with the purge written into the training range.

### Economics

The `economics` object (and a Markdown section) holds, for the pooled series: Sortino, Calmar (`cagr / |max drawdown|`), CVaR 95 and 99 (the mean return of the worst `ceil(n (1 - level))` periods, a negative number), the break-even cost in bp per side (`base * mean(gross) / mean(gross - net)`, since cost is linear in the per-side rate), the cost ladder at 0.5x, 1x, 2x and 3x the cost, and the daily Sharpe and Sortino (`sqrt(365)`) when the stamps are epoch seconds. Daily returns compound the periods of one UTC day.

The benchmark table has the same fields for the strategy, `cash`, `buy_and_hold` (one charged entry and exit per symbol), `equal_weight_universe` (always long, no cost), `constant_long_flat` and `best_long_flat_baseline` (the best mean of cash, buy-and-hold and constant, ties to the earlier). Exposure is long rows over valid rows. Beta is on the equal-weight universe. Every benchmark series starts each window flat, like the controls.

### Block uncertainty (`--bootstrap B`)

* **Circular block bootstrap** of the pooled net series with block length `L >= max(h, holding period)` periods (default `max(h, hold, ceil(n^(1/3)))`; a shorter `--block-len` is refused). It reports the mean and annualised Sharpe with percentile 95% CIs.
* **Design effect**: the variance of the bootstrap mean with blocks over the same with iid rows (`L = 1`). `effective_n = n / design_effect`, in periods (hours on an hourly grid). The `p_mean_positive` field is the one-sided centred p-value `(1 + count) / (B + 1)`.
* **N_eff**: the participation ratio of the candidate correlation matrix, `N^2 / sum c_ij^2`; identical candidates give 1 and uncorrelated ones give `N`. With `--trials N` the effective trials are `N * PR / candidates`.
* **DSR** (Bailey and Lopez de Prado) at `N_eff` and `10 N_eff`, with `SR0 = sqrt(V) ((1-g) Z(1-1/N) + g Z(1-1/(N e)))`, `V` the variance of the candidate Sharpes (the Mertens estimator variance with one candidate), `T` the effective n, and skew and raw kurtosis of the series. All inputs are in the report.
* **PBO** by CSCV with `--pbo-blocks` (default 16) blocks, per-period Sharpe as the measure; needs two candidates.
* **SPA** (Hansen 2005, joint circular block bootstrap, consistent, lower and upper recentering) of the candidates against the best long/flat baseline.
* **BH-FDR** at `--fdr-q` over the per-candidate one-sided block-bootstrap p-values.

Candidates are evaluated on the test periods with their own validation threshold only for these diagnostics; the selection stays on validation.

### Nulls (`--null-draws B`)

* **Hour-block permutation**: within each test window the cross-sections of targets are permuted in blocks of `L` periods (the same permutation for all symbols); positions and costs stay fixed.
* **Matched-exposure random long/flat**: the strategy's own long and flat runs shuffled inside each symbol (see Long-only controls), `B` times.
* **Always-long**: the buy-and-hold series, reported without a p-value.

`p = (1 + #{null >= observed}) / (B + 1)` for the mean and for the per-period Sharpe. A search-budget null is `--null-block` in the search.

### Candidate ledger and the locked test

`--ledger FILE` makes a development run append a `candidate` line (content-addressed id over the procedure, the data SHA-256 and the build commit; `test_read:false`). Each line holds the SHA-256 of the previous line, so an edit breaks the chain. A development window that touches the locked test (`--lockbox START,END`, else the range declared with the data, else 2025-07-01 to 2026-07-01; `--lockbox none` opts out) is refused.

`rcsr protocol seal --ledger L --out locked_candidates.json` writes the sealed list and records its hash in the ledger. `--locked-test --sealed locked_candidates.json` then reads the locked test: the candidate must be in the sealed list, the data and build hashes must equal the sealed entry (otherwise the run is void), and the candidate must not have `test_read:true`. The read appends a `test_read` line with the report hash and the new `read_count`; a second read is refused. `--verify-read --expect-report-sha256 H` recomputes the read and compares hashes without recording anything. The report's `protocol` object holds the lockbox rows, date range and read count; `run.json` carries a `lockbox` object (`--lockbox`, `--ledger`) with the rows in the range, their first and last stamps and the read count.

`rcsr ladder --scheme locked --locked-test --ledger L --sealed S` reads the locked test for the baseline ladder. The sealed list must be recorded in the ledger and the ledger must hold no `baseline_read`; the run appends `{"event":"baseline_read",...}` with the data and report hashes, and a second call is refused. Without `--locked-test`, `ladder` and `ablation score` refuse any fold that touches the declared range.

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

The printed and written JSON ends with a `ledger` object (#284): the phase ledger of the run with `io_s` (store row-group or CSV reads), `pnl_s` (scoring, windows, controls, `--batch-pnl`), `host_other_s`, `wall_s`, `unaccounted_s`, `closure` and `device_busy_s`, the union of the `--batch-pnl` device tile intervals (`host_envelope` basis; `--device-pnl` is not timed). The ledger holds wall-clock seconds, so `report_sha256` and `--expect-report-sha256` hash the report without it. See `docs/benchmarks.md`.

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
