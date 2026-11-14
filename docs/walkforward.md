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

`rcsr walkforward --front run.json` takes every `pareto[k].expression` of a `run.json` (its `expression_exact` when present) as one candidate, labelled `front_rank_<k+1>` in archive order. It replaces `--expr` (passing both exits 2). With one member the walk-forward is the single-expression case. An empty front, a run without `split.train_time` (no time column) and an expression that uses a column missing from `--data` exit 3.

Defaults taken from the run, each overridden by its explicit flag:

| setting | run.json field | flag |
|---|---|---|
| variables | `variables` | `--vars` |
| return column | `target_spec.return_column` | `--target` |
| trials of the deflated Sharpe | the trials that selected on the scored rows (rule below); `search.candidates_evaluated` (offspring scored over the run, summed over islands and generations, kept across `--resume`) only when a window is in sample | `--trials` |
| position rule | `objective.spec.threshold`, used as a fixed threshold | `--threshold-fixed`, `--threshold-quantile` |
| per-side cost | `objective.spec.per_side_bps` | `--cost-bps` |

Default trials of the deflated Sharpe (#1027), shared by `--front` and `--from-run`. The report states the source in `multiple_testing.n_trials_source`:

| case | trials | `n_trials_source` |
|---|---|---|
| `--trials N` | `N` | `flag` |
| `--trials N` with `N` below the candidate count `k` | `k` (the report's `trials` is the number the DSR used) | `candidate_count` |
| `--from-run`, no in-sample window | 1 | `out_of_sample_book` |
| `--front`, no in-sample window | front size `k` | `front_size` |
| a window is in sample (`--allow-in-sample-windows`), run has `search.candidates_evaluated > 0` | `candidates_evaluated` | `run.candidates_evaluated` |
| the same without `candidates_evaluated` | front size | `front_size` |
| `--expr` | candidate count | `candidate_count` |

An out-of-sample window is not in the sample the search compared, so the search trials do not shift the null of its Sharpe. With `N = 1` the expected maximum Sharpe is 0 and the value is the probabilistic Sharpe ratio (PSR), reported as `multiple_testing.psr`. Deflate across the books of a campaign with `rcsr trials`.

A run with an `--objective` selects under `1[pred > threshold]`. Without the carried threshold the walk-forward would use its 0.70 training quantile and score the front under a different book than the one it was selected on.

In-sample guard: `run.json` records `split.train_time` and `split.validation_time` as `[first, last]` raw time keys. With `seen = max(train_time[1], validation_time[1])` and `purge = split.required_purge_seconds` (else `split.purge_seconds`), a window with `test_start < seen + purge` exits 2:

```
error: window 'w1' tests rows the run trained or selected on (test_start 150 < last seen 179 + purge 5)
```

`--from-run` applies the same rule: it reads the same `run.json`, so a window inside the rows the run trained or selected on exits 2, and a run without `split.train_time` refuses every window. `--allow-in-sample-windows` downgrades the refusal: the window is scored and its JSON carries `"in_sample":true`. Use it for diagnostics only. `rcsr ladder --sr-run` and `rcsr ablation score` apply the same rule (`RunFile::in_sample`) to their folds: they exit 3 naming the in-sample fold, and `--allow-in-sample` labels it `oof:false` and leaves it out of every pooled statistic.

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
* `--band-units score|sd` (#925, default `score`, needs `--policy band`, exit 2
  otherwise) says how `--band B` is read. `sd`: per window and candidate,
  `β = B * sd_train`, the sample sd (`n - 1`) of the candidate's finite scores
  on the training rows of the window (`train_start <= t < train_end` and before
  the validation start; never validation or test rows), in the tree order of
  `trading::score_sd`, the same value the search fitness uses. A degenerate sd
  (fewer than 2 scores, 0 or non-finite) gives `β = 0`. The test book of the
  window takes that `β` and never computes an sd from test rows.
  `windows[].band` records `band` (the multiple), `band_units`, `band_sd`,
  `band_sd_rows`, `band_sd_max_time` (the latest training time in the sd),
  `band_resolved` and `band_sd_degenerate`, and `band_width` is the resolved
  `β`; `policy.band_units` names the units. The search prices the same book
  with `--fitness pnl --band B --band-units sd` (docs/optimisation.md).
* **The run's band (#946).** `--from-run` and `--front` price the book the search selected. A run that priced a band or a minimum hold (`objective.spec.band`, `band_units`, `band_resolved`, `band_sd`, `min_holding`) is priced with `--policy band`, the run's `band_resolved` as the half-width in score units and its `min_holding`. The half-width is the recorded value: a window never resolves an sd from its own rows, so no window or test row sets the band. `--front` gives every candidate its member's `validation_book.band_resolved` (sd units; score units use the run's `band_resolved`) and the run's minimum hold. A run without the keys, or with `B = 0` and `min_holding = 1`, prices the threshold book. A `--policy`, `--band`, `--band-units` or `--min-hold` that differs from the run exits 2 and names the run's value; the same values are accepted. A run with `band_resolved` but no `band_units` is refused (exit 3). `policy.band_source` is `run` when the band comes from the run and `flag` otherwise; `windows[].band` records the run's `band`, `band_units`, `band_sd`, `band_sd_rows`, `band_sd_max_time` and `band_resolved`. `ablation score` and `ladder --sr-run` price the threshold rule only and refuse a run that priced a band or a minimum hold (exit 3).
* `--policy band` refuses `--device-pnl` (#810): the device arm prices the threshold book, so its delta against the band book would measure the band rule, not the kernel. The error is a usage error (exit 2) raised before any data work.
* `--policy dp` keeps the reported book on the threshold rule and adds the
  exact hindsight-optimal DP as a diagnostic `hindsight_bound`. The DP is the
  best path with hindsight, so it always flatters the strategy and is **never**
  reported as the strategy's P&L. The report labels it `"label":
  "hindsight_bound"`, `"is_strategy": false`, and carries the threshold book's
  net, the gap and `rule_le_bound`.
* `--policy top-tercile` (#493) is the cross-sectional long-only book: per timestamp, `position = 1` for the top
  `floor(n/3)` of the `n` rows with a finite score (ties by row order), else `0`. The ranks use the scores of
  that timestamp only and run before the target mask, so a row with a NaN target stays in the cross-section.
  A timestamp with fewer than 3 finite scores holds nothing. There is no threshold: `--threshold-quantile`,
  `--threshold-fixed`, `--band`, `--min-hold`, `--select-band`, `--device-pnl` and `--batch-pnl` are refused
  with a usage error (exit 2), the window `threshold` is `null` and `threshold_source` is `none`. Costs, the final exit and the equal-weight
  period aggregation are the threshold rule's. `--from-run` of a search run with `--position top_tercile`
  selects the policy; it conflicts with an explicit `--policy` of another kind. Pair it with an `xs_rank` target
  (`--target lbl_h1_xs --target-spec ...` resolves to its `return_column`; `docs/time-series.md`).
* `--policy vote --quorum Q` (#930) builds one long/flat book from the K repeated `--expr` members. Every member
  keeps its own threshold (`--threshold-fixed`, or the member's validation or training quantile); nothing is fitted
  on test rows, so the members and thresholds do not depend on any test value. Per row, `v` counts the members
  with `score > threshold`; the row is long when `v >= ceil(Q x K)` (at least 1). `Q` must lie in `(0, 1]`
  (usage error, exit 2) and is required. The shared valid-row mask covers all members: a member with a
  non-finite score drops the row (flat, no return), as in the threshold book. `K = 1, Q = 1` is the threshold book
  bit for bit. `Q = 0.67` with `K = 3` needs 3 votes (`ceil(2.01)`); pass `0.66` for a majority of three.
  `--band B` (a whole number of votes) applies after voting: a flat book enters at `v >= ceil(Q x K)` and a long book
  exits when `v < ceil(Q x K) - B`; `--min-hold` then holds a long book for that many rows (cooldown scan). The
  vote count is priced as one strategy (one trial for the DSR unless `--trials` is given); the report lists
  the members in `candidates` and `policy.vote.members`, with `policy.vote.quorum` and `votes_required`.
  `--device-pnl`, `--batch-pnl`, `--select-band`, `--locked-test` and `--band-units` are refused (exit 2).
* `--cost-bps` (default 10; a `--target-spec` cost wins, see "One book from search to evaluation") is charged **per side** on `|position - previous position|`,
  separately inside each `--group-by` series. Every series starts flat, so an
  entry costs one side and its exit (including the **final** exit of every
  group) costs the other: a round trip is `2 * cost_bps`. #145 changed the
  final exit from free to charged; a missing exit would flatter the strategy
  by one side.
* Rows sharing a timestamp are equal-weighted, exactly as in the pilot: the
  period return is the mean net return over the groups present at that
  timestamp. With no `--group-by` the whole table is one series.
* Every window is checked against the guarded box on **both** sides: the declared
  locked test (`--lockbox`, the store sidecar), else for undeclared data, the protocol default and `--lockbox none` the 2026H2 lockbox
  `[2026-07-01, 2027-01-01)`. A training range inside the
  holdout is refused even when the test block is in 2027. `--allow-lockbox` (it opens the 2026H2 box only, never a declared lock) is
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

# no-trade band of one training-score sd per window and candidate
rcsr walkforward ... --policy band --band 1.0 --band-units sd --min-hold 3

# threshold strategy plus the exact hindsight bound (never a strategy)
rcsr walkforward ... --policy dp

# top-tercile long-only book per timestamp (no threshold)
rcsr walkforward ... --policy top-tercile

# one book from three expressions: long when 2 of 3 members are long
rcsr walkforward ... --expr "<f1>" --expr "<f2>" --expr "<f3>" --policy vote --quorum 0.66 --threshold-fixed 1e-30

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
`backend`, `segment_rows`, `tile_candidates`, `windows`, `rows`,
`max_abs_net_delta`, `device_windows`, `bit_mismatch_windows`,
`count_mismatch_windows` and `f32_flip_rows`. `backend` is `cuda` when the
kernel ran and `cpu-canonical` on a non-CUDA build or when no device is
available.

The kernel reads the `f64` score of every row and compares it with the `f64`
threshold (rule R: a row is long iff its score is finite and strictly greater
than the threshold). A score below the f32 range, such as `1e-46`, is long, as
in the scalar evaluator, the exact refit and a user's formula. Three
comparisons run on every window, and each detects a different fault:

* Device totals against the canonical CPU reduction (`trading_gpu::reduce`) on
  the same inputs: `bit_equal`, the bits of every `f64` sum and every integer
  count. A kernel fault, down to one ulp of one P&L term, makes it `false`.
  `max_abs_net_delta` cannot see such a fault: summation-order noise between
  the scalar evaluator and the canonical reduction is orders of magnitude above
  one ulp.
* Device net against the scalar evaluator's net: `net_delta` per window and
  `max_abs_net_delta` in the summary. This is the reported book; the tolerance
  is 1e-9.
* Device integer counts against the scalar book: `counts_equal`. It finds a
  position disagreement also on rows with a return of 0, where the net delta is
  0. It is `null` under `--policy band`, whose scalar book is not the threshold
  book.

`f32_flip_rows` counts the rows whose position changes when the score is
narrowed to `f32`. It is reported per window and summed in the summary. No gate
or selection reads it; it shows how many rows the search's `f32` screening book
can place differently from the reported book. With `--select-band`, every
`band_selection.candidates[]` entry carries the same count on the validation
rows.

`bit_mismatch_windows` and `count_mismatch_windows` count the windows where
`bit_equal` is `false` or `counts_equal` is `false`. `device_windows` counts the
windows whose device block ran on the CUDA kernel.

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
`candidates`, `windows`, `cells`, `rows`, `row_evals`, `tile_candidates`,
`device_bytes`, `host_tile_bytes`, `device_tiles`, `host_tiles` and
`fallback_candidates`. The wall-clock fields (`wall_s`, `device_s`, `score_s`,
`layout_s`, `reference_s`, `candidates_per_s`, `row_evals_per_s`,
`device_fraction`) are in the printed `batch_timings` object, outside the
report hash.
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
* Rows are equal-weighted per period. A period is the bucket `floor(t / S)` of width `S` seconds, the label horizon (`TradingSpec::period_seconds`; without it one period per timestamp). The Sharpe is annualised with `sqrt(periods_per_year)`, ddof 1, where `periods_per_year = 31536000 / S` (the median step of the period stamps when `S` is unknown). `annualisation` is always `per_period`. `net_per_row_annualised` is `net_pnl / rows * periods_per_year`. A series with no variance has a `NaN` Sharpe, which is unusable and never zero.
* The book reports gross and net P&L, total return, log growth (`sum ln(1 + R)`), Sharpe, Sortino, maximum drawdown, Calmar, turnover, trades, entries, exits, trades per year, exposure (time in market), hit rate on long rows and the break-even cost per side. The break-even cost is `gross P&L / sum |delta p|` including exits; at that cost the net P&L is zero. It is `null` when gross P&L is not positive.
* Every book also carries a cash book (flat the whole time, earning `cash_return`), a buy-and-hold book (long from the first row to the end, paying the entry and the exit) and a cost ladder that re-prices the same positions at 0, 5, 10, 20 and 50 bp per side.

A `.rcsb` store read with `--vars` and `--time` is projected to the columns the run names. With a trading objective the projection also keeps the target spec's `return_column` and the `--funding` column. A run without `--objective` keeps its projected column list. A `return_column` that is not in the store is an input error (exit 3) that names the target spec.

Finalist selection with `--objective` scores every Pareto member this way on the validation slice only. The winner has the lowest objective value (`-net P&L`, `-Sharpe` or `-log growth`; ties keep the archive order, so the smaller expression wins). Candidates below `--min-trades` or with a non-finite value are unusable. The test block is scored once, for the winner.

With `--fitness pnl` the search ranks every candidate by the regret `(B - net) / B` of this same book on the training rows, `B` being the hindsight DP bound (`trading_gpu::dp_bound`) at the same cost. `trading::FitnessBook` prices the threshold rule, or with `--band` and `--min-holding` (#925) the `--policy band` book of this page (no `tau`, `--max-position 1`): `net` is the excess net over cash, `sum p_i * (r_i - funding_i - cash) - c_side * (flips + exits)`, which equals `PathStats::net_pnl - cash * rows`. See section 3.5 of [`docs/optimisation.md`](optimisation.md).

## One book from search to evaluation (#487)

The search objective, `rcsr walkforward`, `rcsr ablation score` and `rcsr ladder` price one long/flat book with one cost, one return unit and one position rule. The declarations live in the target spec (`<data>.targets.json`) that `rcsr enrich --label long:<h>,fee=<bp>` writes, and in the `run.json` of the search.

* **Cost.** `rcsr enrich --label long:<h>` needs `fee=` (exit 2; `fee=0` is allowed). The spec carries `cost_bps` (round trip, `2 * fee + spread + slip`) and `cost_per_side_bps = cost_bps / 2`. Every evaluator charges `cost_bps / 2` per side. `--cost-bps` that differs from the spec is refused (exit 2); `--cost-override <bp>` prices a different cost on purpose. Without a spec cost the flag or the default applies (walk-forward 10 bp, ladder and ablation 12 bp per side; the search needs the flag). A spec whose label is already net of cost (`cost_in_label`) declares no per-side cost.
* **The run's band in the ladder and ablation (#951).** `rcsr ladder --sr-run` and `rcsr ablation score --run` and `--from-run` price a run that priced a band or a minimum hold with the same book: the run's `objective.spec.band_resolved` as the half-width (score units and sd units alike, never resolved again from fold rows), its `min_holding`, and every group and fold starting flat. The ladder reports the book in `sr_run` (`policy`, `band`, `band_units`, `band_resolved`, `min_holding`, `band_source: run`) and in the rung's `params`; `ablation score` reports `band_book`. `--from-run` prices the run's fixed threshold; `--run` applies the band and hold around each quantile threshold of `--thresholds`. A run without band keys, or with `B = 0` and `min_holding = 1`, prices the threshold book as before. A band block that is malformed exits 3. Neither command takes a front or a band flag.
* **Units.** A spec with `return_units: "log"` makes the walk-forward, the ladder and `ablation score` convert the return column with `exp_m1` before any P&L, as the objective does. Without a spec the column is read as a simple return.
* **One tree.** A search run writes two strings per Pareto member and for `objective.finalist`: `expression` (constants rounded to 4 decimals, for people) and `expression_exact` (each constant as the shortest decimal that parses back to the same bits, written from the same tree). `walkforward --front`, `walkforward --from-run`, `ladder --sr-run` and `ablation score --run/--from-run` read `expression_exact` first, so the candidate priced after the search is the tree the search priced, and `selection.candidates` holds the exact strings (the candidate id binds them). `windows[].expression` stays the display form. Each report states `expression_source` (`selection.expression_source` in the walk-forward, `sr_run.expression_source` in the ladder, `expression_source` in an ablation result): `exact`, or `rounded` for a run written before the field existed. `walkforward --locked-test` and `ladder --locked-test` refuse a `rounded` source (exit 3, message names `expression_exact`): only the exact tree is the selected book. `simplify_display` still folds constants such as `1/s` before printing, so the exact string can differ from the search tree by rounding noise of the folded constants; no runtime check compares them, the tests do.
* **Position rule.** `rcsr walkforward --from-run run.json` and `rcsr ablation score --from-run run.json` take the expression, the fixed threshold `objective.spec.threshold`, the per-side cost `objective.spec.per_side_bps` and the target spec (return column, units, horizon, entry offset, delay) from the run. A run without an `objective` block is refused (exit 3). `--expr`, `--threshold-quantile`, `--threshold-fixed` and `--target-spec` with `--from-run` are usage errors (exit 2). The quantile rule of a plain `walkforward --expr` is a different book: for a constant finalist every quantile threshold is flat, and the command warns. `walkforward --from-run` refuses a window inside the rows the run trained or selected on like `--front` does (exit 2); `--allow-in-sample-windows` labels it `in_sample:true`.
* **Target spec in walk-forward.** `--target-spec <file>` fills `--horizon`, `--entry-offset`, `--delay`, the return column, the units and the cost from the entry whose `column`, `name` or `return_column` is `--target`. A flag that disagrees is refused like the cost.
* **Ladder grid.** With a spec cost the default `--cost-grid` is 1x, 2x and 3.33x the per-side cost. With a flag or the default cost it stays `12,20,40`. An explicit `--cost-grid` is absolute.
* **Reports.** The `objective` block of `run.json`, the walk-forward report, the ladder report and the ablation result carry `cost: {per_side_bps, source, return_units}` with `source` one of `target_spec`, `flag` or `default`. A `flag` source priced against a spec adds `spec_per_side_bps`. `objective.finalist.constant` is true when the finalist reads no variable.

Verify: `tests/cost_contract.rs` (`one_cost_one_book_across_evaluators`, `walkforward_from_run_prices_the_selected_book`, `a_log_return_column_is_converted_with_exp_m1_in_every_evaluator`, `the_cost_flag_must_agree_with_the_target_spec`, `the_cost_source_is_reported`, `from_run_refuses_a_run_without_an_objective_and_competing_rules`, `from_run_prices_the_band_book_of_the_search`, `from_run_without_a_band_keeps_the_threshold_book`, `from_run_refuses_a_conflicting_band_flag`, `front_candidates_take_their_own_band`, `from_run_records_the_band_source`, `a_long_label_needs_an_explicit_fee`) and the `resolve_cost_*` unit tests in `src/target.rs`.

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
| rows / periods | test rows and periods in the return series (buckets `floor(t / S)` with a declared horizon, else distinct timestamps) |
| total cost / total variation | charged cost and `sum |position change|`; pooled values include the carried seam |

**Period rule (#1016).** One period is one label horizon wide. The width `S` is, in order: the target spec's horizon (`S = max(horizon_seconds, stride_seconds)`, #1053) (`annualization.source` `target_spec`), `--horizon` (`horizon_flag`), else the median positive step of the distinct times of rows with a finite target (`inferred_labelled_rows`). Unlabelled rows never enter the inference. `periods_per_year = 31536000 / S`: 8760 for hourly labels, 365 for daily, 52.1429 for 168 h. When `S` is declared, the period series is the bucket `floor(t / S)` with the equal-weight mean of its rows, so a stride at staggered phases yields one return per calendar bucket, not one per timestamp. For hourly labels and for anchored strides (every group on the same stamps, one stamp per `S`) the series equals the per-timestamp series. Empty buckets are absent. `annualization.period_seconds` and `windows[].inferred_period_seconds` report `S`.

**Stride above the horizon (#1053).** A label whose stride is longer than its horizon has one independent return per group every stride. `rcsr enrich` writes `stride_seconds` (`stride * bar_seconds`) into `<out>.targets.json` when the stride is wider than the horizon; the field is absent otherwise, so a spec and its hash are unchanged for `stride <= horizon`. `S = max(horizon_seconds, stride_seconds)` in the walk-forward and in the search objective (`TradingSpec::period_seconds`, which robust selection also uses for its period buckets). Example: `long:1,fee=0,stride=24,anchor=2021-01-04` gives `S` 86400 and `periods_per_year` 365; `rcsr walkforward` agrees with the same rows declared as a one-day label. A spec without `stride_seconds` (an older store, a hand-written spec) is refused with exit 2 when the smallest per-group step of the labelled test rows (`split.min_group_step_seconds`) is at least `2 * horizon_seconds`: the message names the step and asks for `stride_seconds`. `--horizon` without a spec keeps `S = horizon`. The search objective does not apply that refusal. `--periods-per-year` is checked against `S` as above. The target spec field is optional and must be at least `horizon_seconds`.

`--periods-per-year` that differs from `31536000 / S` by more than 1e-9 relative, with `S` from the target spec, exits 2 and names both values. With `--horizon` alone, an explicit value overrides the derived one (`source` `explicit`). Without a declared horizon the value is used as given. The default bootstrap and null block length is `max(ceil(horizon / S), hold, ceil(n^(1/3)))` periods, so the horizon is one period for stride = horizon. The report records the value in `risk_model` with the annualisation mode.

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

Training fits, validation selects, development out-of-sample evaluates the procedure, and a locked test is read once per sealed candidate (once in total after a final seal). [`protocol/windows.csv`](../protocol/windows.csv) holds the UTC boundaries (`rcsr protocol windows` regenerates it, and a test pins the file to the generator):

| range | UTC |
|---|---|
| warm-up (indicators only, never trained on) | 2021 |
| development | 2022-01-01 to 2025-06-28T11:00 (locked start minus `purge_seconds`) |
| expanding folds | quarterly tests from 2023-01-01, train from 2022-01-01 |
| rolling folds | quarterly tests from 2024-01-01, train on the previous 24 months |
| locked test | 2025-07-01 to 2026-07-01 |
| reserved | 2026H2 (the engine lockbox) |

Each row has `train_end = test_start - purge_seconds` (219600 s: 1 h entry plus a 60 h label). A development test window ends at `min(quarter end, locked start - purge_seconds)`, so `expanding-2025Q2` and `rolling24-2025Q2` end at 2025-06-28T11:00:00: a label at a later stamp reads bars of the locked test. Outside `--locked-test`, `walkforward`, `ladder` and `ablation score` refuse (exit 3) a window that reaches `[locked start - purge, locked end + max(embargo, purge))` (purge = entry offset + horizon + delay for `walkforward`; the larger of `--purge-seconds` and the `--target-spec` purge for the others; embargo = the lookback embargo of the store, `--lockbox-embargo` widens it): the features of a row after the range end read locked bars, and its label shares returns with the last locked labels. A row stamped `t` reaches a range `[S, E)` when its half-open label window `[t, t + purge)` overlaps it, so the row at `S - purge` is kept; the label report, the search lock-box check and the `walkforward` read-time drop and window guard (`ProtocolOptions::locked_reach`) share this predicate (`Lockbox::reaches`) and refuse the same rows (`ceil(purge_seconds / bar_seconds) - 1` rows per symbol before `S` on a dense grid). `ladder` and `ablation score` end the fit rows one purge before `validation_start` and the validation rows one purge before `test_start`, so a table written with a shorter purge than the label (219600 s against 608400 s at 168 h) loses the validation rows whose label would reach the test block instead of training on them. In `--locked-test` mode the other declared ranges stay sealed and every window or fold must clear their `[start - purge, end + embargo)` too; in `--locked-test` mode the training range must end at or before `locked start - purge` and `validation_start` six calendar months before the test start. `--windows protocol/windows.csv --windows-scheme expanding` loads one scheme; the table's `validation_start` replaces `--validation-frac`. The validation slice sets the threshold and selects among `--expr` candidates. Select the procedure, not the lucky candidate: pool the development series of the chosen procedure.

The table's `validation_start` is the validation slice of every window: it sets the threshold and selects among repeated `--expr` candidates, `--front` members and `--select-band` (`selection.selected_on` is `validation`). `--validation-frac` together with the table exits 2 (`mutually exclusive`); repeated `--expr` with neither exits 3.

### Overlapping returns

The book earns the per-row forward return on every long row and charges one cost per position change. With rows closer than `--horizon` in one group, a long run earns the same price move several times. `rcsr walkforward` therefore refuses (exit 2, `overlapping returns`) any window whose test rows have two rows of one group less than `--horizon` seconds apart; a step equal to the horizon passes, a gap passes, equal timestamps are skipped. Rows without a finite return (for example the off-stride rows of a `stride=h` label) do not count. `--horizon 0` skips the check. The report carries `split.min_group_step_seconds`, the smallest positive step. `rcsr ladder` and `rcsr ablation score` apply the same check to the test rows of every fold (horizon from the target spec, or `--horizon-seconds` for the ladder), refuse with exit 2 and the same message, and report `min_group_step_seconds`.

### Purge

The required purge is `--entry-offset + --horizon + --delay` seconds. A window whose `test_start - train_end` is shorter is refused, and the report carries `split.required_purge_seconds`, `split.purge_seconds` (the smallest gap) and per window `train_last_time`: the stamp of the last training row, whose label ends at most `required_purge_seconds` later, before `test_start`. When `<data>.labels.json` holds the report of the target spec column (written by `rcsr enrich --label long:...`), `split.fit_label` records its `name`, `stride`, `average_uniqueness`, `effective_n` and `effective_n_hours`: rows are not independent observations, and a stride-h label has `average_uniqueness` 1. `rcsr ladder` records the same object as `split.fit_label`. The object copies the label sidecar and stays out of the reproduction and read hashes, so a recorded read keeps its hash. The purge is in time and applies to every symbol, so a late listing or a hole in one symbol cannot bring a training label into a test window.

`--gap` counts rows and does not apply to explicit `--train-select`/`--test-select`. `rcsr protocol selectors --windows protocol/windows.csv --label expanding-2023Q1 --time-col ts` prints selectors with the purge written into the training range.

### Economics

The `economics` object (and a Markdown section) holds, for the pooled series: Sortino, Calmar (`cagr / |max drawdown|`), CVaR 95 and 99 (the mean return of the worst `ceil(n (1 - level))` periods, a negative number), the break-even cost in bp per side (`base * sum(gross) / sum(gross - net)`, since cost is linear in the per-side rate; `null` with `break_even_status` `gross_nonpositive` when the summed gross return is not positive, and `null` with `no_cost` when no cost was paid, otherwise `defined`; the Markdown shows `n/a (gross <= 0)` for `gross_nonpositive`), the cost ladder at 0.5x, 1x, 2x and 3x the cost, and the daily Sharpe and Sortino (`sqrt(365)`) when the stamps are epoch seconds and the period `S` is shorter than a day. Daily returns compound the periods of one UTC day. For `S` of a day or more, `daily` is `null` and `daily_unavailable` is `"period >= 1 day"`: the main series is already daily or slower.

The benchmark table has the same fields for the strategy, `cash`, `buy_and_hold` (one charged entry and exit per symbol), `equal_weight_universe` (always long, no cost), `constant_long_flat` and `best_long_flat_baseline` (the best mean of cash, buy-and-hold and constant, ties to the earlier). Exposure is long rows over valid rows. Beta is on the equal-weight universe. Every benchmark series starts each window flat, like the controls.

### Block uncertainty (`--bootstrap B`)

* **Circular block bootstrap** of the pooled net series with block length `L >= max(h, holding period)` periods (default `max(h, hold, ceil(n^(1/3)))`; a shorter `--block-len` is refused). It reports the mean and annualised Sharpe with percentile 95% CIs.
* **Design effect**: the variance of the bootstrap mean with blocks over the same with iid rows (`L = 1`). `effective_n = n / design_effect`, in periods (hours on an hourly grid). The `p_mean_positive` field is the one-sided centred p-value `(1 + count) / (B + 1)`.
* **N_eff**: the participation ratio of the candidate correlation matrix, `N^2 / sum c_ij^2`; identical candidates give 1 and uncorrelated ones give `N`. With `--trials N` the effective trials are `N * PR / candidates`.
* **DSR** (Bailey and Lopez de Prado) at `N_eff` and `10 N_eff`, with `SR0 = max(0, sqrt(V) ((1-g) Z(1-1/N) + g Z(1-1/(N e))))` (the bracket is negative for `1 < N < 1.2836`; the floor keeps `SR0 >= 0`, so the DSR never exceeds the PSR), `V` the variance of the candidate Sharpes (the Mertens estimator variance with one candidate), `T` the effective n, and skew and raw kurtosis of the series. All inputs are in the report.
* **PSR** (`multiple_testing.psr`): the same formula at `N = 1` (`SR0 = 0`), the probabilistic Sharpe ratio against a zero benchmark. `--from-run` with no in-sample window has `N = 1`, so `psr` equals `dsr_at_n_eff`. `n_trials_source` names where `trials` comes from (see the default table under `--front`).
* **MinBTL** (`multiple_testing.min_btl_years`): `inference::min_btl_years(sharpe_annual, n_eff)`; 0 at `n_eff = 1` and wherever the floored `SR0` is 0, NaN (null) when the annual Sharpe is not positive.
* **PBO** by CSCV with `--pbo-blocks` (default 16) blocks, per-period Sharpe as the measure; needs two candidates.
* **SPA** (Hansen 2005, joint circular block bootstrap, consistent, lower and upper recentering) of the candidates against the best long/flat baseline.
* **BH-FDR** at `--fdr-q` over the per-candidate one-sided block-bootstrap p-values.

Candidates are evaluated on the test periods with their own validation threshold only for these diagnostics; the selection stays on validation.

### Nulls (`--null-draws B`)

* **Hour-block permutation**: within each test window the cross-sections of targets are permuted in blocks of `L` periods (the same permutation for all symbols); positions and costs stay fixed.
* **Matched-exposure shifted long/flat**: one circular time shift per draw and window, shared by all symbols. Each symbol's long and flat series is rotated by the same fraction of the window (a joint period shift when every symbol has every period), so each symbol keeps its exposure and the symbols keep their timing relative to each other. The offset is drawn uniformly from `[lo, P - lo]` with `P` the periods of the window and `lo = min(L, P / 4)` (at least 1), so a draw is never the observed book inside its own holding time. Costs are priced on the shifted book, so a run cut by the wrap pays its flips. The report field is `inference.nulls.matched_exposure_random` with `scheme` (`"shared_circular_shift"`), `distinct_shifts` (the product over windows of `P - 2 lo + 1`, saturating), `p_mean`, `p_sharpe` and the 95 % quantiles. With fewer than 20 distinct shifts `p_mean` and `p_sharpe` are `null` (a p-value below 0.05 cannot be reached); the quantiles stay.
* **Always-long**: the buy-and-hold series, reported without a p-value.

Why the shift is shared: a market-wide signal is long in many symbols at the same hours, so the book return carries the common component of the returns. Shuffling each symbol alone removes that component and makes the null far too narrow (a true null was rejected 40 % of the time at 5 % in a simulation with 50 symbols and common share 0.5). Limits: the null assumes the return process is stationary inside a window, and it tests the timing given the book's own exposure path, not the exposure level. The per-window benchmark `matched_exposure_random_long` keeps the per-symbol shuffle and has no p-value.

`p = (1 + #{null >= observed}) / (B + 1)` for the mean and for the per-period Sharpe. A search-budget null is `--null-block` in the search.

### Candidate ledger and the locked test

`--ledger FILE` makes a development run append a `candidate` line (content-addressed id over the procedure, the data SHA-256 and the build commit; `test_read:false`). Each line holds the SHA-256 of the previous line, so an edit breaks the chain. A development window that ends after the locked test start minus the label purge (`--lockbox START,END`, else the range declared with the data, else 2025-07-01 to 2026-07-01; `--lockbox none` opts out) is refused.

Each ledger line and each marker line goes out in one `write` call on an append-mode file, under an exclusive lock on the file, followed by `fsync`; a concurrent appender waits for the lock and extends the head it finds. Every reader refuses a ledger or marker whose last line has no newline, and a line that is not one complete JSON object, with a message that names the file and says "a write was cut" or "not one complete JSON object". `rcsr protocol` commands and `protocol verify` exit 1 (`protocol::LEDGER_FAULT_EXIT`); a locked read, a development run and `--verify-read` exit 3. A ledger event without the fields a read depends on (`id`, `report_sha256`, `read_count` of a `test_read`; `data_sha256`, `report_sha256` of a `baseline_read`; `sealed_sha256` of a seal and `max_reads` of a final seal; `id` of a candidate), or a marker line without `ledger_head` or `report_sha256`, is refused the same way, and the message names the line and the field; a count must be a non-negative integer. An output whose stat fails for any reason but "not found" is refused with exit 3 before anything is recorded. `verify_chain` also fails on a line that holds more than one `"prev":"`. The tool never repairs the file, because the ledger is an audit record: inspect it, restore it from a copy or edit the partial line by hand, then rerun. A marker line with no newline blocks every locked read of the store until a person fixes it. The hash input is unchanged, so ledgers written before this change verify.

`rcsr protocol seal --ledger L --out locked_candidates.json` writes the sealed list and records its hash in the ledger. `--locked-test --sealed locked_candidates.json` then reads the locked test: the candidate must be in the sealed list, the data and build hashes must equal the sealed entry (otherwise the run is void), and the candidate must not have `test_read:true`. The read appends a `test_read` line with the report hash, the new `read_count` and the locked range it read (`"locked_test":{"start","end","source"}`); a second read is refused. A `candidate` line and the sealed list entry hold the range the candidate was developed against. A final read reports the store's `data_sha256` and the path of its read marker in `protocol.lockbox`, so a reviewer can check both. A copy of a store without its `<data>.locked_reads.jsonl` marker is not detected: the marker lives beside the store and nothing else records the read.

A read whose resolved range differs (another `--lockbox`, or a declaration that changed after development) is refused with exit 3: `candidate <id> was developed against the locked test A..B (<source>); this read resolves C..D`. A candidate of an old ledger without the member is refused unless `--allow-unbound-ledger` is given; the read is then reported as unbound. `--verify-read --expect-report-sha256 H` recomputes the read and compares hashes without recording anything (the same flags work for `rcsr ladder --locked-test`; see Report files of a locked read). The report's `protocol` object holds the lockbox rows, date range and read count; `run.json` carries a `lockbox` object (`--lockbox`, `--ledger`) with the rows in the range, their first and last stamps and the read count.

`rcsr ladder --scheme locked --locked-test --ledger L --sealed S` reads the locked test for the baseline ladder. The sealed list must be recorded in the ledger and the ledger must hold no `baseline_read`; the run appends `{"event":"baseline_read",...}` with the data and report hashes and the locked range (the report hash covers the printed report without the per-rung `wall_seconds`, the `ledger` object and file paths: `lockbox.declared_in[].file` becomes `carrier` and `sr_run.run` is left out, so two identical reads hash equal at any path), and a second call is refused. Every candidate of the sealed list must carry the range the ladder resolves (`--allow-unbound-ledger` accepts old entries without one). Without `--locked-test`, `ladder` and `ablation score` refuse any fold that reaches the declared range from `start - purge` to `end + embargo`.

#### Final seal: one read, one candidate

`rcsr protocol seal --final [--max-reads N] [--report-candidates] --ledger L --out locked_candidates.json` closes the ledger for the final result. It needs a ledger with no seal and no locked read, and it appends `{"event":"seal","final":true,"max_reads":N,...}` (N defaults to 1). Afterwards:

* `walkforward --ledger` without `--locked-test` and both `protocol seal` forms are refused (`the ledger is final`; exit 3 and 1). All candidate selection happens on development data before the final seal.
* At most N `test_read` lines (one `baseline_read` for `ladder`) are accepted. A read of a second sealed candidate after the Nth is refused with exit 3.
* The first locked read writes `<data>.locked_reads.jsonl` beside the store: ledger path, the ledger head the read extends, event and report SHA-256. A read of the same store through another ledger (a fresh file, a copy, an edited chain) is refused: `<data> was already read through ledger <path>`. The marker lives with the store, so a fresh ledger cannot reset the count. Copying the store without its marker file is outside this check.
* A vote book (`--policy vote`, #961) is one candidate: the sealed entry and the candidate id bind the ordered member expressions, `--quorum`, `--band`, `--min-hold`, the threshold rule and the other spec fields. A changed member, member order, quorum, band, min-hold or threshold is another candidate and exits 3 with `is not in the sealed list`; a member subset is refused the same way. The read scores all K members on the locked rows with thresholds from training or validation rows only, appends one `test_read` line and refuses a second read of the book. `--verify-read --expect-report-sha256 H` recomputes it. `K = 1, Q = 1` reports the same pooled metrics as the single-candidate read. `--device-pnl` and `--batch-pnl` stay refused.
* A read of a multi-candidate procedure reports the validation-selected candidate only. No other candidate is evaluated on the test block: `--bootstrap` has no candidate columns, `multiple_testing` has `candidates:1`, one p-value and `pbo_error` `final read reports the selected candidate`, and `--batch-pnl` is refused with exit 2. `--report-candidates` at seal time keeps the full report and records the choice in the seal line.
* `rcsr protocol verify --ledger L --data D --expect-reads N [--sealed S]` exits 1 unless the ledger has N locked reads, exactly one seal line that is final, no more reads than the seal allows and a marker that matches the ledger line for line. With `--sealed` it also checks that the file is the final list and that every read id is in it. Before the read, `--expect-reads 1` exits 1 and `--expect-reads 0` exits 0.

A crash after the marker line and before the ledger line leaves a marker the ledger does not record. The next call finds the same ledger head and reads once; `verify` flags the state until then. Deleting the marker makes `verify` report a ledger read without a marker line.

#### Report files of a locked read

`rcsr walkforward --locked-test` and `rcsr ladder --locked-test` write their report files around the ledger record in one order: compute the report and its hash, write every `--out` and `--markdown` file to `<file>.tmp` beside its target and `fsync` it, record the marker and ledger line, then rename each staged file over its target. `--markdown -` prints after the record. An `--out` or `--markdown` path that cannot be written (a directory, a missing parent directory, a full disk, no permission) is refused with exit 3 before any record: the staged files are removed and the locked read stays available. `--out` and `--markdown` must name different files: the check resolves the parent directory and compares inodes, so `a.json` against `./a.json`, `b/../a.json`, a symlink or a hard link is refused with exit 3 before any record. `rcsr ladder --locked-test --verify-read [--expect-report-sha256 H]` is the same route as the walk-forward's: it recomputes the report, compares its hash with the recorded `baseline_read`, writes `--out` and `--markdown` the same way and records nothing. `--verify-read` without `--locked-test` is a usage error (exit 2) in both commands. `rcsr walkforward --locked-test --verify-read` on a candidate with no recorded read exits 3 with `no recorded read for this candidate` before any locked row is scored, and `rcsr walkforward --locked-test --expect-report-sha256 H` without `--verify-read` exits 2 before the authorization. `rcsr ladder --locked-test --verify-read` with no recorded `baseline_read` exits 3 with `no recorded baseline read` before the panel is loaded; on a store whose data hash differs from the `data_sha256` of the recorded `baseline_read` it exits 3 with `data hash <got> differs from the recorded baseline read <recorded>; the verify is void` before the panel is loaded, and on a store with no read marker it exits 3 with `has no read marker`, and `rcsr ladder --locked-test --expect-report-sha256 H` without `--verify-read` exits 2 before the authorization. A locked read is scored only after the ledger allows it, and a refused run computes and prints no report hash. A wrong `--expect-report-sha256`, or a hash that differs from the recorded read, exits 3 and writes no file, so a good report on disk stays.

| State | What is on disk | A plain rerun | `--verify-read` |
|---|---|---|---|
| Report not computed, or a write to `<file>.tmp` failed | no record, no target change | reads and records | refused: no recorded read |
| Staged and synced, killed before the record | no record, `<file>.tmp` may remain | reads, overwrites `<file>.tmp` | refused: no recorded read |
| Marker line written, ledger line not | marker only | finds the marker's ledger head, writes no second marker line, records the read | refused: no recorded read |
| Recorded, killed before the first rename | marker and ledger line, `<file>.tmp` holds the report, target absent or old | refused: the read is recorded | writes the report, exit 0 |
| Recorded, killed between the renames of `--out` and `--markdown` | one target new, the other absent or old | refused | writes both, exit 0 |
| Recorded and renamed | complete | refused | recomputes, same hash, exit 0 |

After a recorded read whose report is lost, run the same command again with `--verify-read` and the recorded hash (`--expect-report-sha256`). The error after a failed rename names the staged file and this command.

Every locked read stores `binding_sha256` in its ledger line: the SHA-256 of the resolved window set and of every other setting that binds the report beyond the sealed entry (the whole walkforward spec and protocol options, `--gpu` included because it changes the report, with the `--front` path left out because the run file's SHA-256 binds its content and another spelling of the same file verifies; the whole ladder option set, columns, window rows, scheme, embargo, the resolved lock (`--lockbox`, `--retired`) and `--allow-in-sample`, with the `--sr-run` path left out for the same reason). `--verify-read` compares it with the recorded read before the panel is loaded and scores nothing when it differs: exit 3, `the window set or another setting that binds the report differs from the recorded read`, no report hash printed, ledger and marker unchanged. A read recorded without the field (written by an older build) cannot be compared: `--verify-read` refuses it (exit 3, `the recorded read stores no binding digest`) and the report hash stays unrecomputable. The digest is built from derived `Debug` text of the resolved structs, so a verify needs the same build line that recorded the read: another Rust version or a renamed field changes every digest and the verify is refused with the same message.

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

`pooled.period_net` is the pooled out-of-fold net return of every period (`pooled.periods` values, in pooled order) and `pooled.period_time` the epoch-second stamp of each period. They are part of the report hash, so a hash recorded by an earlier build does not match. `rcsr trials` reads them; see `docs/overfitting.md` section 3.14.

The printed and written JSON ends with a `ledger` object (#284): the phase ledger of the run with `io_s` (store row-group or CSV reads), `pnl_s` (scoring, windows, controls, `--batch-pnl`), `host_other_s`, `wall_s`, `unaccounted_s`, `closure` and `device_busy_s`, the union of the `--batch-pnl` device tile intervals and of the `--device-pnl` CUDA reduction calls (`host_envelope` basis; the canonical CPU backend adds none). The ledger and the `batch_timings` object (`--batch-pnl`: per seam `wall_s`, `device_s`, `score_s`, `layout_s`, `reference_s`, `candidates_per_s`, `row_evals_per_s`, `device_fraction`) hold wall-clock seconds, so `report_sha256`, `--expect-report-sha256` and `--verify-read` hash the report without them. They also leave out the `--front` path (`selection.run`; `run_sha256` stays), so `--front run.json` and `--front ./run.json` verify the same read. A hash recorded by an earlier build for a `--front` read does not match. The printed JSON keeps every path. See `docs/benchmarks.md`.

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
