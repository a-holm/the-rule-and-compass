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
`train_type7`, `validation_type7` or `fixed`, and `train_fitness` or
`validation_fitness` for a fitted cut (next section). An approximate/sketch threshold
may be used to screen candidates, but it is labelled screening-only and cannot
place a reported position.

### Fitted cut per window (`--threshold-fit fitness`, #1094)

`--threshold-fit fitness` gives each window the cut with the lowest block fitness on the window's source rows and freezes it on the test block. It needs a reference cut: `--threshold-fixed`, or the run threshold of `--from-run` (`--from-run` accepts this flag). Without one it exits 2. `--threshold-quantile`, `--policy band`, `vote`, `dp` and `top-tercile` (also the band book of a `--from-run` run), `--select-band`, `--device-pnl` and `--batch-pnl` exit 2 with a message that names the flag. `--cut-min-entries`, `--cut-blocks` and `--cut-lambda` without `--threshold-fit` exit 2.

* Source rows: the validation slice of the window, or its training rows when it has none (the rows the quantile reads). A row with a non-finite score or target is dropped (the shared valid-row mask); the rest are sorted to the canonical `(group, time, row)` order.
* Book: `trading::FitnessBook` on the source rows at the per-side cost of the run, `--cut-blocks K` time blocks (1 to 8, default 4) and `--cut-lambda L` (finite, at least 0, default 1). The fitness is the search fitness `mean_b r_b + L sd_b r_b` with `r_b = (B_b - net_b) / B_b` ([`optimisation.md`](optimisation.md), section 3.5), `min_trades` 1 and no spacing check.
* Sweep (`FitnessBook::cut_sweep`): cut `t` in `0..=n` is long on the `n - t` rows of highest score. One sort and one pass over difference arrays give the fitness, net, flips, exits and distinct entry timestamps of every cut. A cut that splits equal scores (`-0.0` equals `+0.0`) is not valid and has fitness `+inf`.
* Rule (`CutSweep::best`): among valid cuts with `0 < t < n`, a finite fitness and at least `--cut-min-entries E` (default 30) distinct entry timestamps, the lowest fitness. Ties go to the cut nearest the reference cut, then to the smaller `t`. The always-long and the flat book are never chosen. An entry timestamp is a stamp at which at least one row enters a long run, so one market-wide entry on many symbols counts once. The rule is a pure function of the source rows; it draws no random number.
* Threshold: the midpoint of the two scores around the cut, or the lower score when the midpoint is not strictly below the upper one, so `score > threshold` reproduces the cut on the source rows.
* Fallback: the window keeps the reference cut, `threshold_source` is `fixed` and `cut.fallback` names the reason: `no_rows` (fewer than 2 source rows), `book_error` (the book cannot be built, for example a time block without a profitable path; `cut.fallback_detail` holds the message), `no_valid_cut` (every source score is equal) or `floor` (no cut has `E` entry timestamps).
* No look-ahead: the source rows lie in `[train_start, train_end)`, the purge gap ends their labels before `test_start`, and rows that reach a sealed range are dropped when they are read. The cut is fitted in the candidate selection, which reads no test row; a final read (`--locked-test`) fits it before any test row is scored. `fitted_cut_ignores_the_test_rows` (`tests/walkforward.rs`) replaces every test score and test return and gets the same cut bits, and a cut that pays only on the test block is not chosen.

Each window gets a `cut` object: `source`, `value` (the window threshold), `index` (source rows at or below the cut), `rows` (source rows), `fitness`, `net`, `flips`, `exits`, `entry_times` and `exposure` (long rows over source rows) of the cut the window uses, `reference` (`value`, `index`, `fitness`, `net`, `entry_times`, `exposure` of the reference cut), `plateau` (`lo`, `hi`, `rows_lo`, `rows_hi`: the widest run of valid cuts around the fitted one whose fitness stays within `0.1 * (1 - f)` of the best fitness `f`; the fitted cut alone when `f` is 1 or more; `null` on a fallback), `floor` (`min_entry_times`), `blocks`, `lambda`, `fallback`, `fallback_detail` and `profile`: 32 cuts at long-row shares `0.5 * 0.7^j` and `1 - 0.5 * 0.7^j`, `j = 0..15`, each moved up to the next valid cut, with `long_share`, `index`, `value`, `fitness`, `net` and `entry_times`. Every number of the `cut` object is measured on the source rows, where the cut was chosen, so it is optimistic and is no out-of-sample result. The top-level `threshold` object gets `fit` (`fitness`), `fit_method`, `fit_sources`, `cut_min_entries`, `cut_blocks` and `cut_lambda`, and the Markdown report a `Fitted cut (source rows)` table with one line per window. Without the flag no field is added and the reports are byte-identical (`default_reports_are_pinned`, `tests/walkforward.rs`).

Trials: the cut search compares cuts on the source rows only. On the test block each window has one frozen book, so the cut is no trial on the scored rows, like the type-7 quantile threshold: `multiple_testing.trials` and `n_trials_source` follow the table under `--front` unchanged (`threshold_fit_keeps_the_trial_count`, `tests/walkforward.rs`). A reader who runs the fitted and the fixed cut on the same windows and keeps the better one has compared two books on the scored rows: deflate with `--trials 2`, or with `rcsr trials --threshold-arms` over both reports (without the flag `rcsr trials` refuses the pair).

Sealed reads: the flag and its three values are part of the candidate id (a `threshold_fit=fitness,...` line of the procedure text, written only with the flag, so the id of a run without it is unchanged) and of the read binding. The fixed arm, another floor or another lambda is another candidate (exit 3, `is not in the sealed list`), and a `--verify-read` with another cut rule scores nothing (`threshold_fit_binds_the_sealed_read`, `tests/final_seal.rs`).

```sh
# the cut of every window fitted on its validation slice, the run threshold as reference
rcsr walkforward --data store.rcsb --from-run run.json --time ts --group-by symbol \
  --windows protocol/windows.csv --windows-scheme expanding --threshold-fit fitness --json
```

### Shared valid-row mask (#305)

A row enters the P&L denominator only when **both** its score and its target
are finite (`trading_gpu::ValidMask`). This is deliberately stricter than
`position_for`, which only checks the score: in the pinned cell six rows have a
finite score and a `NaN` target, and keeping them would change exposure, hit
rate and every pooled return. `position = 1` iff the mask says the row is valid
and `score > theta`; `score == theta` is flat.

### Rows with no finite score (`--missing-score`, #1203)

A row with a finite target and no finite score (a NaN feature in a warm-up, a domain error of the formula) is a decision the book has to make. `--missing-score flat|hold` names the rule. The walk-forward test book and `trading::evaluate` (the validation and test books of `rcsr --objective`, which use `flat`) share one implementation, `trading::score_runs` and `trading::hold_positions`. The search fitness book (`trading::FitnessBook`, CPU and device) gives any non-finite prediction the regret `+inf`, so it never prices such a row.

* `flat` (default): the row ends the run of its group and is not priced. The run pays its exit at its last priced row, and the next priced row starts flat and pays its entry, so a long held on both sides of the stretch pays a round trip. The band and cooldown state restarts with the next run. Every run is its own book for the controls, the nulls and the hindsight bound, and the device scans and the `--device-pnl` reduction read the same runs. Timing: the missing score of row `t+1` is known when row `t+1` arrives, and the exit is charged in the period of row `t`, whose return ends at `t+1` (the book refuses rows of one group closer than the horizon). This is the convention of the final exit (#145) and of the exit of a gate on-run (#1180); no later score enters the decision.
* `hold`: the row keeps the position of the row before it in its group and earns its return, with no cost. The band and cooldown scans read the scored rows only, so their state carries across the row; top tercile and vote hold the previous position. A group's leading rows without a score hold the position the seam carries in (flat in a window that starts flat).
* A row with a finite score and no finite target is priced under neither rule (its return is unknown) and does not end a run. A row absent from the data makes no decision: the position carries to the next row of the group.
* Seams: a group whose first labelled row in a window has no score adds no seam term, since no score decides that row: under `flat` it starts the window flat (the carried position closed at the earlier block's end), under `hold` it keeps the carried position. Under `flat` a group whose last labelled row has no score leaves the window flat. The seam rules (`--pool-reset-flat`, a gap between test intervals) are unchanged.
* The report carries `book: {missing_score, missing_score_exits, unpriced_target_rows}`. `missing_score_exits` counts the pooled exits paid at the last priced row before a missing score. `unpriced_target_rows` counts the rows with a finite score and no finite target inside a run of the book that the return of the run's previous priced row does not cover (`t >= t_prev + horizon`), so the off-stride rows of a strided label do not count. `procedure_spec` adds `missing_score=<rule>`, so the candidate id names the rule. `rcsr trials --wf`, `--compare` and `--cpcv-assemble` guard `book.missing_score` (exit 2 naming the field). A report without the field comes from a build that kept the position open across the stretch, booked no return there and paid no exit, so it never mixes with a `flat` or a `hold` report.

`flat` is the default because it is the position rule above (a non-finite score is flat), it prices a stretch with no score like a gate-off stretch (#1180), and a long-only book with no signal holds cash. On the synthetic fixture of #1203 (two symbols, a 40-row stretch with no score over an 18 % fall), the `flat` book equals the book split into two windows at the stretch, and the `hold` book equals the book of a score with no gap. `hold` suits a signal that is meant to persist through a gap in its inputs.

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

Interval-list runs (`--split-table`, #1173) compare intervals instead of the last seen time. `run.json` records `split.intervals`, `split.interval_purge_seconds` and `split.embargo_seconds`; the reader resolves the kept parts of the training and validation intervals again (`split.effective_intervals` shows them) and calls a window `[test_start, test_end)` in sample when a kept row stamped `t` reaches it: `t + purge > test_start` and `t < test_end + embargo` (`RunFile::in_sample_window`, the predicate of the locked-range guard). A combinatorial test group that lies before later training rows is therefore out of sample, because the run itself purged the rows before it and embargoed the rows after it; a window one second wider on either side is refused (exit 2). `ladder --sr-run` and `ablation score` apply the same rule to `(test_start, test_end)` of every fold. A malformed `split.intervals` block is an error, never the old rule. A run without `split.intervals` keeps the last-seen rule and its messages unchanged.

`--windows <splits table> --split-id <id>` takes the test intervals of one split of a `split_id,role,start,end` table as windows, labelled `<id>-test<k>`. Such a window has no training range of its own (one second ending at `test_start` minus the purge gap), so it needs the fixed threshold of `--from-run` or `--threshold-fixed`. `--split-id` without a splits table exits 2. Test intervals of a split are separate books (#1189): the pooled path starts every group flat at each window and pays each window's final exit, so `pooled.period_net` is the concatenation of the window series, `pooled_seam.seam_cost` is 0 and `pooled_seam.reset_flat` is true. Windows from a splits table whose test intervals touch (the end of one is the start of the next) keep the carry across that seam and reset only at gaps; `--batch-pnl` refuses such a mix with exit 2 unless `--pool-reset-flat` is set (every seam then starts flat); a table whose test intervals all touch keeps the carry with `--batch-pnl` too. Windows from a windows table keep the carry.

The report's `selection` block gains `source: "front"`, `run`, `run_sha256` (SHA-256 of the file) and `candidate_labels`. A `--from-run` report gains `run` and `run_sha256` (#1238), so a sealed read names the run file it priced and `rcsr protocol verdict` binds it to the pick whose freeze map named that file; the candidate id does not depend on the file hash. `--select-band` and `--batch-pnl` work unchanged on the front's candidates.

Composite marker (#1237, broad-pilot decision D19): a run.json of `rcsr compose` (#1181) holds `compose.chosen` (`type` C0 to C3, `form`, `gate`). `--from-run` and `--front` of such a run add `"composite":{"source":"rcsr compose","type":"C1","form":"if(gate, f, cash)","gate":"g1"}` to the `selection` block (after `candidate_labels` under `--front`; `gate` is null for C0), so `rcsr trials --table` can cap the composites among its picks (D8). Every member of an `--all-members` report of a compose run belongs to the composition family and takes the report's marker. A report of any other run carries no marker and is byte-identical to the report of an earlier build.

### Every member out of sample (`--all-members`, #1176)

`--front run.json` selects one member per window by validation error, so its report holds the test book of the selected member only, and `--batch-pnl` gives net per window and member, not per period. A selection rule over several out-of-sample metrics (#1177) needs the test book of every member. `--all-members` adds it without changing the selection:

```sh
rcsr walkforward --data panel.rcsb --front s07.run.json --windows cpcv62/windows_s07.csv \
  --all-members --period-grid target --json --out wf_s07_members.json
rcsr trials --wf wf_s07_members.json --members all --out members_s07.json
```

* Every member is priced alone on every window with its own carried threshold and band (`validation_book.band_resolved` under sd units), exactly as `--expr <member>` with the run's threshold, cost and band would price it: the same rows, the same missing-score rule (`--missing-score`), the same seams (`--pool-reset-flat`, the flat start after a gap between the test intervals of a splits table), the same regime gate (`--pnl-mask` of the run) and the same period grid. The scan reads the data once for all members.
* `members[]`, in front order: `archive_rank`, `label` (`front_rank_<k>`), `complexity`, `expression` (the string priced; `selection.expression_source` says whether it is the exact or the rounded form), `constant` (the expression reads no input, so the book takes one position on every row), `dropped_rows` (the member's own count under the row-drop rule of the scalar evaluator), `windows` (label, threshold, threshold source, row counts and metrics per window), `pooled` (the pooled metrics, `net_mean`, `entry_times`, `seam_cost`, `missing_score_exits`, `unpriced_target_rows`, `break_even_bps_per_side`, `break_even_status`, `cost_ladder`) and `series` (the `--emit-series` series below, and `period_rows` under `--period-grid target`).
* `selection.source` is `front_all_members` and the report has `diagnostic: true`. The top-level windows, pooled block, economics and inference are those of `--front --emit-series`: no member enters the selection, `selection.test_used_for_selection` stays false and the deflated Sharpe keeps the front size as its trials (#1027). A member carries no bootstrap, null or batch section.
* Windows without a validation slice (a splits table with `--split-id`): plain `--front` refuses to select among several members (exit 3). With `--all-members` and no validation slice in any window nothing is selected: the report book of every window is the first front member, `selection.selected_on` says so, and every member is priced as above. When a windows table gives a validation start for only some windows, those windows select by validation error and the others keep the first front member; `selection.selected_on` is `validation (k of n windows); first front member where no validation slice`.
* `--period-grid target` puts every member on one period grid, so `rcsr trials --members all` and the candidate table of #1177 join the members period by period. Under the score grid a member with no score on a stretch has no period there.
* `--select-band`, `--threshold-fit`, `--policy vote` and `--locked-test` exit 2: the flag is a development diagnostic that selects nothing, and a locked read reports the sealed candidate only. `--all-members` without `--front` exits 2. A front of one member gives one member equal to the top-level book, and to `--from-run` when the run's finalist is that member.

### Per-period series (`--emit-series`, #1176)

`--emit-series` (implied by `--all-members`) writes, beside `pooled.period_time` and `pooled.period_net`, the `--period-series` series `period_gross`, `period_exposure` (mean position of the period's priced rows), `period_entries` (rows entering a long run) and `period_bh` (the cost-free equal-weight mean of the same rows), and adds:

* `period_cost`: `period_gross - period_net` in f64, the per-period cost the cost ladder and the break-even use. The net at `m` times the cost is `period_gross - m * period_cost`; with the same f64 operations it reproduces every `cost_ladder` row bit for bit, and at `m = 1` it returns `period_net` up to one rounding (exactly when the difference is exact). Weighted by the priced rows of each period it sums to `pooled.total_cost` up to rounding.
* `period_entry_stamp`: 1 for a period where at least one row enters a long run, else 0.
* `period_bh_net`: the net of the buy-and-hold control on the same rows at the report cost (long on every priced row, the entry paid on the first row of a book and the exit on its last, every window started flat): the series behind `economics.benchmarks` `buy_and_hold`. Excess returns are measured against it or against the equal-weight universe, never against an interest rate.
* `pooled.entry_times`: distinct timestamps at which at least one row enters a long run; one market-wide entry on many symbols counts once, as the fitted-cut sweep of #1094 counts them.
* `windows[].entry_times` (#1236): the distinct entry stamps of each window's pooled book; the windows of a report sum to `pooled.entry_times`. Windows do not overlap, so a group of windows (a CPCV test group, also one cut by a locked hole) has the sum of its windows' counts, which `rcsr trials --cpcv-assemble` carries into each path.

Every series has the length of `period_time` and, under `--period-grid target`, 0 on an empty period. Without `--emit-series` and `--all-members` the report is byte-identical to the report of an earlier build.

## Position rule, costs and aggregation

* `position = 1` if `score > threshold`, the score is finite and the shared
  valid-row mask keeps the row (target finite too), otherwise `0`. There is no
  short leg and no leverage; `rcsr::walkforward::position_for` exposes the
  score-only comparison and `rcsr::trading_gpu::long_positions` the masked
  vector.
* `--policy threshold` (default) is the rule above.
* `--policy band` is a long-only no-trade band. With threshold `θ` and
  half-width `β = --band`, an entry needs `score > θ + β`; a long book stays
  long while `score > θ − β` and exits at or below `θ − β` (a row with no
  finite score follows `--missing-score`). `--min-hold K` (default 1) delays an exit until the book has been
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
* **The run's band (#946).** `--from-run` and `--front` price the book the search selected. A run that priced a band or a minimum hold (`objective.spec.band`, `band_units`, `band_resolved`, `band_sd`, `min_holding`) is priced with `--policy band`, the run's `band_resolved` as the half-width in score units and its `min_holding`. The half-width is the recorded value: a window never resolves an sd from its own rows, so no window or test row sets the band. `--front` gives every candidate its member's `validation_book.band_resolved` (sd units; score units use the run's `band_resolved`) and the run's minimum hold. A run without the keys, or with `B = 0` and `min_holding = 1`, prices the threshold book. A `--policy`, `--band`, `--band-units` or `--min-hold` that differs from the run exits 2 and names the run's value; the same values are accepted. A run with `band_resolved` but no `band_units` is refused (exit 3). `policy.band_source` is `run` when the band comes from the run and `flag` otherwise; `windows[].band` records the run's `band`, `band_units`, `band_sd`, `band_sd_rows`, `band_sd_max_time` and `band_resolved`. `ablation score` and `ladder --sr-run` price the threshold rule only and refuse a run that priced a band or a minimum hold (exit 3). Under `--pnl-mask` the recorded `band_sd` is the sd over the gate-on training rows (#1246).
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
  (usage error, exit 2) and is required. A row where some member has no finite score has no vote count, so it
  follows `--missing-score` like a row with no score in the threshold book. `K = 1, Q = 1` is the threshold book
  bit for bit. `Q = 0.67` with `K = 3` needs 3 votes (`ceil(2.01)`); pass `0.66` for a majority of three.
  When `Q x K` lies strictly above a whole number by at most 0.05, `rcsr walkforward` prints one `warning: --quorum ... requires N votes` line on stderr with the largest two-decimal quorum that needs fewer votes; the exit code and the report (including the reproduction hash) do not change.
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
scalar evaluator, which applies `--missing-score` to such rows; its cells
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
The `group` of the draw is the group id: the hash of the label for a text
`--group-by` column, the value for a numeric one (`-0` and `0` are one group, NaN
is one group). A CSV and a `.rcsb` store of the same rows give the same ids and
the same controls, whether or not a dictionary is kept for the column.

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
* Under the plain threshold rule with no cash return and no funding, the period metric block (`period` in the Rust API) equals the scalar walk-forward evaluator bit for bit. The strict `>` comparison, the finite-return mask and the missing-score rule (`TradingSpec::missing_score`, `flat` in a search run) are the same.
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
* **Position rule.** `rcsr walkforward --from-run run.json` and `rcsr ablation score --from-run run.json` take the expression, the fixed threshold `objective.spec.threshold`, the per-side cost `objective.spec.per_side_bps` and the target spec (return column, units, horizon, entry offset, delay) from the run. A run without an `objective` block is refused (exit 3). `--expr`, `--threshold-quantile`, `--threshold-fixed` and `--target-spec` with `--from-run` are usage errors (exit 2). `walkforward --from-run --threshold-fit fitness` keeps the run threshold as the reference cut and fits the cut of each window (see "Fitted cut per window"). The quantile rule of a plain `walkforward --expr` is a different book: for a constant finalist every quantile threshold is flat, and the command warns. `walkforward --from-run` refuses a window inside the rows the run trained or selected on like `--front` does (exit 2); `--allow-in-sample-windows` labels it `in_sample:true`.
* **The run's regime mask (#1180).** A run searched with `--pnl-mask` records `objective.spec.pnl_mask` (with `state_rule`). `walkforward --from-run` and `--front` price that book: the gate state of every test row comes from the gate pass under `rcsr-gate-state/1` (below), every on-run of the labelled rows is its own book that starts flat and pays its exit, a row with the gate off drops out of the book, and a row without a finite target (an unlabelled row of a strided label) neither starts nor ends an on-run (#1278; the definition is in `docs/optimisation.md`, `--pnl-mask`). The windows do not carry a position over a seam under a mask. The gate columns must exist in the data (exit 2 otherwise); a gate column of the label family, the target, its return column or a construction column exits 5. `--unmasked` prices the raw formula as a diagnostic (exit 2 when the run priced no mask). `--device-pnl`, `--batch-pnl`, `--threshold-fit`, `--policy top-tercile`, `--policy vote` and `--policy dp` with a masked run exit 2 (pass `--unmasked`). A single window equal to the run's test block reproduces `objective.test`: its rows, trades, entries and `total_cost` equal the run's, and since a period return is the mean over the rows priced at its stamp, the run's `net_pnl` is `mean_return * periods` times the number of symbols only when every symbol is priced at every priced stamp; with gates that differ by symbol it is the sum of the windows of each symbol alone. `procedure_spec` adds `pnl_mask=<expr|band bits|dwell>|rule=rcsr-gate-state/1` for a masked procedure, so the candidate id of a masked run differs from the id of its `--unmasked` pricing and of a run with another gate expression, band or dwell; an unmasked procedure keeps its text and id.
* **The switch of a composed run (#1214).** `rcsr compose` writes `objective.finalist.switch` and `pareto[].switch` for every composite whose gate decides between two branches (`docs/compose.md`). `--from-run` prices the finalist's switch and `--front` the switch of each member: the score of a row is the score of the branch its gate state holds, and the existing book code prices it at threshold 0 (one book: a flip pays the cost of the actual position change; a row whose held branch has no score follows `--missing-score`). `windows[].expression` is the display form. `--device-pnl`, `--batch-pnl`, `--threshold-fit`, `--select-band`, a `--policy` other than threshold, a band and `--unmasked` with a switch exit 2; a gate column of the label family, the target, its return column or a construction column exits 5; a run with `compose.gate.carried` and no switch exits 3. `procedure_spec` adds one `switch=<state_rule>|<expr>|<band bits>|<dwell>|on=<on>|off=<off>` line per candidate (`switch=none` for a plain member) when a candidate is a switch, so the gate and both branches enter the candidate id; a run without a switch keeps its text and id.
* **One gate-state rule (#1214).** The run's mask, every switch and every `--regime-expr` read their states from one gate pass under `rcsr-gate-state/1`: the kernel over every row of the data file with a finite time, per group in file order, from the group's first row up to the last window end, never over the window rows alone. The pass reads the time, the group and the gate columns only (a second store projection), never a label or a row filter, so the state of a row is the same in every window, slice and command, and a window start inside a running dwell keeps the dwell. A row inside a locked range this run does not open, or within its lookback embargo after the end, is not read, and the state restarts at a group's first readable row after it; the purge rows before a locked start are read for the gate. A sealed read opens its own range, so its warm-up is the development rows before the locked start; the gate pass over the locked rows runs after the ledger authorized the read. Rows of a group out of time order exit 2. The report gains `gate` (`state_rule`, `gate_rows_read`, `barriers` with `start`, `end` and `kind`, and `gates` with `source`, `expr`, `columns`, `band` and `dwell`; the fields of a single gate also stand at the top) and `windows[].gate`: `source`, `start` (the warm-up at the test start: `warmup_rows`, `warmup_first_time`, `warmup_last_time`, `groups`, `on_at_start`, `dwell_running_at_start`) and, for a switch, `regime`, the object `rcsr compose` writes for its validation and test books (per state rows, share, net, exposure, buy-and-hold net and alpha; the seam; switches; `switch_cost` against `total_cost`). A run without a mask, a switch or a regime expression has no gate pass and an unchanged report.
* **Target spec in walk-forward.** `--target-spec <file>` fills `--horizon`, `--entry-offset`, `--delay`, the return column, the units and the cost from the entry whose `column`, `name` or `return_column` is `--target`. A flag that disagrees is refused like the cost.
* **Labelled rows (#1291).** The walk-forward prices the rows of the search's books. A row is labelled (priced, counted in `rows`, read by the spacing and stride checks, the threshold, the constant control and the period inference) when its return is finite and, with a target spec (`--target-spec`, `--from-run`, `--front`), its fit label is finite too: the spec entry's `column`, the column the search fits and `--fill drop` filters on. Under a return column that is denser than the fit label (a daily return beside a label every third day, or the volatility warm-up rows of an `rcsr enrich` label, where `lbl_<name>_fwd` is finite and `lbl_<name>_y` is not) the rows without a fit label drop out, as in the search. A run searched with `--book-target COL` priced every row with a finite return of its book label: `run.json` records that label's spec as `objective.book_target_spec`, and `--from-run` and `--front` price that return column, horizon, stride and cost on every row with a finite book return. A `--book-target` run without `objective.book_target_spec` exits 2 (search again with this build). For such a run `--target` names the book label or its return column; the fit label as `--target` exits 2 (`--target <col> is not the spec's return column ... or label column ...`), because the run's books were priced on the book label's return column and walk-forward prices the same column. Data without the fit label column exits 2 under `--from-run` and `--front` (the run's books need it); with `--target-spec` the return alone labels a row, stderr prints a note and `split.labelled_rows` says `return`. Three cases keep the finite return alone, because the search priced every finite return there too: a top-tercile book (#493: a row whose label alone is missing stays in the search's cross-section and earns its return), a run whose `preparation.fill` filled missing labels (`--fill` mean, median, zero, previous or interpolate; drop and none fill nothing), and a spec whose `return_column` is its `column` (or no spec). The report's `split.labelled_rows` names the rule: `{"rule":"fit_label","column":<label>}`, `{"rule":"book_target","column":<book label>}` or `{"rule":"return"}`; like `split.fit_label` it stays out of the reproduction hash. `rcsr ladder` applies the same rule (one constructor, `RowRule::of_search`, through `ladder::search_rows`) to its folds: a row needs a finite return and a finite `--target`; every finite return counts when the `--target` spec entry is of kind `xs_rank` (the search prices that label only as the top-tercile book, so the ladder's top-tercile rung holds the rows of `objective.test`), when no `--target-spec` names a fit label, and when the `--sr-run` run was searched with `--book-target` or filled missing labels. Its report carries `split.labelled_rows`. `rcsr ablation score` applies the same rule through `ladder::search_rows` (the ladder's own function) to the run's spec and run file (`--run` and `--from-run`), and its report carries `labelled_rows` (`{"rule":...}`) next to `purge_source`. The candidate id binds the rule and every input of it (#1312): `procedure_spec` holds one line `rows=rule=<return|fit_label|book_target>|fit=<fit label column>|return=<return column>|fill=<run preparation.fill>|top_tercile=<true|false>|book=<book label>`, each column as a JSON string and `-` for an absent input, built by one function (`RowRule::of_search`, which calls `LabelledRows::of_search`) for every walk-forward: plain `--expr`, `--target-spec`, `--from-run`, `--front` and `--book-target`. `fit` is the fit label column: the `column` of the `--target` entry of `--target-spec`, or the run's `target_spec.column` under `--from-run` and `--front`. Two entries on one return column are two candidates, and so are two runs that differ only in `preparation.fill` or only in `target_spec.column`; a sealed candidate read under another fit label is not in the sealed list (exit 3).
* **Book label (#1212).** `--book-target COL` with `--target-spec` prices the long/flat book of a formula fitted on `--target` (the fit label) on another entry of the same spec file, matched by `column`, `name` or `return_column`, for a cell that fits one label cadence and trades another (a weekly `lbl_h168` fit on a daily `lbl_h24` book). The book label's return column, horizon, stride and cost price the book on every row with a finite book return (`split.labelled_rows` rule `book_target`, the rule of a run searched with `--book-target`, one function `LabelledRows::of_search`), so the period width, the pooled period grid, `periods_per_year` and the bootstrap block follow the book label: a daily book has one pooled period per day and `periods_per_year` 365. The fit label only sets a floor on the purge: `required_purge_seconds = max(book entry_offset + horizon + delay, fit entry_offset + horizon + delay)`, used for every window gap, the splits-table gap and the locked-range boundary; a window whose gap is below the fit label's requirement exits 2 and names the fit label and both requirements. Refusals (exit 2): no `--target-spec`, `--from-run` or `--front`; a label the spec does not hold; a non-`forward_return` label on either side or a book label without a `return_column` (the rules of the search's `--book-target`, one function `target::book_pair_refusal`); a data file without the book return column; `--device-pnl` or `--batch-pnl` with a book label. The fit label itself as `--book-target` changes nothing: the report and its `report_sha256` equal the run without the flag byte for byte. `--from-run` and `--front` price the run's `objective.book_target_spec` without the flag (and `--front --all-members` prices every member on it); `--book-target` with them names that label, the fit label for a run without `objective.book_target`, and another label exits 2. A `--from-run` or `--front` report of a run without `objective.book_target` is unchanged. The report's `book` block adds `target` (`column`, `name`, `return_column`, `horizon_seconds`, `stride_seconds`), `fit_target` (`column`, `horizon_seconds`, `required_purge_seconds`) and `source` (`flag` or `run`), inside the reproduction hash; `split.fit_label` holds the label report facts of the fit label. `procedure_spec` adds `book_target=<book column>|fit_label=<fit column>|fit_purge=<seconds>`, so the candidate id binds both labels and a locked read of another book label is not the sealed candidate (exit 3).
* **Book label in the ladder and ablation (#1311).** `rcsr ladder --sr-run` and `rcsr ablation score` (`--run` and `--from-run`) of a run searched with `--book-target` price the run's `objective.book_target_spec` as `walkforward --from-run` does, through the same `BookLabel`: the book label's return column, horizon, stride and cost price every ladder rung and the ablation finalist on every row with a finite book return (`labelled_rows` rule `book_target`), so a weekly fit traded on a daily book has one period per test day. The purge is the larger of the two labels' `entry_offset + horizon + delay`, then the protocol default and `--purge-seconds` as before; `purge_source` is `target_spec` when a label binds. `--target` names the fit label or the book label (`column`, `name` or `return_column`) and both price the same book (the ladder panel's target is the book label); another label, a `--return-col` other than the book label's, and a run with `objective.book_target` but no `objective.book_target_spec` exit 2. Both reports add `book` (`target`, `fit_target`, `source: run`); the ladder's `split.fit_label` is the fit label's. The run's own out-of-fold check keeps the run's recorded purge, which the search took as the larger of the two labels. A run without `objective.book_target` is priced as before, byte for byte. Under rule `fit_label`, `ablation score` reads the fit label column (`target_spec.column`, or the `--target-spec` entry's) whatever `--target` names, so `--from-run` without `--target` prices the rows `walkforward --from-run` prices; a `--target` that is not the fit label exits 2.
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

**Ladder (#1129).** `rcsr ladder` uses the same width `S`: the target spec's `max(horizon_seconds, stride_seconds)` (`annualization.source` `target_spec`), else `--horizon-seconds` (`horizon_flag`), else the median step of the distinct labelled test times (`inferred_labelled_rows`). `periods_per_year = 31536000 / S` and, when `S` is declared, one equal-weight period per `floor(t / S)` bucket. The report carries `annualization {source, period_seconds}` (`period_seconds` is `S`, or the median step when inferred). A staggered stride (`long:1d,stride=24`, no anchor) gives 365 and one period per day instead of the former 8760 and one period per stamp; hourly labels and anchored strides with stride equal to the horizon give the same rung fields as before. The cross-sectional rungs (top-tercile basket, rank IC) still rank within one timestamp. `rcsr ablation score` applies the same refusals (exit 2) and forms `net_period` (and so the periods of its deflated Sharpe) from the same buckets when a target spec is set; with no spec it keeps one period per timestamp.

**Stride above the horizon (#1053).** A label whose stride is longer than its horizon has one independent return per group every stride. `rcsr enrich` writes `stride_seconds` (`stride * bar_seconds`) into `<out>.targets.json` when the stride is wider than the horizon; the field is absent otherwise, so a spec and its hash are unchanged for `stride <= horizon`. `S = max(horizon_seconds, stride_seconds)` in the walk-forward and in the search objective (`TradingSpec::period_seconds`, which robust selection also uses for its period buckets). Example: `long:1,fee=0,stride=24,anchor=2021-01-04` gives `S` 86400 and `periods_per_year` 365; `rcsr walkforward` agrees with the same rows declared as a one-day label. A spec without `stride_seconds` (an older store, a hand-written spec) is refused with exit 2 when the smallest per-group step of the labelled test rows (`split.min_group_step_seconds`) is at least `2 * horizon_seconds`: the message names the step and asks for `stride_seconds`. `--horizon` without a spec keeps `S = horizon`. The search objective, `rcsr ladder` and `rcsr ablation score` apply the same refusal (exit 2, message `missing stride_seconds`), measured on the labelled rows of their validation and test slices (search) or fold test rows (ladder, ablation score). A declared `stride_seconds` wider than the smallest per-group step of those rows is refused too (exit 2, message `stride_seconds <s> is wider than the smallest per-group step <d> s`): one period would average rows that are independent returns. Gaps from missing bars lengthen the step and never trigger it. `--horizon` without a spec is never refused. `--periods-per-year` is checked against `S` as above. The target spec field is optional and must be at least `horizon_seconds`. `procedure_spec` adds `return_units=log` when the target spec declares log returns and `period_width=<seconds>` when the period width `S` exceeds the horizon, so the candidate id changes with either field and a sealed candidate cannot be read under another target spec value (#1223). A spec with simple units and a stride at most the horizon keeps its text and id.

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

### Regime strata (`--regimes`, `--regime-expr`, #1175)

`--regimes calendar.csv` splits the pooled out-of-fold periods by a regime calendar. `--regime-expr <expr>` (repeatable, with `--regime-name`, `--regime-band`, `--regime-dwell`) adds causal on/off strata: the expression reads only columns of the row, goes through the hysteresis and dwell gate of `regime_gate` under `rcsr-gate-state/1` (the state of a row from the whole readable prefix of the data file, section "One gate-state rule"), and a period is on when more than half of its priced rows are on (under `--period-grid target`, of its rows with a finite target before the gate; section "Period grid"). Non-finite values are off and counted. A column of the label family, the target, its return column or a construction column is refused (exit 5).

The report gains a `regimes` object (JSON and `--markdown`). Per stratum: periods, years, net mean (annualised, at the report cost and at twice it), Sharpe, Sortino, max drawdown, exposure, entries, alpha against the cost-free equal-weight `market` series (no risk-free rate), and a circular block bootstrap interval of the mean with `--bootstrap`. Expression strata add the switch count and switches per year. A calendar stratum is assigned by the bucket start of each period. Periods outside every calendar row are `undated`. A stratum with too few periods keeps its counts and null statistics. Without any regime flag the report is byte-identical to the report of an earlier build. `--period-series` also writes `pooled.period_gross`, `period_exposure`, `period_entries` and `period_bh`, which `rcsr trials --regimes` reads.

```sh
rcsr protocol regimes --data prices.csv --time time --out calendar.csv
rcsr walkforward ... --regimes calendar.csv --regime-expr 'close > sma_200' --regime-name above200
```

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
* **Matched-exposure shifted long/flat**: one circular time shift per draw and window, shared by all symbols. Each symbol's long and flat series is rotated by the same fraction of the window (a joint period shift when every symbol has every period), so each symbol keeps its exposure and the symbols keep their timing relative to each other. The offset is drawn uniformly from `[lo, P - lo]` with `P` the periods of the window and `lo = min(L, P / 4)` (at least 1), so a shift stays at least `lo` periods from the window ends. `lo` equals the block length only when `P >= 4 L`; in a shorter window `lo` is below `L` and a shift can fall inside the holding time. When the product `D` of the per-window counts `P - 2 lo + 1` is at most `--null-draws`, the null is exact: it evaluates each of the `D` joint shifts once (no random draw), so `p = (1 + #{null >= observed}) / (D + 1)` never falls below `1 / (D + 1)`. When `D` exceeds `--null-draws`, each draw picks a shift at random, with replacement. The block permutation null is unchanged either way. Costs are priced on the shifted book, so a run cut by the wrap pays its flips. The report field is `inference.nulls.matched_exposure_random` with `scheme` (`"shared_circular_shift"`), `distinct_shifts` (the product over windows of `P - 2 lo + 1`, saturating), `p_mean`, `p_sharpe`, `null_mean_q95` and `null_sharpe_q95`. The two quantiles are taken over the `D` enumerated shifts when `D` is at most `--null-draws`, else over the random draws. With fewer than 20 distinct shifts `p_mean` and `p_sharpe` are `null` (a p-value below 0.05 cannot be reached); the quantiles stay. A group that covers only part of a window is rotated by its own rows, not by periods, so the joint period shift holds only when every symbol has every period; the effect on the size of the null is not simulated.
* **Always-long**: the buy-and-hold series, reported without a p-value.

Why the shift is shared: a market-wide signal is long in many symbols at the same hours, so the book return carries the common component of the returns. Shuffling each symbol alone removes that component and makes the null far too narrow (a true null was rejected 40 % of the time at 5 % in a simulation with 50 symbols and common share 0.5). Limits: the null assumes the return process is stationary inside a window, and it tests the timing given the book's own exposure path, not the exposure level. The per-window benchmark `matched_exposure_random_long` keeps the per-symbol shuffle and has no p-value.

`p = (1 + #{null >= observed}) / (B + 1)` for the mean and for the per-period Sharpe. A search-budget null is `--null-block` in the search.

### Candidate ledger and the locked test

`--ledger FILE` makes a development run append a `candidate` line (content-addressed id over the procedure, the data SHA-256 and the build commit; `test_read:false`). Each line holds the SHA-256 of the previous line, so an edit breaks the chain. A development window that ends after the locked test start minus the label purge (`--lockbox START,END`, else the range declared with the data, else 2025-07-01 to 2026-07-01; `--lockbox none` opts out) is refused.

Each ledger line and each marker line goes out in one `write` call on an append-mode file, under an exclusive lock on the file, followed by `fsync`; a concurrent appender waits for the lock and extends the head it finds. Every reader refuses a ledger or marker whose last line has no newline, and a line that is not one complete JSON object, with a message that names the file and says "a write was cut" or "not one complete JSON object". `rcsr protocol` commands and `protocol verify` exit 1 (`protocol::LEDGER_FAULT_EXIT`); a locked read, a development run and `--verify-read` exit 3. A ledger event without the fields a read depends on (`id`, `report_sha256`, `read_count` of a `test_read`; `data_sha256`, `report_sha256` of a `baseline_read`; `sealed_sha256` of a seal and `max_reads` of a final seal; `id` of a candidate), or a marker line without `ledger_head` or `report_sha256`, is refused the same way, and the message names the line and the field; a count must be a non-negative integer. An output whose stat fails for any reason but "not found" is refused with exit 3 before anything is recorded. `verify_chain` also fails on a line that holds more than one `"prev":"`. The tool never repairs the file, because the ledger is an audit record: inspect it, restore it from a copy or edit the partial line by hand, then rerun. A marker line with no newline blocks every locked read of the store until a person fixes it. The hash input is unchanged, so ledgers written before this change verify.

`rcsr protocol seal --ledger L --out locked_candidates.json` writes the sealed list and records its hash in the ledger. `--locked-test --sealed locked_candidates.json` then reads the locked test: the candidate must be in the sealed list, the data and build hashes must equal the sealed entry (otherwise the run is void), and the candidate must not have `test_read:true`. The candidate id is `sha256(spec_sha256, data_sha256, build_commit)` with `spec_sha256` the SHA-256 of `procedure_spec`, so a ledger registered by another build (an older engine included, whose `procedure_spec` has no `rows=` line) is not readable for a sealed read by this one. The refusal is exit 3, `candidate <id> is not in the sealed list; the runner refuses it`, followed by the cause: `the list holds no candidate of build <this build> (it was sealed by build <builds>)`, `the list holds no candidate of build <b> on data <sha>`, or, when the list holds candidates of this data and build, `the sealed candidates of this data and build are other procedures` (another expression, row rule or other `procedure_spec` setting). The read appends a `test_read` line with the report hash, the new `read_count` and the locked range it read (`"locked_test":{"start","end","source"}`); a second read is refused. A `candidate` line and the sealed list entry hold the range the candidate was developed against. A final read reports the store's `data_sha256` and the path of its read marker in `protocol.lockbox`, so a reviewer can check both. A copy of a store without its `<data>.locked_reads.jsonl` marker is not detected: the marker lives beside the store and nothing else records the read.

A read whose resolved range differs (another `--lockbox`, or a declaration that changed after development) is refused with exit 3: `candidate <id> was developed against the locked test A..B (<source>); this read resolves C..D`. A candidate of an old ledger without the member is refused unless `--allow-unbound-ledger` is given; the read is then reported as unbound. `--verify-read --expect-report-sha256 H` recomputes the read and compares hashes without recording anything (the same flags work for `rcsr ladder --locked-test`; see Report files of a locked read). The report's `protocol` object holds the lockbox rows, date range and read count; `run.json` carries a `lockbox` object (`--lockbox`, `--ledger`) with the rows in the range, their first and last stamps and the read count.

`rcsr ladder --scheme locked --locked-test --ledger L --sealed S` reads the locked test for the baseline ladder. The sealed list must be recorded in the ledger and the ledger must hold no `baseline_read`; the run appends `{"event":"baseline_read",...}` with the data and report hashes and the locked range (the report hash covers the printed report without the per-rung `wall_seconds`, the `ledger` object and file paths: `lockbox.declared_in[].file` becomes `carrier` and `sr_run.run` is left out, so two identical reads hash equal at any path), and a second call is refused. Every candidate of the sealed list must carry the range the ladder resolves (`--allow-unbound-ledger` accepts old entries without one). Without `--locked-test`, `ladder` and `ablation score` refuse any fold that reaches the declared range from `start - purge` to `end + embargo`. A null store (`rcsr panel permute-labels`) with `--locked-test` exits 2 before the read is authorised and records nothing (#1304).

#### Final seal: one read, one candidate

`rcsr protocol seal --final [--max-reads N] [--report-candidates] --ledger L --out locked_candidates.json` closes the ledger for the final result. It needs a ledger with no seal and no locked read, and it appends `{"event":"seal","final":true,"max_reads":N,...}` (N defaults to 1). Afterwards:

* `walkforward --ledger` without `--locked-test` and both `protocol seal` forms are refused (`the ledger is final`; exit 3 and 1). All candidate selection happens on development data before the final seal.
* At most N `test_read` lines (one `baseline_read` for `ladder`) are accepted. A read of a second sealed candidate after the Nth is refused with exit 3.
* The first locked read writes `<data>.locked_reads.jsonl` beside the store: ledger path, the ledger head the read extends, event and report SHA-256. A read of the same store through another ledger (a fresh file, a copy, an edited chain) is refused: `<data> was already read through ledger <path>`. The marker lives with the store, so a fresh ledger cannot reset the count. Copying the store without its marker file is outside this check.
* A vote book (`--policy vote`, #961) is one candidate: the sealed entry and the candidate id bind the ordered member expressions, `--quorum`, `--band`, `--min-hold`, the threshold rule and the other spec fields. A changed member, member order, quorum, band, min-hold or threshold is another candidate and exits 3 with `is not in the sealed list`; a member subset is refused the same way. The read scores all K members on the locked rows with thresholds from training or validation rows only, appends one `test_read` line and refuses a second read of the book. `--verify-read --expect-report-sha256 H` recomputes it. `K = 1, Q = 1` reports the same pooled metrics as the single-candidate read. `--device-pnl` and `--batch-pnl` stay refused.
* A read of a multi-candidate procedure reports the validation-selected candidate only. No other candidate is evaluated on the test block: `--bootstrap` has no candidate columns, `multiple_testing` has `candidates:1`, one p-value and `pbo_error` `final read reports the selected candidate`, and `--batch-pnl` is refused with exit 2. `--report-candidates` at seal time keeps the full report and records the choice in the seal line.
* `rcsr protocol verify --ledger L --data D --expect-reads N [--sealed S]` exits 1 unless the ledger has N locked reads, exactly one seal line that is final, no more reads than the seal allows and a marker that matches the ledger line for line. With `--sealed` it also checks that the file is the final list and that every read id is in it. Before the read, `--expect-reads 1` exits 1 and `--expect-reads 0` exits 0.

A crash after the marker line and before the ledger line leaves a marker the ledger does not record. The next call finds the same ledger head and reads once; `verify` flags the state until then. Deleting the marker makes `verify` report a ledger read without a marker line.

#### Frozen inputs and the verdict (#1238)

`rcsr protocol freeze --ledger L --role thresholds|development|subwindows|years --file PATH` records the SHA-256 and byte count of a file the sealed-read verdict reads, before the seal: one file per role, refused with exit 1 after any seal or locked read. `rcsr protocol verdict` then reads the ledger, the final sealed list, the frozen files, the locked reads (bound to the picks of the frozen table by `selection.run_sha256`) and the lock calendar of `protocol regimes --after-read`, and writes the gates S1-S5, R1, R2, P1 and D1-D3 and the verdict of every read (`rcsr-verdict/1`). See [`docs/overfitting.md`](overfitting.md#315-sealed-read-verdict-rcsr-protocol-verdict-1238).

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

## Two books and a gate (`rcsr switch`, #1185)

A formula that trades one book in one market regime and another book in the other has the form `if(greater(g, c), A, B)`. `--sample-rows` exits 2 here: gate state runs over every row of the data file (#1247). `rcsr switch` computes two exact quantities for two fixed books and a gate on the training rows: the gate-cut profile of the composite at every cut `c`, and the k-switch hindsight bound of the pair.

```sh
rcsr switch --data dev.rcsb --target lbl_h1_y --target-spec dev.targets.json --vars "$VIEW" \
  --time ts --group-by symbol --split temporal --validation-frac 0.3 \
  --objective sharpe --threshold 1e-30 --cost-bps 10 --pnl-blocks 4 \
  --a "<expression>" --b flat --gate btc_r672 --gate mtf_1d_ma_ratio \
  --cuts 16 --switches 2,8,32 --json --out switch.json
```

Rows and book:

* The command takes the flags of a search run with `--fitness pnl` (it adds `--fitness pnl` and `--cpu` when absent), builds that run's `trading::FitnessBook` on the canonical training rows (cost, threshold, `--pnl-blocks`, `--pnl-block-lambda`, `--pnl-stat`, `--pnl-cvar-q`, `--pnl-excess`, interval segments, locked ranges and every refusal of the run) and stops before the search. The validation and test rows are never read: replacing their gate values and returns leaves the report byte-identical (`cuts_read_only_the_training_rows`, `tests/switch.rs`).
* `--a` and `--b` are expressions of columns in original units, or `flat` (the zero book). A row is long in a branch when the value is finite and above `--threshold`; a non-finite value is flat and is counted in `nonfinite_rows`. The values are the raw column values of the training rows, as `walkforward --expr` reads them, so `--lags`, `--lag-set`, `--fill`, `--outlier-iqr` and `--smooth` exit 2. A and B that hold the same book on the training rows exit 2.
* `--gate` is an expression of columns, repeatable. Every column that `--a`, `--b` or `--gate` reads passes the `--pnl-mask` column checks before any row is read: a label-family column, the target, its return column or a construction column exits 5 (D13), a column missing from the data exits 3, an aggregate or meta operator exits 2. The columns join the store projection and `--embargo auto`.
* A band or a minimum hold (the book is not plain), `--pnl-mask` (the gate is the mask) and `--gpu` (no device arm) exit 2. `--pnl-blocks` takes 1 to 64 blocks.
* Memory (#1248): `switch::gate_sweep_bytes(rows, blocks)` estimates the sweep: `2 * blocks * (rows + 2) * 4` bytes for the two block arrays plus 125 bytes per row for the linear arrays, rounded up. The estimate is `book.sweep_bytes` in the JSON report and a `gate sweep memory:` line in the text report. After the distinct-value check of a gate, an estimate above `--max-memory` (else the detected budget) exits 2 and names the estimate, the budget, the block count and the two ways out (fewer `--pnl-blocks`, a larger `--max-memory`); `--force` skips the check. The table guard still runs first and refuses an input that does not fit (exit 3).
* `--pnl-stat` and `--pnl-cvar-q` (#1178) set the statistic of every block fitness of the report (`fitness` of a cut and of `a` and `b`); `book.stat` and `book.cvar_q` name it and the text report prints `fitness statistic:`, both omitted under `mean-sd`. Net, gross, the counts and the bound do not read the statistic (`switch_reports_the_block_statistic`, `gate_sweep_fitness_follows_the_block_statistic`).
* `--pnl-excess xs-mean` (#1179, #1232) builds the book on the excess per row `(r - m_t) - funding`, `m_t` the equal-weight mean return of the training rows at the row's timestamp (a timestamp with one row gives that row an excess of `-funding`). The sweep code is the same: the per-row term of the cut-0 sum and of every row that moves from A to B is that excess. So `gross(t)` is the sum of `r - m_t` over the long rows (the absolute gross is `gross(t)` plus the sum of `m_t` over the long rows), `net(t) = gross(t) - side * (flips + exits)` with the same cost term, and `fitness`, `V_k`, every `gain` and `share` are on the excess. `m_t` does not depend on positions, so flips, exits, long rows, entry timestamps, gate flips, switches, seam pairs and the seam identity are those of the absolute run, and `V_k` still bounds every time-level gate cut of the same book. The report gets `excess: xs_mean` (top level and a text line); `walkforward --expr` prices the absolute book and does not reproduce these nets (`switch_json_reports_excess_label`, `gate_sweep_equals_the_composite_book_xs_mean`, `the_seam_identity_holds_xs_mean`, `the_switch_bound_is_the_path_optimum_xs_mean`).

Gate-cut profile (`switch::gate_sweep`):

* The rows with a non-NaN gate value are sorted by value (`-0.0` equals `+0.0`; `-inf` and `+inf` order below and above every finite value, as `greater` compares them). Cut `t` in `0..=m` holds A on the `m - t` rows of highest gate value and B on every other row; a NaN gate value holds B, as `if(greater(NaN, c), A, B)` does. A cut that splits equal gate values is not valid.
* Along the cut axis the flip, entry and seam terms of a row pair of one book run are constant on three rank intervals, split at the lower and the higher rank of the pair: below both ranks both rows hold A, above both both hold B, and between them the row of higher rank holds A and the other B. Three range updates per pair and one prefix pass give, for every valid cut, the gross, net, block fitness ([`optimisation.md`](optimisation.md), section 3.5), flips, exits, long rows, distinct entry timestamps, gate flips (row pairs of one run whose state differs), time-level switches and seam pairs. Cost `O(n log n)` per gate; memory is a few vectors of row length plus two `i32` arrays of `blocks x (rows + 2)` entries. `gate_sweep_equals_the_composite_book` checks every cut of random panels (7 groups, 300 stamps, skipped stamps, interval segments, 1, 2 and 4 blocks, 5 and 10 bp, gates with ties, NaN, infinities and both zeros) against `FitnessBook::block_totals` of the composite position vector: counts exactly, net and gross within `1e-12`.
* Cut value: the midpoint of the two neighbouring gate values, or the lower one when the midpoint is not strictly below the upper one, so `greater(g, c)` reproduces the cut on the training rows.
* Best cut: among the valid inner cuts (`0 < t < m`), the highest net; ties go to the lower cut. It is reported over all cuts and over the grid of `--cuts Q` uniform quantiles (default 16; cut `floor(j m / Q)`, `j = 1..Q-1`, moved up to the next valid cut). The plateau of the best cut is the widest run of valid inner cuts around it whose net stays within a tenth of its gain over the better branch, in rows (`plateau.rows`) and in gate units (`plateau.width`).
* At the best cut: `seam` (below), and `expression`, the composite `if(greater(<gate>, c), <A>, <B>)` with every constant exact; a `flat` branch prints as the constant `threshold - max(1, |threshold|)`. `walkforward --expr` on a window whose test block is the training rows reproduces the cut's net and costs (`the_printed_expression_reproduces_the_book`, `tests/switch.rs`; with `--pnl-blocks 1`, since block seams restart the book).

Seam identity. With `masked_a` the book that holds A on the rows in state A and is flat elsewhere (every run of those rows starts flat and pays its exit) and `masked_b` likewise for B, `composite = masked_a + masked_b + 2 side J`, with `J` the row pairs of one book run whose state changes with a long row on both sides: at such a flip the masked books pay one exit and one entry, the composite nothing. The identity holds for any row state, so it holds for a gate that varies inside a timestamp too (`the_seam_identity_holds` checks it on 200 random panels against the masked `FitnessBook` of `--pnl-mask`). For a gate with `K` flips on `N` groups, `0 <= composite - masked_a - masked_b <= 2 side N K`: for a slow gate the best composite is, up to that term, the pair of masked optima, so the branches can be searched separately.

k-switch bound (`switch::switch_bound`):

* A time-level path holds A or B at each distinct training timestamp, for every group at once. With `W_u(s)` the gross, entries, exits and same-stamp flips of the rows of timestamp `u` in state `s` and `F_u(p, s)` the flips between a row at `u - 1` in state `p` and its successor at `u` in state `s`, `V_k(u, s) = W_u(s) + max(V_k(u-1, s) - side F_u(s, s), V_(k-1)(u-1, s') - side F_u(s', s))`. `V_k` is the best net of any path with at most `k` switches, in `O(T k)`; `V_0 = max(net_a, net_b)`, `V_k` does not decrease in `k`, and the free bound equals `V_(T-1)` (`the_switch_bound_is_the_path_optimum` checks every `k` against all `2^T` paths for `T <= 12`).
* A row pair of one book run that skips a timestamp (a group with a missing stamp) is charged with the cheaper of the two earlier states, which keeps `V_k` an upper bound; the report counts these pairs (`gap_pairs`) and `exact` is true only without them (`a_skipped_timestamp_keeps_the_bound_above_every_path`).
* Every time-level gate with at most `k` flips is such a path, so its composite net is at most `V_k` at every cut (`the_bound_dominates_every_gate_cut`, with and without skipped stamps). A gate whose value varies inside a timestamp is not a time-level path: its `time_level` and `bound_applies` are false and its `switches` are null.
* `--switches` lists the budgets (default `2,8,32`); a budget above `T - 1` is clamped and reported (`clamped: true`). The free bound is always reported.
* The bound is a necessary condition and a pruning rule, not evidence of an edge. On synthetic panels the share of the bound's gain that the best gate reaches does not separate a world with a true gate from a world with none, and a gate chosen on training rows is one more fitted parameter: it must pass an out-of-sample or description-length test before it is believed (#1184).

Report (`rcsr-switch/1`, `--json`, or text without it; `--out` writes the JSON too): `rows` (`train`), `excess` (`xs_mean`, only under `--pnl-excess xs-mean`), `target`, `train_rows`, `groups`, `timestamps`, `time_first`, `time_last`, `book` (`per_side_bps`, `threshold`, `blocks`, `lambda`, `min_trades`, `segments`, and `stat` with `cvar_q` under cvar when the statistic is not `mean-sd`), `a` and `b` (`expr`, `flat`, `net`, `fitness`, `flips`, `exits`, `long_rows`, `nonfinite_rows`), `bound` (`label: pruning_bound`, `is_evidence: false`, `applies_to`, `exact`, `gap_pairs`, `timestamps`, `reference` = `max(net_a, net_b)`, `free`, `free_gain`, `k[]` with `requested`, `k`, `clamped`, `value`, `gain` = `V_k - reference`) and `gates[]` (`expr`, `rows`, `nan_rows`, `ordered_rows`, `cuts` (valid inner cuts), `time_level`, `bound_applies`, `best` with the cut fields, `gain`, `share[]` (`gain / bound_gain` per budget and for the free bound, null when the bound's gain is not positive), `plateau`, `seam` (`net_a`, `net_b`, `masked_a`, `masked_b`, `pairs`, `seam`, `sum`, `composite`) and `expression`, and `grid` with `q`, `best` and `profile`, the grid cuts). The cut fields are `index`, `value`, `rows_a`, `rows_b`, `net`, `gross`, `fitness`, `flips`, `exits`, `long_rows`, `entry_times`, `gate_flips`, `switches` and `seam_pairs`. Every number is a training-row value, measured where the cut was chosen; none is an out-of-sample result. The report draws no random number and is byte-identical across runs.

Edge cases: a gate with fewer than 2 distinct non-NaN values on the training rows exits 2; a cut value that is not finite (a gate whose lowest value above `-inf` is `-f64::MAX`) prints the expression as null.

`rcsr compose` (#1181) builds such composites from whole pools and a pre-registered gate menu, prunes pairs with the k-switch bound, chooses one on the validation rows and writes a run.json that `--from-run` prices; see [`compose.md`](compose.md).

## CPCV

Combinatorial purged cross-validation (López de Prado 2018, *Advances in Financial Machine Learning*, ch. 12) cuts the development span into N calendar groups and tests every choice of k of them. There are C(N, k) splits, every group is tested in C(N-1, k-1) of them, and the test results join into phi = k C(N, k) / N = C(N-1, k-1) complete backtest paths, each an out-of-sample book over the whole span. Walk-forward never tests the first groups; CPCV tests every group, and the path distribution replaces one lucky or unlucky path.

| scheme | splits | tests per group | paths |
|---|---:|---:|---:|
| CPCV(6, 1) | 6 | 1 | 1 |
| CPCV(6, 2) | 15 | 5 | 5 |
| CPCV(4, 2) | 6 | 3 | 3 |

```sh
rcsr protocol cpcv --start 2017-12-15 --end 2023-11-01 --groups 6 --test-groups 2 \
  --purge-seconds 7200 --embargo-seconds auto --vars "$VIEW" --bar-seconds 3600 \
  --validation-share 0.2 --data panel.rcsb --out cpcv62
# one search per split
rcsr --data panel.rcsb ... --split-table cpcv62/splits.csv --split-id s07 --json > s07.run.json
# one walk-forward per split, every test group its own book
rcsr walkforward --data panel.rcsb --time ts --group-by symbol --from-run s07.run.json \
  --windows cpcv62/windows_s07.csv --pool-reset-flat --period-grid target --emit-series --json --out wf_s07.json
rcsr trials --cpcv-assemble wf_s*.json --splits cpcv62/splits.csv --out-dir paths --summary paths/summary.json
rcsr trials --wf paths/path_*.json other-config/paths/path_*.json --baseline ladder.json
rcsr trials --wf paths/path_*.json --table --rule rule.json --regimes calendar.csv --out table.json
```

### Splits (`rcsr protocol cpcv`)

* Groups: `--start` and `--end` are UTC midnights. The locked ranges of `--data` (or `--lockbox`, `--lockbox-embargo`, resolved as for a search) are holes `[start - purge, end + max(embargo, purge))` widened to midnights. Of the D usable days, group g ends at usable day `floor(g D / N)`, so group lengths differ by at most one day and a group may hold a hole. The pilot span 2017-12-15..2023-11-01 gives the boundaries 2018-12-07, 2019-11-30, 2020-11-22, 2021-11-15 and 2022-11-08.
* Splits are numbered `s01`, `s02`, ... in lexicographic order of their test-group tuples: CPCV(4, 2) has s01 = (G1, G2), s02 = (G1, G3), s03 = (G1, G4), s04 = (G2, G3), s05 = (G2, G4), s06 = (G3, G4).
* Validation: the last `--validation-share` of every training group (whole days, floored) is a validation interval; `--select-blocks intervals` gives the robust rule one block per validation interval.
* Purge and embargo: the table holds whole test groups and training and validation intervals already trimmed by the predicate of `--split-table` (`IntervalSplit::effective`, #1173): a row is left out when its label window `[t, t + purge)` reaches a test group or when it lies in `[end, end + embargo)` after one, and training rows are purged on both sides of every validation interval. The search applies the same trim again and keeps every row (the table is a fixed point). `--embargo-seconds auto` (the default) is the larger of the purge and the longest lookback of the `--vars` columns at a bar of `--bar-seconds` (rule `rcsr-lookback/1`); an explicit value below it exits 2. A search that resolves a larger embargo for its own `--vars` trims more; a smaller one trims nothing more.
* Refusals: k >= N, more than 10000 splits, a non-midnight `--start` or `--end`, a training group or validation interval with no time left after the holes, purge and embargo, and a group with no usable day exit 2 and name the split and group. An interval that still reaches a locked range exits 3, as `--split-table` does.
* Files: `splits.csv` (the `--split-table`), `groups.csv` (`group,start,end,days`, one row per piece), `windows_<id>.csv` (one walk-forward window per test group, labelled `G1`..`GN`, with a one-second training range that ends `--purge-seconds` before the test start) and `cpcv.json` (`rcsr-cpcv/1`: counts, purge, embargo and its source, holes, `scheme_sha256`, `splits_sha256`, the test groups of every split and the path assignment). The output depends only on the arguments and the lock: two runs write identical bytes.

### Paths (`rcsr trials --cpcv-assemble`)

* Input: the walk-forward reports of one configuration, one per split. A report belongs to the split whose test intervals equal its windows' test intervals; a report of no split, two reports of one split and a split without a report exit 2.
* Assignment: path j of group g takes the test series of the j-th split, in lexicographic order, that tests g. Every path covers every group once and every (split, test group) result is on exactly one path.
* Seam rule: every group of a path is one book segment that starts flat and pays its final exit, whichever split it comes from. Two groups of one path usually come from two models, and a position that one model holds was never decided by the next, so no position carries across a group boundary. This is the flat reset of #1189 (`--pool-reset-flat`) at every group seam; two touching groups of one split follow the same rule, so there is one rule and no definition of touching. The assembler refuses a report whose `pooled_seam.reset_flat` is not `true` (exit 2); with it, the pooled series of a report is the concatenation of its per-window books, and a group series is cut from it by period time.
* Guards: the split reports must agree on `purge`, `cost_model.per_side_bps`, `cost`, `annualization.periods_per_year`, the threshold rule (`threshold.quantile`, `threshold.fixed` and the fit fields) and the policy (`policy.name`, `policy.band_width`, `policy.min_hold`), the period grid rule (`pooled.grid.rule`, absent for the score grid), the missing-score rule (`book.missing_score`, #1203), the fields `rcsr trials --wf` guards, and every split that tests a group must give it the same period grid. A mismatch exits 2 and names the field (for a grid, the group). `window_definition` differs per split and is not compared.
* Period grid: the C(N-1, k-1) splits that test a group run as many searches and usually hold different formulas. Under the score grid a formula with other NaN coverage or another gate gives the group another grid and the assembly exits 2. Price the split reports with `--period-grid target` (section "Period grid"): the grid of a group then comes from its rows with a finite target, the same in every split, and a split whose formula has no score or whose gate is off books zeros there. Reports on one group that still differ hold different data or windows.
* Path files: `path_<j>.json` in the walk-forward schema: `window_definition` is `scheme_sha256` (the hash of N, k and the group pieces, not of purge or embargo, so configurations with other lookback embargoes on the same groups compare), the guarded objects of the first report and its `book.missing_score`, `pooled.period_time` and `pooled.period_net` (the chosen group series in group order, bit for bit), `pooled.sharpe`, `pooled.max_drawdown`, `pooled.net`, `pooled.entries`, `pooled.active_rows`, and `cpcv.segments` (group, split, report path and SHA-256, expression, periods, net). Target-grid split reports add `pooled.grid` (`rule`, `periods`, `empty_periods` summed over the groups of the path) and `pooled.period_rows`. `rcsr trials --wf` reads the paths of all configurations like any reports: `N_eff`, DSR, SPA and PBO across them.
* Series (#1236): each of `pooled.period_gross`, `period_exposure`, `period_entries`, `period_bh` (`--period-series`) and `period_cost`, `period_entry_stamp`, `period_bh_net` (`--emit-series`) that the split reports carry goes into every path as the concatenation of the chosen group series in group order, cut by the index lists of `period_net`, so a target-grid empty period keeps the values walk-forward wrote for it. `pooled.entry_times` of a path is the sum of the chosen groups' `windows[].entry_times`, and every `cpcv.segments` entry gets the `entry_times` of its group. The fields follow `period_rows` in the order of the walk-forward pooled block, so `rcsr trials --table` computes `net20`, `exposure`, `entry_times`, `alpha_bh` and the strata metrics on path rows. A field that some split reports carry and others lack exits 2 naming the field and the first report without it; a series of another length than `period_net` or with a non-finite value, `pooled.entry_times` without `windows[].entry_times` (a report of an earlier build) and window counts that do not sum to `pooled.entry_times` exit 2 naming the field and the report. Without the series the path files and the summary are byte-identical to those of an earlier build, and the summary never holds the series. The benchmarks stay cash at 0 and buy-and-hold of the same rows; no interest input exists.
* Composite marker (#1237): when every split report carries `selection.composite` (the split runs come from `rcsr compose`), every path gets `"composite":{"source":"rcsr compose","types":[...]}` after `pooled_seam`, one type per group in path order, and every `cpcv.segments` entry its group's `composite_type`. Composed and plain split reports in one assembly exit 2 naming the first plain report. Without markers the path files are byte-identical to those of an earlier build. The candidate table reads the path marker per row (`docs/overfitting.md` 3.14).
* Null-store marker (#1272): when every split report carries `null_store` (they are priced on a store of `rcsr panel permute-labels`), every path gets the same `null_store` object after `pooled_seam`. Split reports with different markers, or with and without one, exit 2 naming the first report and the first that differs. Without markers the path files are unchanged. The candidate table reads a marked path only through `--null-wf` (`docs/overfitting.md` 3.12).
* Summary (`rcsr-cpcv-summary/1`, `--summary` or stdout): per path the net at the report cost (sum of `period_net`), annualised mean net, Sharpe and max drawdown; the path Sharpe distribution (values, mean, sd, min, median, max); path net mean, median, minimum and the share of positive paths; per group the net on each path. Target-grid reports add `grid.rule`, `per_path[].empty_periods` and `per_group[].empty_periods` (one value per path). PBO needs several configurations, so the summary of one configuration names `rcsr trials --wf` for it.

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

A report priced on a null store (`--data` whose store header has a `null_store` member, #1304, or whose `<data>.manifest.json` has schema `rcsr-null-store/1`, both written by `rcsr panel permute-labels`, #1272) ends with `null_store`, the `null` object of the store (label, block, seed, order, source SHA-256, permuted columns). The header member wins over the manifest; it travels with the store bytes. It is outside `report_sha256`. `--ledger` on a null store exits 2 before any ledger read or write, so it never registers a candidate, and `--locked-test` (which needs `--ledger`) never spends a lockbox read on it. `rcsr trials` reads such a report only through `--table --null-wf` (`docs/panel.md`, null store).

### Period grid (`--period-grid`, #1199)

`--period-grid score|target` says which periods the pooled series has. A period is one bucket `floor(t / S)` of the run's period width (one timestamp without a declared width), stamped with the first row of the bucket.

* `score` (the default): a period exists where some test row has a finite score and a finite target. The grid depends on the formula: a formula whose columns are NaN for every symbol on a stretch (the warm-up after a data gap, a long lookback at the start of the span) has no period there, and under `--pnl-mask` a market-wide gate removes the periods where it is off. Under `--missing-score hold` a held row is priced, so it makes a period. The report has no `pooled.grid` field.
* `target`: a period exists where some test row of the window has a finite target before the gate. The grid depends only on the data and the windows, so every formula, gate and configuration on one store gets the same `pooled.period_time`. A period where the book prices no row (no finite score, or the gate off) books net 0. The window's period width is read from the same rows. The report adds `pooled.grid` (`rule`, `periods`, `empty_periods`: the periods with no priced row) and `pooled.period_rows` (priced rows per period, 0 for an empty period).

The grid never changes the book. Positions, costs, exits, `pooled.total_cost`, `pooled.total_abs_delta`, `pooled.total_return`, `pooled.max_drawdown`, trades, entries and the sum of `pooled.period_net` are bit-identical under both rules; the target series is the score series with zeros inserted. An empty period books no return, no cost and no turnover. Across rows with no finite score inside a window the book follows `--missing-score` (#1203): under `flat` the run pays its exit in its last priced period and the stretch is empty; under `hold` the rows are priced at the held position, so the stretch has no empty period and `empty_periods` falls. Across rows absent from the data the position stays as the book left it. Across a gate-off stretch every on-run is its own book: it pays its exit and the next on-run starts flat (#1180). Across a seam between windows the seam rule of the run applies (`--pool-reset-flat`, and a flat start after a gap between test intervals, #1189). A `score` and a `target` report of one formula therefore hold the same positions and differ only by the zero periods.

Everything computed from the pooled series uses the grid: the `pooled` metrics, `economics` (the benchmark series of the same rows get the same zeros), the bootstrap, PSR and DSR, SPA and the nulls (an empty period has no row and no cost). The candidate columns of the in-report PBO and SPA (and the per-candidate p-values) sit on the window's target grid under both rules (#1203): every column, the selected one included, is the candidate's own book padded with 0 where it prices no row, and the SPA baseline of the columns sits on the same grid, so CSCV compares every candidate on the same periods whichever candidate a window selected. With full coverage the two grids are one grid and the columns are the score columns. The regime strata (`--regimes`, `--regime-expr`, `--period-series`) use it too: `pooled.period_gross`, `period_exposure`, `period_entries` and `period_bh` have the grid's length, with 0 on an empty period, and every stratum sums to the pooled series. A calendar stratum is assigned by the bucket start of the grid period. The state of a regime expression in a period is the majority state of the rows of its bucket that define the grid (finite target, before the gate), not of the priced rows, so an empty period gets the state of the expression at its timestamp and every formula on one store gets the same strata periods. The per-window metrics stay on each window's priced periods. Consequences, for a book that prices rows on a share `f` of the grid periods: the mean per period falls by `f` and the standard deviation by about `sqrt(f)`, so the annualised Sharpe is about `sqrt(f)` times the Sharpe on its own periods; CAGR and the annual `turnover` count the empty periods as elapsed time; total return and max drawdown do not change. `docs/overfitting.md` section 3.14 gives the effect on PSR, DSR and PBO. Edge cases: a formula with no score anywhere is reported with `empty_periods` equal to `periods`, and a gate that is off for a whole window gives that window a series of zeros.

The printed and written JSON ends with a `ledger` object (#284): the phase ledger of the run with `io_s` (store row-group or CSV reads), `pnl_s` (scoring, windows, controls, `--batch-pnl`), `host_other_s`, `wall_s`, `unaccounted_s`, `closure` and `device_busy_s`, the union of the `--batch-pnl` device tile intervals and of the `--device-pnl` CUDA reduction calls (`host_envelope` basis; the canonical CPU backend adds none). The ledger and the `batch_timings` object (`--batch-pnl`: per seam `wall_s`, `device_s`, `score_s`, `layout_s`, `reference_s`, `candidates_per_s`, `row_evals_per_s`, `device_fraction`) hold wall-clock seconds, so `report_sha256`, `--expect-report-sha256` and `--verify-read` hash the report without them. They also leave out the `--front` and `--from-run` path (`selection.run`; `run_sha256` stays), so `--front run.json` and `--front ./run.json` verify the same read. A hash recorded by an earlier build for a `--front` read does not match, and neither does one for a `--from-run` report written before #1238, which carried no `selection.run_sha256`. The printed JSON keeps every path. See `docs/benchmarks.md`.

## Store handling and memory

A `.rcsb` path is opened with `store::FileStore` and scanned one row group at a
time; the resident state is one group plus the selected test rows and one score
per training row per window. A CSV path is read as a resident `data::Table` and
accumulated with the same rule. The JSON `memory_bound` field states the bound
for the run.

The scan reads only the columns the run uses: the variables that a candidate
expression names, the target, an `f64` time column and a numeric group column. The exact `i64` time and the
dictionary codes of a text time or group column are read separately. A group in which no row of the current
pass is wanted is skipped before its value read when its time is an exact `i64` or a dictionary column; an `f64`
time column is part of the value read, so such a group is always read. A `--locked-test` final read runs the
group loop twice, training rows first and test rows second. The printed JSON carries a `scan` object next to
`ledger`: `store_columns`, `named_variables`, `projected_columns`, `groups_read`, `groups_skipped` (both summed
over the passes of a final read) and `value_bytes`. The table path has no `scan` object. `scan` is outside
the reproduction hash, like `ledger`.

## Performance

Measured with the release binary on an Intel i7-3770K (8 threads), 31 GiB,
Linux, using the 3,943,856 x 56 hourly store, streamed one group at a time:
`rcsr walkforward --group-by symbol --horizon 172800` takes 4.06 s with a peak
RSS of 107.7 MiB. The corpus, window, command and measurement protocol are in
[`eval-benchmark.md`](eval-benchmark.md).
