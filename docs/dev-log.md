# The development log (`--dev-log`) and `rcsr dev-report`

`rcsr` writes a **development log** while it searches. It is on by default: a
concise, bounded NDJSON record — the in-engine replacement for the old external
long-run monitor — that says how a run is going and, afterwards, why it behaved
as it did.

```sh
# the default: rcsr.dev.ndjson in the working directory, summary level
rcsr --data examples/line.csv --target y --vars x --seed 3 --generations 200

# summarise it
rcsr dev-report rcsr.dev.ndjson
```

The final `--json` says *what* a run found. The development log says *why*: which
phase consumed the wall clock, whether the run was fit-bound or
evaluation-bound, how many candidates de-duplication threw away, how often the
GPU fell back to the CPU and for what cause, how the front and the convergence
readings moved, and where a 10-hour run is stuck. `rcsr dev-report` turns the
file into that summary with a ranked list of bottlenecks. Both are Rust; there is
no separate Python evaluator.

| flag | default | meaning |
|---|---|---|
| `--dev-log-level summary\|full` | `summary` | how much detail each generation event carries |
| `--dev-log <path>` | `rcsr.dev.ndjson` next to `--json-out` (else `--save`), else the working directory | write the log here instead |
| `--no-dev-log` | off | disable the log entirely |
| `--dev-log-every <N>` | level-dependent (below) | write a generation event every Nth generation |
| `--dev-log-max-bytes <size>` | `64MiB` | cap on generation and checkpoint events |

The log is independent of `--json`, `--ndjson`, `--progress`, `--out`,
`--json-out` and `--run-manifest`: it never shares their streams. It is refused
with `--eval-expr` (no search to log) and, when a log flag is given explicitly,
with `--report benchmark` (one file describes one search). `--report benchmark`
then runs its repeats without a log, and `--null-runs` runs its permutation
controls with the hook removed, so every event in the file belongs to the logged
search.

## The two levels

**`summary` (default)** is the monitoring record: `run_start`, a `generation`
event every few generations, `checkpoint` events and `run_end`. A generation
event carries the generation number, elapsed seconds, wall, the best and median
population error, the Pareto size, the five phase windows (plus the two summed
CPU times), the fit/GPU/fallback counts, the GPU kernel/wall/duty, the fallback
pool cost and `percent_converged`. That is enough to watch error, front growth,
where the time goes and whether the device is being used.

**`full`** writes every generation with the complete record: all error
quantiles, the archive hypervolume and hash, archive offers/admissions, the
variation counters (attempts, duplicates, too-complex, offspring, fitted,
refits, dedup rate), the fit fallback breakdown by cause, QD coverage/entropy,
the per-island best errors and the island RNG key.

The default cadence keeps the file small however long the run is:

| level | events written |
|---|---|
| `full` | every generation |
| `summary` | every `clamp(generations / 10, 10, 25)` generations — and every generation when the budget is 10 or fewer |

So a 200-generation run writes 10 summary events (6 KiB in the reference run
below), a 10-hour run at 5 000 generations writes 200, and `--dev-log-every`
overrides the cadence at either level.

## Where the file goes

1. `--dev-log <path>` — exactly there.
2. otherwise next to `--json-out` (`<dir>/rcsr.dev.ndjson`),
3. otherwise next to `--save` (`<dir>/rcsr.dev.ndjson`),
4. otherwise the working directory (`./rcsr.dev.ndjson`).

A launcher that runs several windows must name one log per run — the tracked
launchers do (`scripts/run-long.sh`, `rcsr-launch` write
`<label>.dev.ndjson`) and `tests/launchers.rs` enforces it. A failure to open an
explicit `--dev-log` path is an input error (exit 3); a failure to open the
*default* path is a warning and the search runs on without a log, so a read-only
working directory never stops a run. `--quiet` suppresses that warning, like
every other note.

The default file name is in `.gitignore`: a run in the repository (including
`cargo test`) must not leave an untracked artifact.

## What is guaranteed

**Trajectory-neutral.** The log is a *reading*. It never draws from an RNG,
never reorders candidates, never changes variation, fitting, selection or the
archive. The same seed with the log off, at `summary` and at `full` produces a
byte-identical front (`--out`), an identical result document (the only differing
fields are `elapsed_seconds` and the `timing`/`knn` wall clocks) and the same
canonical configuration hash in the run manifest. Pinned by
`tests/dev_log.rs::logging_on_or_off_leaves_the_archive_results_and_config_hash_identical`,
which compares all three arms.

**Atomic.** Each event is one `write` of one complete line, newline included, to
a file opened with `O_APPEND`. A killed process therefore leaves a **valid
prefix**: every complete line parses, and at most the final line is torn.
`rcsr dev-report` skips that torn line, says how many lines it skipped, and still
reports the prefix.

**Bounded.** Generation and checkpoint events stop at `--dev-log-max-bytes`
(default 64 MiB); each dropped event is counted and `run_end` reports the count.
`run_start` and `run_end` are terminal events and are always written, so the
file is bounded by `cap + two events` and never grows without limit. The file is
truncated when the run starts: one log describes one run.

**No raw rows.** The log records counts, timings, hashes and error statistics.
It never writes an input row, a target value, a prediction or a formula. The
data appears only as its path, its SHA-256 and the training/test fingerprints
the checkpoint uses (`tests/dev_log.rs::the_log_contains_no_raw_data_rows`).

**Stable schema.** Every line carries `"schema":"rcsr-devlog/1"` and `"v":1`,
and every generation event says which `"level"` produced it. Fields may be added
within this version; removing or redefining one needs a new version. `rcsr
schema` publishes the schema id, the levels, the default path, the cadence rule
and the flags under `dev_log`.

## Cost

Measured on a **fit-heavy** cell (3 000 rows, two inputs, `--fit-iters 12
--pop 120 --generations 80 --metric squared --cpu`, ≈0.55 s per generation,
≈44 s per run). Same seed and data; four interleaved rounds per arm; a paired
`--no-dev-log` baseline in every round:

| quantity | no log | `summary` (default) | `full` |
|---|---:|---:|---:|
| wall clock (median) | 43.80 s | 43.70 s (−0.24 %) | 43.75 s (−0.13 %) |
| paired wall delta | – | −0.47 % (pairs +1.2, −0.8, −0.9, −0.2) | +0.23 % (pairs +0.5, −1.4, −0.1, +0.6) |
| process CPU time | 43.12 s | 43.08 s (−0.09 %) | 43.08 s (−0.09 %) |
| peak RSS (`/usr/bin/time %M`) | 7.7 MiB | +22 KiB (+0.28 %) | −58 KiB (−0.74 %) |
| archive/results | – | identical | identical |
| file size, 80 generations | – | 6.1 KiB (10 events) | 81.7 KiB (82 events) |

Single runs on this host swing by ±20 %, so both differences are **within
run-to-run noise** and inside their targets (**summary < 1 %**, **full < 2 %**).

Per-event cost, measured directly
(`src/devlog.rs::one_generation_event_costs_well_under_a_millisecond`):

```
summary 7.5 µs/event (560 bytes)   full 15.6 µs/event (1 014 bytes)
```

At 0.55 s per generation that is 0.001–0.003 % of a generation.

The **fixed start cost** is the one-time provenance in `run_start` — SHA-256 of
the binary and of the data file, plus the git commit probe. Measured on a
0.020 s run (median of 7): no log 0.020 s, `summary` 0.070 s, `full` 0.070 s, so
the log adds **≈50 ms once per run**, at either level and independent of its
length. The cost model is therefore

```
overhead ≈ 50 ms (once) + 7–16 µs per written generation event
```

— 0.11 % + 0.003 % of the reference cell. A run shorter than a second pays the
fixed part as a larger share; `--no-dev-log` avoids it entirely. The same
provenance is what `--run-manifest` (#140) records, at ≈80 ms.

Peak RSS impact is the open handle, one `Mutex` and no retained history: every
event is written and forgotten, so the log adds no growth over the run. The
measured deltas above are inside the allocator noise of a 7.7 MiB process.

## Events

Every event is one line, `{"schema":"rcsr-devlog/1","v":1,"event":…}`. Times are
seconds, sizes are bytes, hashes are `0x`-prefixed 64-bit hex. `generation`,
`fit.attempts/gpu/fallback`, `gpu.duty`, `checkpoint` and `run_end` are written at
both levels; the fields marked **full** only exist in a `full` log.

### `run_start` (terminal)

| field | meaning |
|---|---|
| `command` | the command line, verbatim |
| `version`, `commit`, `dirty` | binary version and the commit recorded when the binary was built (`manifest::build_info`; #256) |
| `binary_sha256` | SHA-256 of the running executable |
| `data.path`, `data.sha256` | the input path and its SHA-256 (`null` for stdin). The digest is cached in `<data>.sha256.json` (size, mtime, inode); a changed file is hashed again. A read-only directory skips the cache. |
| `data.full_rows`, `data.sampled_rows` | #112: `true`/`null` full load, `false`/`N` explicit sample, `null` for a CSV |
| `data.estimate_bytes` | #112: the memory estimate behind the load decision (`null` for a CSV) |
| `data.train`, `data.test` | `rows` and the checkpoint `fingerprint` of each sample |
| `config_hash` | FNV-1a of the canonical search configuration (#140) |
| `config` | the full resolved configuration, key by key, from the same canonical text |
| `seed`, `backend`, `device` | the effective seed, `cpu`/`gpu` and the device name |
| `threads`, `host.os`, `host.arch` | available parallelism and the host |
| `start` | UTC start time |
| `level`, `every`, `max_bytes` | the resolved log level, cadence and cap |
| `resumed_from` | the checkpoint path when the run continues one |

### `generation`

Emitted at the same boundary as `--progress`, once per completed generation
(subject to the cadence). `generation` is 1-based.

| field | level | meaning |
|---|---|---|
| `level` | both | `summary` or `full` |
| `t` | both | seconds since the search started |
| `wall` | both | the generation's wall up to the archive/convergence boundary |
| `phases.prep` | both | per-generation sample draws and the carried-population refit |
| `phases.search` | both | the island-thread scope: variation, fitting and inline scoring |
| `phases.eval` | both | the batched evaluator call and its CPU fallback rescore; the fit stage is not in it (#256) |
| `phases.fit` | both | #256: the constant-fit stage (GPU fit call, exact CPU fallback fits, write-back), taken out of the eval, prep and island windows it ran in |
| `phases.select` | both | screened successive-halving promotion and population selection |
| `phases.archive` | both | Pareto insertions and the best/quantile bookkeeping |
| `phases.fit_cpu` | both | constant-fitting CPU time, **summed over the island threads**, plus the exact CPU fit-fallback wall seconds (#256) |
| `phases.eval_cpu` | both | candidate-evaluation CPU time, **summed over the island threads**, plus the evaluator's CPU fallback pool wall seconds (#256) |
| `error.best` | both | best finite error in the generation's population |
| `error.median` | both | median of the same population |
| `error.p05`, `p25`, `p75`, `p95` | full | the rest of the population distribution (Hyndman–Fan type 7) |
| `archive.size` | both | Pareto members |
| `archive.hypervolume` | full | fixed-reference hypervolume, reference error 1.0 |
| `archive.hash` | full | the front's canonical hash (`Archive::hash`) |
| `archive.offers` | full | population candidates offered to the archive |
| `archive.admissions` | full | offers the archive accepted as non-dominated when they arrived |
| `fit.attempts` | both | candidates offered to the batched GPU fit (#169) |
| `fit.gpu` | both | candidates the GPU fit returned usable constants for |
| `fit.fallback` | both | candidates refit by the exact CPU arm |
| `fit.refits` | full | exact CPU refits recorded for this generation |
| `fit.host_seconds`, `fit.fallback_seconds`, `fit.fallback_workers` | full | #256: wall of the whole fit stage in this generation, the exact CPU fit-fallback inside it, and the width of the **fit** fallback pool |
| `fit.gate_seconds` | full | #498: wall of the exact host gate of device proposals inside `fit.host_seconds` (log_loss, correlation, correlation_signed, signed; 0 otherwise) |
| `fit.fallback_by_cause` | full | CPU fallback counts labelled by cause (`unsupported_plan`, `too_few_rows`, `too_many_constants`, `ill_conditioned`, `non_finite`, `launch`, …) |
| `fit.fallback_improved`, `fit.fallback_improved_by_cause` | full | fallbacks whose exact CPU refit lowered the exact error against the device (or incoming) constants, in total and by cause |
| `gpu` | both | `null` without a GPU batch: `batches`, `programs`, `kernel_s`, `wall_s`, `duty`, `native`, `lowered`, `fallbacks`, `fallback_s`, `fallback_workers` |
| `fallback.seconds`, `fallback.workers`, `fallback.scope` | both | the **evaluator** fallback pool (`scope` is `evaluator_pool`): what it cost and how wide it was. The fit fallback is `fit.fallback_*`. |
| `variation.attempts` | full | candidate generation attempts |
| `variation.duplicates` | full | attempts rejected as structural duplicates (`--no-dedup` makes this 0) |
| `variation.too_complex` | full | offspring rejected by `--max-complexity` |
| `variation.offspring` | full | candidates that survived generation and were kept |
| `variation.fitted` | full | offspring whose constants were fitted on the CPU |
| `variation.refits` | full | exact CPU refits of carried/promoted survivors |
| `variation.dedup_rate` | full | `duplicates / attempts` |
| `variation.reevals` | full | #500: carried individuals scored again over every training row at the start of the generation: pending (`NaN`) ones, and `+inf` ones whose first 2048 rows are all finite. A `+inf` individual with a non-finite prediction in those rows, or one the dimension filter rejects, stays `+inf` without the pass. The probe covers the plain unweighted metrics; every other shape takes the full pass |
| `variation.constant_rejected` | full | offspring rejected before the fit because no input reaches the rows outside an aggregate (`--reject-constant-offspring`; 0 without the flag) |
| `qd` | full | `null` without `--qd`: `occupied`, `reachable`, `coverage`, `entropy` |
| `convergence.percent_converged` | both | the #18 headline (`docs/convergence.md`) |
| `convergence.confidence/stability/maturity` | full | its three components |
| `islands` | full | best error of each island's population, in island order |
| `rng` | full | hash of the island and main MT19937 states at this boundary |

The six `phases` (excluding the two summed CPU times) are disjoint windows on
the search's main thread and add up to `wall`. `fit_cpu` and `eval_cpu` overlap
them: the islands run in parallel, so those are CPU-time sums, not wall time.
That is why the analyzer reports the fit share of the island CPU budget as
`fit_cpu / (search × islands)`.
When `fit_cpu + eval_cpu` is under 1 ms (a device-fit run, where only fallback seconds remain) the analyzer prints "no CPU fitting in the islands" with severity 0 and omits the fit share rows.

On the flat postfix path (`--variation postfix`, chosen automatically for a
large, fit-free GPU run) there is no per-candidate fit, and the arena
de-duplicates as one pass: `variation.attempts` and `variation.offspring` are
the batch size, `variation.duplicates` counts slots that share a canonical
postfix program, `fit.*` is zero, and `phases.select`/`phases.prep` are zero
because that loop has neither.

### `checkpoint`

| field | meaning |
|---|---|
| `generation` | the generation the snapshot was taken at |
| `path` | the checkpoint file |
| `bytes`, `hash` | its size and FNV-1a hash after the atomic save |

### `run_end` (terminal)

| field | meaning |
|---|---|
| `generations` | generations completed |
| `stop` | `solved` (error at or below the solved floor; never for AIC, AICC, BIC), `patience`, `time_budget`, `generations` (budget reached) or `already_stopped` (a resumed checkpoint was already finished) |
| `wall`, `search` | seconds from the CLI taking the clock to the end of the search, and the search phase alone |
| `archive` | the final `size`, `best_error` and `hash` |
| `json` | where the result JSON went (`null` = stdout) |
| `peak_rss_bytes` | the process high-water mark from `/proc/self/status` `VmHWM` (`null` off Linux) |
| `gpu_wall_s` | measured evaluator full-path wall (the evaluator window only) |
| `device_seconds_incl_fit`, `device_hours`, `device_fraction_of_run` | #256: evaluator wall plus fit kernel seconds, that over 3600, and that over `wall` |
| `run_wall_s`, `device_busy_s`, `duty_run` | #284 run ledger: run wall, **union** of device intervals, and their ratio |
| `unaccounted_s` | `run_wall_s − Σ stage_s`; >1 % of wall is a telemetry defect |
| `device_busy_s` basis | the union uses CUDA event timestamps when they are available; the stage seconds (including `io`), the placement basis (`events`, `host_envelope`, `mixed`) and the structural-zero stages (`features`, `pnl`) are in `--json` `gpu.ledger`, not in this event |
| `stages.vary_s`, `stages.fit_s` (mixed arm) | with `--gpu --fit-strategy exact` the island threads fit on the CPU; each island-scope window is split into `fit_s` (summed island fit time divided by island threads, capped at the window) and `vary_s` (the rest). The split is apportioned, not a measured interval |
| `fit_kernel_s`, `fit_fallback_s` | raw device fit-kernel seconds and measured exact-CPU fit-fallback seconds |
| `fit_kernel_lm_s`, `fit_kernel_polish_s` | `fit_kernel_s` apportioned to the damped LM loop and to the polish phase by per-block cycle sums (`--fit-polish`) |
| `fit_refit_s`, `fit_refits` | #495: wall seconds and count of the exact CPU finalist refits after the last generation (GPU runs; `null` on a CPU run). `rcsr dev-report` ranks the refit as its own window beside the generation phases and prints it in the run table |
| `events`, `bytes`, `dropped` | log bookkeeping **before** this event; `dropped` counts events lost to the byte cap |
| `end` | UTC end time |

Diagnostics, baselines and report generation happen after `run_end` is written,
so a later failure still leaves a complete log. The run manifest's phase timings
remain the place to read the whole-process wall. A search that panics writes
`run_start` and the completed generations but no `run_end`.

## `rcsr dev-report`

```sh
rcsr dev-report rcsr.dev.ndjson
```

```
# rcsr development log

`rcsr.dev.ndjson` — schema `rcsr-devlog/1`, `summary` level, 10 event(s),
8 of 80 generation(s) logged, 0 checkpoint(s)

> This is a **summary** log: one event every few generations with the
> monitoring fields. Per-generation quantiles, variation counters, archive
> hypervolume/hash and RNG keys are only in `--dev-log-level full`.

## Run
| item | value |
|---|---|
| data | `fit_heavy.csv` (2250 train + 750 test rows) |
| config hash | `0x…` |
| seed | 7 |
| backend | cpu |
| generations | 80 of 80 requested |
| stop reason | `generations` |
| wall | 43.7 s (search 43.6 s) |
| peak RSS | 7.7 MiB |

## Where the generation wall went
| phase | total s | median s/gen | share of generation wall |
|---|---:|---:|---:|
| search | 4.36 | 0.545 | 99.9% |
| eval | 0.000 | 0.000 | 0.0% |
| prep | 0.000 | 0.000 | 0.0% |
| select | 0.000 | 0.000 | 0.0% |
| archive | 0.001 | 0.000 | 0.0% |
| **generation wall** | 4.37 | 0.546 | 100.0% |

## Fit vs evaluation
| constant fitting (CPU, summed over islands) | 4.30 s |
| candidate evaluation (CPU, summed over islands) | 0.03 s |
| fit share of candidate CPU time | 99.4% |
| fit share of the island CPU budget | 98.6% |

## Variation effort
Not recorded at `summary` level: re-run with `--dev-log-level full` for
attempts, duplicates, too-complex rejects, archive offers/admissions and refits.
…

## Bottleneck ranking
1. **search** — 4.36 s (99.9% of generation wall): island variation: selection, crossover, mutation, fitting, scoring
2. **archive** — 0.001 s (0.0% of generation wall): Pareto insertions and the best/quantile bookkeeping
…

## Other findings
- **(severity 0.99)** constant fitting is 99.4% of candidate CPU time (4.30 s of 4.33 s); the run is fit-bound
```

The analyzer reads only the events this engine writes and adapts to the level:
a `summary` log reports the monitoring sections and says explicitly which
sections (variation counters, archive hypervolume, per-island best, RNG keys,
fallback causes) need `--dev-log-level full`; it never prints `NaN` for a field
the level did not carry.

* **Bottleneck ranking** — the five disjoint wall windows over the logged
  generations, largest first, with its share and a one-line reading of what the
  phase does. The top entry is where the wall went.
  When `run_end` carries `fit_refit_s` (a device-fit run), the post-loop exact
  finalist refit (#495) is a sixth window and the shares are of the generation
  wall plus the refit.
* **Other findings** — cross-cutting readings ranked by severity (0–1): the fit
  share of candidate CPU time, the duplicate rate (**full**), the archive
  admission rate (**full**), GPU kernel-time duty, the GPU-fit fallback rate, and
  a convergence plateau (the best error stopped improving more than half way
  through).
* **Fit vs evaluation** — is this run fit-bound? Fitted CPU time, evaluation CPU
  time, their share, and the fit share of the island CPU budget.
* **Fallback causes** — GPU-fit attempts/successes/fallbacks, fallback seconds
  and workers, and the per-cause table (**full**).
* **Convergence** — first and last best error, the generation of the last
  improvement, the final `percent_converged` and the first generation at or
  above 0.75.
* **Archive growth** — Pareto size (first/last/max), hypervolume (**full**), the
  final archive hash, and the last per-island best error (**full**).
* **Run** — data, fingerprints, config hash, seed, backend, threads,
  generations, stop reason, wall, peak RSS, device-hours, and the log's own
  event/byte/dropped counters.

Exit codes: `0` success, `2` usage error (missing path, unknown flag), `3` the
file cannot be read or contains no `rcsr-devlog/1` events.

## A worked example: what the log found

On a fit-heavy cell (3 000 rows, `--metric squared --fit-iters 10 --pop 200
--generations 30`, CPU exact fitting) a `full` log made `dev-report` say:

```
## Fit vs evaluation
| constant fitting (CPU, summed over islands) | 12.78 s |
| candidate evaluation (CPU, summed over islands) | 0.12 s |
| fit share of candidate CPU time | 99.0% |

## Other findings
- **(severity 0.99)** constant fitting is 99.0% of candidate CPU time (12.78 s of 12.90 s); the run is fit-bound
```

That reading points at one lever, the batched GPU fit already in the engine
(#169). Re-running the identical command with `--gpu --fit-strategy gpu` produced,
in its own dev log, `search` 6.15 s instead of 13.18 s (`wall` 13.2 s → 6.2 s,
**2.1×**), `fit_cpu` 0 and the fit moved into the eval window (5.50 s, GPU kernel
0.012 s; since #256 it is booked as `phases.fit`) with the fallback causes `ill_conditioned: 2`, `unsupported_plan: 5` —
which is the next thing to look at, because 99.8 % of that window is host-side
work, not the device. The archive differed (`2.928` vs `2.696` best error on the
same seed) because the GPU path fits in `f32` and polishes in `f64`: it is a
different, not a faster, search.

## The 10-hour run

The development log replaces the external monitor: the engine itself writes the
per-generation record, so a long run can be followed with `tail -f` and explained
afterwards with `dev-report`:

```sh
# tracked launcher: one manifest, one result, one checkpoint, one dev log per label
scripts/run-long.sh --data data.csv --target y --vars x --label btc-usd-2019h1 \
    --seed 7 --generations 5000 --gpu

tail -f runs/btc-usd-2019h1.dev.ndjson        # watch it (summary level)
rcsr dev-report runs/btc-usd-2019h1.dev.ndjson   # explain it
```

`scripts/run-long.sh` and `rcsr-launch` name one
`<label>.dev.ndjson` per run at `summary` level. The old scratch
`monitor_10h.py` was never part of the documented flow; it is tracked for
deletion in #200. What the engine log does **not** replace is 1 Hz *process*
sampling (CPU, RSS, `nvidia-smi`): that belongs outside the engine, and
`run_end` records the process peak RSS and the measured device-hours instead.

## Limitations and scope

* One log describes one search. `--report benchmark` and `--eval-expr` are
  refused when a log flag is given explicitly; a default-on benchmark run simply
  writes no log.
* A `summary` log cannot answer per-generation questions (which candidate,
  which fallback cause): re-run with `--dev-log-level full` (larger file, same
  guarantees).
* Optional candidate dumps are a separate, opt-in feature and are not part of
  this log; it deliberately records no expressions.
* Hypervolume is the fixed-reference form with reference error `1.0` and exists
  only in a `full` log. It is monotone within a run but only comparable across
  runs whose errors are on the same scale; `--json` carries the data-referenced
  validation hypervolume for cross-run comparison.
* `--dev-log-every N` writes generations `N, 2N, …`; if a run stops early, its
  last generation is not written unless it is a multiple of `N`. `run_end`
  always carries the final archive, best error and hash.

See issue #211 for the motivation and the acceptance criteria.
