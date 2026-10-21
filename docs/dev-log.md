# Development log (`--dev-log`) and `rcsr dev-report`

The final `--json` says *what* a run found. It does not say *why* it behaved the
way it did: which phase consumed the wall clock, whether the run was fit-bound
or evaluation-bound, how many candidates structural de-duplication threw away,
how often the GPU fell back to the CPU and for what reason, how the Pareto front
and the convergence readings moved, or where a 10-hour run was stuck.

`--dev-log <path>` writes that record — one NDJSON event per generation plus a
start, checkpoint and end event — and `rcsr dev-report <path>` summarises it.
Both are implemented in Rust; there is no separate Python analyzer.

```sh
rcsr --data examples/line.csv --target y --vars x --seed 3 --generations 200 \
     --dev-log run.ndjson
rcsr dev-report run.ndjson
```

| flag | default | meaning |
|---|---|---|
| `--dev-log <path>` | off | write the NDJSON development log to this path |
| `--dev-log-every <N>` | `1` | write a `generation` event only every Nth generation |
| `--dev-log-max-bytes <size>` | `64MiB` | cap on generation and checkpoint events |

`--dev-log` is independent of `--json`, `--ndjson`, `--progress`, `--out`,
`--json-out` and `--run-manifest`: it never shares their streams. It is refused
with `--eval-expr` (there is no search to log) and with `--report benchmark`
(one file describes one search; the repeats would interleave). `--null-runs`
and the benchmark repeats deliberately do **not** write into the log: they run
with the hook removed, so every `generation` event belongs to the logged run.

## What is guaranteed

**Off by default, and free when off.** Without `--dev-log` the search's
`Config::dev_log` is `None`. Every measurement the log needs (per-phase clocks,
atomic counters, the population quantile scan, the RNG-state hash) is skipped;
the only residue is one predictable branch per counted candidate.

**Trajectory-neutral.** The log is a *reading*. It never draws from an RNG,
never reorders candidates, never changes variation, fitting, selection or the
archive. The same seed with the log on and off produces a byte-identical front
(`--out`), an identical result document (the only differing fields are
`elapsed_seconds` and the `timing`/`knn` wall clocks) and the same canonical
configuration hash in the run manifest. Pinned by
`tests/dev_log.rs::logging_on_or_off_leaves_the_archive_results_and_config_hash_identical`.

**Atomic.** Each event is one `write` of one complete line, newline included, to
a file opened with `O_APPEND`. A killed process therefore leaves a **valid
prefix**: every complete line parses, and at most the final line is torn.
`rcsr dev-report` skips that torn line, says how many lines it skipped, and
still reports the prefix.

**Bounded.** Generation and checkpoint events stop at `--dev-log-max-bytes`
(default 64 MiB); each dropped event is counted and `run_end` reports the count.
`run_start` and `run_end` are terminal events and are always written, so the
file is bounded by `cap + two events` and never grows without limit. The file
is truncated when the run starts: one log describes one run.

**No raw rows.** The log records counts, timings, hashes and error statistics.
It never writes an input row, a target value, a prediction or a formula. The
data appears only as its path, its SHA-256 and the training/test fingerprints
the checkpoint uses (`tests/dev_log.rs::the_log_contains_no_raw_data_rows`).

**Stable schema.** Every line carries `"schema":"rcsr-devlog/1"` and `"v":1`.
Fields may be added within this version; removing or redefining one needs a new
version. `rcsr schema` publishes the schema id, the event names and the flags
under `dev_log`.

## Cost

Measured on a **fit-heavy** cell (`tests`-style synthetic table, 3 000 rows, two
inputs, `--fit-iters 12`, `--pop 120`, 80 generations, CPU, ~0.4 s per
generation). Every run used the same seed and data; the table is the median of
5 paired runs in alternating order:

| quantity | log off | log on | difference |
|---|---:|---:|---:|
| wall clock | 32.27 s | 32.01 s | −0.8 % (inside run-to-run noise) |
| process CPU time | 32.22 s | 31.97 s | −0.8 % |
| engine `timing.search_seconds` | 32.20 s | 31.95 s | −0.8 % |
| peak RSS (`/usr/bin/time`, `%M`) | 7 824 KiB | 7 684 KiB | −140 KiB |

A paired off-versus-off control on the same machine moved the medians by up to
±1 % and single runs by ±20 %, so the honest reading is **no measurable
overhead**: the difference is smaller than the noise of this host and inside
the 2 % target either way. The same comparison on a 2 s cell gave a median
+1.5 % (range −11 %…+64 % per pair), i.e. noise again.

The per-event cost is measured directly as well
(`src/devlog.rs::one_generation_event_costs_well_under_a_millisecond`): about
**10 µs** to render the record, format the JSON and append one ~1 000-byte line.
At 0.4 s per generation that is 0.002 % of the generation.

The **fixed start cost** is the one-time provenance of the first event —
SHA-256 of the binary and of the data file, plus the git commit/dirty probe.
Measured on a 0.13 s run (median of 7): a plain run takes 0.130 s, with
`--dev-log` 0.170 s, so the log adds **≈ 40 ms** once per run, independent of
its length. For comparison, `--run-manifest` (#140), which records the same
facts, adds ≈ 80 ms. The cost model is therefore

```
overhead ≈ 40 ms (once) + ~10 µs per logged generation
```

— about 0.13 % of the 32 s fit-heavy cell above, and 4 % of a sub-second GPU
cell (measured: 0.94 s → 0.98 s wall, +2.2 % CPU). `--dev-log-every N` divides
the second term; nothing divides the first, so a run shorter than a second is
the only case where the flag is worth turning off.

Peak RSS impact is the open handle, one `Mutex`, and no retained history: every
event is written and forgotten, so the log adds no growth over the run. The
measured difference (−140 KiB, −1.8 %) is inside the RSS noise of the process
allocator for a 7.5 MiB run; the reference logs for this page cost 12 KiB per
80-generation CPU run and 16 KiB per 12-generation GPU run.

`--dev-log-every N` divides both the event count and the per-generation cost by
`N` when a long run makes even that matter.

## Events

Every event is one line, `{"schema":"rcsr-devlog/1","v":1,"event":…}`. Times are
seconds, sizes are bytes, hashes are `0x`-prefixed 64-bit hex.

### `run_start` (terminal)

| field | meaning |
|---|---|
| `command` | the command line, verbatim |
| `version`, `commit`, `dirty` | binary version and git state (`manifest::git_info`) |
| `binary_sha256` | SHA-256 of the running executable |
| `data.path`, `data.sha256` | the input path and its SHA-256 (`null` for stdin) |
| `data.train`, `data.test` | `rows` and the checkpoint `fingerprint` of each sample |
| `config_hash` | FNV-1a of the canonical search configuration (#140) |
| `config` | the full resolved configuration, key by key, from the same canonical text |
| `seed`, `backend`, `device` | the effective seed, `cpu`/`gpu` and the device name |
| `threads`, `host.os`, `host.arch` | available parallelism and the host |
| `start` | UTC start time |
| `every`, `max_bytes` | the resolved log flags |
| `resumed_from` | the checkpoint path when the run continues one |

### `generation`

Emitted at the same boundary as `--progress`, once per completed generation
(subject to `--dev-log-every`). `generation` is 1-based.

| field | meaning |
|---|---|
| `t` | seconds since the search started |
| `wall` | the generation's wall up to the archive/convergence boundary |
| `phases.prep` | per-generation sample draws and the carried-population refit |
| `phases.search` | the island-thread scope: variation, fitting and inline scoring |
| `phases.eval` | the batched evaluator call and its CPU fallback rescore |
| `phases.select` | screened successive-halving promotion and population selection |
| `phases.archive` | Pareto insertions and the best/quantile bookkeeping |
| `phases.fit_cpu` | constant-fitting CPU time, **summed over the island threads** |
| `phases.eval_cpu` | candidate-evaluation CPU time, **summed over the island threads** |
| `error.best` | best finite error in the generation's population |
| `error.p05`, `p25`, `median`, `p75`, `p95` | Hyndman–Fan type-7 quantiles of the same population |
| `archive.size` | Pareto members |
| `archive.hypervolume` | fixed-reference hypervolume of the front, reference error 1.0 (`qd::fixed_reference_hypervolume`) |
| `archive.hash` | the front's canonical hash (`Archive::hash`) |
| `archive.offers` | population candidates offered to the archive |
| `archive.admissions` | offers the archive accepted as non-dominated when they arrived |
| `variation.attempts` | candidate generation attempts |
| `variation.duplicates` | attempts rejected as structural duplicates (`--no-dedup` makes this 0) |
| `variation.too_complex` | offspring rejected by `--max-complexity` |
| `variation.offspring` | candidates that survived generation and were kept |
| `variation.fitted` | offspring whose constants were fitted on the CPU |
| `variation.refits` | exact CPU refits of carried/promoted survivors |
| `variation.dedup_rate` | `duplicates / attempts` |
| `gpu` | `null` without a GPU batch: `batches`, `programs`, `kernel_s`, `wall_s`, `duty`, `native`, `lowered`, `fallbacks`, `fallback_s`, `fallback_workers` |
| `fit.attempts` | candidates offered to the batched GPU fit (#169) |
| `fit.gpu` | candidates the GPU fit returned usable constants for |
| `fit.fallback` | candidates refit by the exact CPU arm |
| `fit.refits` | refits recorded for this generation |
| `fit.fallback_by_cause` | CPU fallback counts labelled by cause (`unsupported_plan`, `too_few_rows`, `too_many_constants`, `ill_conditioned`, `non_finite`, `launch`, …) |
| `fallback.seconds`, `fallback.workers` | what the CPU fallback pool cost and how wide it was |
| `qd` | `null` without `--qd`: `occupied`, `reachable`, `coverage` (reachable support), `entropy` (bits) |
| `convergence` | `confidence`, `stability`, `maturity`, `percent_converged` (#18) |
| `islands` | best error of each island's population, in island order |
| `rng` | hash of the island and main MT19937 states at this boundary |

The five `phases` (excluding the two summed CPU times) are disjoint windows on
the search's main thread and add up to `wall`. `fit_cpu` and `eval_cpu`
overlap them: the islands run in parallel, so those are CPU-time sums, not wall
time. That is why the analyzer reports the fit share of the island CPU budget
as `fit_cpu / (search × islands)`.

On the flat postfix path (`--variation postfix`, chosen automatically for a
large, fit-free GPU run) there is no per-candidate fit, and the arena
de-duplicates as one pass: `variation.attempts` and `variation.offspring` are
the batch size, `variation.duplicates` counts slots that share a canonical
postfix program, and `fit.*` is zero. `phases.select` and `phases.prep` are
zero because that loop has neither.

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
| `stop` | `solved` (error at or below the solved floor), `patience`, `time_budget`, `generations` (budget reached) or `already_stopped` (a resumed checkpoint was already finished) |
| `wall`, `search` | seconds from the CLI taking the clock to the end of the search, and the search phase alone |
| `archive` | the final `size`, `best_error` and `hash` |
| `json` | where the result JSON went (`null` = stdout) |
| `peak_rss_bytes` | the process high-water mark from `/proc/self/status` `VmHWM` (`null` off Linux) |
| `gpu_wall_s`, `device_hours` | measured GPU full-path wall and `gpu_wall_s / 3600` |
| `events`, `bytes`, `dropped` | log bookkeeping **before** this event; `dropped` counts events lost to the byte cap |
| `end` | UTC end time |

Diagnostics, baselines and report generation happen after `run_end` is written,
so a later failure still leaves a complete log. The run manifest's phase
timings remain the place to read the whole-process wall.

## `rcsr dev-report`

```sh
rcsr dev-report run.ndjson
```

```
# rcsr development log

`run.ndjson` — schema `rcsr-devlog/1`, 82 event(s), 80 generation(s), 0 checkpoint(s)

## Run
| item | value |
|---|---|
| data | `fit_heavy.csv` (3000 train + 1000 test rows) |
| config hash | `0x…` |
| seed | 7 |
| backend | cpu |
| generations | 80 of 80 requested |
| stop reason | `generations` |
| wall | 32.0 s (search 31.9 s) |
| peak RSS | 7.5 MiB |

## Where the generation wall went
| phase | total s | median s/gen | share of generation wall |
|---|---:|---:|---:|
| search | 31.7 | 0.396 | 99.9% |
| eval | 0.000 | 0.000 | 0.0% |
| prep | 0.000 | 0.000 | 0.0% |
| select | 0.000 | 0.000 | 0.0% |
| archive | 0.001 | 0.000 | 0.0% |
| **generation wall** | 31.7 | 0.396 | 100.0% |
…

## Fit vs evaluation
| item | value |
|---|---:|
| constant fitting (CPU, summed over islands) | 31.4 s |
| candidate evaluation (CPU, summed over islands) | 0.27 s |
| fit share of candidate CPU time | 99.1% |
…

## Variation effort
## Fallback causes
## Convergence
## Archive growth
## Bottleneck ranking
1. **search** — 31.7 s (99.9% of generation wall): island variation: selection, crossover, mutation, fitting, scoring
2. **archive** — 0.001 s (0.0% of generation wall): Pareto insertions and the best/quantile bookkeeping
…

## Other findings
- **(severity 0.99)** constant fitting is 99.1% of candidate CPU time (31.4 s of 31.7 s); the run is fit-bound
- **(severity 0.90)** structural de-duplication rejected … of … attempts (…%): …
```

The analyzer reads only the events this engine writes:

* **Bottleneck ranking** — the five disjoint wall windows, largest first, with
  its share of the summed generation wall and a one-line reading of what the
  phase does. This is the ranked list: the top entry is where the wall went.
* **Other findings** — cross-cutting readings that are not wall-clock phases,
  ranked by severity (0–1): the fit share of candidate CPU time, the duplicate
  rate, the archive admission rate, GPU kernel-time duty (when a GPU ran), the
  GPU-fit fallback rate, and a convergence plateau (the best error stopped
  improving more than half way through).
* **Fit vs evaluation** — answering "is this run fit-bound?" with fitted CPU
  time, evaluation CPU time and the party's share, plus the fit share of the
  island CPU budget.
* **Fallback causes** — GPU-fit attempts, GPU successes, CPU fallbacks, the
  fallback seconds and workers, and the per-cause table.
* **Convergence** — first and last best error, the generation of the last
  improvement, the final `percent_converged` and the first generation at or
  above 0.75.
* **Archive growth** — Pareto size (first/last/max), hypervolume (first/last),
  the final archive hash and the last per-island best error.
* **Run** — data, fingerprints, config hash, seed, backend, threads,
  generations, stop reason, wall, peak RSS, device-hours, and the log's own
  event/byte/dropped counters.

Exit codes: `0` success, `2` usage error (missing path, unknown flag), `3` the
file cannot be read or contains no `rcsr-devlog/1` events.

## A worked example: what the log found

The log's job is to name a bottleneck you would otherwise guess at. On the
fit-heavy cell above (3 000 rows, `--metric squared --fit-iters 10 --pop 200
--generations 30`, CPU exact fitting) `dev-report` says:

```
## Fit vs evaluation
| constant fitting (CPU, summed over islands) | 12.78 s |
| candidate evaluation (CPU, summed over islands) | 0.12 s |
| fit share of candidate CPU time | 99.0% |

## Other findings
- **(severity 0.99)** constant fitting is 99.0% of candidate CPU time (12.78 s of 12.90 s); the run is fit-bound
```

That reading points at one lever, the batched GPU fit already in the engine
(#169). Re-running the identical command with `--gpu --fit-strategy gpu`
produced, in its own dev log, `search` 6.15 s instead of 13.18 s (`wall`
13.2 s → 6.2 s, **2.1×**), `fit_cpu` 0 and the fit moved into `phases.eval`
(5.50 s, GPU kernel 0.012 s) with the fallback causes `ill_conditioned: 2`,
`unsupported_plan: 5` — which is the next thing to look at, because 99.8 % of
that window is host-side work, not the device. The archive differed (`2.928`
vs `2.696` best error on the same seed) because the GPU path fits in `f32` and
polishes in `f64`: it is a different, not a faster, search.

## Limitations and scope

* One log describes one search. `--report benchmark` and `--eval-expr` are
  refused; the null control and benchmark repeats skip the log.
* The log is not a replacement for the external long-run monitor: 1 Hz process
  CPU/RSS/GPU sampling belongs outside the engine. This is the per-generation
  record the engine can make from the inside.
* Optional candidate dumps are a separate, opt-in feature and are not part of
  this log; it deliberately records no expressions.
* Hypervolume is the fixed-reference form with reference error `1.0`. It is
  monotone within a run but only comparable across runs whose errors are on the
  same scale; `--json` carries the data-referenced validation hypervolume for
  cross-run comparison.
* `--dev-log-every N` writes generations `N, 2N, …`; if a run stops early, its
  last generation is not written unless it is a multiple of `N`. `run_end`
  always carries the final archive, best error and hash.
* A search that panics or exits through an error path writes `run_start` and
  the completed generations but no `run_end`; `dev-report` then says there is
  nothing to analyse when no generation event was written at all, or reports
  the prefix it has otherwise.

See issue #211 for the motivation and the acceptance criteria.
