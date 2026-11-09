# Per-stage dispatch and the calibrated route ledger (#290)

Dispatch decides **which stage runs where**, not which backend a whole run uses.
Scoring 512 candidates over 2.4 M rows, fitting four constants on one candidate,
selecting a tournament winner and writing a checkpoint have different work
shapes and different transfer/launch overheads. The old gates
(`maximum_gpu_fit_worthwhile`, `median_gpu_fit_worthwhile`, `defers_fit_to_gpu`)
were boolean special cases of the model in `my-version/src/dispatch.rs`.

## The model

For `W` candidates, `N` rows, `m` nodes and `p` constants:

```text
R      = W · N · m                    row-evals
T_dev  = R / r_eval                   device arithmetic
T_fit  = W / r_fit(N, p, sweeps)      device fit
T_xfer = bytes / BW                   H2D + D2H
T_cpu  = R / cpu_eval                 host arithmetic
route  = Device iff T_dev + T_fit + T_xfer + launch < T_cpu / (1 + h)
```

`plan(stage, workload)` returns:

```text
Route  = Device | Cpu | Split
Reason = cost_device | cost_cpu | hysteresis_device | split_cost |
         cpu_contract | no_device | unsupported_plan | too_few_rows |
         too_many_constants | launch | ill_conditioned | non_finite |
         metric_not_native | device_memory | exactness | no_calibration
```

Every stage answers: `vary`, `eval`, `fit`, `select`, `archive`, `features`,
`pnl`, `io`, `refit`. Selection, archive insertion, I/O, feature preparation,
P&L and the exact finalist rebuild are **CPU by contract** and say so
(`cpu_contract`); they are not timing decisions. `vary` is CPU until a
fixed-capacity device pool lands (#291).

## Hysteresis

`h = 0.3` is a band, not a tie-breaker. From the CPU a stage must beat the CPU
by `1/(1+h)` (≈0.77×) to move to the device. Once on the device it stays until
it becomes `(1+h)` (1.3×) more expensive than the CPU. A workload hovering at
the crossover therefore has a stable route instead of switching every
generation. The ledger labels the held case `hysteresis_device`.

## Calibration (#733)

A run prices its stages from a **pinned calibration**, so one seed gives one route
ledger and one front in every process. No run times anything unless
`--calibration probe` is given. The source is chosen with
`--calibration anchors|probe|<path>`:

| source | rates | `dispatch.calibration.source` |
|---|---|---|
| (default) | the file `$XDG_CACHE_HOME/rcsr/calibration.json` (`~/.cache/rcsr/calibration.json` when unset) if it exists, else the anchors below | `file` or `anchors` |
| `anchors` | the fixed corpus anchors below; any file is ignored | `anchors` |
| `<path>` | the file at `<path>`, written by `rcsr calibrate` | `file` |
| `probe` | `Calibration::probe()` and, with `--gpu`, `dispatch_gpu::device_rates()` at startup; routes can differ between processes | `startup_probe` |

A file the user named, or the default file when it exists, must be valid. A
missing, truncated, non-JSON, wrong-schema, non-finite or non-positive file exits
2 with a message naming the file; there is no silent fallback to the anchors. A
file measured on another device than the one the run uses exits 2 naming both
devices. A CPU run, or a build without `--features cuda`, reads the host terms
only and keeps the anchors for the device terms.

A checkpoint records `config calibration_source` (`anchors`, `probe` or `file`) and, for a file, `config calibration_sha256`. Neither line is part of the config hash. `--resume` exits 2 when the resolved source or the file hash differs from the record, naming both. A checkpoint without the lines (an older binary) resumes only with an explicit `--calibration`, and the run prints a `# resume:` notice. A `probe` run is not reproducible, so it resumes only with `--calibration probe`, also with a notice.

`rcsr calibrate [--out <path>] [--gpu | --cpu] [--no-enrich] [--strict]` is the only writer. It runs
`Calibration::probe()` (host, median of 31 reps) and `dispatch_gpu::measure()`
(device) and writes the file atomically (a sibling temporary file, `fsync`, then
`rename`). The file is JSON, schema `rcsr-calibration/2` (a `rcsr-calibration/1` file still loads, with `enrich: null`): `written` (UTC),
`device` (name or `null`), `build`, the rates `r_eval`, `r_fit`, `bw`, `bw_h2d`,
`bw_d2h`, `launch_s`, `cpu_eval`, `cpu_fit`, `hysteresis`, and the `spread` of
each probed term. Floats are written in the shortest form that reads back bit
for bit. A search run never writes it.

The file also records `load1_start` and `load1_end`, the one-minute load average (`/proc/loadavg`) when the probe started and when it ended; both are left out on a host without `/proc/loadavg`, and a file without them loads as before. No rate and no route reads them. The probe times host work at idle priority, so a busy machine moves some terms: `r_eval` and `r_fit` hold, while `bw`, `launch_s`, `cpu_eval` and the enrich rates (elementwise, segmented-rank, mtf-join) move by up to 2.5x. When `load1_start` is above half the CPUs of the process, or any term has a spread above 1.5, `rcsr calibrate` writes the file and prints one line on stderr:

```text
# calibrate: measured under load 8.65 (limit 4.00); noisy terms: bw 1.89, launch_s 2.26, cpu_eval 21.68
```

`--strict` prints the line, exits 3 and writes nothing. A search that prices with a file whose `load1_start` is above that limit prints `# calibration: <path>: measured under load ...` once and routes as it would without the fields; it does not refuse, because that would invalidate files that exist. A load below the limit does not make the terms quiet: the line also names every term with a spread above 1.5, whatever the load. `rcsr calibrate` takes under 60 s at a load below 2.5 on the RTX 3060 and longer above it.

The probes behind `rcsr calibrate` and `--calibration probe` work as follows. The
host terms that route stages (`cpu_eval`, `cpu_fit`) stay within a fixed factor of
their anchors; the raw probe rate is reported unclamped as `host_rate` with its
spread. The device probe times the device once per process and
`Calibration::with_device_rates` installs the result. The launch term is small
in the model and its band is wide; the other terms dominate a route. Without a
device the device terms stay the corpus anchors:

| term | anchor | device probe |
|---|---:|---|
| `r_eval` | 20e9 row-evals/s (RTX 3060 saturated batch) | `eval_batch` kernel time over a resident batch, rows x candidates x nodes per second |
| `r_fit` | 1 / 89.5 ms at 2.456 M rows (batched LM corpus) | `fit_batch` kernel time, scaled to the reference shape by rows, constants and sweeps |
| `BW` | 12 GB/s (effective PCIe 3.0 x16) | pinned-host `cudaMemcpy` of 16 MiB each way, `2*bytes/(h2d+d2h)` per rep; `bw_h2d` and `bw_d2h` are recorded too |
| `launch` | 5 µs (3–7 µs measured range) | host time in the launch calls of a one-program batch |

Each device term is the median of its timed reps after untimed warm-up reps
(15 for eval and bandwidth, five for fit, 33 for launch). Its
spread is the largest rep over the smallest rep. Nothing is clamped. Two probes
agree on a term when their values differ by no more than the larger of the two
probes' spreads (`Calibration::agreement`); the ignored tests
`host_probe_three_runs_print_their_agreement` and the device counterpart print
the ratios and bands and assert nothing about them; they are not part of a normal run.

The result JSON (`--json` → `dispatch.calibration`, `--ndjson` done event →
`dispatch`) carries the provenance labels only: `source` (`anchors`, `file` or
`startup_probe`), `file_path` and `file_sha256` for a file, `device_source`
(`file`, `probe` or `anchor`), the sample counts, the hysteresis and the fit
reference shape. The rates stay out of it. The
`--run-manifest` `dispatch.calibration` object records the same source fields and adds `probe_s`, `r_eval`,
`r_fit`, `bw`, `bw_h2d`, `bw_d2h`, `launch_s`, `cpu_eval`, `cpu_fit`, `host_rate` and
`spread` (`r_eval`, `r_fit`, `bw`, `launch_s`, `cpu_eval`).

## Route ledger

Each run records one row per stage:

```json
{"stage":"eval","route":"device","reason":"cost_device","calls":4,
 "candidates":2048,"predicted_s":0.031,"actual_s":0.028,
 "actual_over_predicted":0.90,"bytes":540672,"switches":0}
```

`predicted_s` is the sum of the planned wall for the calls that ran;
`actual_s` is the measured stage time. The ratio is the calibration residual:
it makes a wrong route or a tuning regression visible without reading source.
A CPU-only run still gets the ledger (with `no_device`/`cpu_contract` reasons).

`--dispatch device` records reason `forced` for a stage routed to the device without a cost comparison; refusals and `cpu_contract` still win. With the pinned calibration, `auto` routes the same under `compute-sanitizer` as natively, so a sanitizer run exercises the kernels of the native run.

## Refusals are a cause set, never a stopwatch

Structural and arithmetic refusals are checked **before** any timing
comparison:

`launch`, `unsupported_plan`, `too_few_rows`, `too_many_constants`,
`ill_conditioned`, `non_finite`, `metric_not_native`, `no_device`, `exactness`.

A fast device can never turn an arithmetic failure into a CPU route, and a slow
measurement can never turn a supported plan into a "fallback". The
`FallbackCause::is_timing` predicate is always `false`, and a test forces every
cause against an absurdly fast device to prove the cause wins.

The fit arm has one extra policy label, `dispatch_cost`: the model predicted the
CPU as cheaper for a valid plan. It is deliberately separate from
`ill_conditioned`/`non_finite`, so a tuning route is never reported as an
arithmetic defect.

The fit arm plans the candidates that pass their own row floor (`32p` rows)
once per batch, as `W` fits with the largest constant count `p`, because the
device runs them as one launch and one transfer. The decision applies to every
candidate of the batch, and the route ledger holds one `fit` call carrying
`candidates = W`.

## Exactness

Routing never changes the reported numbers. The device may guide the search with
`f32` rows, but every archived error, constant and finalist is rebuilt by the
exact CPU `f64` path. `refit` is CPU by contract. Changing the route can change
which candidates the search visits (the trajectory), so the `--cpu` escape hatch
remains the reproducible cross-backend comparison; the reported numbers still
come from the `f64` rebuild.

## Tests

`my-version/src/dispatch.rs` pins:

* a route and machine reason for every stage;
* monotonicity in rows, candidates, nodes, constants and sweeps;
* hysteresis at the crossover and stable re-entry;
* the cause-set rule (timing cannot override a refusal);
* the split route;
* the pinned calibration: anchors twice give equal fields, a written file reads back bit for bit, and a corrupt, wrong-schema, unusable or other-device file is refused;
* manifest/JSON and ledger fields;
* the compact-cell batch still routes to the device.

The existing arm floors (`median`/`interquartile` 4 096 rows, `maximum` 512
rows) are expressed through the same planner with a zero-hysteresis
compatibility calibration. The `maximum` fit arm also needs the cost model to prefer the device, which holds above about 3 300 / p rows (#733).

## What the anchors decide

The compiled anchors (`DEFAULT_R_EVAL`, `DEFAULT_R_FIT`, `DEFAULT_BW`, `DEFAULT_LAUNCH_S`, `DEFAULT_CPU_EVAL`, `DEFAULT_CPU_FIT`) price every run with `--calibration anchors`. Some are off by large factors from a measured file. That moves a route only where the device and the CPU are close. `rcsr dispatch-plan` shows where:

```sh
rcsr dispatch-plan --calibration anchors --against tests/fixtures/calibration_rtx3060.json
rcsr dispatch-plan --calibration anchors --against my-calibration.json --json
```

It routes a fixed grid of 4,950 cells under both calibrations and prints one line per cell, then `grid cells 4950, flips <n>, largest flip cost <s> s, largest flipped work <w>`. The grid covers 16 to 3,911,395 rows, 1 to 1,024 candidates, 1 to 31 nodes (eval) or 1 to 14 constants and 3 or 8 sweeps (fit), planned from the CPU and with the stage held on the device. It reads no clock and opens no device.

Two tests keep a stale anchor from moving a route unseen:

* `pilot_sizes_route_to_the_device_under_a_tenfold_rate_error` scales each of the six rates of the anchors by 0.1 and by 10, one at a time, and asserts the device route for 1 to 1,024 candidates, at least 65,536 rows and every complexity and constant count of the grid. It reads no clock. A planner change that shrinks the margin fails it, whatever the anchors say.
* `anchor_flips_stay_below_the_launch_scale` routes the anchors against the fixture `tests/fixtures/calibration_rtx3060.json` (an RTX 3060 file) and asserts at most 60 flips, each with at most 4,096 units of work and at most 8 microseconds of wall cost per call.

A route that does not flip at a size is not decided by the difference between the two calibrations. A changed anchor gives a new binary SHA-256 and moves every `predicted_s`.

## The run-level fit rule

The stage model above decides one candidate batch. The fit arm of a whole run (`exact` or the batched GPU LM for `squared`, `aic`, `aicc`, `bic`) is a separate, pure rule that reads no timing: `min(islands, cores) * (s * k + (1 - s) / W) * E * (1 + h) < 1`, with `k = 0.05`, `E = 1.32`, `h = 0.3`, `W` the fallback workers (`cores` is `--threads` when given, else 8, so the rule does not follow the CPU affinity of the process; `W = min(cores, 8)` unless `RCSR_CPU_FALLBACK_THREADS` is set) and `s = 1` only when every configured operator has a device fit opcode. It is resolved once before generation 0; see "Default fit arm" in `gpu-backend.md`. The per-candidate model cannot decide it: its device and CPU fit costs scale with the same row, constant and sweep factors, so the device always wins there and the floor of `32p` rows is the only guard.

## Enrich planner rates

`rcsr enrich --gpu auto` prices each feature family with the rates of `enrich::EnrichRates`. The anchors are the constants in `enrich.rs` (see `docs/ohlcv.md`). `rcsr enrich --calibration anchors|<path>` selects the source with the same rules as the search flag: the default is the default calibration file when it exists, else the anchors; a missing, invalid or other-device file exits 2 and there is no silent fallback. `probe` is refused. `--write-bw` overrides the file's bandwidth.

The optional `enrich` object of a `rcsr-calibration/2` file holds `rows`, `context_ms_per_mrow`, `fixed_overhead_ms`, `write_bps`, `classes` (per operation class name: `cpu_fixed_ms`, `cpu_col_ms`, `gpu_cells_per_s`; the `compact-pass` class also takes the optional `cpu_fixed_ms_no_benchmark`) and `spread`. A class the file leaves out keeps its anchor. Every `*_ms`, `write_bps` and rate must be finite; the rates and bandwidths must be positive and the CPU cost parts not negative, otherwise the file is invalid. The planner limits each value to 0.25x to 4x of its anchor. A `/1` file, a host-only file, `--cpu` and `--no-enrich` give `enrich: null` and the anchors.

`rcsr calibrate` fills the object (skip with `--no-enrich`) by running the real enrich producers on a synthetic panel of 109 symbols and 9,175 bars each (about 1,000,000 rows), held in the temp directory. For each class it times a small and a large column set on the CPU and, with a device, on the GPU, and fits the CPU `fixed + per column` cost and the device throughput; the write bandwidth is the median over classes of the output write seconds the same device runs report, so a busy disk at probe time does not move the plans; without a device `measure_write_bandwidth` supplies it. The small and the large column set of a class each run twice on the device, interleaved. Timing noise on a fixed workload only adds time, so the smallest sample of each set is used. A class's device rate is what the extra cells cost after the bus and the reported write (the write noise of a run cancels in its own term), limited to 4x of the anchor. A kernel term that does not exceed the largest minus smallest of the two per-pair figures is hidden behind the transfers: the class keeps its anchor. `RCSR_PROBE_TRACE=1` prints the kernel seconds, the noise and the rate of every class to stderr. The `trend-join` class (#932) is timed with one and with all 27 trend columns. A pass class (`compact-pass`, `py15m-pass`) is fitted from the same runs; the compact pass is timed twice, with the probe benchmark symbol and without it (`cpu_fixed_ms_no_benchmark`), on the first 27 symbols of the panel (a shared pass costs the same per row on any number of symbols; the benchmark series keeps its full length). The classes `calendar-trig`, `rolling-sum`, `rolling-extrema`, `wilder-scan` and `join` are not probed (to keep the probe short, or because no column set avoids a shared pass or a benchmark series), so they keep their anchors. The `spread` of a device rate is the largest per-pair rate over the smallest for a resolved class (1.0 when the class keeps its anchor); the other terms are timed once and report 1.0. The device context (host read plus resident upload) comes from a fresh `rcsr enrich` process, because the device initialisation is part of what a real run pays,. The planner never reads a clock: only `rcsr calibrate` times anything.

The enrich JSON carries `plan_rates` (`source` `anchors` or `file`, plus `file_path`, `file_sha256` and `schema` for a file) and `<out>.enrich.json` records the same object. A family's shared CPU pass (`compact-pass`, `py15m-pass`) is costed once per run and saved only when every family that needs it goes to the device; the `plan[]` entries carry `pass_shared`, `pass_cpu_ms` and `benchmark_present`, and the `# enrich-plan` lines print the same three fields. `benchmark_present` is true when the store holds the benchmark series (`--cross-symbol`, `--cross-data`): the compact pass joins it per row and costs several times more than without it, so the planner prices the pass with `classes.compact-pass.cpu_fixed_ms` when it is true and with `cpu_fixed_ms_no_benchmark` when it is false. A file that leaves the field out keeps the anchor.
