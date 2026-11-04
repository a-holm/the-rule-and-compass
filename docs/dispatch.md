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

`rcsr calibrate [--out <path>] [--gpu | --cpu]` is the only writer. It runs
`Calibration::probe()` (host, median of 31 reps) and `dispatch_gpu::measure()`
(device) and writes the file atomically (a sibling temporary file, `fsync`, then
`rename`). The file is JSON, schema `rcsr-calibration/1`: `written` (UTC),
`device` (name or `null`), `build`, the rates `r_eval`, `r_fit`, `bw`, `bw_h2d`,
`bw_d2h`, `launch_s`, `cpu_eval`, `cpu_fit`, `hysteresis`, and the `spread` of
each probed term. Floats are written in the shortest form that reads back bit
for bit. A search run never writes it.

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
`host_probe_three_runs_agree_within_the_larger_of_their_own_spreads` and the
device counterpart measure this and are not part of a normal run.

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
compatibility calibration, so their measured crossover is unchanged.

## The run-level fit rule

The stage model above decides one candidate batch. The fit arm of a whole run (`exact` or the batched GPU LM for `squared`, `aic`, `aicc`, `bic`) is a separate, pure rule that reads no timing: `min(islands, cores) * (s * k + (1 - s) / W) * E * (1 + h) < 1`, with `k = 0.05`, `E = 1.32`, `h = 0.3`, `W` the fallback workers (`cores` is `--threads` when given, else 8, so the rule does not follow the CPU affinity of the process; `W = min(cores, 8)` unless `RCSR_CPU_FALLBACK_THREADS` is set) and `s = 1` only when every configured operator has a device fit opcode. It is resolved once before generation 0; see "Default fit arm" in `gpu-backend.md`. The per-candidate model cannot decide it: its device and CPU fit costs scale with the same row, constant and sweep factors, so the device always wins there and the floor of `32p` rows is the only guard.
