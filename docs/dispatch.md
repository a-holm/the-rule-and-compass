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
         metric_not_native | exactness | no_calibration
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

## Calibration

`Calibration::probe()` is the startup probe (well under 100 ms on the reference
host). It measures the host evaluation rate and anchors the device rates from
the measured corpus:

| term | anchor | source |
|---|---:|---|
| `r_eval` | 20e9 row-evals/s | RTX 3060 saturated batch |
| `r_fit` | 1 / 89.5 ms at 2.456 M rows | batched LM corpus |
| `BW` | 12 GB/s | effective PCIe 3.0 x16 |
| `launch` | 5 µs | 3–7 µs measured range |

The host term scales the device anchors by at most 0.75–1.33×, and the values
are clamped so three probes on one host agree within the stated band
(`dispatch.calibration.band = 2.0`). A hardware probe can replace the anchors
with `Calibration::with_device_rates`. The calibration, its source, sample count,
reproducibility band and (in the manifest) probe time are written to:

* `--json` → `dispatch.calibration`
* `--ndjson` done event → `dispatch`
* `--run-manifest` → top-level `dispatch` object

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
* three-probe reproducibility within the stated band;
* manifest/JSON and ledger fields;
* the compact-cell batch still routes to the device.

The existing arm floors (`median`/`interquartile` 4 096 rows, `maximum` 512
rows) are expressed through the same planner with a zero-hysteresis
compatibility calibration, so their measured crossover is unchanged.
