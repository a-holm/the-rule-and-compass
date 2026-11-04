# How everything works

This page shows how the parts of the repository fit together: the engine, the GPU backend, the data pipeline, the protocol and lockbox, the pilot runbook and the agent workflow. It explains the flow and links each fact to the document that owns it. It holds no flag lists and no measured numbers.

## Engine

`rcsr` is a symbolic-regression engine in pure `std` Rust. Given a table of columns, it searches for formulas `y = f(x)` and reports the Pareto front of error against complexity: the most accurate formula at each complexity level. The behaviour contract is [`SPEC.md`](../SPEC.md).

The search is an island-model genetic program. Each island evolves a population of expression trees. Mutation, crossover and simplification change the trees. Every candidate gets a complexity score and an error on the training rows. A shared archive keeps the non-dominated set.

Constants inside a candidate are not guessed. They are fitted against the configured `--metric` with Levenberg-Marquardt, or with an exact solver chosen by `--fit-strategy`. Reported values are always exact CPU `f64` values, whichever backend guided the search.

The data is split before the search. The training rows drive the fit. A validation slice, set with `--validation-frac`, selects the final model. The test block is read for the winner only. Time-ordered splits for ordered data are in the protocol section.

A run can stop and continue. `--save` writes an atomic checkpoint and `--resume` restores the search state exactly, so a split run and an uninterrupted run give the same front. `--run-manifest` records the binary, data and configuration, and `rcsr verify-manifest` checks them later. `--report` turns a run into a Markdown document for another reader.

The main use case is a long-only buy/sell signal trained on profit. `--objective pnl` (or `sharpe`, `log_growth`) scores finalists as a book that is long or flat only: no short, no leverage, a cost per side. `--fitness pnl` goes further and ranks every candidate in the search by that book's regret on the training rows, so profit drives the search and not only the final choice. `--pnl-blocks` cuts the training rows into equal time blocks and penalises a formula that earns in one regime only. `--fitness pnl` refuses `--cv`, `--stability` and `--null-runs`, with exit 2.

`rcsr schema` prints a machine-readable description of the CLI. `--help` and `README.md` list every option.

Details:
* [`docs/how-symbolic-regression-works.md`](how-symbolic-regression-works.md): plain-language walkthrough.
* [`docs/optimisation.md`](optimisation.md): theory and measurements behind the search and the fitting.
* [`docs/complexity.md`](complexity.md): what complexity means and how to choose it.
* [`docs/convergence.md`](convergence.md): how to tell whether a run converged.
* [`docs/checkpoint.md`](checkpoint.md): checkpoint format and resume rules.
* [`docs/reports.md`](reports.md): summary, sensitivity and benchmark reports.
* [`docs/optimisation.md`](optimisation.md), section "`--fitness pnl`": the ranking, `--pnl-blocks` and the refusals.

## GPU backend

The backend is optional. A binary built with `--features cuda` links the CUDA code statically, probes the device once at start and uses it when one is usable. A default build is pure `std` and runs on the CPU only. `--cpu` forces the CPU.

The device scores candidates in `f32`. It is a fast ranking of many candidates, not the source of reported numbers. Finalists are rebuilt and judged in CPU `f64`, and a device proposal for constants is accepted only after an exact CPU check. Device and CPU results can therefore differ in the search path but not in the reported values.

Dispatch decides per stage, not per run, where work runs: on the device, on the CPU or split. A cost model chooses from a pinned calibration, and the reason for each route is recorded. The calibration is a file written by `rcsr calibrate` or the fixed corpus anchors, so one seed gives one route and one front in every process. Some stages are CPU by contract, such as selection and the exact finalist rebuild. `--dispatch device` skips the cost comparison only.

An explicit `--gpu` fails closed. If the device is missing or fails after the probe, the run exits non-zero and writes no result, and the last checkpoint stays for `--resume --gpu`. Without `--gpu`, a CUDA build prefers the device and falls back to the CPU with a recorded reason. A checkpoint records its backend, and `--resume` refuses a backend change unless you pass `--allow-backend-change`.

A device out-of-bounds access can crash the driver and freeze the machine. Any change to CUDA code, and any test of an allocation-failure path, runs first under `compute-sanitizer` memcheck with small inputs. Device allocation and CUB calls go through the guard helpers in `my-version/src/cuda/`. `--gpu-memory` caps what a run may plan on.

Details:
* [`docs/gpu-backend.md`](gpu-backend.md): build, kernels, exactness contracts and verified results.
* [`docs/dispatch.md`](dispatch.md): the cost model, routes, reasons and the route ledger.
* [`docs/memory-model.md`](memory-model.md): bytes per cell and the memory guard.

## Data pipeline

Market data flows through small, separate commands. Each step writes a file the next step reads.

1. `rcsr-acquire` in [`tools/acquire/`](../tools/acquire/) lists, downloads and checksum-verifies the spot kline archive. It is a separate pure-std crate that shells out to `curl`, so the core crate holds no HTTP client. It also filters the symbol universe from pinned input files.
2. `rcsr prep` decodes the raw archives, resamples, joins timeframes causally and builds features and labels into a `.rcsb` store. `rcsr import` writes a store from a CSV.
3. `rcsr panel` builds point-in-time membership, the panel store, costs and the lockbox declaration. Membership uses only data before each rebalance, so delisted coins stay selectable while they traded.
4. `rcsr enrich` appends computed columns to a store without rebuilding it. It also declares the lookback embargo for the lockbox.
5. `rcsr audit` runs the leakage detectors. They compare a pipeline run on inputs that differ only where the answer must not change.
6. `rcsr verify-manifest` checks a finished run against its recorded binary, data and configuration.

The `.rcsb` store is a versioned binary column store with a header the loader can read without scanning the data. A label column can carry a target spec sidecar, `<data stem>.targets.json`, that declares the horizon, the entry offset and the cost. The memory guard refuses an input that does not fit, before any resident copy is built.

Details:
* [`docs/data-pipeline.md`](data-pipeline.md): which command owns which step.
* [`docs/data-format.md`](data-format.md): the store, the target spec and `--max-memory`.
* [`docs/data-preparation.md`](data-preparation.md): missing values, outliers, smoothing and scaling.
* [`docs/panel.md`](panel.md): membership, panel build, costs and the lockbox.
* [`docs/ohlcv.md`](ohlcv.md): bars, resampling, features and look-ahead traps.
* [`docs/leakage.md`](leakage.md): the detector suite behind `rcsr audit`.
* [README, "Acquiring the Binance Vision archive"](../README.md#acquiring-the-binance-vision-archive): the `rcsr-acquire` commands.

## Protocol and lockbox

A fit on ordered data looks good easily and means little. The protocol makes the evidence honest by fixing the rules before any search runs.

Splits are time-ordered. Training comes before validation, and validation before test. A purge removes rows whose label window reaches into the next block, and an embargo guards the far side of a block. With a target spec, the purge comes from the label horizon. `--time` and `--group-by` name the time column and the symbol column of a panel.

`rcsr walkforward` evaluates a formula on a sequence of windows with a long/flat position rule, costs and trading metrics. The fixed fold table is [`protocol/windows.csv`](../protocol/windows.csv), written by `rcsr protocol windows`. The feature-family ablation is defined in [`protocol/ablation.json`](../protocol/ablation.json) and run with `rcsr ablation`. `rcsr ladder` scores the simple baselines on the same folds and costs.

The lockbox is a final block of time that no development step may read. Every guard refuses a fit, a validation slice or a window that touches it, including the purge before it and the lookback embargo after it. `--lockbox none` is the only opt-out, and the report records it. Only a sealed read opens it: `rcsr protocol seal` freezes the candidate list and records its hash in a ledger, and a walk-forward with `--locked-test` reads the block once. A second read is refused.

Run integrity backs this up. A manifest names the binary, data and configuration of a run. Checkpoint writes are atomic. A seed contract keeps runs reproducible, and `--seed-expr` starts a search from a known formula.

Details:
* [`docs/time-series.md`](time-series.md): the checklist to run before believing a result.
* [`docs/data-splitting.md`](data-splitting.md): split modes, purge and embargo.
* [`docs/leakage.md`](leakage.md): differential leakage detectors.
* [`docs/walkforward.md`](walkforward.md): windows, costs, metrics, the ledger and the sealed read.
* [`docs/overfitting.md`](overfitting.md): diagnostics, null controls and multiple testing.
* [`docs/run-integrity.md`](run-integrity.md): manifests, atomic output and launchers.
* [`docs/seeds.md`](seeds.md): seeding a search with prior solutions.

## Pilot runbook

The pilot is a long-only crypto study that uses every part above. A stepped runner, `run.sh` with a `runbook.md`, lives in the git-excluded work folder at `arbeidsmappe/crypto-rcsr/pilot/v2/runbook/`. It is not in the repository and is not linked from here. Its header comment states the contract.

The runner executes numbered steps in order. A step that has a done-file is skipped, and `FORCE=1` reruns it. `run.sh list` prints the steps and their state. `PROFILE=smoke` runs the same commands on a small slice with small budgets, to test the runner before a real run.

Every development step reads a development-only store that holds no row at or after the lockbox start. Only the `seal` step and the single `sealed` step open the full store, and both refuse to run without `SEALED=1`. `seal` registers the chosen candidate and seals the ledger. `sealed` is the one locked-test read, and it is last.

GPU work runs one command per lock hold. The runner counts kernel Xid errors before and after each GPU command and stops when the count rises. A search runs in chunks, one hold per chunk, with a pause between them so other jobs can take the lock. A resume refuses a changed thread count, so the runner pins the CPUs it uses. Each command logs under `/mnt/lager/tmp/rcsr-gates`.

Details:
* [`docs/walkforward.md`](walkforward.md): the ledger, the seal and the sealed read the runner drives.
* [`docs/checkpoint.md`](checkpoint.md): the resume rules behind the chunked search.
* [`docs/run-integrity.md`](run-integrity.md): the manifest each step verifies.

## Agent workflow

Work follows one rule: one issue, one worktree, one branch, one pull request. The branch is `issue/<n>-<slug>` and the worktree lives under the git-ignored `worktrees/`. No agent commits to `main`. Every non-trivial change starts from a GitHub issue with acceptance criteria.

Pull requests target the integration branch `integration/gpu-hardening`, not `main`. A separate clerk agent checks the evidence of each pull request before the orchestrator merges it there. The full gate, build plus tests, runs once on the integration branch after all tickets are merged. Only then does the branch go to `main`, on the lead's instruction.

The machine is shared with a desktop, so builds and tests queue. Builds run in a build slot, up to three machine-wide. Tests and every GPU run take the single test lock. Both run at low priority (`SCHED_IDLE`, nice 19, idle I/O) and give way to the desktop. A ticket runs only the tests it adds or touches. Each worktree keeps its own `target/`.

Code is Rust only. Numerical experiments, fixtures and helpers are Rust, never Python, and a test guards this. Docs describe the present state and commits carry the history.

The workshop in [`verksted/`](../verksted/AGENTS.md) holds the recurring routines as numbered pipelines: issue, research, orchestrate, gate, diagnose, data, long run, close, handoff and machine.

Details:
* [`AGENTS.md`](../AGENTS.md): the non-negotiables for agents.
* [`docs/issue-workflow.md`](issue-workflow.md): issues, worktrees, pull requests, the lock and parallel work.
* [`verksted/AGENTS.md`](../verksted/AGENTS.md): the workshop and its routing.
