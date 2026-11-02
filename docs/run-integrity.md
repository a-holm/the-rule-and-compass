# Run integrity: manifests, atomic output and reproducible launchers

A long run is evidence only if you can name the binary, the data and the
configuration that produced it. Issue #140 adds four guarantees:

1. a checkpoint records its **backend** (`cpu`/`gpu`) and a canonical
   **config hash**, and `--resume` refuses a silent backend change;
2. every checkpoint/metadata write is **atomic** (temp file, `fsync`, rename)
   and the previous checkpoint generation is kept as `<path>.prev`;
3. `--run-manifest <path>` records a verifiable JSON manifest at the start and
   updates it atomically at the end;
4. the final JSON result can be published atomically with `--json-out <path>`,
   and progress is never mixed into the final document.

See [`checkpoint.md`](checkpoint.md) for the checkpoint format and the resume
rules, and [`seeds.md`](seeds.md) for the deterministic seed contract.

## Run manifest

```sh
rcsr --data data.csv --target y --vars x --seed 7 --gpu \
     --generations 500 --run-manifest run.manifest.json \
     --json-out run.result.json --progress
```

The manifest is written before the search starts (so a killed process still
leaves evidence) and rewritten atomically at the end with `end_utc` and phase
timings. It contains:

| field | meaning |
|---|---|
| `tool`, `version` | engine name and crate version |
| `start_utc`, `end_utc`, `completed` | UTC timestamps; `start_utc` is the process start (before the data is read and hashed) and `end_utc` the end of the process work, after baselines and diagnostics (#256); `end_utc` is `null` until the run finishes |
| `commit`, `dirty` | #256: the commit the binary was built from, recorded by `build.rs` when the binary is built (`RCSR_BUILD_COMMIT` overrides it for a source export), never by a runtime `git` call; `dirty` is the work-tree state when the build script last ran. The binary SHA-256 is the identity; the commit is a label. |
| `binary.path`, `binary.sha256`, `binary.build_commit` | the actual `current_exe`, its SHA-256 (two CUDA-linked binaries at the same commit can differ) and the commit compiled into it |
| `data.path`, `data.sha256` | raw data file and its SHA-256 (`null` for stdin) |
| `data.full_rows`, `data.sampled_rows` | #112: `true`/`null` for a full load, `false`/`N` for an explicit `--sample-rows N` sample, `null`/`null` for a CSV |
| `data.estimate_bytes` | #112: the memory estimate that justified the load decision (`null` for a CSV) |
| `data.train`, `data.test` | prepared-sample row count and FNV-1a fingerprint, exactly as the checkpoint records them |
| `config.hash`, `config.seed`, `config.canonical` | the canonical configuration text, its FNV-1a hash and the search seed |
| `backend`, `device` | `cpu`/`gpu`, device name, driver version and CUDA version |
| `dispatch` | #290: the calibrated `r_eval`/`r_fit`/`bw`/`launch` values, probe wall time, source and band, plus the per-stage route ledger (`route`, `reason`, `calls`, `candidates`, `predicted_s`, `actual_s`, `bytes`, `switches`) |
| `fit_subsample.fit_rows`, `fit_subsample.hash`, `fit_subsample.fixed`, `fit_subsample.scored_rows` | present with `--fit-rows`/`--fit-samples` below the training rows: the subsample size, the FNV-1a hash of its ascending row indices (generation 0 for the CPU screening arm), whether the device fits the same rows for the whole run, and the rows every candidate is scored on; `fit_subsample.source` is `default` when the row-count default chose the size (#602) and `flag` when `--fit-rows`/`--fit-samples` did |
| `resume.from`, `resume.backend_change` | checkpoint path and an allowed backend change, if any |
| `phases_seconds` | `data_prepare`, `search`, `baselines`, `diagnostics`, `report`, `total`. `total` is exactly `timing.total_seconds` of the result JSON: the one runtime of the process (#256). `search` is `elapsed_seconds`. |

`rcsr verify-manifest <path>` re-checks the manifest against the **current**
environment:

* the SHA-256 of the running `rcsr` binary, or of the file named by
  `--binary <file>` (#256: the preserved binary of a finished run; the commit
  line then reports the manifest's build-time label and the SHA-256 is the
  proof);
* the commit compiled into the running binary (no `git` call);
* the data file named in the manifest (or `--data <file>`) by SHA-256;
* the canonical configuration text by recomputing its hash.

It prints one `PASS`/`FAIL` line per check and exits `0` only if every check
passes, `1` on any mismatch, `2` on a usage error. A stale binary, changed data
or edited configuration therefore fails loudly instead of being mistaken for a
reproduction.

```console
$ rcsr verify-manifest run.manifest.json
PASS binary sha256 9c1f...
PASS commit 4e2a...
PASS data sha256 data.csv 7b31...
PASS config hash 4f0c2a1c74b1e0a9
PASS manifest run.manifest.json
```

## Atomic publication

All engine-owned files use the same pattern: write `<path>.tmp`, `fsync`,
rename over `<path>`; the destination is never a partially written stream.

* `--save` checkpoints retain `<path>.prev`;
* `--run-manifest` is published at start and updated at the end;
* `--json-out` is the final JSON object; without it, `--json` prints one object
  to stdout;
* `--out` (the Pareto TSV) is atomic too.

A killed process leaves the old file (and, for a checkpoint, `.prev`) intact.
The output stream rules are unchanged: stdout carries the machine-readable
result, stderr carries the human log. `--progress` writes generation lines to
stderr, or NDJSON lines to stdout when `--ndjson` is selected; the final result
is never interleaved with progress in the same file.

## Reproducible launcher contract

A launcher should:

* choose `(label, seed)` and derive the numeric seed with a **stable** hash —
  never a per-process salted hash;
* state the backend explicitly (`--gpu` or `--cpu`), even when CPU is the
  default, and pass the same one on resume;
* record the exact command line and the resolved seed;
* pass `--run-manifest`, `--json-out` and `--progress`;
* name one development log per run — `--dev-log <label>.dev.ndjson` (the engine
  writes `rcsr.dev.ndjson` by default, which several windows in one directory
  would share) — see [`dev-log.md`](dev-log.md);
* keep stderr in a file rather than `/dev/null`;
* resume with `--resume checkpoint.rcsr` and, only if intentional, with
  `--allow-backend-change`.

Two tracked examples implement the contract:

* [`scripts/run-long.sh`](../scripts/run-long.sh) — one uninterrupted or
  resumed run; requires `--gpu`/`--cpu` and writes a manifest, atomic result,
  command record, progress log and a per-label development log.
* [`rcsr-launch`](../my-version/src/bin/rcsr_launch.rs) (`my-version/target/release/rcsr-launch`) — several
  walk-forward windows; derives `SHA-256(label:seed)`, reuses the same backend
  on resume, and writes one manifest/log/result set per window. `--dry-run`
  prints the resolved commands without searching.

The scratch launchers in the git-ignored `arbeidsmappe/crypto-rcsr/pilot/`
directory are **retired**. Their classification and the Rust command that
replaces each one are in
[`python-pipeline-retirement.md`](python-pipeline-retirement.md); one-off probes
are archived under `pilot/archive/` and must not be used. In particular, do not
resume a long run with them (per-process salted seeds, `/dev/null` stderr, plain
`--json` redirected to a file without a manifest). Use the tracked launchers
above, or the `rcsr walkforward` command directly. The repository contains no Python; `my-version/tests/no_python.rs` fails on a
tracked `.py` file or a `python` shebang.

`my-version/tests/launchers.rs` enforces the contract on every tracked
`scripts/*.sh` and on `rcsr-launch`: no `DEVNULL`, an explicit backend, and
`--run-manifest`/`--json-out`/`--progress`. It also runs `run-long.sh`
end-to-end and checks the manifest with `rcsr verify-manifest`.

## A long run, start to finish

The flow the tracked launchers implement, and the one to use for a multi-hour
run (the "10-hour run"):

1. **Launch** with an explicit backend, a manifest, an atomic result, a
   checkpoint and a per-run development log:
   `scripts/run-long.sh ... --gpu` (or the per-window
   `rcsr-launch`).
2. **Watch** the development log instead of an external monitor:
   `tail -f runs/<label>.dev.ndjson` shows a summary event every 10–25
   generations (error, front size, phase times, fit/GPU/fallback counts, GPU
   duty). The old scratch `monitor_10h.py` is not part of this flow; it is
   tracked for deletion in #200. 1 Hz *process* sampling (CPU, RSS,
   `nvidia-smi`) is still an outside-the-engine job, and `run_end` records the
   process peak RSS and measured device-hours.
3. **Resume** on the same backend with `--resume` when the wall-clock budget
   ends (a `time_budget` stop is not terminal, and `--time-budget` may be set
   again); the log is truncated per run, so a resumed run gets its own file.
   The checkpoint records the build identity; `--resume` refuses a different
   binary, commit, device, driver or thread count unless `--allow-build-change`
   is given. A run with a target spec also records the spec sha256;
   `--resume` with a different spec (or with none, or onto a checkpoint
   written without one) is refused with exit 2 and a message naming both
   hashes, before the data fingerprint is compared.
4. **Explain** the finished run with `rcsr dev-report runs/<label>.dev.ndjson`:
   phase table, fit vs evaluation share, fallback causes, convergence, archive
   growth and a ranked bottleneck list. Use `--dev-log-level full` when the
   per-generation detail is needed (`full` keeps every generation).
5. **Verify** with `rcsr verify-manifest runs/<label>.manifest.json`.

## What the result fields measure (#256)

| field | scope |
|---|---|
| `timing.total_seconds` | the runtime: the whole process from argument parsing to the end of the report phase. `prep + search + baselines + diagnostics + report + unaccounted_seconds = total_seconds`. |
| `elapsed_seconds`, `timing.search_seconds` | the search phase only |
| `gpu.eval_kernel_duty` | kernel seconds over the evaluator window (`gpu.eval_window_s`); it says nothing about the rest of the run |
| `gpu.ledger.duty_run` | device-busy union over the run wall |
| `gpu.device_seconds_incl_fit` | evaluator full-path wall plus fit kernel seconds |
| `gpu.device_fraction_of_run`, `gpu.eval_fraction_of_run` | the above over `timing.total_seconds`, as value, numerator, denominator, units, scope |
| `gpu.batches_per_s` | evaluator batches per evaluator second. It is not a generation rate. |
| `gpu.quality_per_gpu_hour` | decades of error reduction against the constant-mean baseline per evaluator-window hour; both errors are MSE (the baseline is the training-row variance). `null` for any metric but squared. |
| `converged_early`, `stop_reason` | see [`convergence.md`](convergence.md): a time-budget stop is not convergence |

## Acceptance checks

The integration branch runs the tests that pin this behaviour:

* `tests/checkpoint.rs`: backend/config-hash round trip, backend-mismatch
  refusal and override, injected partial write leaving the previous generation
  intact, `.prev` retention;
* `tests/report_integrity.rs` (#256): the manifest carries the build-time
  commit and the process start; `manifest.total == timing.total_seconds`;
  `verify-manifest --binary` passes on a copied binary; a time-budget stop is
  resumable and adds generations; the exit checkpoint's final archive hash
  equals the reported one;
* `tests/manifest.rs`: manifest pass, stale binary hash, changed data and
  changed config all fail the verifier; `--json-out` is one atomic JSON object
  and never streams to stdout.
