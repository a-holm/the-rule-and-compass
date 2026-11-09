# Save and resume a run

`rcsr` can write its complete search state to a checkpoint and continue from it
later. Resuming is **not** a heuristic restart: the saved random-number states,
populations, archive, patience counters and completed generation are restored
exactly, so a split run and an uninterrupted run produce the same Pareto front,
bit for bit.

```sh
# run 20 generations and save on exit (and every 25 generations by default)
rcsr --data examples/line.csv --target y --vars x \
     --generations 20 --seed 7 --save run.rcsr

# continue the same run for another 20 generations
rcsr --data examples/line.csv --target y --vars x \
     --generations 40 --seed 7 --resume run.rcsr
```

## Options

| option | meaning |
|---|---|
| `--save <file>` | write a checkpoint on exit and every `--save-every` generations |
| `--save-every <N>` | generation interval for `--save` (default `25`, requires `--save`). A checkpoint is also written when 15 minutes of wall time have passed since the last one (#256; `RCSR_CHECKPOINT_INTERVAL_S` overrides the 900 s). |
| `--resume <file>` | continue from a checkpoint, refusing a data or backend mismatch |
| `--allow-backend-change` | permit a CPU↔GPU resume, recorded on stderr and in the run manifest |
| `--interim-chunk` | #855: a chunk that stops on `--time-budget` saves its state and skips the end-of-run report. Needs `--save` and `--time-budget`. See [Interim chunks](#interim-chunks) |

A checkpoint is written at a **generation boundary**, after the generation's
evolution, migration and island restarts have all completed. The final write on
exit happens before the full-data archive rebuild, so the saved archive is the
search archive at the last completed generation. Saving only reads the state; it
never consumes a random number or mutates a population, so `--save` alone cannot
change the result.

Each save is **atomic**: the new text is written to `<file>.tmp`, flushed with
`fsync`, and renamed over `<file>`. The previous good generation is retained as
`<file>.prev` (hard link where possible, copy otherwise). A process killed
during a save therefore leaves the last complete checkpoint readable; the
fault-injection test in `tests/checkpoint.rs` proves this.

`--save` and `--resume` apply to the main search. Auxiliary searches
(`--cv`, `--stability`, `--null-runs`, `--report benchmark`) are reported as
usual but are not checkpointed.

## Interim chunks

A search run in chunks (`--time-budget` per process, `--resume` between them) pays the end-of-run report at every chunk end, although only the last report is read. The saved state is captured before the report, so the report cannot change the next chunk. `--interim-chunk` skips it.

When a chunk stops on `--time-budget` under `--interim-chunk`:

- The checkpoint is written with the search state and no `final` archive.
- The exact finalist refit, prune, test scoring, baselines, diagnostics, gates, QD refit and the report do not run.
- `--json-out` receives a short document: `tool`, `version`, `interim` (`true`), `backend`, `backend_reason`, `device`, `threads`, `seed`, `stop_reason` (`time_budget`), `generations_run`, `elapsed_seconds`, `timing`, `gpu` and `checkpoint`. It has no `pareto` array.
- The development log `run_end` and the run manifest carry `interim true`. Its archive fields describe the search archive, not a reported front.
- `rcsr walkforward --from-run`, `--front` and every other reader of a result file refuse the short document by name (exit 2 for `--from-run`). `rcsr verify-manifest` refuses an interim manifest with exit 2.

A chunk that stops for another reason (`generations`, `patience`, `solved`, `front_stable`) is the last one in fact and writes the full report. The flag is an execution setting: it is not saved and not part of the config hash.

The last chunk is run without the flag. Its checkpoint, archive and `run.json` equal those of the same chunks run without the flag, apart from timing fields. `tests/checkpoint.rs` pins the checkpoint bytes and the report across interim chunks, plain chunks and one uninterrupted run.

### Cost of a chunked run

The checkpoint carries no timestamp, so a prefix run reproduces it byte for byte. The cost of the chunks before this process lives in `<save path>.cost`, written atomically (temp file and rename) in the same step as every checkpoint save: `{"v":1,"generation":G,"chunks":N,"search_seconds":S,"ledger_wall_s":W,"device_seconds_incl_fit":D}`. `--resume` reads the sidecar of the `--resume` path and uses it only when `generation` equals the checkpoint generation. A missing, unreadable or other-generation sidecar (a checkpoint from before this file existed, or a crash between the two writes) gives `search.cumulative` `{"complete":false}`, and the chain stays incomplete. A save inside a running chunk records `ledger_wall_s` and `device_seconds_incl_fit` as null; the exit save of the chunk records them from the GPU ledger (null for a CPU run).

Scope of the result fields after a chunked run:

| field | scope |
|---|---|
| `generations_run`, `checkpoint.generation`, `search.candidates_evaluated` | cumulative |
| `elapsed_seconds`, `timing.*`, `gpu.ledger`, `gpu.generations`, `gpu.totals`, `gpu.evals`, `gpu.batches`, `gpu.quality_per_gpu_hour` | last process only |
| `search.cumulative.chunks`, `.elapsed_seconds`, `.ledger_wall_s`, `.device_hours` | all chunks (`chunks` is 1 without `--resume`) |
| `gpu.quality_per_gpu_hour_run` | best error of the cumulative archive over `search.cumulative.ledger_wall_s` when known, else over this process's ledger wall |
| dev log `run_end` `wall`, `search`, `device_hours`, `run_wall_s` | last process only (one dev log per chunk) |
| dev log `run_end` `search_cumulative` | all chunks; `rcsr dev-report` prints it as "search wall (all chunks)" |

`elapsed_seconds` keeps its per-process meaning; `search.cumulative` carries the total. `rcsr ladder --sr-run` takes the SR rung wall from `search.cumulative.elapsed_seconds` when `complete` is true (`equal_cost.sr_wall_basis` is `cumulative`), else from `elapsed_seconds` (`process`). `--sr-wall` overrides both and leaves `sr_wall_basis` null. The interim `run.json` carries the same `search.cumulative` block.

If the run stops early and the last chunk was an interim chunk, no report exists. `rcsr ... --resume <file> --time-budget 0` without `--interim-chunk` runs one more generation and writes the full report.

## Format, version 1

The format is plain UTF-8 text, one record per line. It is hand-written and
read by the engine itself; no third-party parser is involved.

Every encoded tree is bounded to **4096 nodes** and a root-to-leaf **depth of
256** (a single node has depth 1). `decode_tree` returns an error such as
`checkpoint tree depth 257 exceeds the limit of 256` or `checkpoint tree has
more than 4096 nodes`; it never recurses without a bound.

```text
rcsr-checkpoint v1
config generations 40
config islands 1
config pop 200
config max_complexity 30
config complexity.op_costs 1.0 1.0 ... 1.0          # 49 values, one per Op
config complexity.var_weight 1.0
config complexity.const_weight 1.0
config complexity.depth_weight 0.0
config metric absolute
config seed 7
config ops 2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,27,28,29,30,31,36
config migration_interval 25
config restart_interval 60
config init_depth 3
config fit_iters 8
config dedup true
config time_budget none
config patience 40
config predictor_samples none
backend cpu
config_hash 4f0c2a1c74b1e0a9
generations 20
stopped_early false
best_prev 1.9968344769643271
no_improve 0
data train 30 5091143174517734903
data test 10 6767427338702826278
archive 1 4000
sol 2 2.0 1.9968344769643271 11 0.0 1 1 0.0 0
rng main 8 2443250962 1093594115 ... 10512            # 624 u32 words
island 0 0
rng 0 12 315011378 977967129 ... 30                  # 624 u32 words
ind 0 1.9968344769643271 2.0 11 0.0 1 1 0.0 0
```

Every record:

| record | contents |
|---|---|
| `config <key> <value>` | one search setting. All settings that influence the trajectory are present; `complexity.op_costs` carries one cost per `Op`, in discriminant order. `units`, `units.target` and `require_dimensions` are optional: they are written only when `--units` is active, and a checkpoint without them loads as units-free. `front_parents` is optional too: it is written only when `--no-front-parents` is used, and a checkpoint without it loads with the Pareto-front feedback on. `reject_constant_offspring` is optional too: it is written only as `true` when `--reject-constant-offspring` is used, and a checkpoint without it loads with the filter off. `reject_fixed_position` is optional in the same way: it is written only as `true` when `--reject-fixed-position` is used, and a checkpoint without it loads with the filter off.  `mutation_mix`, `crossover_internal` and `adaptive_mix` are optional too: each is written only when it differs from `classic`, `0` and `false`. The QD settings (`qd_bins`, `qd_cell_cap`, `qd_cvt`, `qd_policy`) are optional and written only when the layer is enabled; a checkpoint without them loads with QD off. `selection` is optional: it is written only for `--selection afp`, and a checkpoint without it loads as `lexicographic`. An `afp` checkpoint also carries one `ages <island> <age>...` record per island, in population order; a lexicographic checkpoint has none. |
| `backend <cpu\|gpu\|unknown>` | evaluator that produced the state (#140). New checkpoints always write `cpu` or `gpu`; `unknown` only appears when reading a checkpoint from before this record existed. |
| `config_hash <hex>` | FNV-1a 64-bit hash of the canonical `config` lines, excluding the `generations` budget (resuming may raise it). The loader recomputes it and refuses a checkpoint whose configuration text does not agree. |
| `generations <n>` | completed generations (the next generation to run). |
| `stopped_early <bool>` | the saved run declared itself finished: it reached the solved floor or its patience limit. Resuming it does not run more generations. A time-budget stop is **not** terminal (#256): it is saved with `stopped_early false`, so `--resume` continues it and `--time-budget` may be set again. |
| `best_prev <f64>` | lowest error seen so far, used by the patience comparison. |
| `no_improve <n>` | consecutive generations without improvement. |
| `data train <rows> <hash>` | fingerprint of the training sample. |
| `data test <rows> <hash>` | fingerprint of the test sample. |
| `error_kind <f32_proxy\|f64\|unknown>` | #256: what the `sol` and `ind` errors are. A GPU search stores `f32_proxy` (optimistic by up to a few `1e-7` relative); a CPU search stores `f64`. |
| `target_spec <sha256>` | #345: sha256 of the declared target spec of the writing run. Absent without a spec. `--resume` refuses a different value, or a missing one on either side, with exit 2 naming both hashes. |
| `build <field> <value>` | #256: the process that wrote the file: `version`, `commit` (recorded when the binary was built), `binary_sha256`, `device`, `driver`, `threads` (the CPU count of the process, `available_parallelism`). `--resume` refuses a mismatch unless `--allow-build-change` is given, except in `threads`: a different CPU count prints a `# checkpoint: build identity changed (threads: saved 8 -> current 4)` notice and resumes (#860). A checkpoint without `build` records loads and is not checked. There is no timestamp, so a prefix run reproduces an earlier checkpoint byte for byte (apart from `config generations`). |
| `archive_hash <hex>` | #256: `Archive::hash` of the `sol` members, verified on load. |
| `archive <members> <cap>` | number of Pareto members and the archive capacity. |
| `sol <complexity> <cost> <error> <tree>` | one archive member (the search archive, with `error_kind` errors). |
| `final <members> <cap> <hash>` | #256: header of the post-refit archive. Present only in the exit checkpoint. `<hash>` is the `Archive::hash` the run reports (`archive_hash` in the dev log, `hash` in `run_end`); the loader verifies it. |
| `fsol <complexity> <cost> <error> <tree>` | #256: one post-refit member, in reported order, with the exact CPU `f64` errors of the result. A resume ignores the block and continues from `sol`. |
| `island <i> <stagnant>` | per-island stagnation counter. |
| `mix <i> <numbers>` | #471: the adaptive edit counts of island `i` (`--adaptive-mix`), oldest generation first, 14 numbers per generation (attempts and successes for each of the seven edit kinds), at most 10 generations. Written only when the island has counts. |
| `rng <i\|main> <idx> <624 words>` | exact MT19937 state (state array and next index). `main` is the RNG that built the initial populations. |
| `ind <i> <error> <complexity> <tree>` | one population member, in population order. |

Floating-point values are written with Rust's shortest round-trip `Debug` form
(`1.9968344769643271`, `-0.0`), with `inf`, `-inf` and `NaN` for non-finite
values. Reading a value back yields the identical bit pattern.

### Tree encoding

A tree is a prefix token sequence:

```text
op_index value arity child_0 ... child_{arity-1}
```

`op_index` is the `Op` discriminant (`Const = 0`, `Input = 1`, …,
`Constrain = 48`). For example, `cosh(x)` is:

```text
11 0.0 1  1 0.0 0
```

which reads as "operator 11 (`Cosh`), value `0.0`, one child: operator 1
(`Input`), value `0.0`, no children". The loader checks that the declared arity
equals the operator's arity, that every operator index is in range, that the
tree is complete and that no tokens trail the root. `expr::to_string` is not
used anywhere in the format: it rounds constants and would corrupt a run.

## Versioning rule

The first line is the version marker. This build reads and writes exactly:

```text
rcsr-checkpoint v1
```

Any change that is not backward-compatible (new/removed fields, changed meaning,
different tree encoding) bumps the version to `v2`; a `v1` reader refuses it with
`unsupported checkpoint version ...; this build reads 'rcsr-checkpoint v1'`.
Within `v1`, unknown records and unknown configuration keys are errors rather
than being ignored, and duplicate records are errors. A future reader may accept
older versions, but a v1 checkpoint is never silently reinterpreted. The
`backend` and `config_hash` records added in #140 are optional on read: a v1
checkpoint written before them loads with `backend unknown` and a freshly
computed hash, while every new checkpoint writes both.

## Data fingerprint

The fingerprint is the row count plus an FNV-1a 64-bit hash over the dimensions
and every stored `x`, `y` and weight value (using `f64::to_bits`). Row order and
weights are part of the hash. Both the training and test samples are recorded.

On resume the fingerprints are recomputed from the prepared samples. If either
differs, the run stops with an input error (exit code `3`) such as:

```text
checkpoint training data mismatch: saved 240 row(s) with fingerprint
0a1b..., current data has 240 row(s) with fingerprint 9f3c...; refusing to
resume a search on different data
```

This also catches preprocessing changes (`--test-frac`, `--split`, `--fill`,
`--scale`, …) that produce different prepared rows, even when the input file is
the same.

## Override rule

The saved configuration is authoritative. On `--resume`:

* every saved search setting is used, and an explicitly supplied flag that
  contradicts it is a **usage error** (exit code `2`) naming the flag and the
  saved value;
* omitting a flag is always fine;
* `--generations` is the one exception: it is the **budget**, so it may be
  raised to run further. It may not be lowered below the saved budget or below
  the number of completed generations. Without `--generations`, the saved budget
  is used.

Non-search flags (input file, column names, output format, reports, `--top`, …)
are free to change, subject to the data fingerprint. `--save` may be combined
with `--resume`; saving to the same path overwrites it at the next boundary.

Execution settings are not saved. A resumed run takes them from its own command line: `--gpu`, `--cpu`, `--dispatch`, `--pipeline-depth` (default `2`, not the earlier chunk's value), `--threads` (and the CPU count of the process), `--gpu-memory`, and the post-loop options `--prune`, `--prune-rows`, `--prune-budget`, `--fit-report` and the selection rule (`--select-rule`, `--select-blocks`, `--select-se`, `--select-block-len`; `--reselect` re-selects a finished checkpoint without a search, see overfitting.md). None of them changes the search trajectory. Settings that do change it, such as `--fit-polish`, `--fit-irls` and `--fit-rows`, are saved and a mismatch is refused.

### Backend identity

The saved `backend` is checked before the search starts. A GPU checkpoint
resumed without `--gpu` (or a CPU checkpoint resumed with `--gpu`) is refused
with exit code `3`:

```text
checkpoint backend mismatch: saved run used 'gpu', this run uses 'cpu';
pass --allow-backend-change to resume across backends (the trajectory and
reported errors will differ)
```

`--allow-backend-change` is the only override. It prints
`# checkpoint: backend changed gpu -> cpu (allowed by --allow-backend-change)`
and records the pair under `resume.backend_change` in `--run-manifest`.
This check exists because a GPU run saves f32 proxy fitness; continuing it with
the CPU f64 evaluator would silently select against a different objective.
A checkpoint written before this record existed loads as `backend unknown` and
is allowed to resume (there is no saved identity to check).

## Determinism guarantee and its limits

Given the same data, seed, configuration and a checkpoint taken at a generation
boundary, the resumed run produces the **same archive**: the same expressions,
the same error and cost bit patterns, in the same order, and the same completed
generation count as an uninterrupted run. `tests/checkpoint.rs` checks exactly
that for 40 generations versus 20 + 20.

The guarantee holds because the checkpoint carries all of the trajectory state:

* the complete `Config` (including the complexity model and predictor-sampling
  settings);
* the completed generation, per-island populations and their exact errors and
  complexities;
* each island's MT19937 state and the main initialization RNG;
* the stagnation counters and the patience bookkeeping (`best_prev`,
  `no_improve`);
* the archive, with its members and capacity.

The CPU count and `--threads` are not part of the trajectory. Every stage that
reads the thread count hands out independent items and writes each result at its
own index: batch scoring and constant fits (`par_map`), the exact CPU fallback
fits, the device-proposal gate, the finalist refit, the screened refit and the
k-NN baseline queries. Islands are one thread each, and their RNG streams come
from the seed and the island index. The island count is `--islands`, a saved
setting, never derived from the CPU count. `tests/checkpoint.rs`
(`search_results_do_not_depend_on_the_thread_count`,
`a_device_search_does_not_depend_on_the_thread_count`,
`resume_under_another_cpu_count_is_allowed`) checks the archive, test errors and
checkpoint bytes at 1, 2, 3 and 12 threads and across a resume from 12 to 1.
The one place `--threads` meets a routing rule is the start of a fresh run: with
`--fit-strategy auto` it enters the device fit cost check once (`--threads` when
given, else 8). The resolved strategy is saved, and a resume does not resolve it
again.

The time budget is saved with the checkpoint (`config time_budget`). A `--resume` without `--time-budget` runs with the saved budget, not with none; `--time-budget` replaces it for this process, and the next checkpoint saves the budget in force. A resume prints `# checkpoint: time budget <s> s (saved with the checkpoint)` or `(from --time-budget)`, and nothing when no budget applies.

Limits:

* **Wall-clock budgets are not reproducible.** The budget is a limit of one
  process: the clock starts when the resumed process starts. Two runs with the
  same checkpoint and `--time-budget` can therefore stop at different
  generations. A run that stopped because of the budget is resumable (#256):
  the generation in progress finishes, the checkpoint is saved with
  `stopped_early false`, and `--resume` adds generations. `--time-budget` is
  not part of the saved search settings, so a resume may set a new budget. For
  bit-identical comparisons, leave `--time-budget` unset.
* Checkpoints are taken only at generation boundaries. Killing the process in
  the middle of a generation loses that generation; the last completed
  generation is intact.
* A different build whose operator set, metric or search code changed is not
  covered by the version guarantee. The version line changes when the format
  changes; algorithm changes are a new release, and a checkpoint should be
  restarted rather than assumed identical across releases.
* Data that is not bit-identical is refused by the fingerprint. There is no
  "close enough" mode.

## Exit codes

| situation | code |
|---|---|
| checkpoint cannot be read, is truncated, has a bad version/op index/arity, or has a duplicate/missing record | `3` (input) |
| data fingerprint mismatch | `3` (input) |
| saved `backend` differs and `--allow-backend-change` was not given | `3` (input) |
| saved `target_spec` differs from this run's spec (or one side has none) | `2` (usage) |
| saved `build` identity (binary, commit, version, device, driver) differs and `--allow-build-change` was not given | `3` (input) |
| `--resume` contradicts a saved search setting; `--save-every` without `--save`; invalid flag value | `2` (usage) |
| checkpoint cannot be written | `3` (input) |

## Worked save/resume transcript

```console
$ rcsr --data examples/line.csv --target y --vars x \
       --generations 20 --seed 7 --save run.rcsr
# checkpoint: saved to run.rcsr at generation 20
...
# elapsed: 0.42s | generations: 20 | pareto members: 14

$ head -1 run.rcsr
rcsr-checkpoint v1

$ rcsr --data examples/line.csv --target y --vars x \
       --generations 40 --seed 7 --resume run.rcsr
# checkpoint: resumed from run.rcsr at generation 20
# elapsed: 0.39s | generations: 40 | pareto members: 17
...
```

The resume command keeps the saved `seed`, `islands`, `pop`, metric, operator
list, complexity model and every other search setting; only the generation
budget grows. Passing a conflicting `--pop` (or a different seed, metric, …)
is refused before the search starts. Continuing without `--generations` simply
finishes the saved budget.

In `--ndjson` mode the same actions are machine-readable events:

```json
{"event":"checkpoint","action":"resumed","path":"run.rcsr","generation":20}
{"event":"checkpoint","action":"saved","path":"run.rcsr","generation":40}
```

and `--json` includes:

```json
"checkpoint": {"resumed_from": "run.rcsr", "saved_to": "run.rcsr", "generation": 40}
```

The text-mode `# checkpoint: ...` line goes to stderr, like the rest of the
human-readable log.
