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
| `--save-every <N>` | generation interval for `--save` (default `25`, requires `--save`) |
| `--resume <file>` | continue from a checkpoint |

A checkpoint is written at a **generation boundary**, after the generation's
evolution, migration and island restarts have all completed. The final write on
exit happens before the full-data archive rebuild, so the saved archive is the
search archive at the last completed generation. Saving only reads the state; it
never consumes a random number or mutates a population, so `--save` alone cannot
change the result.

`--save` and `--resume` apply to the main search. Auxiliary searches
(`--cv`, `--stability`, `--null-runs`, `--report benchmark`) are reported as
usual but are not checkpointed.

## Format, version 1

The format is plain UTF-8 text, one record per line. It is hand-written and
read by the engine itself; no third-party parser is involved.

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
| `config <key> <value>` | one search setting. All settings that influence the trajectory are present; `complexity.op_costs` carries one cost per `Op`, in discriminant order. `units`, `units.target` and `require_dimensions` are optional: they are written only when `--units` is active, and a checkpoint without them loads as units-free. |
| `generations <n>` | completed generations (the next generation to run). |
| `stopped_early <bool>` | the saved run had already converged, hit its patience limit or its time budget. Resuming it does not run more generations. |
| `best_prev <f64>` | lowest error seen so far, used by the patience comparison. |
| `no_improve <n>` | consecutive generations without improvement. |
| `data train <rows> <hash>` | fingerprint of the training sample. |
| `data test <rows> <hash>` | fingerprint of the test sample. |
| `archive <members> <cap>` | number of Pareto members and the archive capacity. |
| `sol <complexity> <cost> <error> <tree>` | one archive member. |
| `island <i> <stagnant>` | per-island stagnation counter. |
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
older versions, but a v1 checkpoint is never silently reinterpreted.

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

Limits:

* **Wall-clock budgets are not reproducible.** A saved time budget is still
  enforced, but the clock is measured from the moment the resumed process
  starts, not from the original start. Two runs with the same checkpoint and
  `--time-budget` can therefore stop at different generations, and a run that
  stopped because of the budget is saved with `stopped_early true` (resuming it
  does not run more generations). For bit-identical comparisons, leave
  `--time-budget` unset.
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
