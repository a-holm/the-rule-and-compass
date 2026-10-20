# Seeding the search with prior solutions

A user often knows part of the answer: the shape of a law, a textbook formula,
or an expression a previous run printed. `--seed-expr` and `--seed-file` give
that expression to the search instead of making it rediscover the shape from
random trees.

```sh
rcsr --data spring.csv --target period --vars m,k \
     --seed-expr "sqrt(m / k)" --generations 15
```

```sh
rcsr --data spring.csv --target period --vars m,k \
     --seed-file seeds.txt --seed-expr "2 * sqrt(m / k)"
```

Both flags are repeatable and may be combined. Every accepted seed is reported:

```text
# seeded: complexity=12 train_error=0.000000  y = sqrt(((-0 + m) / (0 + k)))
```

## What a seed is

A seed is a formula in **original units**, written exactly like `--eval-expr`
and like the right-hand side of a printed row. Variables are the names from
`--vars` (the `x0`, `x1`, … fallback also works); operators, constants and
function calls follow the parser grammar described in the README. Seeds are
read by the same recursive-descent parser as every other user formula, so a
syntax error names the 1-based column.

## Processing order

Each seed is processed once, in this order, **before generation 1**:

1. **parse** against the `--vars` names (a bad expression is exit code `2`);
2. **simplify** (`expr::simplify`) — folds constants and removes algebraic
   identities such as `+ 0`;
3. **fit** its numeric constants on the full training sample with
   `--fit-iters` Levenberg–Marquardt iterations (the same fitter offspring
   use; `--no-fit` leaves constants untouched);
4. **evaluate** its error on the full training sample under `--metric`;
5. **insert** it into the Pareto archive and into the initial population of
   **every** island, so the seed cannot be lost before migration.

Structurally identical seeds (for example `sqrt(m/k)` and `sqrt(m/k) + 0`) are
inserted once. The reported error and the archived error are the same number.

Because the search standardises inputs (and only inputs) before it runs, a seed
is rewritten into that space internally: each raw variable `x_i` becomes
`x_i * std_i + mean_i`. The reported `expression` is denormalised back to
original units, just like a Pareto row, so a printed seed can be pasted back
into `--eval-expr`.

## The elite rule: `--max-complexity`

`--max-complexity` governs **generated offspring**: mutation and crossover
discard candidates above the budget. A seed is not generated — it is an
**elite** the user supplied — so it is inserted even when its configured
complexity exceeds the budget. When that happens the run says so explicitly:

```text
# note: seeded expression 'sqrt((m / k))' has complexity 12 above --max-complexity 3; kept as an elite
```

The note is written to stderr in every output format. Complexity is measured
with the same configured model (`--op-complexity`, `--complexity-weights`,
`--depth-weight`) as every candidate, in the search's normalised-input
representation.

## Seed file format

`--seed-file <path>` reads plain text:

* one expression per line;
* a line whose first non-space character is `#` is a comment;
* blank (or whitespace-only) lines are ignored;
* there are no inline comments, so the whole remaining line is the expression;
* `--seed-file` and `--seed-expr` may be used together and are processed in
  command-line order (`--seed-expr` values first, then file lines in file
  order).

```text
# seeds.txt
# a coarse guess first
sqrt(m / k)

# and a scaled variant; the fitter will find the scale
2 * sqrt(m / k)
```

An unreadable file is an **input error** (exit code `3`); a malformed
expression is a **usage error** (exit code `2`) that names the flag (or the
file and line), quotes the offending text and carries the parser's column.

## Deterministic seed contract

A reproduction is only a reproduction if the search seed is derived the same
way every time. The contract is:

* a run is identified by `(label, seed)`, where `label` is a stable string
  chosen by the launcher (for example `btc-usd-2019h1`) and `seed` is an
  integer;
* the search seed is a fixed function of those two values, computed with the
  engine's deterministic hash. The engine's own `--seed` is used directly and
  echoed in every output format;
* **do not use Python's `hash(label)`**: it is salted per process
  (`PYTHONHASHSEED`), so the same label gives a different seed in each
  interpreter. The old scratch launcher did exactly that and a "reproduction"
  was therefore a different run;
* a launcher must write the resolved numeric seed and the exact command line
  into its log or run manifest, so the mapping is evidence rather than
  folklore.

The engine's MT19937 stream is seeded by `Mt::new(seed)`; islands use
`seed ^ 0x9E3779B9 * (i + 1)`. No wall clock, PID or environment value enters
the stream. `--time-budget` is the one exception to reproducibility: stopping
on wall-clock time cannot be replayed bit for bit (see
[`checkpoint.md`](checkpoint.md)).

## Reporting

* **text** prints one `# seeded: complexity=… train_error=…  y = …` line per
  accepted seed, on stderr with the rest of the human log;
* **`--json`** adds a `seeds` array to the result object:

  ```json
  "seeds": [
    { "expression": "sqrt((m / k))", "complexity": 12, "train_error": 2.5e-11 }
  ]
  ```

* **`--ndjson`** emits one `{"event":"seed", …}` line per seed before the
  solution events.

## Scope and interactions

* Seeds apply to the **primary search**. Auxiliary searches started by
  `--null-runs`, `--stability` and the repeats of `--report benchmark` begin
  unseeded, so their comparison stays a comparison of the search itself.
* Seeds are inserted at the start of a **fresh** run and are not part of the
  checkpoint format. `--resume` therefore refuses seed flags with a usage
  error: the checkpoint already contains any seeded individuals in its archive
  and populations.
* `--eval-expr` does not search, so combining it with a seed flag is a usage
  error.

## Measured effect

The target is the known-truth `spring_period` problem (`p = sqrt(m / k)`) from
`src/bench.rs`, at zero noise, with the same budget and search seed in every
pair. Budget: 15 generations, population 60, one island, metric `absolute`.
The seed is `sqrt(m / k)`. "First generation ≤ 1e-6" is the first generation
whose reported best training error crossed the threshold (`none` means it
never did inside the budget).

| search seed | unseeded best error (gen 15) | seeded best error (gen 15) | first ≤ 1e-6, unseeded | first ≤ 1e-6, seeded |
|---:|---:|---:|---:|---:|
| 1 | 1.485211e-1 | 0 | none | 1 |
| 2 | 1.245400e-1 | 0 | none | 1 |
| 3 | 5.903883e-2 | 0 | none | 1 |

The seeded run has the exact law in its front from the first generation; the
unseeded control does not get below `1e-3` at all in 15 generations. This is
the easy case for seeding — the user supplied exactly the right form — and it
is what the feature is for. It does not say a seed always helps when the form
is wrong: a wrong-shaped seed is one more individual, and because it changes
which individuals are selected, a bad seed can change the trajectory. What is
guaranteed is that leaving the flags off is exactly the historical search
(asserted by `no_seeds_is_the_historical_trajectory`).

The comparison is asserted in `tests/seeds.rs`
(`a_seed_reaches_a_lower_error_at_the_same_small_budget`,
`the_seed_crosses_the_threshold_first`).
