# rule-and-compass (🎻 `rcsr`)

A **symbolic-regression** engine written in Rust.

Given a table of numeric columns, `rcsr` searches for short closed-form
expressions `y = f(x1, x2, ...)` that fit the data, trading **error** against
**complexity**, and reports the non-dominated set (the *Pareto front*).

> The name is an old-fashioned English pub sign (1700s style, like *The Three
> Compasses*): the **rule** is the law/equation we recover, the **compass** is
> the measurement — the instrument that finds it.

## What it does

Symbolic regression finds formulas, not just fitted parameters. `rcsr` evolves a
population of expression trees, scores each on **two axes (error and
complexity)**, and returns the front of non-dominated equations — the most
accurate equation at each level of complexity.

Constants inside a candidate are not guessed: each candidate's constants are
fitted by **Levenberg–Marquardt**, so a candidate with the right *shape* is
immediately usable even if its numbers started out wrong.

See [`docs/how-symbolic-regression-works.md`](docs/how-symbolic-regression-works.md)
for a plain-language walkthrough,
[`docs/optimisation.md`](docs/optimisation.md) for the theory and measurements
behind the search strategy, and [`SPEC.md`](SPEC.md) for the precise behaviour.

## Build & run

Requires a Rust toolchain (`rustup` recommended). No third-party crates are used.

```sh
cd my-version
cargo build --release

# demo on the bundled sample: recover y = 2x + 1
cargo run --release -- \
  --data examples/line.csv --target y --vars x \
  --generations 200 --islands 4 --pop 200 --metric absolute
```

Output (human-readable form):

```
  1  complexity=1   train_err=1.455120      test_err=1.216192       y = (0.8446 * x)
  2  complexity=2   train_err=1.096793      test_err=0.609935       y = exp((0.8446 * x))
  3  complexity=3   train_err=0.999970      test_err=1.000090       y = (2.0044 * x)
  4  complexity=4   train_err=0.651494      test_err=0.423752       y = (exp((0.8446 * x)) + (0.8446 * x))
  5  complexity=5   train_err=0.000001      test_err=0.000001       y = (1 + (2 * x))
```

The last row is the answer: `y = 1 + 2x`, exact to the rounding of the data, and
the smaller rows are the cheaper approximations it beat along the way.

```sh
# a trigonometric example: recover y = sin(x)
cargo run --release -- --data examples/sine.csv --target y --vars x \
  --generations 300 --islands 4 --pop 300 --metric squared
```

`cargo test` runs the unit tests and the end-to-end recovery tests
(`y = x`, `y = x²`, `y = x1·x2`), plus a regression test that constant fitting
reaches machine precision.

## Using it from other programs

`rcsr` is built to be driven by scripts and agents.

* **stdout carries only the result; stderr carries the human log.** So
  `rcsr ... --json > result.json` and `rcsr ... --json | jq .pareto[0]` do the
  right thing.
* **`rcsr schema`** prints a self-describing JSON document (every option, its
  type, default and meaning; plus the exit codes). An agent can discover the
  interface without reading the source.
* **`--json`** emits one JSON object; **`--ndjson`** streams one object per line
  (`start`, `progress`…, `solution`…, `done`) for long runs.
* **Deterministic**: the same `--seed` gives the same answer, and the seed is
  echoed in the output.
* **Bounded time**: `--time-budget <seconds>` guarantees the process returns.
* **stdin**: `--data -` reads the table from standard input.
* **Stable exit codes**: `0` success, `2` usage error, `3` input error,
  `4` no finite solution.

```sh
# machine-readable
rcsr --data examples/sine.csv --target y --json | jq '.pareto[-1].expression'

# discover the interface
rcsr schema

# stream progress as NDJSON while it runs
rcsr --data examples/sine.csv --target y --ndjson --progress --time-budget 5

# read the table from a pipe
cat examples/line.csv | rcsr --data - --target y --top 3
```

```jsonc
// rcsr --data examples/line.csv --target y --json
{
  "tool": "rcsr", "version": "0.1.0", "seed": 0, "metric": "absolute",
  "target": "y", "variables": ["x"], "train_rows": 30, "test_rows": 10,
  "generations_run": 200, "converged_early": false, "elapsed_seconds": 1.9,
  "pareto": [
    { "rank": 5, "complexity": 5, "train_error": 5.3e-7,
      "test_error": 5.2e-7, "expression": "(1 + (2 * x))" }
  ]
}
```

### Full option list

```
rcsr --data <file|-> --target <col> [--vars a,b,..]
     [--metric absolute|squared|correlation] [--generations N] [--islands I]
     [--pop P] [--max-complexity K] [--seed S] [--test-frac f] [--ops list]
     [--fit-iters N] [--no-fit] [--no-dedup] [--patience N]
     [--time-budget seconds] [--format text|json|ndjson] [--progress]
     [--top N] [--quiet] [--out file.tsv]
```

Input may be CSV, TSV, or whitespace-delimited, with a header row. 25% of the
rows are held out for testing; the search only sees the rest.

## Features

- expression trees over a broad primitive set (`+ - * / ^`, `exp ln sqrt abs
  sin cos tan sinh cosh tanh logistic gauss square cube`, `if`), with domain
  guards;
- **Levenberg–Marquardt constant fitting** using adjoint (backpropagation)
  derivatives — one backward sweep gives every constant's gradient;
- an **island-model, multi-objective** evolutionary search over an
  error/complexity **Pareto front**, with mutation, crossover, migration and
  restarts, run in parallel across islands;
- **ε-dominance** archiving and **lexicographic parsimony pressure**, so the
  front stays small and expressions stay short;
- **structural de-duplication** and **convergence/time-budget stopping**, so the
  same expression is never evaluated twice and the search stops when there is
  nothing left to gain;
- bottom-up simplification with affine normalisation, and denormalisation back
  to original units.

See [`docs/optimisation.md`](docs/optimisation.md) for the measured before/after
numbers and the hardware plan.

## Layout

```
SPEC.md                       behaviour specification
docs/how-symbolic-regression-works.md   plain-language explainer
docs/optimisation.md          theory, literature and measurements behind the search
docs/gpu-backend.md           the optional CUDA evaluator and the DGX Spark plan
my-version/
  Cargo.toml
  build.rs      compiles the CUDA backend when --features cuda is used
  src/
    rng.rs        MT19937 RNG for reproducible runs
    expr.rs       expression trees, primitives, evaluator, mutation, simplification
    fit.rs        Levenberg–Marquardt constant fitting
    metrics.rs    error metrics
    pareto.rs     the error/complexity Pareto front
    data.rs       CSV/TSV loading, normalisation, train/test split
    search.rs     island-model GP loop
    gpu.rs        bytecode compiler + FFI for the optional GPU evaluator
    cuda/eval.cu  the CUDA kernel
    main.rs       the `rcsr` CLI
    bin/
      gpu_bench.rs  GPU vs CPU evaluation benchmark (--features cuda)
  tests/
    recover.rs    end-to-end recovery tests + unit tests
  examples/
    line.csv, sine.csv
```

## License

MIT.
