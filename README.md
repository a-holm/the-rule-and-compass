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

See [`docs/how-symbolic-regression-works.md`](docs/how-symbolic-regression-works.md)
for a plain-language walkthrough and [`SPEC.md`](SPEC.md) for the precise
behaviour.

## Build & run

Requires a Rust toolchain (`rustup` recommended).

```sh
cd my-version
cargo build --release

# demo on the bundled sample
cargo run --release -- \
  --data examples/line.csv --target y --vars x \
  --generations 200 --islands 2 --pop 200 --metric absolute
```

`cargo test` runs the unit tests and the end-to-end recovery tests
(`y = x`, `y = x²`, `y = x1·x2`).

### CLI

```
rcsr --data <file> --target <col> [--vars a,b,..] [--metric absolute|squared|correlation]
     [--generations N] [--islands I] [--pop P] [--max-complexity K]
     [--seed S] [--test-frac f] [--ops list] [--out file.tsv]
```

Input may be CSV, TSV, or whitespace-delimited, with a header row. 25% of the
rows are held out for testing; the search only sees the rest.

## Features

- expression trees over a broad primitive set (`+ - * / ^`, `exp ln sqrt abs
  sin cos tan sinh cosh tanh logistic gauss square cube`, `if`), with domain
  guards;
- column-wise evaluation and three error metrics (absolute, squared,
  correlation);
- complexity = node count, with a path-length constraint;
- an **island-model, multi-objective** evolutionary search over an
  error/complexity **Pareto front**, with mutation, crossover, **constant
  fitting**, migration and restarts;
- bottom-up simplification and denormalisation back to original units.

## Layout

```
SPEC.md                       behaviour specification
docs/how-symbolic-regression-works.md   plain-language explainer
my-version/
  Cargo.toml
  src/
    rng.rs        MT19937 RNG for reproducible runs
    expr.rs       expression trees, primitives, evaluator, mutation, simplification
    metrics.rs    error metrics
    pareto.rs     the error/complexity Pareto front
    data.rs       CSV/TSV loading, normalisation, train/test split
    search.rs     island-model GP loop
    main.rs       the `rcsr` CLI
  tests/
    recover.rs    end-to-end recovery tests + unit tests
  examples/
    line.csv, sine.csv
```

## License

MIT.
