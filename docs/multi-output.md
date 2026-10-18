# Multi-output regression

`--target y1,y2` searches for **one model shape that explains several outputs**
at once. Examples are a system with several sensors driven by the same inputs, or
a quantity measured on two scales.

```sh
rcsr --data panel.csv --target y1,y2 --vars x1,x2 --metric absolute
```

## The model: one shape, one constant vector per output

A candidate is a single expression **structure**. For each output `j` the engine
fits its own constants on that output's target column:

```text
y1 = c0*x1*x2 + c1        (c0, c1 fitted to y1)
y2 = c0*x1*x2 + c1        (same structure, c0, c1 fitted to y2)
```

The structure is identical; only the fitted numbers differ. This is the point of
the design: **sharing the shape is what buys statistical strength**. A single
constant vector serving every output would be strictly less expressive — it could
not represent `y1 = a*x1*x2` and `y2 = b*x1*x2 + c`, because the first output
wants the offset to be zero and the second does not. The engine therefore does
**not** pretend that one set of constants fits all outputs.

Constants are fitted per output by the same Levenberg–Marquardt solver used for
single-output runs. The search itself evolves one structure; every candidate is
scored by fitting each output and combining the errors below.

## Combined error

With targets `y1 … ym` and metric `L`, the combined error is the arithmetic mean
of the per-output errors:

```text
combined_error = ( L(f, y1) + … + L(f, ym) ) / m
```

If any output's error is non-finite, the combined error is `+inf`. The Pareto
archive ranks candidates on `(combined_error, shape_complexity)`: complexity is
counted **once for the shared structure**, not per output. That is why a shared
model can dominate two independent models at the same complexity.

Metrics that use the constant count (AIC/AICc/BIC) receive the constants of the
output being scored. `--weight` applies the same row weights to every output.

## Reporting

* **Text** prints a combined line per Pareto member and then **one line per
  target**:

  ```text
    1  complexity=5  combined_train_err=0.000001  combined_test_err=0.000002
       y1 = (2 * (x1 * x2))          (train_err=0.000001 test_err=0.000001)
       y2 = (1 + (3 * (x1 * x2)))    (train_err=0.000001 test_err=0.000003)
  ```

* **`--json`** keeps the usual top-level fields and adds `"multi_output": true`,
  `"target": "y1,y2"` and `"targets": ["y1","y2"]`. Each Pareto entry carries
  the combined `train_error`/`test_error` and a `targets` array:

  ```json
  {"rank":1,"complexity":5,"k_constants":4,
   "train_error":0.000001,"test_error":0.000002,
   "targets":[
     {"name":"y1","expression":"(2 * (x1 * x2))","constants":[2.0],
      "train_error":0.000001,"test_error":0.000001,
      "fold_errors":[...],"constant_errors":{...}},
     {"name":"y2","expression":"(1 + (3 * (x1 * x2)))","constants":[3.0,1.0],
      "train_error":0.000001,"test_error":0.000003,
      "fold_errors":[...],"constant_errors":{...}}
   ]}
  ```

* **`k_constants` is the total across outputs** (`shape constants × targets`).
  `constant_errors` and the printed expression are **per output**, because each
  output has its own fitted vector. `train_error`/`test_error` on the Pareto
  entry are the combined means; the per-target values sit inside `targets`.
* `--out` writes one TSV line per target with the combined row repeated.

Test errors never refit on the test rows: the per-output constants are fitted on
the training rows and then evaluated on the held-out rows. This is the same
honesty rule as the single-output path.

## Limits

Multi-output mode is intentionally narrower than the single-output pipeline:

* `--ndjson` is refused. Use `--json`.
* `--cv`, `--null-runs`, `--bootstrap-error`, `--stability` and `--report` are
  refused: their summaries assume a single target.
* `--save`/`--resume` and `--seed-expr`/`--seed-file` are refused.
* `--eval-expr` is refused; there is no single formula to score.

These are honest refusals (exit code 2) rather than half-filled reports. The
single-target path is unchanged bit for bit: with one target the engine takes the
historical code path.

## Measured comparison

`tests/multi_output.rs` builds two outputs that share the term `x1*x2`:

```text
y1 = a*x1*x2
y2 = b*x1*x2 + c
```

It runs the shared-shape search and two independent single-target searches at the
same generation/population budget. On the recorded run (`--seed 7`, 120
generations, population 160, absolute error):

| run | y1 error | y2 error | combined (mean) |
|---|---:|---:|---:|
| shared shape | 0.000000000 | 0.000000000 | **0.000000000** |
| independent y1 | 0.000000000 | — | — |
| independent y2 | — | 0.508548266 | — |
| independent mean | | | 0.254274133 |

The shared run recovered the one shape `poly[c, x0*x1]` — `c0 + c1*x1*x2` — and
fitted `c0 = 0` for `y1` and `c0 = 1` for `y2`. The independent runs share the
budget but not information: `y1` alone finds the product, while `y2` alone
spends its budget on the offset and misses it. The test asserts the shared
combined error is no worse than the independent mean and that both outputs keep
the same structure. These numbers are printed by the test and repeated in the
pull request for issue #15.
