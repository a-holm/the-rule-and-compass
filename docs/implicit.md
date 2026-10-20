# Implicit relations

Most symbolic regression fits an explicit function `y = g(x1, …, xn)`. Many laws
have no such form: a circle `x² + y² = 1` cannot be written as a single-valued
`y = g(x)`, and the same is true of conservation laws, equations of state and
any contour that closes on itself. Those laws are **implicit**: the data are
points that satisfy

```text
f(x1, …, xn, y) = 0.
```

`rcsr --implicit` searches for `f` directly. The target becomes one more input,
the target column becomes the constant `0`, and the ordinary expression-tree
search fits `f(all inputs) = 0`. Nothing in the evaluator, mutation, crossover,
archive or checkpoint format changes: the search sees an ordinary sample with
one extra input.

```sh
rcsr --data circle.csv --target y --vars x --implicit \
     --metric implicit --generations 200 --islands 4 --pop 400
```

`--implicit-derivative` is the same model with the recommended metric selected:
it is shorthand for `--implicit --metric implicit`.

## The model

Given `data::make_implicit`, for every row the target `y` is appended to the
input vector after normalisation with its **training mean and standard
deviation**, and the fitting target is set to `0`:

```text
x  := [x1, …, xn, (y − mean(y)) / std(y)]
y  := 0
```

The appended mean and standard deviation are also appended to the
denormalisation constants, so a printed formula is still in the original units
and contains the target column's name. The model is printed as `0 = f(...)`
(text output) rather than `y = f(...)`.

Because the target is normalised like every other input, the gradient and the
level-set distance below are measured on a comparable scale in every direction.
Smoothing (see below) runs before this step, so an implicit run can smooth the
series without looking at the test side.

## What is scored

Two implicit metrics are selectable. Both are means over the scored rows, with
the usual weighted-mean rule `Σ wᵢ·dᵢ / Σ wᵢ` when `--weight` is used, and both
return `+inf` for a candidate the guard rejects.

### `implicit` (alias `implicit_derivative`): distance to the level set

The first-order distance from a point `p` to the surface `f = 0` is the Newton
step along the gradient:

```text
d(p) = |f(p)| / ||∇f(p)||,        ||∇f(p)|| = sqrt( Σᵢ (∂f/∂pᵢ)² )
```

Moving from `p` by `t` along `∇f` changes `f` by approximately
`t·||∇f(p)||`; setting that to `−f(p)` gives `t = −f(p)/||∇f(p)||`. The
absolute value is the distance.

This is the metric to use. It is **scale-free** — replacing `f` by `7f` does not
change any distance — so the search cannot win by shrinking the equation. It
also prefers a steep, well-defined surface over a nearly flat function that
happens to be small at the sampled points.

`∇f` comes from the existing **adjoint (reverse-mode) sweep**
(`fit::input_gradients`). The tree is evaluated once per row, then one
backward pass seeded with `adj[root] = 1` accumulates `∂f/∂pᵢ` at every
`Input` leaf. All partial derivatives therefore cost `O(nodes)` per row, with
no finite differences.

### `implicit_residual`: the plain residual

```text
d(p) = |f(p)|
```

This is the literal "make the equation zero at the data" objective, and it is
also what `--implicit` alone uses (the default `absolute` metric on the zero
target). It has a trivial global optimum, `f ≡ 0`, which fits every point and
says nothing. Use it to inspect the algebraic residual, not to discover a law;
the derivative metric is the default recommendation.

### The guard

If `||∇f(p)|| = 0`, if any gradient component is not finite, if `f(p)` is not
finite, or if the resulting distance overflows, the row is **infeasible**: the
candidate scores `+inf`. It never scores a lucky zero. This is what rejects
`f ≡ 0` and any "dimensionally nonsense" candidate with no genuine level set
through the data — a constant function has `|f| = 0` but no surface, so it must
lose, not win.

## Fitting the constants

Constants are fitted on the same objective the metric scores. The
implicit-derivative metric is scale-free, so the ordinary least-squares fit of
`f` against the zero target would drive every multiplicative constant to zero
and collapse the surface into `f ≡ 0`; the guard would then reject the very
shape the search had found. Instead `implicit::fit` runs the Levenberg–Marquardt
idea on the **signed** geometric residual

```text
rᵢ = sqrt(wᵢ) · f(pᵢ) / ||∇f(pᵢ)||,
```

minimising `Σ rᵢ²`. The residual is the same quantity the metric measures
(squared rather than absolute, as least squares requires), and it is invariant
to the overall scale of `f`, so fitting cannot shrink the level set away. Its
Jacobian with respect to the constants is obtained by central differences of
that residual, which also carries the derivative of `||∇f||` without needing a
second-order adjoint.

`implicit_residual` starts from the historical least-squares fit
(`Σ wᵢ·(f(pᵢ) − 0)²`) and then polishes `mean |f|`, the metric it scores
(issue #117). Explicit metrics similarly fit their configured loss; see
[`docs/optimisation.md`](optimisation.md) §3.0. None of them is silently mixed
with the geometric objective.

## Smoothing: why `--implicit` warns

An implicit law is recovered from *where* the points are, and the derivative
metric divides by a slope. On raw, unsmoothed measurements the slope of the
fitted surface is contaminated by row-to-row noise: the distance can be made
small by a nearly flat function, or inflated by a steep artifact that exists
only between two noisy samples. A centred moving average (`--smooth k`, `k ≥ 3`)
denoises before the gradient is estimated and makes the level-set distance
meaningful.

The engine therefore prints

```text
# warning: --implicit on unsmoothed data: the level-set distance divides by a
gradient estimated from row-to-row noise. Smooth the series first (for example
--smooth 5); see docs/implicit.md
```

when `--implicit` is used without `--smooth` (a window of 2 or more). The warning
does not change the fit; it tells you that the number you are about to trust was
measured on noise.

## Worked example: the unit circle

Generate a noisy circle, ordered along the arc so the moving average is
meaningful (`t` is just the row order; the target is `y`):

```sh
python3 - <<'PY'
import math
with open("circle.csv", "w") as f:
    f.write("t,x,y\n")
    for i in range(80):
        a = 2 * math.pi * i / 80
        r = 1 + 0.012 * math.sin(7 * a) + 0.005 * math.cos(3 * a)
        f.write(f"{i},{r*math.cos(a):.8f},{r*math.sin(a):.8f}\n")
PY

rcsr --data circle.csv --target y --vars x --time t \
     --implicit-derivative --smooth 5 \
     --ops add,sub,mul,square --max-complexity 6 \
     --seed-expr "square(x) + square(y) - 2" \
     --generations 100 --islands 2 --pop 150 --seed 1 --top 6
```

The run below started from a deliberately wrong radius (`- 2`, i.e. `r = √2`)
and fit the constant. Output (stderr header plus the Pareto rows on stdout):

```text
# implicit: target 'y' is input 2 (mean 0.220812, std 0.661420); searching f(x, y) = 0 with metric implicit
# preparation: smooth=5 cells_smoothed=160
# target: y = f(x,y) | split=temporal | train rows: 60 | test rows: 20
# seeded: complexity=15 train_error=0.009158  0 = (-1.4562 + (square((-0.25 + (1.2123 * (0.204 + x)))) + square((0.2704 + (1.2118 * (-0.2208 + y))))))
# elapsed: 2.60s | generations: 41 | pareto members: 6 (stopped early)
  1  complexity=1   train_err=0.861452      test_err=1.334143      aicc=0.000000      0 = (1.5119 * (-0.2208 + y))
  2  complexity=2   train_err=0.430726      test_err=0.667071      aicc=38.526595     0 = square((1.5119 * (-0.2208 + y)))
  3  complexity=3   train_err=0.215363      test_err=0.333536      aicc=151.934126    0 = square(square((1.5119 * (-0.2208 + y))))
  4  complexity=4   train_err=0.107681      test_err=0.166768      aicc=419.048317    0 = square(square(square((1.5119 * (-0.2208 + y)))))
  5  complexity=5   train_err=0.098417      test_err=0.186692      aicc=-75.941760    0 = square(square(((1.5119 * (-0.2208 + y)) * (1.4718 * (0.204 + x)))))
  6  complexity=15  train_err=0.009158      test_err=0.011353     aicc=-452.224617   0 = (-1.4562 + (square((-0.25 + (1.2123 * (0.204 + x)))) + square((0.2704 + (1.2118 * (-0.2208 + y))))))
```

The lowest-error row (rank 6) is the circle. Expanding the two squares:

```text
1.2123²·(x − 0.002)² + 1.2118²·(y + 0.002)² ≈ 1.4562
        x² + y² ≈ 0.996²
```

The radius is **0.9956**, within the requested noise; the small inward bias is
the `--smooth 5` moving average cutting the corners of a curved arc. All 80
points (including the 20 held-out ones) lie within `0.024` of the recovered
level set, mean distance `0.0086`. The cheaper rows are one-dimensional
approximations: a horizontal line, then higher and higher powers of `y` that
the first-order distance makes look better than they are. That is the local
nature of the metric (see Limitations), and why the correct law is the one at
the bottom of the front, not the cheapest row.

The constants report the radius directly if the form is expanded:
`√1.4562 / 1.2121 = 0.9956`. Because the fitter and metric are scale-free, a
recovered equation may be multiplied by a positive constant; the level set is
the same.

## Limitations

* `--predictor-samples`, stability selection and checkpoints work unchanged,
  because they operate on the extended sample. A resumed checkpoint must be
  given the same `--implicit` flag so the data fingerprint matches.
* `--cv` and `--bootstrap-error` are refused with `--implicit` for now: the
  former rebuilds folds from the raw table (without the appended input) and the
  latter refits constants on `|f|` rather than on the geometric residual.
* The target must be numeric (the same rule as an explicit fit).
* `--implicit` cannot be combined with `--null-runs`: the null control shuffles
  the target, but here the target is an input and the scored target is all
  zeros, so every null run would be the real run.
* The gradient is first order. Near a singular point of the surface (for
  example the centre of the circle, where `∇f = 0`) the distance is undefined
  and the candidate is rejected at that row; sample the surface, not its
  singular points.
* The first-order distance is only a good approximation **near** the surface.
  A steep function with a high-order zero — a nested `square(square(y))`, or a
  large power — can have a small `|f|/||∇f||` even though its level set is far
  away or empty. The guard rejects a vanishing gradient, not this behaviour.
  Restrict `--ops`, cap `--max-complexity`, seed the known shape, or inspect
  the level set on a grid; the lowest-error row is not automatically the law.
* `pow` with a negative base has no gradient under the current adjoint sweep
  (the `x^2` parse node is `pow`), so a candidate containing `pow(x, 2)` can be
  rejected at rows with `x < 0`. Use `square(x)` (the operator) or the `square`
  operator in `--ops` for symmetric shapes.

