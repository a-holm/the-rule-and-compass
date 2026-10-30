# Row weights

`--weight <col>` gives every row its own weight. A weight says **how much that
row counts**, not how large its target is: recent measurements, more reliable
sensors or duplicated observations can be emphasised, and a row that should not
influence the model at all can be given weight zero.

```sh
rcsr --data d.csv --target y --vars x --weight w --metric squared
```

Weights are **relative**. Multiplying every weight by the same positive constant
changes nothing, because every supported metric is normalised by `Σw`. An empty
weight vector means "every row weighs 1", which is the behaviour of a run
without `--weight`.

## What a weight means: the weighted mean

For a loss `ℓᵢ` on row `i`, the weighted metric is

```text
score = Σ wᵢ·ℓᵢ / Σ wᵢ
```

not `Σ wᵢ·ℓᵢ / n`. The denominator is the **sum of weights**, not the row count.
This is the whole point of the feature:

> Doubling one row's weight is exactly the same as duplicating that row.

If row `k` originally has weight `wₖ`, doubling it adds `wₖ·ℓₖ` to the numerator
and `wₖ` to the denominator. A duplicate copy of the same row contributes
exactly the same two terms to an unweighted mean. With a plain division by `n`
the numerator would grow but the denominator would not, and a weight of 2 would
mean something else entirely. The tests in `my-version/tests/weights.rs` assert
the equivalence to `1e-12` for the metrics and for the fitted constants.

A zero weight removes the row from the numerator and adds nothing to the
denominator, so the row has no influence on the score, the fit or the reported
constants — changing its target value cannot move anything. The row is invisible
even when its **prediction or target is non-finite**: the scorer's finiteness
check applies only to rows with `wᵢ > 0`, so an undefined branch (say `ln` of a
negative value) in a masked row cannot make the candidate infeasible. This
matches the fitter, which skips `w == 0` rows before forming a residual. The
unweighted path is unchanged and still rejects any non-finite prediction.

## Which metrics support weights

| metric | weighted form |
|---|---|
| `absolute` | `Σ wᵢ·|eᵢ| / Σ wᵢ` |
| `squared` | `Σ wᵢ·eᵢ² / Σ wᵢ` |
| `signed` | `|Σ wᵢ·eᵢ / Σ wᵢ|` |
| `log_squashed` | `Σ wᵢ·ln(1 + eᵢ²) / Σ wᵢ` |
| `correlation` | `1 − |r_w|`, weighted Pearson `r_w` |
| `correlation_signed` | `1 − r_w`, weighted Pearson `r_w` (direction preserved) |
| `implicit` | `Σ wᵢ·(|f(pᵢ)| / ||∇f(pᵢ)||) / Σ wᵢ` |
| `implicit_residual` | `Σ wᵢ·|f(pᵢ)| / Σ wᵢ` |
| `log_loss` | `Σ wᵢ·[max(−uᵢ,0) + log1p(exp(−|uᵢ|))] / Σ wᵢ`, `uᵢ = tᵢ·pᵢ` |
| `brier` | `Σ wᵢ·(σ(pᵢ) − lᵢ)² / Σ wᵢ` |
| `hinge` | `Σ wᵢ·2σ(−uᵢ) / Σ wᵢ` (the bounded sigmoid surrogate) |
| `hinge_ordinary` | `Σ wᵢ·max(0, 1 − uᵢ) / Σ wᵢ` |
| `hinge_squared` | `Σ wᵢ·max(0, 1 − uᵢ)² / Σ wᵢ` |
| `aic`, `aicc`, `bic` | weighted RSS, `n = number of rows with wᵢ > 0` |

For the classification family the label is still the midpoint rule `lᵢ = [yᵢ >
(min y + max y)/2]`, but **the midpoint and the class counts are taken over the
positive-weight rows only**. A zero-weight row is invisible, so a wild, missing
or `±∞` target in it cannot relabel a row that counts. The prediction is the
model output read as a **logit** `pᵢ`; `σ` is the stable logistic and
`tᵢ = 2lᵢ − 1`. These are frequency weights like every other weighted mean: a
weight of 2 is exactly a duplicated row (the tests assert the equivalence to
`1e-12`). To use a class-balance prior, give the positive and negative rows
different weights yourself; the engine does not silently rebalance the
objective, so a weighted log-loss remains an estimate of the weighted
distribution and not a calibrated probability under the original prior.

The weighted Pearson correlation uses weighted means and weighted covariance:

```text
mₐ = Σ wᵢ·aᵢ / Σ wᵢ
cov(a,b) = Σ wᵢ·(aᵢ − mₐ)(bᵢ − m_b)
r_w = cov(a,b) / sqrt(cov(a,a)·cov(b,b))
```

**Weights are ignored nowhere.** For the remaining metrics the engine *refuses*
`--weight` with a usage error (exit code 2) that lists the supported metrics:

* `median`, `maximum`, `interquartile` are order statistics. A weighted quantile
  is a different estimator with its own definition (and no single standard one);
  it is not the same as duplicating rows.
* `slope` compares adjacent first differences. Row weights do not give an
  unambiguous weighted difference, and duplicating a row changes the adjacency.
* `rank_correlation`, `rank_correlation_signed` and `auc` depend on ranks or on
  class counts; a weighted rank/ROC needs a separate definition for ties and
  class prevalence. `auc` therefore keeps the usage error even though
  `log_loss`, `brier` and the hinge forms are weighted.

`--metric aic`, `aicc` and `bic` use the weighted residual sum of squares
`Σ wᵢ·eᵢ²` and the number of **positive-weight** rows as `n`, so a zero-weight
row does not buy a degree of freedom.

## The fitter: weighted least squares

The constant fitter minimises the weighted residual sum of squares

```text
RSS_w = Σ wᵢ·(f(xᵢ) − yᵢ)².
```

It multiplies each residual by `sqrt(wᵢ)` before the Jacobian products, so the
Gauss–Newton/Levenberg–Marquardt step solves the weighted normal equations

```text
(JᵀWJ) δ = −JᵀWr,     W = diag(w₁, …, wₙ).
```

A zero-weight row is skipped entirely: it neither enters `JᵀJ`, `Jᵀr` nor the
returned RSS, so even a wild target or a non-finite value in that row cannot
move the constants. `fit::fit_weighted` is the weighted entry point;
`fit::fit` delegates to it with an empty weight vector and is bit-for-bit
unchanged.

The standard errors of the fitted constants use the same weighted normal matrix:
`cov = σ²·(JᵀWJ)⁻¹` with `σ² = RSS_w / dof` and

```text
dof = (number of rows with wᵢ > 0) − (number of fitted constants).
```

The weights are treated as **frequency weights** here. That matches the
"weight 2 = two copies" interpretation; if you use weights as measurement
precision rather than replication, the degrees of freedom are a modelling
choice, not a theorem.

## Weights on the CUDA evaluator

The device evaluator carries the per-row weight vector and the classification
label sign, uploaded once per sample (#289). A classification run therefore
keeps its weights when the search is scored on the device: the kernel returns
`Σ wᵢ·lossᵢ`, the host divides by `Σw`, and the zero-weight skip happens
**before the program runs**, so a masked row with an undefined branch is never
evaluated. The device value is the search fitness only; every reported number
is the exact CPU `f64` rebuild of the same weighted mean. The weighted
**regression** kernels are not part of #289: a weighted `squared`, `absolute`
or `log_squashed` run still keeps the CPU evaluator, and the backend note names
the reason rather than dropping the weights.

## What else is weighted, and what is not

* **Search**: `search::eval_error` scores with the sample's weights, and every
  constant fit uses `fit_weighted`.
* **Stability selection**: each bootstrap resample draws rows *and their
  weights* together, so a stability run solves the same weighted problem.
* **Cross-validation**: `cv_folds_weighted` carries the weight column into each
  fold's training and test samples.
* **`--eval-expr`**: scores the parsed formula with the same weighted metrics.
* **Null control**: the y-randomisation control shuffles the target and keeps
  the weights.
* **Not weighted**: `diagnostics::Baselines` and `diagnostics::ResidualTests`
  are diagnostic descriptions of the data and the residuals; they stay
  unweighted even when a weight column is present. A text report says so.
* **Refused**: `--weight` cannot be combined with `--bootstrap-error`. The
  `.632+` estimator resamples and refits without weights, so applying weights to
  only part of it would be misleading; the CLI refuses the combination (exit
  code 2).

## Validation

The weight column must exist, be numeric, and contain only finite values
`>= 0`, with at least one strictly positive value. Validation happens after
`--exclude` and `--filter` have selected the population, and the message names
the column and the number of offending rows. All of these are input errors
(exit code 3):

```text
weight column 'w' not found (columns: [...])
weight column 'w' is not numeric; weights must be finite and >= 0
weight column 'w' has 2 negative value(s); weights must be finite and >= 0
weight column 'w' has 1 non-finite value(s); weights must be finite and >= 0
weight column 'w' is all zero; at least one row must have a positive weight
```

The weight column is excluded from the default input variables. It is available
as a feature only if you name it explicitly in `--vars`; the weighting itself
still comes from the column values.

## Worked example

```csv
x,y,w
0,1,1
1,3,2
2,5,1
3,7,0
4,9,1
```

```sh
rcsr --data d.csv --target y --vars x --weight w --metric squared --test-frac 0
```

The report starts with

```text
# weights: column w, sum 5 over 5 rows (1 zero-weight)
# weights: null baselines and residual tests are unweighted
```

and the search recovers `y = 1 + 2x`. Row 3 (`x = 3`, weight 0) is not fitted or
scored; row 1 counts twice because its weight is 2. The same fit and the same
squared error would be obtained from a table in which row 1 appears twice and
row 3 is absent (apart from the row counts reported for the split).

`--json` reports the same summary at the top level:

```json
"weight_column":"w","weight_sum":5,"weight_rows":5,"weight_positive_rows":4
```

`weight_sum` is the sum of the **training** weights actually used by the search,
`weight_rows` is the number of training rows, and `weight_positive_rows` is the
effective `n` (rows with a positive weight). Without `--weight` all fields are
`null`.
