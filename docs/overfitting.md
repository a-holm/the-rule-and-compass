# Overfitting: how to avoid it and how to prove you did

A symbolic search is deliberately good at *fitting*. Given enough operators and
constants it can drive the training error to zero on almost anything — including
random numbers. So a small error, even an error of exactly zero, is **not**
evidence that you found a law. This document is the practical answer to two
questions:

1. what do we do to *avoid* overfitting, and
2. how do we *tell* a real fit from an overfitted one?

It is written for someone meeting the topic for the first time, but every test
is a real method from the literature and all of them are either implemented
here or named with a deliberately honest "not yet".

---

## 1. Why symbolic regression is unusually exposed

Ordinary curve fitting searches *parameters*. Symbolic regression also searches
the **form** — the structure — so the space of candidate models is astronomically
larger. Two consequences:

* **The best of many is always good.** If you generate a million random
  expressions and keep the one with the lowest error, that expression looks
  impressive on the data you generated it from, even if the data had no signal.
  This is the *selection / multiple-comparisons* problem.
* **There is usually an exact fit that is meaningless.** For `n` data points a
  big enough expression can pass through all of them. Passing through every
  point is called *interpolation*, and it says nothing about what happens
  between or beyond the points.

So the question is never "how small is the error?" but "**could this fit have
arisen from noise, and does it survive data the search never saw?**"

### The failure modes, by name

| Failure | What it looks like | Detected by |
|---|---|---|
| Noise fitting | tiny train error, poor test error | hold-out / CV |
| Spurious structure | an invented term that is really noise | permutation control |
| Extrapolation failure | good inside the data, absurd outside | out-of-distribution test |
| Complexity bloat | bigger expressions for no real gain | AICc / BIC, Pareto front |
| Selection bias | "best of a million" reads as signal | permutation control |
| Leakage | test rows influenced the model | time-aware splits, purging |

---

## 2. Strategies to avoid overfitting

These are decisions made *before* you trust a result.

### 2.1 Never score on the data you fit
The first and largest defence. `rcsr` splits the rows, normalises using
**training statistics only**, and reports train and test error side by side. If
you only look at one number, look at the test error.

### 2.2 Split the way the data behaves
Shuffling assumes rows are exchangeable. Time series and panel data are not:
see [`data-splitting.md`](data-splitting.md). Use `--split temporal` or
`--split temporal-group` and, for financial or sensor data, an embargo
(`--gap`) so that near-in-time rows cannot leak across the boundary. The same
idea is known in the finance literature as **purging and embargo** (López de
Prado 2018). A random split on a time series is the single most common way to
fool yourself.

### 2.3 Keep the model small unless the data pays for it
Every state-of-the-art method uses some parsimony pressure:
* `rcsr` keeps an error/complexity **Pareto front** and applies
  **lexicographic parsimony pressure** (Luke & Panait 2006): a smaller
  expression wins ties in error.
* The front is thinned with **ε-dominance** (Laumanns et al. 2002) so a
  vanishingly small error cannot justify a much larger expression.
* **AICc / BIC** turn "unless the data pays for it" into a number (§3.1).

The practical habit: among the front, pick the **simplest** model whose error is
not materially worse, not the one with the lowest error.

### 2.4 Don't over-iterate against the test set
If you re-run and tweak until the test error looks good, the test set has become
training data. Hold out a *final* set you touch once, or use
cross-validation (§3.2). This is why `rcsr` reports folds (`--folds`) rather
than a single number: an unstable model cannot hide behind one lucky block.

### 2.5 Prefer the answer the search agrees on
A real relationship is usually found by many runs; a noise fit is a lottery
winner. Re-run with different `--seed`s, or bootstrap the rows, and see whether
the same form (or at least the same variables) comes back — **stability
selection** (Meinshausen & Bühlmann 2010). `rcsr` automates this: `--cv`
re-runs the whole search on `k` folds, and `--stability B` bootstraps the rows,
re-runs the search and ranks forms and variables by how often they come back
(§3.11).

### 2.6 Respect the physics when you have it
If the variables carry units, a true equation is dimensionally consistent
(Buckingham's π theorem). Requiring that eliminates huge families of spurious
candidate structures before any statistics. `rcsr` makes this available without
making it mandatory: `--units x=m,t=s --require-dimensions` rejects candidates
whose dimensions cannot be consistent (see [`units.md`](units.md)), and
`--function` supplies a known sub-formula as a building block.

### 2.7 Feed it clean data
Noise, outliers and gaps are exactly what gets fitted. Smoothing, outlier
removal and sensible missing-value handling reduce what there is to overfit.
Always keep an untouched copy for evaluation.

---

## 3. How to tell a real fit from an overfitted one

A perfect fit is credible only when the checks below agree.

### 3.1 Information criteria — implemented
Fit always improves with more constants; the criteria subtract a price for
them. For least squares with `k` fitted constants on `n` points and residual sum
of squares `RSS`:

```
AIC  = n·ln(RSS/n) + 2k
AICc = AIC + 2k(k+1)/(n − k − 1)      # small-sample correction (Burnham & Anderson 2002)
BIC  = n·ln(RSS/n) + k·ln n
```

Lower is better. Reported per solution as `aicc` and `bic`. Two cautions, both
real: they compare models **on the same data** (not across datasets), and they
assume the residuals are roughly Gaussian.

### 3.2 Cross-validation — implemented
The direct test: error on rows the search never saw. `--cv` makes this a
first-class mode: it rebuilds the split as `k` folds, re-runs the **whole
search** on each fold's training rows (normalised on those rows alone), and
reports the mean and spread of the held-out errors.

```sh
rcsr --data clean.csv --target y --vars x --cv k-fold --cv-k 4
# cross-validation (k-fold, k=4): mean test=0.000001 +/- 0.000000 | mean train=0.000001
#   fold 3/4: train=0.000001 test=0.000000 cx=5  y = (3 + (2 * x))
```

Three schemes, chosen to match how the data behaves:

* **`k-fold`** — a shuffled partition; the right default for exchangeable rows.
* **`rolling`** — forward chaining: the training window expands and the next
  contiguous block is held out. For time-ordered data, and the only scheme that
  never trains on the future.
* **`purged`** — `k` contiguous parts with `--gap` rows embargoed either side of
  each test block, so a leaking neighbour cannot help (López de Prado 2018).

Read the **gap between mean train and mean test error**: that difference is the
optimism a single split hides, and it is what a flexible model cannot fake.
`--cv` costs about `k` extra searches.

### 3.3 The permutation control (y-randomisation) — implemented
The decisive test for "is this just what the search does to noise?". Shuffle the
target so it is *independent* of the inputs, run the **identical** search, and
look at the best error it reaches. Repeat. If the real fit is no better than the
shuffled-target fits, the "law" is manufactured.

```sh
rcsr --data data.csv --target y --vars x1,x2 --null-runs 19
# ... null control (19 shuffled-target runs): best=..., median=..., p=0.05 — ...
```

This is *y-scrambling* (Rücker et al. 2007; Wold & Eriksson), long standard in
cheminformatics. With `r` control runs the smallest attainable p-value is
`1/(r+1)`, so use at least 19 runs for p ≤ 0.05 and 99 for p ≤ 0.01.

**Honest limitation:** it tests the *combination* of data and search power. A
weak search can fail to beat the control on real data (it simply cannot find the
law), so a high p-value is not proof of no signal — only that the search has not
demonstrated one. This is inherent to the method and worth stating out loud.

### 3.4 Residual diagnostics — implemented
If anything is left in the residuals, the model is incomplete. Four checks run
on the residuals of the best solution and are reported in text and JSON.

**Durbin–Watson** (Durbin & Watson 1950/51),

```
d = Σ(e_t − e_{t−1})² / Σ e_t²
```

is ≈2 for no autocorrelation, →0 for positive, →4 for negative. It only uses
lag 1 and only makes sense when the row order is real (temporal splits) — `rcsr`
says so in the output when the order is arbitrary.

**Ljung–Box** (Ljung & Box 1978) generalises it to the first `h` lags and turns
it into a p-value:

```
Q = n(n+2) Σ_{k=1}^{h} ρ_k² / (n−k)      ~ χ²_h   under no autocorrelation
```

where `ρ_k` is the sample autocorrelation at lag `k` and `h = min(10, n/5)` by
default (a common rule of thumb). A small p-value means autocorrelation the
model did not explain is still there.

**Breusch–Pagan** (Breusch & Pagan 1979) tests *heteroscedasticity*: regress the
squared residuals on the fitted values and take

```
LM = n · R²_aux      ~ χ²_1   under constant residual variance
```

A small p-value means the residual spread grows or shrinks with the fitted
value — the errors are not identically distributed even if their average is
zero.

Finally, the residual **skewness** (`m3 / m2^{3/2}`) and **excess kurtosis**
(`m4 / m2² − 3`) are printed. Both are 0 for a Gaussian; a large positive
kurtosis flags the heavy tails of a few big misses, which is exactly what an
overfit outlier-driven model leaves behind.

### 3.5 Compare against a null model — implemented
A real model should beat the trivial baselines: the mean, a straight line, and a
flexible non-symbolic learner. `rcsr` fits three and reports each one's training
and test error in the search's own metric:

* the **constant mean** of the training target;
* a **least-squares straight line** (hyperplane) with intercept, solved from the
  normal equations;
* a **k-nearest-neighbour** predictor with `k = 5` — small enough to follow a
  gentle curve, large enough not to chase single points.

If the searched expression does not beat these, the "law" is not earning its
complexity. The `k-NN` baseline is the one to watch: it is a genuinely flexible
learner, so a symbolic model that only matches it has not found structure the
data did not already contain.

```sh
rcsr --data data.csv --target y --vars x1,x2
# null baselines (absolute): mean train=2.003 test=2.250 | linear train=0.000 test=0.000 | 5-NN train=0.130 test=0.250
```

### 3.6 Out-of-distribution / extrapolation test
Train on part of the input range and test outside it, or hold out an entire
regime (a group, a time period). SRBench made in-distribution vs
out-of-distribution accuracy a standard axis precisely because a formula can
interpolate perfectly and extrapolate absurdly (La Cava et al. 2021).
`rcsr`'s `--split temporal` is a natural extrapolation test.

### 3.7 Known-truth benchmarks
The strongest evaluation is on data whose true law you know: the **Feynman**
equations (Udrescu & Tegmark 2020), the Strogatz systems, and the **SRSD**
suite with physically sensible sampling (Matsubara et al. 2022), collected in
**SRBench** (La Cava et al. 2021). These include noisy variants, so they measure
whether a method recovers the truth *despite* noise rather than memorising it.

Recommended order for a result you intend to trust:

```sh
# 1. honest split, with an embargo for dependent data
rcsr --data d.csv --target y --vars x1,x2 --split temporal --time t --gap 5 --folds 5
# 2. is it just what the search does to noise?
rcsr ... --null-runs 19
# 3. read the diagnostics on the simplest good solution
#    aicc, bic, durbin_watson, overfit_index are printed per solution, and the
#    null baselines and Ljung–Box / Breusch–Pagan residual tests come with them
```

---

### 3.8 Pick a metric that matches the loss you care about
The metric decides what "good" means, so choosing it is itself a modelling
choice. `--metric` selects one of sixteen objectives:

* `absolute` (default) and `squared` — squared punishes large residuals hard;
* `median` and `interquartile` — robust to a few outliers;
* `log_squashed` — squared-like but with bounded influence;
* `maximum` — minimise the single worst row;
* `signed` — penalise a consistent bias;
* `correlation` and `rank_correlation` — care about shape, not level or offset;
* `slope` — match the trend rather than the level;
* `auc`, `log_loss`, `hinge` — classification; the target is read as two classes
  and the model output as a logit;
* `aic`, `aicc`, `bic` — fit measured against the number of fitted constants, so
  parsimony drives the front directly instead of only being reported.

AICc and BIC are printed for every solution regardless of the search metric, so
you can compare them without changing the objective.

### 3.9 The .632+ bootstrap error — implemented

Hold-out and cross-validation need data to spare, and on a small sample both the
in-sample error (optimistic: the model saw every row) and a single hold-out
(pessimistic: it uses only part of the data) mislead. The **.632+ bootstrap**
(Efron & Tibshirani 1997) corrects both.

Write `err_app` for the model's error on the rows it was fit to. Draw `B`
bootstrap samples of the rows (sample `n` rows **with replacement**), refit the
model's **constants** on each sample, and score it on the rows that were left
out — the *out-of-bag* rows. Let `err_oob` be the mean of those scores. Let `γ`
be the *no-information* error rate, the score the model would get if its
predictions carried no information about the responses: pair the fitted values
with a random permutation of the targets and average.

```
R̂         = (err_oob − err_app) / (γ − err_app)     clamped to [0, 1]
ŵ         = 0.632 / (1 − 0.368·R̂)                   so 0.632 ≤ ŵ ≤ 1
err_.632+ = (1 − ŵ)·err_app + ŵ·err_oob
```

`R̂ = 0` (no extra overfitting beyond the apparent error) gives `ŵ = 0.632`;
`R̂ = 1` gives `ŵ = 1`, the fully pessimistic out-of-bag error. Because `ŵ`
never leaves `[0.632, 1]`, `err_.632+` is a convex combination of `err_app` and
`err_oob` and therefore always lies **between** them.

```sh
rcsr --data data.csv --target y --vars x1,x2 --bootstrap-error 200
# bootstrap error (rank 1): apparent=0.407152 out-of-bag=0.413428 .632+=0.411133 (200 resamples, 200 used, weight=0.634)
```

The flag costs `B` constant refits per reported solution; `B = 0` (the default)
switches it off. The three numbers are reported per solution in the text output,
in each JSON row's `bootstrap_error` object, and in the ndjson `solution` events.

### 3.10 Constant standard errors — implemented

A model can fit perfectly and still rest on constants the data does not
determine. The **standard error** of a fitted constant says how tightly the data
pins it down; the **relative uncertainty** `se / |value|` makes that comparable
between constants of different size.

The constants are the parameters of a nonlinear least-squares problem, so the
usual linearised theory applies at the solution. Write `J` for the residual
Jacobian — one row per training point, one column per constant, from the same
adjoint sweep the fitter already uses — and let `RSS` be the residual sum of
squares. Then

```
cov     = sigma^2 · (JᵀJ)^-1
sigma^2 = RSS / (n − k)          # residual variance, n − k degrees of freedom
se_i    = sqrt(cov_ii)
rel_i   = se_i / |value_i|
```

with `k` the number of fitted constants. The standard errors are in the model's
own (normalised) units, so quote the relative uncertainty when comparing.

How to read it:

* **small `rel_i`** — the constant is determined by the data;
* **large `rel_i`** (order 1 or more) — the constant is barely constrained, and
  a different value fits about as well;
* **`ill_conditioned: true`** — `JᵀJ` is (near-)singular because two or more
  constants can trade off (for example `a + b`, or a coefficient pair the data
  cannot separate). The errors are then meaningless and reported as `inf`; treat
  those constants as **unidentified** and drop the model or fix one of them.

`JᵀJ` is scaled to a correlation-like matrix before inversion so the conditioning
test does not depend on the magnitude of each constant, and inverted by Gaussian
elimination with partial pivoting. The errors are only defined when `n > k`;
with no residual degrees of freedom (or no constants at all) the field is
`null` rather than a made-up number.

```sh
rcsr --data data.csv --target y --vars x1,x2
# constant errors (rank 5): 1.013675±1.33e-7 (rel 1.31e-7), 2.495876±1.33e-7 (rel 5.32e-8) | dof=28
```

### 3.11 Stability selection — implemented

A real relationship is **rediscovered** when the data is resampled; a fit to
noise is a lottery winner. `--stability B` bootstraps the training rows `B`
times (with replacement), re-runs the **identical** search on each resample, and
counts what comes back:

* how often each **structural form** is selected, and
* how often each **input variable** appears.

The form is the expression with every fitted constant collapsed to `c`, so
`c + c·x`, `c − c·x` and `c·(c + x)` count as one law — they are the same family
with different free constants. Polynomials are compared by the terms they
contain; anything non-polynomial (`sin`, `exp`, division by a variable, `if`, …)
falls back to its structural shape. The representative of each resample is the
model the search would **recommend**: the Pareto member with the lowest
held-out error. Picking by training error would let a spurious term win by a
numerical hair, which is the very failure this check exists to expose.

```sh
rcsr --data data.csv --target y --vars x1,x2 --stability 8
# stability (8 resamples, 8 found a solution): modal form 88%  y = (0.7536 + (2.285 * x1))
#   variable selection: x1 8/8 (100%) | x2 0/8 (0%)
#   forms: 7/8 (88%) (0.7536 + (2.285 * x1)) | 1/8 (12%) (2.285 * x1)
```

How to read it:

* a variable at or near **100%** is one the data insists on; a variable well
  below is one the search only sometimes needs — usually noise;
* the **modal form's** frequency is the honest confidence in the *shape* of the
  answer, not just in its error;
* every reported solution carries `stability`, its own form's frequency, so a
  low-error solution that no resample reproduced is visibly a coincidence;
* the whole summary is reproducible from `--seed`.

Cost: `B` extra searches, so it is opt-in and best used on the final candidate
set rather than inside a tuning loop.

## 4. What `rcsr` reports today

Every solution carries:

| Field | Meaning |
|---|---|
| `train_error` | error on the fitting rows |
| `test_error` | error on held-out rows (mean over `--folds` blocks) |
| `fold_errors` | the individual held-out blocks |
| `k_constants` | fitted constants — the model's free parameters |
| `aicc`, `bic` | fit vs. constants (lower is better) |
| `durbin_watson` | residual autocorrelation in row order (≈2 clean) |
| `overfit_index` | `test_error / train_error` |
| `constant_errors` | per-constant standard error and relative uncertainty, plus an `ill_conditioned` flag for unidentified constants (§3.10) |
| `stability` | fraction of `--stability` resamples that selected this same structural form (§3.11); `null` when the flag is off |

and, with `--null-runs n`, a top-level `null_control` object with the shuffled
error distribution, its minimum and median, the permutation p-value and a plain
verdict. With `--bootstrap-error n`, every Pareto row additionally carries a
`bootstrap_error` object with the `apparent`, `oob`, `no_information`,
`relative_overfitting`, `weight` and `estimate` values of §3.9. With `--cv
<scheme>`, a top-level `cross_validation` object reports the scheme, `k`, the
`mean_train_error`, `mean_test_error` and `std_test_error`, and the per-fold
`train_error` / `test_error` / `complexity` / `expression`. With `--stability
B`, a top-level `stability` object carries the runs, how many found a solution,
and the ranked `forms` and `variables` with their counts and frequencies.

Alongside the front, `rcsr` reports:

* a top-level **`baselines`** object with the training and test error of the
  constant mean, the least-squares line and the `k`-NN learner (`k = 5`), in the
  search's metric;
* a top-level **`residual_tests`** object for the best (lowest training-error)
  solution:

| Field | Meaning |
|---|---|
| `durbin_watson` | residual autocorrelation, lag 1 (≈2 clean) |
| `ljung_box.statistic`, `ljung_box.df`, `ljung_box.p_value` | autocorrelation over `lags = min(10, n/5)` lags (χ²) |
| `breusch_pagan.statistic`, `breusch_pagan.df`, `breusch_pagan.p_value` | residual variance vs. the fitted value (χ²) |
| `skewness` | residual asymmetry (0 for a symmetric residual) |
| `kurtosis` | excess kurtosis (0 for a Gaussian) |

## 5. Not yet implemented (on the roadmap)

* **Beta / variance-ratio drift tests**, and a **White** general
  heteroscedasticity test (more regressors than just the fitted value).
* **A flexible non-symbolic learner beyond k-NN** (random forest, Gaussian
  process) as an additional baseline.

## 6. References

* Akaike 1974; Burnham & Anderson 2002 — information criteria, AICc.
* Durbin & Watson 1950, 1951 — residual autocorrelation.
* Ljung & Box 1978, Biometrika 65:297–303 — the portmanteau autocorrelation test.
* Breusch & Pagan 1979, Econometrica 47:1287–1294 — heteroscedasticity test.
* Rücker, Rücker & Meringer 2007 — y-randomisation in QSAR.
* Wold & Eriksson — response permutation / chance correlation.
* Meinshausen & Bühlmann 2010, JRSS-B 72:417–473 — stability selection.
* López de Prado 2018 — purged/embargoed, blocked cross-validation.
* La Cava et al. 2021 — SRBench (in- vs out-of-distribution).
* Matsubara et al. 2022 — SRSD / SRSD-Feynman benchmark suites.
* Udrescu & Tegmark 2020 — AI Feynman.
* Cranmer et al. 2020 — PySR / symbolic discovery.
* Bomarito et al. 2022 — Bayesian model selection against GP bloat and
  overfitting.
* Luke & Panait 2006 — lexicographic parsimony pressure.
* Laumanns et al. 2002 — ε-dominance.
* Efron & Tibshirani 1997 — the .632+ bootstrap error estimate.
* Schmidt & Lipson 2009 — distilling laws with noise-robust objectives.
* Bates & Watts 1988; Seber & Wild 2003 — nonlinear least squares: parameter
  standard errors and conditioning from the Jacobian.
