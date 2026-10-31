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
fool yourself. [`time-series.md`](time-series.md) turns the whole sequence —
split, purge, preparation, lockbox, trials and costs — into a checklist.

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

**Block permutation for autocorrelated series.** Row-by-row shuffling destroys
serial dependence, so for overlapping labels or autocorrelated returns the i.i.d.
null is *anti-conservative*: the real fit looks better than a shuffled target
that no longer resembles the data. `--null-block N` permutes contiguous blocks
of `N` consecutive target rows instead (all rows are kept exactly once, and the
same seed reproduces the null):

```sh
rcsr --data data.csv --target y --vars x1,x2 --split temporal --time t \
     --null-runs 99 --null-block 24
# ... null control (99 block-permutation runs, 7 blocks of up to 24 rows,
#     observed=..., p=0.03) ...
```

`N` must be at least the label horizon (how many rows a label spans), so each
block still contains the dependence the label has. The report and JSON record
`method: "block"`, `block_len`, `blocks`, `observed`, the null distribution and
`p_value`. `--null-block` without `--null-runs` runs 19 permutations. Use the
i.i.d. null (`--null-runs` alone) only when rows are genuinely exchangeable;
use the block null whenever time order or overlapping labels create dependence.
The calibration test in `my-version/tests/nulls.rs` shows the difference on
synthetic AR(1) data: i.i.d. shuffling over-rejects, block permutation does not.

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
# knn baseline: reference_rows=2500 train_query_rows=2500 test_query_rows=0 sampled=false
```

**Cost.** The baseline's **reference set is always every training row**, at
every `n`: it is never subsampled, because that would change the estimator.
What changes is how the queries are answered. For up to eight inputs `rcsr`
builds a balanced k-d tree once over the training rows and prunes with
bounding boxes; above eight inputs it uses the exact brute-force path with a
reused distance buffer and `select_nth_unstable`, so it does not allocate and
fully sort per query. Ties break by original row order either way, and the
20 000-row number matches the pre-#69 all-rows value bit for bit.

The k-d tree is not claimed to be `O(log n)` per query: that is the expected
behaviour for fixed dimension, and the measured 20k→40k step is larger. What the
tree removes is the old per-query full sort; the practical claim is a k-d tree
for `d <= 8`. The `d > 8` exact path is still `O(query_rows × reference_rows ×
d)`, so "200 000 rows under a minute" is a **`d <= 8` claim**; for wider data,
sample queries or expect the brute-force cost. Measured on this machine
(`--generations 0 --pop 1`): 200k rows take 12–18 s at `d = 6` and 35 s at
`d = 8`; at `d = 12` the exact 200k run did not finish within 120 s, and the
20k exact runs take 1.1 s (`d = 8`), 6.1 s (`d = 12`) and 6.8 s (`d = 16`).
The cutoff is a practical choice, not a theorem: the literature puts k-d trees
in their intended regime around `d <= 10` and worse than linear search beyond
about 16.

The **query rows** are bounded by default: `--baseline-samples N` scores at
most `N` deterministic query rows per set (training and each test fold), seeded
from `--seed`, and the default is **512**. On a small panel (at most 512 rows)
that is every row, so the result is exact. This is offered only
for the mean-loss metrics **absolute**, **squared** and **log_squashed**, where
the sample average is an unbiased estimate of the full row-average loss. For
median, maximum, correlation, rank correlation, interquartile, slope,
classification and information-criterion metrics a subsample is a different
estimator, so the CLI refuses the flag. When sampling happens the text line
says `sampled=true`, and `--json` carries `baselines.knn.train_query_rows`,
`train_query_sem`, `test_query_rows` and `test_query_sem`. The SEMs use the
finite-population correction for sampling without replacement,
`(s/√q)·√((N−q)/(N−1))`; they are `null` when every row was scored.
`--baseline-samples 0` explicitly means no caps at all: every query row is
scored, which is `O(N_train²·d)` and should be opt-in on large panels.
`baselines.knn.reference_rows`, `train_query_rows`, `test_query_rows`,
`capped`, both SEMs and `seconds` are reported. The reference set is always the
training size.

At 20k rows (15k training) `--baseline-samples 4096` gives a 95% interval of
roughly ±2.5% of the 5-NN value; ask for about 9.5k queries for ±1%. The
training 5-NN includes the query row itself, as it always has, so the training
error is optimistic; the test error is the out-of-sample number.

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
choice. `--metric` selects one of twenty objectives:

* `absolute` (default) and `squared` — squared punishes large residuals hard;
* `median` and `interquartile` — robust to a few outliers;
* `log_squashed` — squared-like but with bounded influence;
* `maximum` — minimise the single worst row;
* `signed` — penalise a consistent bias;
* `correlation` and `rank_correlation` — care about shape, not level, offset or
  **direction**: both score `1 - |r|`, so a perfectly inverted forecast scores
  zero. Use them when either sign is a valid law (e.g. an unsigned relation);
* `correlation_signed` and `rank_correlation_signed` — score `1 - r`, so the
  sign is part of the objective and an inverted forecast is penalised. Use them
  for directional signals (returns, demand, a control response) where the
  forecast has to rise and fall with the target;
* `slope` — match the trend rather than the level;
* `auc`, `log_loss`, `hinge` — classification; the target must have exactly two
  distinct values (or one, which scores as chance). The model output is a
  logit. A `{-1, 0, +1}` target is **rejected** rather than silently split at
  the midpoint; collapse it to two classes first. `auc` is the only
  direction-blind classification metric (it rewards rank separation), while
  `log_loss` and `hinge` also reward the correct logit sign;
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

**Resample size and the BLB cap.** Each draw is an index selection over the
training rows; only the selected rows are materialised, so the per-resample
memory is `O(b·d)` rather than `O(n·d)`. When `n^0.7 >= 1000` (about 19 300
training rows) the draw is bounded with the **bag of little bootstraps**
(Kleiner, Talwalkar, Sarkar & Jordan 2014): draw a subsample of `b = n^0.7`
distinct rows and resample within it. A 200 000-row file with the default 25 %
test split has 150 000 training rows, so `b = 4200`; the cap follows the
*training* size, not the file size. Below the threshold the exact bootstrap runs
unchanged and is bit-identical to the pre-#92 results. The text output and the
`bootstrap_error` object report `resample_rows`, `source_rows` and `capped`;
above the cap the text says `rows=4200 of 150000 (BLB cap b=n^0.7)`.

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

**Structural identification.** The pivot ratio is not a singular value, so it misses an exact alias that rounding hides (for example `c0 + c1/(c2 + exp(x1)) + c3*(c4 - x2)`, where `c0` and `c4` trade off). A cyclic Jacobi eigendecomposition of the scaled matrix gives `rank = #{lambda_i > 1e-8 * lambda_max}` (condition index above 1e4 is a lost rank) and `condition_index = sqrt(lambda_max / lambda_min)`. The constants that carry a lost direction (Belsley variance-decomposition proportion at least 0.5) are `involved`. A rank-deficient model has `ill_conditioned: true` and a `null` standard error for each involved constant. Full-rank errors are unchanged. Runs with the validation gates add `rank`, `condition_index`, `involved`, `std_errors_robust` and `relative_robust` to `constant_errors`. The robust errors use the Driscoll-Kraay sandwich `(JᵀJ)⁻¹ Ω (JᵀJ)⁻¹`; `practically_identified` is true when every robust relative error is at most 1 and is a flag only. A structurally unidentified member stays in the front, labelled, and is never supported or selected.

```sh
rcsr --data data.csv --target y --vars x1,x2
# constant errors (rank 5): 1.013675±1.33e-7 (rel 1.31e-7), 2.495876±1.33e-7 (rel 5.32e-8) | dof=28
```

### 3.10b Validation gates — implemented

A Pareto front lists models; it does not say whether any of them beats a trivial reference. The validation gates answer that on the validation slice, after the search. The search, the in-search archive and its hash do not change. The gates run on time-ordered runs (`--split temporal|temporal-group`, `--time`), whenever a target spec is in force, and with `--require-validation`. Test errors are scoring-only: they never enter a gate.

**Noise.** For two models scored on the same validation rows, `d_i = loss_ref,i - loss_model,i`. The standard error of `mean(d)` is Driscoll-Kraay: per-row influences are summed within each timestamp, then combined across timestamps with Bartlett weights up to a lag `L`. This is valid for cross-correlated panels where the iid standard error is too small. `L` is `--noise-lag`, else the larger of the label horizon in timestamps minus one (from the target spec) and `floor(4 (T/100)^(2/9))`. A random split uses one key per row and `L = 0`. Row losses exist for `squared`, `absolute`, `log_squashed`, `log_loss` and `hinge`; `aic`, `aicc` and `bic` use the squared row loss. Other metrics report `not_gated` and keep the old behaviour.

**Reference rungs.** With a target spec that has a structural term (see `docs/time-series.md` section 2b), `baselines.target_structure.rungs` holds four rungs fitted on fit-train only: `zero_param` (`S`, 0 parameters), `affine` (`a + b*S`, 2), `binned20` (train mean of `y` in 20 equal-count bins of `S`, edges from train, 20) and `construction_ols` (`y` on the scale column, 2). Each reports train, validation and test error. The references compared on every member are `mean`, `linear` and, with a structure, `zero_param` and `affine`.

**Supported front.** The identified members with a finite validation loss are ordered by `(cost, validation loss, structural hash, expression)`. Starting from the reference `R` (`affine` with a structure, else `mean`), a member is *supported* when `mean(loss_last - loss_member) > z * SE`, with `z = --equivalence-z` (default 1). The member then becomes `last`. An unsupported member carries `not_better_than` (the entry that blocked it, `delta`, `se`). This is `Archive::canonicalize()` with "strictly improves" replaced by "improves by more than z standard errors".

**Verdict.** `selection.verdict` is one of:

| verdict | meaning | `selected_rank` |
|---|---|---|
| `signal` | the top supported member beats `R` by more than `z_null` standard errors | its archive rank |
| `no_signal` | no member is supported, or the top one fails `delta > z_null * SE` | null |
| `unvalidated` | no validation slice | null |
| `not_gated` | the metric has no per-row loss | null |

`z_null = Phi^-1(1 - 0.05 / K)`, a one-sided Bonferroni threshold over the `K` eligible members. On `no_signal` the text names the reference as the model to quote; the front is still printed with its labels. Formal inference over many trials (deflated Sharpe, PBO, SPA) is not part of a single run.

**Without a slice.** A time-ordered run without `--validation-frac` prints one warning, reports `unvalidated` and selects nothing; the front is a menu. `--require-validation` turns this into a refusal (exit 2).

**Per-member flags.** Every member reports `comparisons.validation` and `comparisons.test` against each reference (`delta`, `se`, `better|equivalent|worse` at `z`) and the flags `beats_mean_test`, `beats_zero_param_test`, `beats_mean_validation`, `beats_zero_param_validation` (null without a structure) and `validated` (the member is the selected one). With a target spec, `return_space` reports for each slice the correlation of `r_hat = c + m*g*y_hat` with `r` and `R2 = 1 - SSE/SS0` against the zero return forecast, using `TargetSpec::return_from_target`. Rows at a clip bound are excluded because the label only bounds the return there.

**Presentation rule.** A member whose return-space `r2_vs_zero` on the validation sample is negative is not presentable: it reports `presentable: false` with `presentable_reason: "negative_return_r2_on_validation"`, takes no part in the supported-front walk and is never `selected_rank`. It stays in `pareto` with all its comparisons. The test block never decides presentability. Without a target spec every member is presentable.

**Shuffled test.** Permuting the targets of the test block changes `comparisons.test` only. `selection`, `noise` and every `gates` object are byte-identical.

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

**Resample size and the BLB cap.** As in §3.9, the resample is an index draw and
the per-resample search sees only the selected rows. Above the threshold
(`n^0.7 >= 1000`, about 19 300 rows) each run searches a subsample of
`b = n^0.7` distinct rows (Kleiner et al. 2014). This changes the stability
estimator above the cap — the review on #85 notes that a search inside each
resample is not covered by BLB's differentiability theorem — so the summary
reports `resample_rows`, `source_rows` and `capped`. Below the threshold the
summary is bit-identical to the pre-#92 result.

### 3.12 Multiple testing for a long-only book (`rcsr walkforward --bootstrap`)

A walk-forward report on many candidates is a selection problem. The report quantifies it on the pooled development series:

| statistic | what it answers | threshold used by the protocol |
|---|---|---|
| block bootstrap CI and design effect | is the mean or Sharpe distinguishable from zero once labels overlap and returns are serially dependent | CI excludes 0 |
| `N_eff` (participation ratio) | how many independent trials the candidate set is worth | reported |
| DSR at `N_eff` and `10 N_eff` | probability the true Sharpe is positive after deflating for the trials, skew and kurtosis | reported with its inputs |
| PBO (CSCV) | how often the in-sample winner ranks in the lower half out of sample | at most 0.20 |
| SPA vs the best long/flat baseline | does any candidate beat cash, buy-and-hold or the constant book after data snooping | p at most 0.05 |
| BH-FDR | which candidates survive a false discovery rate | q = 0.10 |
| block-permutation and matched-exposure nulls | does the score time returns better than chance | `p = (1+count)/(B+1)` |

Choices that the formulas leave open: block length `max(h, hold, ceil(n^(1/3)))`; 10000 draws is the recommended `--bootstrap`; the SPA uses the circular block bootstrap with Hansen's consistent, lower and upper recentering; PBO uses the per-period Sharpe over 16 blocks. A locked test is read once per sealed candidate (see `docs/walkforward.md`); a small tweak after a failed read is a new trial that inflates `N`.

Verify: `tests/inference.rs` (hand-computed CVaR, break-even, BH, PSR and PBO fixtures, the AR(1) and overlapping-label design effects, SPA on signal and noise) and `tests/eval_protocol.rs`.

## 4. What `rcsr` reports today

### The test lockbox and the validation slice

The holdout is a **lockbox**: a block you may look at to report an honest error,
but not to choose a model. Selecting the best of many candidates by
`test_error` spends the block and makes the reported number optimistic. With
`--validation-frac f`, the last `f` share of the chronological training period
becomes a validation slice, separated from fit-train by the `--gap` embargo.
Selection ranks on validation by default (`--select-by validation|train|aicc`)
and `test_error` stays scoring-only. The synthetic test in
`my-version/tests/validation.rs` demonstrates the leak: ranking by test reports
a smaller test error than the validation-selected candidate, while the
test-selected candidate is the worse model on validation.

Every solution carries:

| Field | Meaning |
|---|---|
| `train_error` | error on the fitting rows |
| `validation_error` | error on the chronological validation slice when `--validation-frac` is set; the default selection criterion |
| `test_error` | error on the held-out **lockbox** (mean over `--folds` blocks). Scoring-only: never rank candidates by it |
| `fold_errors` | the individual held-out blocks |
| `k_constants` | fitted constants — the model's free parameters |
| `aicc`, `bic` | fit vs. constants (lower is better) |
| `durbin_watson` | residual autocorrelation in row order (≈2 clean) |
| `overfit_index` | `test_error / train_error` |
| `constant_errors` | per-constant standard error and relative uncertainty, plus an `ill_conditioned` flag for unidentified constants (§3.10) |
| `stability` | fraction of `--stability` resamples that selected this same structural form (§3.11); `null` when the flag is off |

and, with `--null-runs n`, a top-level `null_control` object with the shuffled
error distribution, its minimum and median, the permutation p-value and a plain
verdict. With `--null-block N` the same object adds `method: "block"`,
`block_len`, `blocks` and the `observed` error, so a report says which null was
used. With `--bootstrap-error n`, every Pareto row additionally carries a
`bootstrap_error` object with the `apparent`, `oob`, `no_information`,
`relative_overfitting`, `weight` and `estimate` values of §3.9, plus
`source_rows`, `resample_rows` and `capped`, so a BLB-bounded resample is never
mistaken for an exact one. With `--cv <scheme>`, a top-level
`cross_validation` object reports the scheme, `k`, the `mean_train_error`,
`mean_test_error` and `std_test_error`, and the per-fold `train_error` /
`test_error` / `complexity` / `expression`. With `--stability B`, a top-level
`stability` object carries the runs, how many found a solution, the same three
resample fields, and the ranked `forms` and `variables` with their counts and
frequencies.

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
* Bentley 1975; Friedman, Bentley & Finkel 1977 — k-d trees and exact k-NN.
* Kibriya & Frank 2007; Beyer et al. 1999 — k-d tree query cost and the
  curse of dimensionality for nearest neighbours.
* Cochran 1977, *Sampling Techniques*; NIST/SEMATECH e-Handbook §1.3.5.2 —
  standard error with the finite-population correction.

