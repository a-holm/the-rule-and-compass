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
selection** (Meinshausen & Bühlmann 2010). `rcsr` currently reports per-fold
errors but does not yet automate a seed/bootstrap stability run; it is on the
roadmap.

### 2.6 Respect the physics when you have it
If the variables carry units, a true equation is dimensionally consistent
(Buckingham's π theorem). Requiring that eliminates huge families of spurious
candidate structures before any statistics. The original tool exposed this as
per-variable metadata and building-block constraints; `rcsr` does not yet.

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

### 3.2 Hold-out and cross-validation
The direct test: error on rows the search never saw. Variants that matter:
* **k-fold** for exchangeable rows (`--folds k` reports per-block error).
* **Forward-chaining / rolling-origin** for time series — train on the past,
  test on the next block, roll forward.
* **Purged, embargoed, blocked CV** for dependent data (López de Prado 2018).

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

### 3.4 Residual diagnostics — Durbin–Watson implemented
If anything is left in the residuals, the model is incomplete. **Durbin–Watson**
(Durbin & Watson 1950/51),

```
d = Σ(e_t − e_{t−1})² / Σ e_t²
```

is ≈2 for no autocorrelation, →0 for positive, →4 for negative. It doubles as a
general misspecification test. Reported as `durbin_watson` and only meaningful
when the row order is real (temporal splits) — `rcsr` says so in the output when
the order is arbitrary. See also Ljung–Box for autocorrelation, and
Breusch–Pagan / White for changing variance.

### 3.5 Compare against a null model
A real model should beat the trivial baselines: the mean, a straight line, and a
flexible non-symbolic learner (k-NN, random forest, Gaussian process). If it does
not, the "law" is not earning its complexity.

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
#    aicc, bic, durbin_watson, overfit_index are printed for every solution
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

and, with `--null-runs n`, a top-level `null_control` object with the shuffled
error distribution, its minimum and median, the permutation p-value and a plain
verdict.

## 5. Not yet implemented (on the roadmap)

* **Cross-validation as a first-class mode** (k-fold / rolling-origin over the
  whole search, not just per-block reporting).
* **Bootstrap and seed stability** — how often the same form reappears
  (stability selection).
* **.632 / .632+ bootstrap error** for small samples (Efron & Tibshirani 1997).
* **Parameter uncertainty** — standard errors of the constants from the
  Gauss–Newton Jacobian (`JᵀJ`), to flag constants the data does not pin down.
* **Null baselines** (mean/linear/k-NN) reported automatically alongside the
  front.
* **Units / dimensional consistency** as a structural constraint.
* **Ljung–Box** and heteroscedasticity tests.

## 6. References

* Akaike 1974; Burnham & Anderson 2002 — information criteria, AICc.
* Durbin & Watson 1950, 1951 — residual autocorrelation.
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
