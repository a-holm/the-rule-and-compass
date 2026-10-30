# Is the search converging? Confidence, stability and maturity

`rcsr` reports not just the best error it has found but whether the search is
still moving. Four numbers describe that, all computed from the Pareto archive
alone:

| name | range | question it answers |
|---|---|---|
| **confidence** | 0 … 1 | did the *best error* stop improving? |
| **stability** | 0 … 1 | do the *structural forms* on the front survive? |
| **maturity** | 0 … 1 | how long has the oldest form been there? |
| **percent converged** | 0 … 1 | one headline combining the three |

They are emitted with every `--progress` event, in the final `--json` object
under `"convergence": {...}`, and in the NDJSON `progress` and `done` events.
Without `--progress`/`--ndjson` the default text output is unchanged; `--json`
is already a machine-readable opt-in and always includes the object. The code
is [`my-version/src/progress_stats.rs`](../my-version/src/progress_stats.rs).

## The window

All three components compare generation `g` with an earlier generation. The
window is

```
w = 20 generations
```

a documented constant rather than a flag: these are *readings* of the search,
not tuning knobs. Twenty generations is long enough that one lucky generation
does not dominate, and short enough to react on the default 200-generation
budget. Before generation 20 the oldest available generation is used, so the
numbers are defined from the first generation onwards.

The cost is **O(front size) per generation**. The tracker hashes the front's
structural forms and keeps at most `w + 1` small snapshots; it never touches a
training row. The tracker is not part of the checkpoint: a resumed run starts a
fresh window at the resumed generation, so its first reading has less history.

## The definitions

Let `F_g` be the set of **structural forms** on the Pareto front at generation
`g`. A form is the expression with every fitted constant collapsed to `c` and
commutative arguments sorted — the same canonical spelling that
`--stability` reports (`stability::shape`). Two members that differ only in the
numerical value of a fitted constant therefore count as the same form.

### Confidence

Let `e_g` be the best (lowest) error on the front at generation `g`, and let
`e_ref` be the best error at the oldest generation still inside the window
(generation `g - w`, or the first generation when `g < w`). Define the relative
improvement over the window

```
r = max(0, (e_ref - e_g) / (1 + |e_ref|))
```

and map it to `0 … 1` with the documented scale `R = 1e-3`:

```
confidence = exp(-r / R)
```

* No movement (`r = 0`) gives **1**.
* A 0.1 % relative improvement gives `e^-1 ≈ 0.37`.
* A large improvement gives a number near **0**.

The `1 + |e_ref|` denominator is the same one the search's patience test uses,
so "no improvement" means the same thing in both places. (Patience calls
`r <= 1e-9` no improvement; that maps to confidence `> 0.999999`.) If there is
no earlier generation, or either error is not finite, confidence is **0**: we
have no evidence yet. If the current best error is at or below the search's
solved floor (`1e-24`), confidence is **1**: there is nothing left to improve,
even when the window contains the jump to the exact answer.

### Stability

For two form sets `A` and `B`, use the Jaccard overlap

```
J(A, B) = |A ∩ B| / |A ∪ B|        (J(∅, ∅) = 1)
```

Let `J_i = J(F_i, F_{max(g0, i-w)})`, where `g0` is the first generation
observed (`1` for a fresh run); for `i < w` this compares against the first
observation. Stability is the mean of the most recent values (at most `w`, and
none before the second observation):

```
stability = (1/k) * sum of the last k values of J,    k <= w
```

Equivalently: the mean Jaccard overlap between each generation in the window
and the generation `w` before it. Identical fronts score **1**; disjoint fronts
score **0**. Before the first comparison there is no evidence, so stability is
**0**.

### Maturity

For each form currently on the front, record the generation at which its
*current continuous run* on the front began. A form that leaves and later
returns starts again. Let `a_g` be the smallest such generation:

```
maturity = (g - a_g) / max(g, 1)
```

An empty front, or generation 0, gives **0**. A form that has survived since
generation 1 and is still present at generation 100 gives
`(100 - 1) / 100 = 0.99`. Replacing every form resets maturity to 0 and it then
climbs from the replacement generation.

### Percent converged

The headline number is the weighted sum

```
percent_converged = 0.5 * confidence
                  + 0.3 * stability
                  + 0.2 * maturity
```

Confidence carries the most weight because "the best error stopped moving" is
the most direct evidence of convergence; stability guards against a front that
is merely stuck on one point; maturity rewards forms that have survived. The
result is clamped to `0 … 1`. We call a run **converged** when

```
percent_converged >= 0.75
```

Nothing in the search depends on this threshold — it changes no stopping rule —
it is only a reading for the caller. Early stopping is still controlled by
`--patience` and `--time-budget`.

The result JSON carries `stop_reason` (`solved`, `patience`, `time_budget`,
`generations` or `already_stopped`) and `converged_early` (#256).
`converged_early` is true only when the engine stopped on `solved` or
`patience` **and** the final `percent_converged` is at least `0.75`. A
`time_budget` stop never sets it: the engine did not declare convergence. The
terminating generation of a `time_budget` stop still emits its progress and
development-log events.

## Reading the numbers

| reading | interpretation |
|---|---|
| confidence high, stability high, maturity high | the search has settled: the best error is flat and the same forms stay on the front |
| confidence low | the best error is still falling; the run is improving |
| stability low | the front's structure is churning; new forms keep replacing old ones |
| maturity low | even if the error is flat, the front is young — a recent restart or a new form |
| percent converged ≥ 0.75 | the conventional "converged" range |

A low stability with a high confidence is a warning: the best error has
plateaued, but the rest of the front is still being rewritten, so the *shape* of
the answer is not settled yet. Conversely, high maturity with low confidence is
the normal picture of slow, steady refinement.

## Evidence: one easy and one hard problem

Both runs use `--islands 1 --pop 120 --metric squared`, a 150-generation budget
and the defaults for everything else. The numbers below are the final
`"convergence"` objects.

**Easy** — `y = 2x + 1`, 40 noiseless rows, seed 1. The search found the law and
stopped early at generation 42:

```json
{"confidence": 1.0, "stability": 1.0, "maturity": 0.97619,
 "percent_converged": 0.995238}
```

**Hard** — `sin(2x) + 0.4 sin(13x)` plus deterministic noise on 80 rows, a
complexity cap of 80, seed 2. It used all 150 generations:

```json
{"confidence": 0.374751, "stability": 0.734167, "maturity": 0.993333,
 "percent_converged": 0.606292}
```

The easy run is comfortably inside the converged range and the hard run is not.
The hard run's maturity is high because some forms do survive; its low
confidence and churning front are what keep the headline down. On a noiseless
problem the search can also find the exact law in two generations and stop on
the search's solved floor; confidence is then 1 by definition, stability is 1
and maturity is only `(2-1)/2 = 0.5`, giving `percent_converged = 0.9`. The
tests in
[`my-version/tests/progress_stats.rs`](../my-version/tests/progress_stats.rs)
run an easy and a hard problem with the same shape of configuration and assert
that the first lands in the converged range and the second does not (the numbers
above are from the CLI runs, not hard-coded test expectations).

## Caveat: this measures the search, not the truth

Convergence means the **search** has stopped changing. It does **not** mean the
formula is correct, or that it will generalise. A search can converge happily on
a wrong law — a noisy target, a too-small complexity cap, or a local optimum all
produce a flat, stable, mature front around the wrong answer. Read the
statistics together with the held-out error, the baselines, the overfitting
diagnostics and `--stability`; a high `percent_converged` is evidence about the
optimiser, not about the data. In particular, do not use it as a substitute for
`--cv`, `--null-runs` or a test split.
