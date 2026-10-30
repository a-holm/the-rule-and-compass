# Quality-diversity layer (#120)

The engine has **reset diversity** (restarts, immigrants, structural
de-duplication). It did not have **niche diversity**: nothing recorded which
behaviour regions the search never reached, so thousands of evaluations could
collapse onto one prediction shape without the search noticing.

The QD layer is opt-in. It has two levels:

* **passive** (`--qd`): record descriptor cells and report coverage, but never
  change parent selection, migration, restarts or the reported Pareto front;
* **policy** (`--qd-policy`): feed the archive back into the search —
  UCB1 parent draws, stratified cell migration and least-occupied restarts.

With QD disabled the trajectory is bit-identical to before (pinned by
`tests/qd.rs`).

## Descriptors

| axis | definition | range |
|---|---|---|
| variable usage | distinct input columns read by the tree, divided by the number of inputs | `[0, 1]` |
| prediction skewness | third standardised moment of the prediction vector, mapped by `0.5 + 0.5·s/(1+|s|)` | `[0, 1]` |

`PredictionStats::from_prediction` accumulates `Σp, Σp², Σp³, Σp⁴`, sign
changes and the mean in one local pass; skewness and kurtosis are derived from
those moments.

**Descriptors are accumulated in the scoring pass.** When the QD layer is on,
`search::eval_error_and_descriptor` scores the candidate and builds the
descriptor from the *same* prediction vector, so the CPU scoring path never
evaluates a tree a second time to classify it. `Ind` carries the descriptor
into the archive (`record_with_descriptors`), into migration's
nearest-to-least-visited selection and into restart bookkeeping. The QD archive
keeps a bounded, sample-tagged structural-hash cache for members it has already
classified.

Two paths still need the evaluating fallback (`qd::descriptor`): a tree scored
on the GPU (the kernel returns the error, not the prediction vector) and a tree
scored on a row subset (`--predictor-samples`/screening) whose descriptor would
describe different rows than the archive. A third case is the squared-metric
SSE reuse: the error is reused bit-for-bit so the passive layer cannot change
the trajectory, and the descriptor then costs one prediction pass — the same
cost as before. Those are identified in the report by a full evaluation, not
silently mislabelled. For every other metric the scoring pass already evaluates
the prediction vector, so the descriptor is free.

Measured descriptor cost (`cargo test --release --lib qd::tests::descriptor_cost
-- --nocapture`): **2.5 µs/candidate on 41 rows** and **46 µs/candidate on 800
rows**, including the tree evaluation in the convenience path. The in-pass form
adds only the moment accumulation to a sweep the search already performs.

## Archive

`--qd` enables the passive archive. `--qd-cells N` sets the grid side or the
CVT cell count `K` (`2..=512`; `K = 512` is the design size) and
`--qd-cell-cap N` bounds each cell's local front (default 4).

* `--qd` alone is the original rectangular `16 × 16` grid.
* `--qd-cvt` (or `--qd-policy`) uses **CVT-MAP-Elites**: `K` centroids are
  initialised deterministically from a Halton (2,3) sequence and updated by
  Lloyd steps over the descriptors actually visited (every 8 `record` calls,
  4 iterations, bounded 4096-descriptor sample). Existing cell elites are
  re-assigned to the moved centroids; their local fronts are preserved.
* Each occupied cell holds a **bounded local Pareto front over `(error, cost)`**
  with the same ε-dominance rule as the global archive. Insertion is
  deterministic: fixed island/population/archive order, dominated candidates
  dropped, equal cost/error keeps the first entry, the middle cost level is
  dropped at the cap.

## Search policy

`--qd-policy` implies `--qd --qd-cvt`. Parent draws use the issue's mix:

| draw | source |
|---|---|
| 60% | UCB1 quality cell: best local error minus `2·sqrt(ln(total+1)/(visits+1))` |
| 25% | uniform occupied cell |
| 10% | ring neighbour of a random occupied cell |
| 5% | global Pareto front (the existing `--front-parents` draw) |

Two equal-budget controls are first-class flags: `--qd-uniform` makes every
occupied cell equally likely, and `--qd-random` hashes descriptors to random
cells so the partition has no behavioural information. A gain that survives
only the proposed policy, not the controls, is search capacity rather than
niche diversity.

Migration keeps the ring but replaces the individual nearest the
**least-visited cell's centroid** instead of the globally worst error.
Restarting a stagnant island aims at the least-visited region: it samples four
random trees and keeps the one whose descriptor is nearest that centroid.
Seeds are inserted into the archive and keep their own niche.

Cell elites and promoted survivors are **refit exactly on all training rows in
f64** before the QD report is returned (`QdArchive::refit_all`), so screening
and `--predictor-samples` never turn the reported QD front into a subset
estimate. Search-time exploration is still allowed to use the cheap fit.

## Reported coverage and hypervolume

Text output adds two lines:

```text
# qd (cvt): 12/19 reachable cells occupied (63.2%), 31 entries, best error 1.145..., entropy 2.31/2.94 bits, qd-score 3.812, hypervolume 0.7143 (self-normalised diagnostic)
# validation hv (test): 4.201337 (ref error 0.912, 9 front points; fixed reference; complexity span 1..9, compare only runs with the same span)
```

`--json` adds a `qd` object and a top-level `validation_hv` object:

```json
"qd":{"mode":"cvt","bins":512,"cells":512,"reachable":19,
      "occupied":12,"occupied_reachable":12,"coverage":0.0234,
      "reachable_coverage":0.6316,"entries":31,"best_error":1.145...,
      "cell_cap":4,"occupancy_entropy":2.31,
      "occupancy_entropy_normalized":0.784,"qd_score":3.812,
      "descriptor_fallbacks":0,"hypervolume":0.7143,"evaluations":12000},
"validation_hv":{"sample":"test","reference_error":0.912,
                 "hypervolume":4.201337,"points":9,
                 "complexity_min":1,"complexity_max":9}
```

### What the coverage numbers mean

* `cells` is the design size (`K`, or `2 x bins^2`). `coverage = occupied /
  cells` is retained for compatibility but is **not a target**: it charges the
  search for regions the descriptor cannot express, and it changes when `K`, the
  grid or the CVT centroids change between arms.
* `reachable` is the number of cells whose Voronoi region intersects an
  *attainable* descriptor. Usage is `distinct inputs / nvars`, so only `nvars+1`
  values occur; skew is treated as continuous on `[0, 1]`. For a grid the
  denominator is the number of usage rows that contain one of those values times
  the skew resolution; for a CVT it is computed geometrically: on the vertical
  line `usage = u`, "centroid i is at least as near as centroid j" is linear in
  skew, so the intersection of those half-lines gives the skew interval where
  `i` wins. A cell is reachable when that interval is non-empty for at least one
  usage level. Ties count as reachable, which can only over-count a
  measure-zero boundary.
* `reachable_coverage = occupied_reachable / reachable` is the honest fraction.
  It is still a **diagnostic** in CVT mode because the centroids move during the
  run; the decision metric is the fixed-reference validation hypervolume below.
* `occupancy_entropy` is the Shannon entropy (bits) of the candidate-visit
  distribution over reachable cells; `occupancy_entropy_normalized` divides it
  by `log2(reachable)`. Coverage cannot distinguish "one offer in each of two
  cells" from "ten thousand offers in one and one in the other"; entropy can.
* `qd_score = Σ 1 / (1 + best_error)` over occupied reachable cells pairs
  coverage with quality. A descriptor map that occupies every cell with useless
  predictions has near-zero QD-score.
* `descriptor_fallbacks` counts archive insertions that needed the evaluating
  fallback (GPU-scored, subset-scored or previously unseen trees). It is `0`
  for a CPU run whose candidates all came through the one-pass scoring path,
  and is the direct evidence that QD no longer re-evaluates every candidate.
* `hypervolume` is the old dominated area **normalised by the arm's own observed
  `(cost, error)` extremes**. It is kept as a diagnostic and must not be
  compared between runs: a worse front can score higher by shrinking its span.

### Fixed-reference validation hypervolume

`validation_hv` is computed from the archive the run actually produced, on the
held-out sample (the chronological validation slice when present, else the test
fold, else training rows), with

```text
HV = Σ (c_{i+1} − c_i) · log10(1 + e_ref / e_i)
```

over the cost-sorted front, where `e_i` is the best error at or below `c_i` and
`e_ref` is the **variance of that sample's target** — a data constant, identical
for every arm of the same split. Because neither axis is rescaled by the arm's
own extremes, `validation_hv` is comparable across baseline, passive, policy,
uniform-cell and random-descriptor arms **that reach the same complexity span**:
the integral covers the cost range the front reaches, so a front that reaches a
higher cost adds area whatever its error. `complexity_min` and `complexity_max`
state the span; compare arms only when the spans match. It is emitted for single-output runs;
the multi-output JSON keeps its existing shape (the QD descriptor is not
accumulated there either).

## Why "material coverage" is not a valid target

The original acceptance box asked for coverage "materially higher" than the
current search at equal budget, with `K = 512` as the design size. That is not a
valid target:

1. `K` is a designer choice and CVT centroids move, so `occupied / K` measures
   the map, not the search, and differs between arms.
2. The descriptor has a small reachable support. With `nvars = 2` usage takes
   only `{0, 0.5, 1}`, so most of a K=512 map can never win; the random control
   fills `K/K` by construction. The PR's `K=16 -> 10-11` and `K=64 -> 18-19`
   numbers are descriptor quantisation, not a search failure or success.
3. Coverage without quality is trivially maximised by a map with no behavioural
   information (`--qd-random`).

Coverage is therefore reported as a **diagnostic** (`reachable_coverage`,
`occupancy_entropy`, `qd_score`), and the decision metric is quality per
resource: `validation_hv` on a fixed reference at equal budget. Any effect size
and seed count must be pre-registered before a claim is made.

## DSR and PBO are not reported, and cannot be

The acceptance box asks for the deflated Sharpe ratio at the effective trial
count and the probability of backtest overfitting for every budget. Neither can
be computed from this engine's current state:

* DSR needs a per-trial Sharpe-like statistic and `N_eff`; "Sharpe" is undefined
  for a generic regression target, and the search retains no trial family.
* PBO/CSCV needs an `S × configurations` matrix with the in-sample best
  recomputed per split. The engine keeps only the final front and population,
  and has no candidate-to-P&L mapping.

Those boxes stay **open**. They need a separate evaluation mode that retains
trial families and a candidate-to-P&L mapping, which is outside the QD layer.


## Checkpoints and exactness

`--save` writes `qd_bins`, `qd_cell_cap`, `qd_cvt` and `qd_policy` into the
checkpoint. The archive itself is rebuilt from the resumed search's evaluated
candidates; it is reporting state, not trajectory state. The passive layer never
changes the Pareto front. With the layer disabled, the exactness fixtures are
unchanged — `tests/qd.rs::qd_is_off_by_default_and_does_not_change_the_front`
and the pinned archive/checkpoint tests cover this.

## Benchmark and controls

The issue asks for equal-budget coverage, validation hypervolume per GPU-hour,
signal-to-search, DSR/PBO and random-descriptor/uniform-cell controls. The
implementation exposes the measurement hooks (reachable coverage, occupancy
entropy, QD-score, fixed-reference `validation_hv`, evaluations and wall time in
`elapsed_seconds`). The recorded numbers and the unmet boxes are reported in the
issue/PR, not baked into this document as a claim. The GPU-hour figure is a
GPU-hour only when the CUDA backend is active; a CPU run reports seconds and the
same quantity per CPU-second, which is the honest denominator for a CPU-only
measurement.

Signal-to-search (`ΔHV_real / ΔHV_null` with a block-permuted target) is
**not** implemented in this change; the fixed-reference HV makes it a small
harness away, but it is not claimed here. DSR/PBO remain open for the reason
above.
