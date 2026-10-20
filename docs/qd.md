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
those moments. The SoA/GPU evaluator can add the same accumulators to its SSE
reduction, so the descriptor does not require a second tree evaluation once a
prediction vector exists. The current search path still calls
`descriptor_from_prediction` with the prediction it already computed; the
`descriptor()` convenience wrapper evaluates once only for standalone callers.

Measured descriptor cost (`cargo test --release --lib qd::tests::descriptor_cost
-- --nocapture`): **3.2 µs/candidate on 41 rows** and **92 µs/candidate on 800
rows**, including the tree evaluation in the convenience path.

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

Text output adds one line:

```text
# qd (cvt): 141/512 cells occupied (27.5%), 318 entries, best error 1.145..., hypervolume 0.7143
```

`--json` adds:

```json
"qd":{"mode":"cvt","bins":512,"cells":512,"occupied":141,"coverage":0.275,
      "entries":318,"best_error":1.145...,"cell_cap":4,
      "hypervolume":0.7143,"evaluations":12000}
```

`coverage` is `occupied / cells`; `hypervolume` is the normalized dominated
area of the archive's `(cost, error)` points; `evaluations` is the number of
candidates offered. The same seed and flags produce the same coverage, cell
fronts and centroids.

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
implementation exposes the measurement hooks (coverage, hypervolume,
evaluations, wall time in `elapsed_seconds`, and the full Pareto front for
validation scoring). The recorded numbers and the unmet boxes are reported in
the issue/PR, not baked into this document as a claim. The GPU-hour figure is a
GPU-hour only when the CUDA backend is active; a CPU run reports seconds and the
same quantity per CPU-second, which is the honest denominator for a CPU-only
measurement.
