# Quality-diversity archive (#120)

The engine has **reset diversity** (restarts, immigrants, structural
de-duplication). It did not have **niche diversity**: nothing recorded which
behaviour regions the search never reached, so thousands of evaluations could
collapse onto one prediction shape without the search noticing.

This increment adds a passive, opt-in quality-diversity layer (`--qd`) over two
cheap behaviour descriptors. It is the first slice of the CVT-MAP-Elites design
in issue #120; the search policy and the empirical budget study are listed as
remaining work below.

## Descriptors

| axis | definition | range |
|---|---|---|
| variable usage | distinct input columns read by the tree, divided by the number of inputs | `[0, 1]` |
| prediction skewness | third standardised moment of the prediction vector, mapped by `0.5 + 0.5·s/(1+|s|)` | `[0, 1]` |

The usage count walks the tree once. The skewness is three passes over the
prediction vector (mean, second and third moments). In one release run
(`cargo test --release --lib qd::tests::descriptor_cost -- --nocapture`), the
complete descriptor, including the tree evaluation, cost **3.2 µs/candidate on
41 rows** and **92 µs/candidate on 800 rows**. A later GPU pass can accumulate
`sum p, p², p³, p⁴` in the same SSE reduction and remove that second
evaluation.

## Archive

`--qd` enables a `16 × 16` rectangular grid (256 cells). `--qd-cells N` selects
`N × N` bins (`N = 23` gives 529 cells, closest to the K = 512 design) and
`--qd-cell-cap N` bounds each cell's local front (default 4).

Each occupied cell holds a **bounded local Pareto front over `(error, cost)`**,
using the same ε-dominance rule as the global archive. Insertion is
deterministic: candidates are offered in a fixed order (island 0..I, population
order, then the global archive), dominated candidates are dropped, and a tie at
equal cost/error keeps the first entry. A cell never grows beyond `--qd-cell-cap`.

The layer is **passive**. It observes evaluated individuals after they have been
scored; it never selects parents, changes migration, restarts or the reported
Pareto front. With the flag off, the search is bit-identical to before.

## Reported coverage

Text output adds one line:

```text
# qd: 15/256 descriptor cells occupied (5.9%), 39 local-front entries, cap 4
```

`--json` adds:

```json
"qd":{"bins":16,"cells":256,"occupied":15,"coverage":0.05859375,
      "entries":39,"best_error":1.145...,"cell_cap":4}
```

`coverage` is `occupied / cells`. The same seed and flags produce the same
coverage and the same cell fronts (pinned by `tests/qd.rs`).

## What is not implemented yet

This is deliberately a grid, not a full CVT-MAP-Elites. The remaining work in
#120, in order:

1. **CVT centroids** — replace the rectangular grid with a centroidal Voronoi
   tessellation (K = 512), fitted once from a descriptor sample. The archive
   contract (cell → bounded local front → deterministic insertion) stays.
2. **Search feedback** — UCB1 parent selection (60% quality / 25% uniform /
   10% neighbour / 5% global front), and islands as demes on a ring with
   stratified cell migration. This is what can raise coverage at an equal
   evaluation budget; the current layer only *measures* coverage.
3. **Restart policy** — reseed the least-occupied region instead of replacing a
   stagnant island with random trees; seeds occupy their own niche.
4. **Same-pass descriptors** — extend the SSE reduction with `sum p, p², p³,
   p⁴`, sign changes and one volatility dot product, so the descriptors are
   nearly free rather than a second evaluation.
5. **Exact promotion** — cell elites and promoted survivors refit exactly on
   all rows in f64 (the existing full-data archive rebuild is the template).
6. **Empirical study** — coverage at equal budget, validation hypervolume per
   GPU-hour at 4× evaluations, `dHV_real / dHV_null`, DSR at effective trial
   count, PBO ≤ 0.20, and random-descriptor/uniform-cell controls. The present
   PR measures descriptor cost and coverage, not those end-to-end claims.
