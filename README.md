# rule-and-compass (🎻 `rcsr`)

A **symbolic-regression** engine written in Rust.

Given a table of numeric columns, `rcsr` searches for short closed-form
expressions `y = f(x1, x2, ...)` that fit the data, trading **error** against
**complexity**, and reports the non-dominated set (the *Pareto front*).

> The name is an old-fashioned English pub sign (1700s style, like *The Three
> Compasses*): the **rule** is the law/equation we recover, the **compass** is
> the measurement — the instrument that finds it.

## What it does

Symbolic regression finds formulas, not just fitted parameters. `rcsr` evolves a
population of expression trees, scores each on **two axes (error and
complexity)**, and returns the front of non-dominated equations — the most
accurate equation at each level of complexity.

Constants inside a candidate are not guessed: each candidate's constants are
fitted by **Levenberg–Marquardt** against the configured `--metric`, so a
candidate with the right *shape* is immediately usable even if its numbers
started out wrong, and the fitted numbers minimise the same loss that ranks the
front. The search may instead use the exact solver portfolio
(`--fit-strategy auto`): closed-form column-pivoted QR when the candidate is
affine in every constant, variable projection (Golub–Pereyra) when it is
separable, and the historical LM path for everything else. `varpro`, `stream`
(out-of-core f64 sufficient statistics) and `gpu` (the #169 batched kernel with
an exact CPU verifier, which fits candidates with at most 20 constants on the
device — least squares for the squared/AIC family, IRLS for `log_squashed`
(#229), L1 IRLS with true-MAE acceptance for `absolute` (#222), Fisher scoring
on the binary label for `log_loss` (#234; the device proposes the constants
and the exact CPU `f64` log-loss judges the proposal and refits every finalist,
and since #289 the search also scores log-loss, Brier and the three hinge forms
with a native label-aware kernel — `hinge` is explicitly the bounded sigmoid
surrogate, not the SVM hinge), and the exact correlation objective with
a frozen affine gauge
(#224) — and labels every larger or unsupported candidate for the exact CPU
arm; the correlation *scorer* is native on the device since #287;
below 1 048 576 training rows it fits the screened top
`max(32, pop/4)` of each island, and at or above that size every candidate with
constants, so the seed-constant screen cannot cost quality at large `N`) can be
forced individually. With no `--fit-strategy`, or with `auto`, a run on the GPU
evaluator with `squared`, `aic`, `aicc` or `bic`, fitting on, device memory for
the f64 fit rows and a run-level cost model that predicts a win resolves to the
batched kernel with every candidate with constants admitted; every other run
keeps `exact` (no flag) or the structural `auto`, so CPU-only runs are
unchanged, and each resolution is labelled with its reason. For `--metric rank_correlation`/`rank_correlation_signed`
the same arm fits the exact integer rank polish on the device; the same exact
integer sums also **score** every candidate on the device from 4 096 training
rows (#288), and finalists are still rebuilt in CPU `f64`; `rank_correlation_signed`
is the same stage under `1 - rho`, so a reflected model is rejected and a
proposal is judged on the exact oriented score, never on SSE or `|rho|` (#231);
`--metric median` does the same with the exact `f64` median and
`--metric interquartile` with the middle-50 % window mean of `|r|` (fit and
native scoring from 4 096 training rows — the measured crossover of the batched
selection, #226, #232 and #288), and `--metric maximum` fits the Chebyshev
objective by **exact bisection on the error level** per constant (from 512
training rows — the measured cost-model floor, #227) and is scored natively by
the #287 maximum reducer. An explicit `gpu` with a metric that has no device fit
objective is a usage error, because the kernel would otherwise fit a different
loss; use `--metric squared`, `--metric log_squashed`, `--metric absolute`,
`--metric log_loss`, a rank metric, `--metric median`, `--metric
interquartile`, `--metric maximum` or a CPU strategy.
Dispatch is planned per stage, not per run: a calibrated cost model returns
`device`, `cpu` or `split` plus a machine reason, with hysteresis at the
crossover and a route ledger in `--json`/`--run-manifest`. Structural and
arithmetic refusals are a cause set, never a timing test. See
[`docs/dispatch.md`](docs/dispatch.md).
Every arm falls back to the default LM path when it cannot prove a safe result,
and the default remains the historical LM path bit for bit. `--metric signed`
(the absolute mean residual) is the one non-squared metric with an exact closed
form: its constants are projected in f64 onto the zero-mean-residual manifold by
a constrained least-squares step (two passes, no optimiser), and when no
constant can move the mean the arm says so instead of fabricating a fit — see
[`docs/optimisation.md`](docs/optimisation.md) §3.0.1.

See [`docs/how-symbolic-regression-works.md`](docs/how-symbolic-regression-works.md)
for a plain-language walkthrough,
[`docs/optimisation.md`](docs/optimisation.md) for the theory and measurements
behind the search strategy, and [`SPEC.md`](SPEC.md) for the precise behaviour.

### Implicit relations

Some laws are level sets rather than functions: a circle `x² + y² = 1` has no
single-valued `y = g(x)`. `--implicit` appends the target as one more input and
searches `f(x1, …, xn, y) = 0`. `--metric implicit` (or the shorthand
`--implicit-derivative`) scores the distance to the level set,
`|f| / ||∇f||`, using the adjoint gradient the fitter already computes. See
[`docs/implicit.md`](docs/implicit.md) for the model, both implicit metrics, the
zero-gradient guard and a worked unit-circle example.

## Build & run

Requires a Rust toolchain (`rustup` recommended). No third-party crates are used.

```sh
cd my-version
cargo build --release

# demo on the bundled sample: recover y = 2x + 1
cargo run --release -- \
  --data examples/line.csv --target y --vars x \
  --generations 200 --islands 4 --pop 200 --metric absolute
```

Output (human-readable form):

```
  1  complexity=1   train_err=1.455120      test_err=1.216192       y = (0.8446 * x)
  2  complexity=2   train_err=1.096793      test_err=0.609935       y = exp((0.8446 * x))
  3  complexity=3   train_err=0.999970      test_err=1.000090       y = (2.0044 * x)
  4  complexity=4   train_err=0.651494      test_err=0.423752       y = (exp((0.8446 * x)) + (0.8446 * x))
  5  complexity=5   train_err=0.000001      test_err=0.000001       y = (1 + (2 * x))
```

The last row is the answer: `y = 1 + 2x`, exact to the rounding of the data, and
the smaller rows are the cheaper approximations it beat along the way.

```sh
# a trigonometric example: recover y = sin(x)
cargo run --release -- --data examples/sine.csv --target y --vars x \
  --generations 300 --islands 4 --pop 300 --metric squared
```

### Score a formula without searching

`--eval-expr` parses a formula, scores it on the split and exits — no search. It
is the inverse of the printed output, so a printed right-hand side can be pasted
straight back in to re-check it:

```sh
# original units: variables by name, *, ^, sin(...), if(...), ...
cargo run --release -- --data examples/line.csv --target y --vars x \
  --eval-expr "2*x + 1"
#   1  complexity=5   k_constants=2   train_err=0.000000      test_err=0.000000      y = ((2 * x) + 1)
```

With one test block `test_err` is the error on that block; `--json-out FILE`
writes the JSON result to the file.

A syntax error names the offending column:
`--eval-expr "2 * (x + )"` fails with
`expected a number, variable, function or '(' (column 10)` and exit code 2.

### Walk-forward evaluation and backtest

`rcsr walkforward` evaluates one supplied expression over time windows on a
`.rcsb` store (CSV is also accepted), applies a long/flat rule and charges costs
on position changes. It reads the store group by group, uses the same f64
evaluator as `--eval-expr`, and reports per-window and pooled Sharpe, CAGR, max
drawdown, turnover, exposure, hit rate, break-even hit rate, trades and rows as
JSON or Markdown:

```sh
rcsr walkforward --data panel_daily_long.rcsb --target fwd1 \
  --expr 'logistic(r60) - tanh(r120)' --vars r60,r120 \
  --time ts --group-by symbol --horizon 172800 \
  --window '2019,0,1546128000,1546300800,1577836800' \
  --cost-bps 10 --threshold-quantile 0.7 --json
```

Test windows are defined only by time; the purge gap must be at least the
horizon, and the threshold comes from training (or an optional trailing
validation slice), never from the test block. Per-window metrics start every
group flat; pooled P&L carries each group's ending position across a window
boundary and charges the seam term `|p_end - q_start|` (pass
`--pool-reset-flat` for the historical reset). Repeat `--expr` to select among
candidates by their validation error; the test block is never a selection
input. `--select-band` selects those repeated candidates with the long-only P&L
band instead: a screening score admits a measured `epsilon` band, the admitted
set is rescored on full validation rows in exact CPU `f64`, and a deterministic
`(net, turnover, complexity, index)` tie-break picks the finalist; the JSON
carries the surrogate/exact ranks, band width and recall audit. The fold sign
test is reported as not applicable for expanding folds
because `test_k` lies inside `train_{k+1}`. Pass `--device-pnl` to recompute
each test block through the fused f64 long-only reduction (`src/trading_gpu.rs`
plus the CUDA kernel `src/cuda/pnl.cu`) and report it with cash and
buy-and-hold benchmarks; the scalar evaluator stays the reported authority.
Every window also lists the long-only controls (cash, buy-and-hold, constant
long/flat from the training mean, matched-exposure random long seeded by
`--control-seed`) with exposure, turnover, beta and excess over cash; see
`src/baselines.rs`.
`--policy band` adds a long-only no-trade band (`--band`, `--min-hold`) whose
associative `Carry2`/`CooldownCarry` is chunk-invariant, and `--policy dp`
adds the exact `O(2n)` hindsight DP as a labelled bound, never as the
strategy's P&L. `--gpu` runs both scans on the CUDA device, bit-identical to the
CPU scans, and names the reason when it falls back to the CPU. See [`docs/walkforward.md`](docs/walkforward.md) for the
semantics and metric definitions, and [`docs/eval-benchmark.md`](docs/eval-benchmark.md) for
the pinned 4M-row benchmark protocol and the measured wall time.

Grammar: `+ - * / ^` (`^` is right-associative and binds tighter than unary
minus), function calls with any arity (`sin(x)`, `min(x, y)`, `if(c, a, b)`),
comparisons (`x > 0`, `x <= y`, `x == 0`), the logical operators
`and`/`or`/`xor`/`not`, the positional constraint `between(x, lo, hi)`,
`require`/`constrain` (a violated constraint makes the candidate infeasible
rather than merely wrong), the opt-in meta operators `complexity`,
`variables`, `constants`, `contains` and `contains_op` (see
[`docs/meta-operators.md`](docs/meta-operators.md)), the opt-in dataset
aggregates `mean(x1)`, `std(x1)`, `median(x1)`, `iqr(x1)`, `sum(x1)`,
`amin(x1)`, `amax(x1)` and `count()` (see
[`docs/aggregation.md`](docs/aggregation.md)), parentheses, the constants
`pi` and `e`, and variable names from your table (the `x0`, `x1`, … fallback
also works). Comparisons are non-associative, so write
`(a < b) and (b < c)`.

`cargo test` runs the unit tests and the end-to-end recovery tests
(`y = x`, `y = x²`, `y = x1·x2`), plus a regression test that constant fitting
reaches machine precision.

## GUI

A local web front end ships as a second binary. It is a self-contained
single-page app embedded in the executable, so it has no CDN, framework or
network dependency at runtime and binds loopback only:

```sh
cd my-version
cargo build --release
./target/release/rcsr_serve --port 8080 --open
# open http://127.0.0.1:8080
```

The tour: paste or open a CSV/TSV and give each column a role (input, target,
time, group, weight, ignore); the Search pane is generated from `rcsr schema`,
with metric, generations, population, islands, complexity cap and seed pinned
and every other flag under **More options**; Run streams NDJSON and grows a
log-scale Pareto chart as solutions arrive; Solutions lists the front and the
selected row shows the formula, constants ± standard errors, train/test error,
AICc/BIC, Durbin–Watson, residual checks, baselines, bootstrap and
cross-validation; **Copy as CLI** emits the exact `rcsr` invocation (including a
here-document when the table was pasted). The page consumes only the JSON
contract: `/api/schema`, `/api/example` and the `/api/run` event stream. See
[`docs/gui.md`](docs/gui.md) for the as-built notes, the event shapes and what
is deferred.
### Start from what you already know

`--seed-expr` (repeatable) and `--seed-file` inject prior solutions into the
search instead of making it start from random trees alone:

```sh
cargo run --release -- --data examples/sine.csv --target y --vars x \
  --seed-expr "sin(x)" --generations 15
```

and reports the processed seed on stderr:

```text
# seeded: complexity=... train_error=0.000000  y = sin(x)
```

Each seed is parsed in original units, simplified, its constants fitted on the
training sample, then inserted into the archive and the initial population of
**every** island. A seed is an elite: it is kept even above `--max-complexity`,
which limits generated offspring, and the run prints a `# note:` when that
happens. `--seed-file` takes one expression per line, skipping `#` comments and
blank lines; both flags can be combined. A malformed seed is a usage error
(exit 2) naming the flag, text and parser column; an unreadable file is an
input error (exit 3). `--json` reports a `seeds` array and `--ndjson` a `seed`
event per seed. See [`docs/seeds.md`](docs/seeds.md) for the processing order,
the elite rule, the file format and the measured before/after numbers.

Every user-written formula (seeds, `--eval-expr`, filters, `--function` bodies
and the HTTP body) is bounded to **4096 nodes** and a root-to-leaf **depth of
256** (a single node has depth 1), and checkpoint tree decoding uses the same
limits. An over-limit or pathologically nested input is rejected with a normal
parser/checkpoint error that names the limit and, for text, the 1-based column;
it never overflows the process stack. `rcsr schema` reports the limits as
`expression_limits`.

## Building blocks

`--function` (repeatable, up to four) supplies a sub-formula the search may
reuse as an operator of its own:

```sh
cargo run --release -- --data examples/sine.csv --target y --vars x \
  --function "wave(a)=sin(a)" --generations 100
```

A definition is `name(a,b)=<expr>` or `name=<expr>` (parameters inferred from
the free variables). The recovered model prints the name, e.g. `y = wave(x)`,
and the definition's constants are shared by every call.
A block costs 1 by default and can be priced with
`--op-complexity "wave=2"` (or `custom0=2`); the Pareto front ranks on that
cost. See [`docs/building-blocks.md`](docs/building-blocks.md) for the syntax,
parameter inference, shared-constant fitting and limits.

## Choosing what "complexity" means

By default complexity is the node count. `--op-complexity "pow=2,sin=1.5"`,
`--complexity-weights "var=1,const=3"` and `--depth-weight 0.5` price
operators, terminals and nesting instead. The configured value is what the
Pareto front is ranked on, what `--max-complexity` limits, and what every row
reports; the defaults reproduce the node count exactly. See
[`docs/complexity.md`](docs/complexity.md) for the formula and worked examples.

## Dimensional constraints

`--units 'x=m,t=s,y=m/s'` declares the dimensions of the inputs and the target;
`--require-dimensions` then rejects every candidate whose dimensions cannot be
reconciled with the target's with an infinite error. The algebra knows the SI
names `m`, `s`, `kg`, `A`, `K`, `mol` and `cd`, and **any other identifier is
its own base dimension**, so `price=USD,volume=shares` catches `price + volume`.
It models dimensions only, not unit identity or scale: declare `volume=m^3`, not
`litre`. Fitted constants are allowed to carry dimensions (a coefficient absorbs
a unit ratio, an intercept absorbs the target's), and each solution reports their
apparent dimensions as a plausibility check. Undeclared names are dimensionless
and column names are never inspected. Units are strictly optional in three
layers: no `--units` changes nothing and adds no output fields; `--units` alone
only reports and never rejects; `--require-dimensions` turns the hard filter on
and is a usage error without `--units`. See
[`docs/units.md`](docs/units.md) for the operator table, the radian/degree and
dB/pH caveats, a worked example and the limitations of a hard filter.

## Meta operators

`complexity(e)`, `variables(e)`, `constants(e)`, `contains(e)` and
`contains_op(e)` describe the shape of a subexpression: node count, distinct
input variables, constant leaves, and whether an input or an operator is
present. They are opt-in (`--ops complexity,...`), return one value per model
rather than one per row, and are cached when the node is built. Because they
describe the model rather than the data, they are useful for self-describing
models and structural conditions, not as a source of row-level signal. See
[`docs/meta-operators.md`](docs/meta-operators.md) for the semantics table and
the caching rule.

## Dataset aggregates

`mean(x1)`, `std(x1)`, `median(x1)`, `iqr(x1)`, `sum(x1)`, `amin(x1)`,
`amax(x1)` and `count()` reduce one input column over the training rows. They
are opt-in (`--ops mean,std,...`), computed once into a small per-column table
before the search and cached in the node, so per-row evaluation cost does not
grow. They are dataset facts, not fitted parameters: no constant slot, no
gradient and no `k_constants`; and the table is built from the training rows
only, never the test split. See
[`docs/aggregation.md`](docs/aggregation.md) for the semantics table, the cache
and leakage rules, and a worked recovery example.

## Save and resume a run

A long search can be checkpointed and continued later. The checkpoint carries
the complete search state — configuration, backend (`cpu`/`gpu`), canonical
config hash, populations, per-island RNG states, stagnation and patience
counters, archive and a data fingerprint — so a resumed run follows the
identical trajectory:

```sh
rcsr --data examples/line.csv --target y --vars x \
     --generations 20 --seed 7 --save run.rcsr
rcsr --data examples/line.csv --target y --vars x \
     --generations 40 --seed 7 --resume run.rcsr
```

`--save-every <N>` sets the interval (default 25 generations); a checkpoint is
also written after 15 minutes of wall time without one, and always on exit. The
exit checkpoint carries the exact f64 post-refit archive and its `archive_hash`
beside the search archive (labelled `error_kind`). A `--time-budget` stop is
resumable: `--resume` continues it and may set a new budget. The checkpoint
records version, build commit, binary SHA-256, device, driver and threads;
`--resume` refuses a mismatch unless `--allow-build-change` is given. It also refuses a different target spec (sha256 recorded in the checkpoint), exit 2. Saves are crash-safe: temp file, `fsync`, atomic rename,
with the previous good generation kept as `run.rcsr.prev`. Saving does not
consume randomness, so it cannot change the result. The saved search settings
win on resume; only the generation budget may be raised, and data that does not
match the saved fingerprint is refused. A CPU checkpoint is never silently
continued on the GPU (or vice versa): `--resume` refuses the backend change
unless `--allow-backend-change` is explicit. See
[`docs/checkpoint.md`](docs/checkpoint.md) for the versioned text format, the
determinism guarantee and its limits.

`--run-manifest <path>` records the evidence needed to reproduce a long run:
the commit recorded when the binary was built, the SHA-256 of the running binary, the data file
checksum and sample fingerprints, the resolved configuration, seed, backend,
device/driver/CUDA, process UTC start/end and phase timings; `total` equals
`timing.total_seconds` in the result JSON. `rcsr verify-manifest` re-checks it
against the current binary (or `--binary <file>` for a preserved one), data and
config and exits non-zero on any mismatch. `--json-out <path>` publishes the final JSON result
atomically instead of streaming it. See
[`docs/run-integrity.md`](docs/run-integrity.md).

## Using it from other programs

`rcsr` is built to be driven by scripts and agents.

* **stdout carries only the result; stderr carries the human log.** So
  `rcsr ... --json > result.json` and `rcsr ... --json | jq .pareto[0]` do the
  right thing.
* **`rcsr schema`** prints a self-describing JSON document (every option, its
  type, default and meaning; plus the exit codes). An agent can discover the
  interface without reading the source.
* **`--json`** emits one JSON object; **`--ndjson`** streams one object per line
  (`start`, `progress`…, `solution`…, `done`) for long runs.
* **Deterministic**: the same `--seed` gives the same answer, and the seed is
  echoed in the output.
* **Bounded time**: `--time-budget <seconds>` guarantees the process returns.
* **stdin**: `--data -` reads the table from standard input.
* **Bounded memory**: `--max-memory 2GB` refuses an input whose estimated
  footprint would exceed the budget, with exit 3 and a fix, instead of being
  SIGABRT/SIGKILL'ed. A CSV is estimated from its measured rows × columns plus
  preparation (calibrated on 6- and 90-column runs, never a multiple of the file
  size). A `.rcsb` store is estimated exactly from its header: resident dtypes
  (values only for numeric columns, the extra `i64` copy, 4-byte codes),
  dictionaries, row groups, the used-column projection and GPU/fit copies, plus
  25 % slack. The full rows load whenever that fits — on the 3.9M × 56 panel the
  estimate is 7.83 GB against a 7.26 GB measured peak, instead of the old
  61.71 GiB that produced a 577k-row sample. A store that does not fit is
  refused; sampling is explicit via `--sample-rows N` (mutually exclusive with
  `--force`, and still budget-guarded) and is recorded as `full_rows=false` in
  the preparation block, manifest and dev log. When a store is read with
  `--vars` and `--time`, the default **projects the read** to those variables,
  targets, `--weight`, `--time` and `--group-by`: only those blocks are read,
  the guard charges only them, and the resolved hash and column list go into
  `--run-manifest` (`--no-project` restores the full-column load for an A/B
  run). `rcsr
  import` writes a versioned `.rcsb` store (format v2: per-column
  `f64`/`i64`/`u32` dtypes, 4-byte dictionary codes, nanosecond-safe i64 time);
  v1 stores still load. See [`docs/data-format.md`](docs/data-format.md).
* **Market data as a typed layer**: `rcsr::ohlcv` resamples OHLCV exactly
  (open=first, high=max, low=min, close=last, flows summed), keeps gaps and
  flags zero-volume/partial bars, understands crypto 24/7 and NYSE/LSE
  sessions with holidays, half-days and DST, applies splits/dividends while
  keeping raw prices, verifies symbol splices, and joins higher timeframes
  only after their bucket has closed. `rcsr::features` supplies causal
  `ta:core`/`ta:full`/`exotic:cheap`/`candle:geometry`/`cross:btc` features and
  a next-open forward label with symmetric cost. See
  [`docs/ohlcv.md`](docs/ohlcv.md).
* **Incremental feature enrichment**: `rcsr enrich --data old.rcsb --out
  new.rcsb --features returns,ma,vol,atr,rsi,bollinger,vz,range,candle,donchian,calendar,lags,xs,btc,cross,mtf`
  appends pre-computed columns without rebuilding rows: existing value/code
  blocks are copied byte for byte, dtypes and dictionaries survive, a duplicate
  name needs an explicit `--replace`, and every change is chunk-size stable.
  Windows are causal and gap-aware, warm-up is reported per column, and the
  `py15m` group reproduces the 35 Python enrichment columns on a committed
  parity fixture, the `compact` group emits the exact 52-column smoke/parity
  subset at the 15 m base, the `flow` group materialises `vwap` plus the nine
  derived OHLCV/flow columns (taker ratios, trade intensity, average trade
  size, Amihud illiquidity, zero-return share, OBV z-score, CVD proxy and VWAP
  deviation) with the documented zero-volume/zero-trade and gap policies, the
  `mtf` group emits all 36 multi-timeframe columns
  (`mtf_<1h|4h|1d>_<leaf>`), joined only after the higher bar has closed with
  explicit `bar_age` staleness (the same resample and causal join dispatch to
  the device under `--features cuda` and `--gpu on`), the engine-tail kernels
  cover return/MA parameters, `don20`/`don96`, stochastic, ATR/ADX/DI,
  Bollinger and the registered exotic statistics with `O(n)` blocked extrema
  and one shared gap policy (`r16`, `r96` and `don96` no longer fall through
  the compact guard), the benchmark family (`btc_*`, `eth_*`, `rel_*`,
  `beta_*`, `cross_ret1`, `cross_beta20`, `beta384`) joins the BTC and ETH
  series by exact timestamp on the device and the per-timestamp market means
  (`mkt_*`, `breadth`, `dispersion`, `rel_mkt*`, `avg_corr`,
  `btc_volume_share`) run one thread per timestamp in the CPU addition order
  (`--features cuda`, tiers per column in `rcsr schema` `gpu_feature_parity`),
  the point-in-time membership and MTF as-of joins (`src/cuda/pit.cu`, #302: `as_of <= t` and
  `h.time + tf <= t` merges with `bar_age`, chain resets and `lookahead_violations`; bitwise vs the
  CPU reference, membership decisions stay on the CPU, `rcsr panel build` is not rewired),
  the `price_action` group adds the continuous
  candle geometry and multi-bar structure of #261 (`clv16 body16 rng_atr
  prev_range_pos inside_s outside_s engulf_s harami_s run8 nr4 trend96`;
  `batr` is shared with `compact`) with strict zero-range/zero-volume and gap
  masks, and the `cross` group adds the complete #273 BTC/ETH benchmark
  context, equal-weight market state, breadth, dispersion, average-correlation
  proxy, ex-self relative returns and `[-1,1]` PIT cross-sectional ranks. On
  the CUDA path one device `(time,symbol,row)` permutation serves every rank
  column, with non-members and NaN keyed last. The
  fused flow and price-action device producers materialise their twelve and
  eleven derived columns from one base pass per family, with a declared
  `bitwise`/`1e-12` tier per column (`feature_registry.parity_tier`). The
  default financial feature set is registered in `rcsr schema`
  (`feature_registry`): every available column has a tested producer, and
  registered-but-unavailable derivative columns fail closed with the missing
  dependency instead of being silently dropped. The `engine_catalogue` block
  enumerates all 254 engine names (216 in
  `all,py15m,ta:full,exotic:cheap,cross:btc` plus 38 compact-only names) with
  family, warm-up, availability, exact formula and the explicit canonical
  mapping for every name shared by several groups. The CPU path is always
  available; a per-family cost model

  decides CPU vs GPU and records the reason, GPU builds read only the required
  source blocks into a bounded pinned ring and copy them straight to the device
  (no full `Table`), report the GPU column share, and the 3.9M-row measurement
  keeps only the families that actually beat the CPU on the GPU. On the
  16.1M-row OHLCV panel the GPU feature phase is **1.49x** the CPU (72.1 % of
  columns dispatched, zero fallbacks) while end-to-end wall is decided by the
  append write; the output checksum is computed while the blocks stream, so a
  multi-gigabyte store is never read twice. GPU output blocks share a 64 MiB
  arena and a pinned chunked D2H copy, and cost model v2 charges the measured
  `BW_write` separately from kernel and transfer time. See
  [`docs/ohlcv.md`](docs/ohlcv.md#incremental-enrichment-rcsr-enrich-issue-200).
* **Point-in-time panels**: `rcsr panel membership|build|audit` ranks coins by
  trailing liquidity **as of** each rebalance, keeps delisted names while they
  trade, never forward-fills across a listing/delisting/migration, verifies
  splice ratios, charges both sides and the final exit, supplies a
  matched-exposure random-long benchmark, refuses 2026H2 in every runner, and
  emits a checksummed provenance manifest with a lockbox-coverage block.
  `panel build --resume` caches one atomic **binary** part per symbol, keyed by
  the input store SHA-256, the membership decisions and the settings; a fully
  warm run reads the parts only (it does not parse the store) and reproduces a
  byte-identical panel. `panel membership --cache` keys the membership table the
  same way. `prep --resume` caches one checksum-keyed spool per symbol;
  `--snapshot` embeds a verified archive snapshot. See
  [`docs/panel.md`](docs/panel.md).
* **Default hourly PIT panel v2**: `rcsr panel snapshot|base|membership|build|validate`
  builds `panel_hourly_default.rcsb` in staged, cacheable steps. `snapshot` hashes
  every archive and lists the duplicate and invalid-OHLC bars. `base` keeps only
  full four-bar hours (`cov15m`), collapses identical duplicate instants, flags
  differing ones and stores the unit of each source. `build` keeps point-in-time
  members, numbers listings, fails on an undeclared level jump and refuses the
  lockbox. `validate` runs V1-V12 and exits 3 on a failure. See
  [`docs/panel.md`](docs/panel.md#default-hourly-panel-v2-259).
* **Stable exit codes**: `0` success, `2` usage error, `3` input error,
  `4` no finite solution.

```sh
# machine-readable
rcsr --data examples/sine.csv --target y --json | jq '.pareto[-1].expression'

# discover the interface
rcsr schema

# stream progress as NDJSON while it runs
rcsr --data examples/sine.csv --target y --ndjson --progress --time-budget 5

# read the table from a pipe
cat examples/line.csv | rcsr --data - --target y --top 3
```

```jsonc
// rcsr --data examples/line.csv --target y --json
{
  "tool": "rcsr", "version": "0.1.0", "seed": 0, "metric": "absolute",
  "target": "y", "variables": ["x"], "train_rows": 30, "test_rows": 10,
  "generations_run": 200, "converged_early": false, "elapsed_seconds": 1.9,
  "null_control": null,
  "baselines": {
    "mean":   { "train_error": 2.0034, "test_error": 2.2496 },
    "linear": { "train_error": 4.8e-7, "test_error": 7.6e-7 },
    "knn":    { "k": 5, "train_error": 0.1299, "test_error": 0.2503,
                "reference_rows": 30, "train_query_rows": 30,
                "train_query_sem": null, "test_query_rows": 10,
                "test_query_sem": null, "capped": false }
  },
  "residual_tests": {
    "n": 30, "lags": 6, "durbin_watson": 2.0,
    "ljung_box":     { "statistic": 2.01, "df": 6, "p_value": 0.919 },
    "breusch_pagan": { "statistic": 0.159, "df": 1, "p_value": 0.690 },
    "skewness": 0.008, "kurtosis": -0.667
  },
  "pareto": [
    { "rank": 5, "complexity": 5, "train_error": 5.3e-7,
      "test_error": 5.2e-7,
      "constant_errors": {
        "dof": 28, "ill_conditioned": false,
        "constants": [ { "value": 1.0, "std_error": 4.1e-7, "relative": 4.1e-7 },
                       { "value": 2.0, "std_error": 3.2e-7, "relative": 1.6e-7 } ] },
      "expression": "(1 + (2 * x))" }
  ]
}
```

A graphical front end for the same contract is designed in
[`docs/gui.md`](docs/gui.md): schema-driven forms, a live Pareto plot, the
split made visible on the grid, and a "Copy as CLI" action on every search.

### Full option list

```
rcsr --data <file|-> --target <col> [--vars a,b,..]
     [--max-memory size] [--force] [--sample-rows N] [--skip-malformed]
     [--dtype col=continuous|boolean|categorical,..] [--max-levels N]
     [--weight col]
     [--metric <name>] [--generations N] [--islands I]
     [--pop P] [--max-complexity K] [--seed S] [--test-frac f] [--ops list]
     [--seed-expr expr] [--seed-file path]
     [--implicit] [--implicit-derivative] [--function def]
     [--op-complexity spec] [--complexity-weights spec] [--depth-weight w]
     [--split random|temporal|group|temporal-group|auto] [--time col]
     [--date-only error|utc-midnight|session-close] [--time-unit s|ms|us|ns] [--tz offset]
     [--group-by col] [--no-project] [--project n|auto] [--view k] [--gap N] [--folds K]
     [--lags K | --lag-set k,..|default] [--history-frac f]
     [--validation-frac f] [--select-by train|validation|aicc] [--require-validation]
     [--equivalence-z z] [--noise-lag n|auto]
     [--allow-lockbox] [--target-spec file] [--allow-target-structure]
     [--gap-break f] [--fill method] [--allow-interpolate] [--outlier-iqr k]
     [--smooth k] [--smooth-centred k]
     [--center mean|median|min|max|none] [--scale std|iqr|range|none]
     [--filter expr] [--exclude expr] [--train-select expr] [--test-select expr]
     [--null-runs N] [--null-block N] [--baseline-samples N] [--bootstrap-error N] [--cv mode] [--cv-k K]
     [--stability B] [--eval-expr expr]
     [--units name=unit,..] [--require-dimensions]
     [--fit-iters N] [--fit-strategy exact|auto|linear|varpro|stream|gpu] [--fit-report] [--fit-rows N] [--fit-fallback-max N] [--no-fit]
     [--no-dedup] [--no-simd] [--predictor-samples N]
     [--pipeline-depth N] [--variation auto|node|postfix]
     [--no-front-parents]
     [--fit-samples N]
     [--patience N]
     [--gpu] [--gpu-memory size] [--no-gpu|--cpu] [--qd] [--qd-cells N] [--qd-cell-cap N]
     [--qd-cvt] [--qd-policy] [--qd-uniform] [--qd-random]
     [--time-budget seconds] [--save file] [--save-every N] [--resume file]
     [--allow-backend-change] [--allow-build-change] [--run-manifest file] [--json-out file]
     [--format text|json|ndjson] [--json] [--ndjson] [--progress]
     [--dev-log-level summary|full] [--dev-log file.ndjson] [--no-dev-log]
     [--dev-log-every N] [--dev-log-max-bytes size]
     [--top N] [--quiet] [--out file.tsv]
     [--report summary|sensitivity|benchmark] [--report-out file.md]
     [--report-repeats N]
```

Subcommands (`rcsr import`, `rcsr prep`, `rcsr enrich`, `rcsr panel`, `rcsr
walkforward`, `rcsr verify-manifest`, `rcsr dev-report`) carry their own option
lists in `rcsr schema`; the block above is the root search command. The former
Python pipeline is classified and retired in
[`docs/python-pipeline-retirement.md`](docs/python-pipeline-retirement.md):
the documented path is these Rust commands plus the acquisition tool
[`rcsr-acquire`](#acquiring-the-binance-vision-archive). `rcsr dev-report` summarises the optional `--dev-log` NDJSON record.

Input may be CSV, TSV, or whitespace-delimited, with a header row. Numeric
columns can be used as inputs or target; text columns (dates, ticker symbols)
are kept and can be used for splitting. Blank lines and `#` comment lines are
ignored. A data row with a different number of fields from the header is
refused with an input error (exit 3) naming the 1-based line and both counts;
`--skip-malformed` drops such rows instead and reports how many it skipped.
A numeric cell may be missing (empty,
`NA`, `NaN` or `null`, case-insensitively); `--fill` chooses whether those rows
are dropped or filled, and the default leaves them untouched.

## Acquiring the Binance Vision archive

`rcsr-acquire` is a separate pure-std crate in `tools/acquire/`. It has no dependencies and uses `curl` as the transport, so `rcsr` itself holds no HTTP client. It reads the spot monthly kline archive at `data.binance.vision` and has four subcommands.

```sh
cargo build --release --manifest-path tools/acquire/Cargo.toml
A=tools/acquire/target/release/rcsr-acquire

$A list   --out archive --interval 15m             # archive/all_symbols_archive.txt, archive/keys_15m.json
$A fetch  --symbols-file archive/all_symbols_archive.txt --interval 15m \
          --from 2017-08 --to 2026-06 --out data --jobs 4
$A verify --dir data --interval 15m --through 2026-12-31 --snapshot snapshot.json
 universe --archive archive/all_symbols_archive.txt --products prod.json \
          --equities nasdaqlisted.txt,otherlisted.txt --out universe
```

* `list` pages through the bucket listing. Without `--symbols` it writes `all_symbols_archive.txt`. With `--interval` it also writes `keys_<iv>.json`, the published months per symbol. Use `--symbols A,B` to index a few symbols with one request each.
* `fetch` writes `<out>/raw_<iv>/<SYM>/<SYM>-<iv>-YYYY-MM.zip`, the layout `rcsr prep --input` reads. A zip larger than 1000 bytes is never requested again. Every request appends one row to `<out>/downloads_<iv>.csv` (`symbol,interval,month,http_status,bytes,fetched_at`). A 404 is a row too: it marks the listing or delisting month. `--skip-404` does not repeat months an earlier run recorded as 404. `--to` defaults to the previous month, and a month that is not published yet is refused. `--jobs` defaults to 4. Each request retries 3 times.
* `verify` downloads the `.CHECKSUM` sibling of every zip under `<dir>/raw_<iv>/`, compares it with the local SHA-256 and writes the snapshot JSON that `rcsr panel build --snapshot` embeds (`docs/panel.md`, "Snapshot provenance"). Per file it records `vendor_url`, `http_status`, `etag`, `last_modified`, `vendor_sha256`, `local_sha256`, `bytes`, `verified` and `fetched_at`. `coverage_last` is the last day of the newest verified month. `complete` is true only when every zip verifies and `coverage_last` reaches `--through`. The exit code is 1 when any zip is not verified.

* `universe` filters the USDT symbols of the archive list to genuine crypto pairs. Every input is a file, so two runs on the same files give byte-identical output and the command makes no request. It writes `universe_all_usdt_pit.txt` (sorted, one symbol per line) and `symbol_exclusions.csv` (`symbol,rule,detail`, one row per matched rule). Rules: `non-ascii`, `tokenised-stock`, `stablecoin`, `commodity-token`, `fiat`, `leveraged-token` (suffix UP, DOWN, BULL or BEAR, with exceptions such as JUP) and `stablecoin-legacy`. `--products` is the exchange product list (JSON `{"data":[{"s":"BTCUSDT","tags":[..]}]}`): tags `bStocks`, `stablecoin` and `tCommodities` trigger a rule, and a symbol listed there skips the equity name heuristic. A symbol with no entry, such as a delisted one, is a tokenised stock when its base minus a trailing `B` is a ticker in `--equities` (comma-separated NASDAQ-format files: header line, ticker in the first `|` field) or when the base is a single-letter equity. `--config` is a TOML-like file of `key = ["A", "B"]` lists that replaces the matching keys of the defaults in `tools/acquire/universe-defaults.toml`. `--tickers` (JSON array of `{symbol,quoteVolume}`) with `--top-by-quote-volume N` also writes `universe_top_N_by_quote_volume.txt`: the N kept symbols by quote volume (ties by symbol), skipping bases shorter than `top_min_base_len` and bases in `top_exclude`. Point-in-time membership by ADV stays in `rcsr panel`.

Exit codes: 0 success, 1 failed requests or unverified zips, 2 usage error. The data is spot only.

## Choosing how rows are split

Which rows go to training and which to testing is the difference between a
number you can trust and one you cannot. `--split` selects the rule:

| `--split` | training rows | test rows | use when |
|---|---|---|---|
| `random` (default) | a random 75% | the rest | rows are exchangeable (i.i.d. samples) |
| `temporal` | the **first** rows | the **last** rows | one time series; the future must stay unseen |
| `group` | whole groups | other whole groups | you must generalise to *unseen* entities |
| `temporal-group` | the past of every group | the future of every group | panel data: many series measured over one period |
| `auto` | inferred from the columns, and printed | | you want the tool to decide and say why |

A random split of a time series lets the model see the future, and a random
split of panel data puts the same stock on both sides. `rcsr` warns when the
rows look non-exchangeable and a random split was requested.

```sh
# One series: train on the past, test on the future, with a 5-row embargo
# between them so autocorrelation cannot leak across the boundary.
rcsr --data prices.csv --target y --split temporal --time date --gap 5

# Panel data (OHLC for several symbols over the same period): train on the
# past of every symbol, test on its future.
rcsr --data panel.csv --target next_close --split temporal-group \
     --time date --group-by symbol

# Generalise to unseen symbols: hold out whole symbols.
rcsr --data panel.csv --target next_close --group-by symbol

# Let it decide, and report the choice:
rcsr --data panel.csv --target next_close --split auto

# You know exactly which rows belong where: say so.
rcsr --data d.csv --target y --vars x1,x2 \
     --train-select "year < 2020" --test-select "year >= 2020"

# Drop rows first, then let a mode split the rest.
rcsr --data d.csv --target y --vars x1,x2 --exclude "warmup == 1"
```

* `--time <col>` may be a number or a date (`2024-01-31`, `2024/01/31`,
  `2024-01-31T09:30:00`).
* `--group-by <col>` may be text (tickers) or a low-cardinality number.
* `--gap N` drops `N` rows after the train/test boundary (an *embargo*), per
  series in `temporal-group` mode.
* `--folds K` splits the test region into `K` contiguous blocks and reports the
  error of each block next to the mean, so an unstable model cannot hide behind
  a lucky pooled number.
* `--validation-frac f` holds out the last `f` of the training period as a
  chronological validation slice (with a `--gap` purge either side) and
  `--select-by train|validation|aicc` chooses the display order; the test
  block is a scoring-only lockbox and is never used to select. See
  [`docs/overfitting.md`](docs/overfitting.md) for the lockbox argument.
* Time-ordered runs, runs with a target spec and runs with `--require-validation`
  run the **validation gates** (section 3.10b of `docs/overfitting.md`): a
  noise-scaled supported front on the validation slice (`--equivalence-z`,
  `--noise-lag`), per-member comparisons against the mean, linear and
  target-structure rungs, a structural identification flag, and a verdict
  `signal | no_signal | unvalidated | not_gated`. Without a validation slice the
  verdict is `unvalidated` and nothing is selected; `--require-validation`
  refuses such a run (exit 2).
* Normalisation statistics are computed from the **training rows only**; the
  test set never influences the model's input scaling.
* A split whose training **or** test rows fall inside the crypto 2026H2 lockbox
  (`2026-07-01 .. 2027-01-01`) is refused unless `--allow-lockbox` is passed.
  The same central guard covers `rcsr run`, `rcsr walkforward`, an explicit
  `rcsr panel build` window and a labelled `rcsr prep`; see
  [`docs/panel.md`](docs/panel.md#the-2026h2-lockbox).
* `--train-select` / `--test-select` assign rows by expression (`> 0` is true)
  and override `--split`; `--exclude` drops rows and composes with any mode.
  A row matching both selections goes to training, one matching neither is
  unused, and both counts are reported — nothing is discarded silently.

See [`docs/data-splitting.md`](docs/data-splitting.md) for the reasoning and the
worked examples, and [`docs/time-series.md`](docs/time-series.md) for the full
time-series and panel audit (purge/embargo, overlapping labels, look-ahead
preparation, lockboxes, per-group reporting, multiple testing and costs).

## Cleaning and preparing data

Real tables have gaps, spikes and rows that should never be scored. Every
preparation option is opt-in, and the defaults are exactly the plain mean/std
run:

| flag | effect |
|---|---|
| `--fill <method>` | what to do with a missing cell: `none` (default), `drop`, `mean`, `median`, `zero`, `previous`, `interpolate` |
| `--outlier-iqr k` | drop rows beyond `k` interquartile ranges from the column median |
| `--smooth k` | `k`-wide **trailing** moving average per column (row `i` reads only itself and earlier rows); edges shrink |
| `--smooth-centred k` | the old **centred** window; reads future bars, is opt-in, and is reported as look-ahead |
| `--allow-interpolate` | permit `--fill interpolate` on time-ordered data (it reads the next bar; otherwise refused) |
| `--allow-lockbox` | permit train, validation or test rows in the 2026H2 crypto lockbox (declared final test only) |
| `--target-spec <file>` | declared target spec (`rcsr-target-spec/1`, default `<data stem>.targets.json`; `long_only` is always true): refuses a label whose zero-return prediction depends on a column (D13c, exit 5), the future return as an input (D13a) and a look-ahead scale (D13b); purges training rows within the label end offset of every validation and test row; reports `baselines.target_structure` |
| `--allow-target-structure` | run a D13c target for a diagnostic replay (warns, records `check.status = overridden` and a proxy table) |
| `--allow-overlap` (`rcsr enrich`, `rcsr prep`) | permit a `--label` stride below the horizon for horizons of 4 h and longer; without it such a label is refused. `--side short` and `--allow-short` are always refused: labels are long-only (action `{0,+1}`, long-side barriers, see [`docs/time-series.md`](docs/time-series.md#3b-long-only-labels-and-their-purge)) |
| `--center <how>` | what to subtract before the search: `mean` (default), `median`, `min`, `max`, `none` |
| `--scale <how>` | what to divide by: `std` (default), `iqr`, `range`, `none` |
| `--filter expr` | keep rows where the expression is `> 0`, e.g. `"volume > 1e5 and close > 0"` |
| `--weight col` | per-row weight column (finite, `>= 0`, at least one positive); a weight of 2 means "this row twice" |
| `--skip-malformed` | drop rows whose field count differs from the header and report the count (default: refuse the file with exit 3) |

```sh
# Robust preparation: fill gaps with the training median, drop extreme
# points, smooth a 3-row window and scale by median/IQR.
rcsr --data panel.csv --target y --vars x1,x2 --split temporal --time date \
     --fill median --outlier-iqr 1.5 --smooth 3 --center median --scale iqr

# Only score rows that pass a condition:
rcsr --data panel.csv --target next_close --filter "volume > 100000"
```

The report says what ran, one commented line per active transformation:

```
# preparation: fill=median rows_filled=12 rows_dropped_missing=0
# preparation: outlier-iqr=1.5 rows_dropped_outlier=2
# preparation: center=median scale=iqr
```

`--json` carries the same numbers in a `preparation` object (methods, filled /
dropped / smoothed / filtered counts), and `--ndjson` emits a `preparation`
event when something is active.

### Boolean and categorical variables

Real tables mix continuous measurements with switches and labels. `--dtype`
says what each input is:

| kind | meaning |
|---|---|
| `continuous` (default) | a number, centred and scaled on the training rows |
| `boolean` | a `0`/`1` switch; validated, and **never** standardised |
| `categorical` | a text or low-cardinality numeric label; expanded to one-hot `col=value` boolean indicators |

```sh
# x1 AND NOT x2, from two 0/1 switches
rcsr --data and_not.csv --target y --dtype x1=boolean,x2=boolean \
     --ops and,not

# One indicator per symbol, all levels kept and sorted
rcsr --data panel.csv --target next_close --dtype symbol=categorical
```

A boolean column must contain only `0`/`1`; any other value is a usage-level
data error naming the row and the value. It is not centred or scaled, because
the evaluator's truthiness rule (`> 0` is true) and comparisons such as
`flag == 1` depend on the literal `0`/`1`. Categorical levels are the distinct
values in the **whole table**, sorted (numerically when they are numbers,
lexicographically otherwise), with missing cells as the final
`col=<missing>` level. `--max-levels N` (default 16) refuses a column with too
many levels rather than silently creating a huge input set. All levels are
kept: the search selects terms, so it does not need a dropped reference level.

`--json` reports the result as
`"variables_detail":[{"name":"flag","dtype":"boolean","source":null}, ...]`,
and the text log prints `# variables: flag (boolean), ...`. See
[`docs/variable-types.md`](docs/variable-types.md) for the exact rules, why
boolean columns bypass normalisation, and the `x1 AND NOT x2` truth table.

### Weighing rows

`--weight <col>` gives every row its own weight. The weight says **how much a
row counts**, not how large its target is: recent measurements, reliable sensors
or duplicated observations can be emphasised, and a row given weight zero has no
influence on the fit or the error.

```sh
# y = 2x + 1, with the second observation counted twice
rcsr --data d.csv --target y --vars x --weight w --metric squared
```

`absolute`, `squared`, `signed`, `log_squashed`, `log_loss`, `brier`, the
three hinge forms, `correlation` and `correlation_signed` become weighted means
(weighted Pearson for the correlations; the weighted classification midpoint
and class counts use positive-weight rows only); `aic`, `aicc` and `bic`
use the weighted RSS and the number of positive-weight rows as `n`. A weight of
2 is exactly a duplicated row because the normalisation is `Σwᵢ·lossᵢ / Σwᵢ`,
not division by the row count. The fitter solves the weighted least-squares
problem (`sqrt(w)` on every residual), and the standard errors use the weighted
`JᵀWJ` with `dof = #positive weights − k`.

Metrics with no standard weighted form (`median`, `maximum`, `interquartile`,
`slope`, `rank_correlation`, `auc`) **refuse** `--weight`
with a usage error that lists the supported ones: weights are never silently
ignored. The text report says

```
# weights: column w, sum 12.5 over 20 rows (2 zero-weight)
```

and `--json` echoes `"weight_column"`, `"weight_sum"` and `"weight_rows"`.
Stability resamples carry the weights with their rows; cross-validation and
`--eval-expr` are weighted too. Null baselines and residual tests remain
unweighted, and the report says so. See
[`docs/weights.md`](docs/weights.md) for the formulas, the doubling proof, why
the other metrics refuse weights, and a worked example.

**No leakage:** filling statistics, outlier bounds and centre/scale are all
computed from the **training rows only** and applied to both sides; smoothing
and `previous`/`interpolate` never cross the train/test boundary. See
[`docs/data-preparation.md`](docs/data-preparation.md) for each method, when
*not* to use it, and a worked example.

### Checking for leakage

Leakage is caught differentially, not by a significance test. The
[`leakage` suite](docs/leakage.md) implements the Tier-A auditors D0–D2 and
D8–D11 (spec/time/resample, universe, corporate actions, join cardinality and
determinism), the differential D5 learner, truncation replay (recompute at cut
points; every completed row must be bit-identical), canaries, as-of joins, the
wall-time `--gap` conversion and `N_eff` for overlapping labels.

```sh
rcsr audit --data panel.csv --target y --time ts --horizon 24 --json
```

`rcsr audit` writes a machine-readable report and exits **5** on any finding.
The same detectors run in `cargo test --release --test leakage` and can be
pointed at your own panel with `Panel::from_csv`; `assert_clean()` panics on a
real leak instead of logging it, naming the detector, column and row.

### Forecasting: history and delayed variables

A forecast usually needs *history*: today's value may depend on yesterday's.
`--lags K` materialises `x_lag1 … x_lagK` for every input, computed inside each
series (group boundary, time order and time gaps respected), so the search can
use them like any other column:

```sh
# y_t = 1 + 3·x_{t-1}:  let the engine look one step back
rcsr --data lag.csv --target y --vars x --split temporal --time t --lags 1
```

`--lag-set 1,4,24,96` (or `default`, the same menu) replaces the blanket `--lags K`: it adds `x_lag<k>` only for the listed bar counts, which keeps the variable count and the mutation dilution down on a wide store. The two flags are alternatives. Stderr and the split report give `nvars` before and after (`nvars 2 -> 10` for two variables and four lags). The first `max(k)` rows of each series are history, as with `--lags`; no normalisation or centred window is involved.

`--project n|auto` (with `--view k`) selects `--vars` from the feature registry instead: a deterministic family-stratified view of 24-48 columns (research cell 96-128), rotated by `--seed` and `--view`, recorded in `--run-manifest`.

`--history-frac f` keeps the first share of every series as *history*: those
rows provide the past for later rows but are not scored, so a delay near the
start of a series never reaches before it. `--gap-break f` cuts a series when
the time step exceeds `f` times the median step, which stops a lag from jumping
a hole in the data.

## Is it real, or is it overfitting?

A symbolic search can fit noise, so a small error proves nothing on its own.
Every solution therefore comes with the checks below, and `--null-runs` adds the
decisive one:

```sh
rcsr --data data.csv --target y --vars x1,x2 --null-runs 19
#  3  complexity=4  train_err=0.0004  test_err=0.0009  aicc=-402.1
# diagnostics: n=90 k=3 constants | overfit index (test/train)=2.25 | Durbin-Watson=2.01
# residual tests (best solution): Ljung-Box h=10 Q=9.4 p=0.494 | Breusch-Pagan LM=1.1 p=0.290 | skewness=0.02 excess kurtosis=-0.41
# null baselines (absolute): mean train=0.88 test=0.91 | linear train=0.42 test=0.46 | 5-NN train=0.31 test=0.35
# null control (19 shuffled-target runs): best=0.61 median=0.72 p=0.05 — the real fit beats the noise control
```

For autocorrelated series or overlapping labels, add `--null-block N` (N at
least the label horizon) to permute contiguous blocks instead of rows; the
report names the method, block length and block count, and the i.i.d. shuffle is
anti-conservative there. See [`docs/overfitting.md`](docs/overfitting.md).

| check | question it answers |
|---|---|
| `train_error` vs `test_error` | does it survive data it never saw? |
| `overfit_index` | how much worse does it get off-sample? |
| `fold_errors` | is the fit stable across blocks of the test set? |
| `k_constants` | how many free parameters did it really earn? |
| `aicc`, `bic` | is the extra complexity paid for by the fit? |
| `durbin_watson` | did anything get left in the residuals? |
| `constant_errors` | are the constants well determined, or do they trade off? |
| `baselines` | does it beat the mean, a straight line and a 5-NN learner? |
| `residual_tests` | Ljung–Box autocorrelation, Breusch–Pagan changing variance, residual skew/kurtosis |
| `null_control` | could the search have produced this from noise? |
| `bootstrap_error` | apparent, out-of-bag and .632+ error estimates (`--bootstrap-error`) |
| `cross_validation` | mean and spread of held-out error over `k` folds (`--cv`) |
| `stability` | how often each form and variable is selected on resampled rows (`--stability`) |

The null baselines are fit on the training rows and scored on the training rows
and the test folds in the search's own metric: a model that does not beat the
5-nearest-neighbour learner has found no structure the data did not already
contain. The k-NN baseline's reference set is always all training rows; for up
to eight inputs a k-d tree answers exact queries, and above that the exact
brute-force path reuses one buffer (still `O(query_rows × reference_rows × d)`,
so the 200k-under-a-minute result is a `d <= 8` claim). Only query rows can be
sampled, with `--baseline-samples N` (default `512`; `0` = every row, and only
for absolute/squared/log_squashed). The default cap keeps the exact
`O(train²·d)` query bounded on large panels; the JSON reports reference/query
rows, `capped`, SEMs and seconds. That is an unbiased estimate of the row-average
loss, and `--json` reports train/test query rows and SEMs with the
finite-population correction. The residual tests (Ljung–Box, Breusch–Pagan,
skewness, excess kurtosis) are reported for the best solution; a small p-value
means the model left autocorrelation or changing variance behind.

The null control shuffles the target, runs the *identical* search, and repeats:
if shuffled targets fit as well as the real one, the "law" is manufactured. With
`r` runs the smallest attainable p-value is `1/(r+1)`, so use at least 19.

On a small sample, `--bootstrap-error n` refits each solution's constants on `n`
bootstrap resamples of the training rows and reports the **apparent**, the
**out-of-bag** and the **.632+** error (Efron & Tibshirani 1997) per solution;
the `.632+` estimate interpolates the optimistic and pessimistic numbers and
always lies between them. Above `n^0.7 >= 1000` rows (about 19 300) the draw is
bounded with a bag-of-little-bootstraps subsample of `b = n^0.7` rows, and the
resample size is reported in text and `--json`.

`--cv k-fold|rolling|purged` goes further: it rebuilds the split as `k` folds,
re-runs the whole search on each fold's training rows (normalised on those rows
alone) and reports the mean and spread of the held-out error, so a single lucky
split cannot flatter the answer. Use `rolling` for time series (it never trains
on the future) and `purged` to embargo `--gap` rows around each test block. It
costs about `k` extra searches.

`--stability B` asks the question that matters most: **does the same law come
back?** It bootstraps the training rows `B` times, re-runs the identical search
on each resample and ranks the structural forms and input variables by how often
they are selected. A variable at 100% is one the data insists on; one near 0% is
noise the search only sometimes needed. Cost: `B` extra searches. The same BLB
cap as `--bootstrap-error` applies above about 19 300 rows, and the resample size
is reported.

See [`docs/overfitting.md`](docs/overfitting.md) for the full checklist, the
formulas, what is still missing, and the literature.

## Reports for other people

`--report` turns a completed run into a Markdown document. One report per run;
a later flag replaces an earlier one. Use `--report-out` for a file, or let a
text run print it to stdout. In `--json`/`--ndjson` the report goes to stderr so
the machine-readable stream is never corrupted.

```sh
# everything a reader needs: command, data, law, constants + errors, diagnostics,
# baselines, the whole Pareto front and a trust verdict
rcsr --data data.csv --target y --report summary --report-out summary.md

# which input actually moves the prediction? analytic d f / d x per row
rcsr --data data.csv --target y --report sensitivity --report-out sensitivity.md

# repeat the search N times and report median/IQR error and wall-clock time
rcsr --data data.csv --target y --report benchmark --report-repeats 9 \
     --report-out benchmark.md
```

The summary's closing paragraph is derived from the statistics in the document
(it counts the baselines beaten, reads the overfit index, the permutation
p-value and the residual tests), so it states what the run does *not* establish
as well as what it does. The sensitivity report uses the **analytic** adjoint
derivative and ranks inputs by `mean(|∂f/∂x_i| · σ_x_i / σ_f)`, which is
scale-free and therefore comparable across columns with different units. The
benchmark honours the same `--generations`, `--pop` and `--time-budget`, derives
its seeds from `--seed`, and reports Hyndman–Fan type-7 quartiles.

See [`docs/reports.md`](docs/reports.md) for one worked example of each report,
with real output, the seed derivation and the stdout/stderr rules.

## Known-truth benchmarks

To check that the engine *recovers* laws rather than overfitting samples, a
separate harness runs it on a suite of **25 problems whose true formula is
known** — small Feynman-style physics laws and elementary dynamical systems —
with and without noise:

```sh
cd my-version
cargo run --release --bin rcsr-bench -- --suite known-truth --noise 0.0,0.01,0.1
cargo run --release --bin rcsr-bench -- --tiny          # fast CI subset, seconds
cargo run --release --bin rcsr-bench -- --tiny --json   # machine-readable
cargo run --release --bin rcsr-bench -- --list          # the problem table
```

A problem counts as **recovered** when the engine's Pareto front contains an
expression reaching test R² ≥ 0.99 on fresh *noiseless* data. The harness prints
a per-run table and a summary (recovery rate overall and per noise level, median
complexity) in text or with `--json`, and a run is reproducible from `--seed`.
See [`docs/benchmarks.md`](docs/benchmarks.md) for the criterion, the problem
list and the recorded results.

## Features

- expression trees over a broad primitive set (`+ - * / ^`, `exp ln sqrt abs
  sin cos tan sinh cosh tanh logistic gauss square cube`, `if`), plus the
  opt-in `floor ceil round mod min max sign step erf erfc atan2 asinh acosh
  atanh`, the conditional family
  `equal less less_or_equal greater greater_or_equal and or xor not require
  between constrain`, the meta operators
  `complexity variables constants contains contains_op` and the dataset
  aggregates `mean std median iqr sum amin amax count` selected with `--ops`
  (all with domain guards; the meta operators describe a subexpression's shape
  and are evaluated once per candidate, not per row — see
  [`docs/meta-operators.md`](docs/meta-operators.md); the aggregates are
  training-column statistics cached once per search — see
  [`docs/aggregation.md`](docs/aggregation.md));
fitted by **Levenberg–Marquardt** against the configured `--metric`, so a
candidate with the right *shape* is immediately usable even if its numbers
started out wrong, and the fitted numbers minimise the same loss that ranks the
front. The search may instead use the exact solver portfolio
(`--fit-strategy auto`): closed-form column-pivoted QR when the candidate is
affine in every constant, variable projection (Golub–Pereyra) when it is
separable, and the historical LM path for everything else. `varpro`, `stream`
(out-of-core f64 sufficient statistics) and `gpu` (the #169 batched kernel with
an exact CPU verifier, which fits candidates with at most 20 constants on the
device — least squares for the squared/AIC family, IRLS for `log_squashed`
(#229), L1 IRLS with true-MAE acceptance for `absolute` (#222), Fisher scoring
on the binary label for `log_loss` (#234; the device proposes the constants
and the exact CPU `f64` log-loss judges the proposal and refits every finalist,
and since #289 the search also scores log-loss, Brier and the three hinge forms
with a native label-aware kernel — `hinge` is explicitly the bounded sigmoid
surrogate, not the SVM hinge), and the exact correlation objective with
a frozen affine gauge
(#224) — and labels every larger or unsupported candidate for the exact CPU
arm; the correlation *scorer* is native on the device since #287;
below 1 048 576 training rows it fits the screened top
`max(32, pop/4)` of each island, and at or above that size every candidate with
constants, so the seed-constant screen cannot cost quality at large `N`) can be
forced individually. For `--metric rank_correlation`/`rank_correlation_signed`
the same arm fits the exact integer rank polish on the device; the same exact
integer sums also **score** every candidate on the device from 4 096 training
rows (#288), and finalists are still rebuilt in CPU `f64`; `rank_correlation_signed`
is the same stage under `1 - rho`, so a reflected model is rejected and a
proposal is judged on the exact oriented score, never on SSE or `|rho|` (#231);
`--metric median` does the same with the exact `f64` median and
`--metric interquartile` with the middle-50 % window mean of `|r|` (fit and
native scoring from 4 096 training rows — the measured crossover of the batched
selection, #226, #232 and #288), and `--metric maximum` fits the Chebyshev
objective by **exact bisection on the error level** per constant (from 512
training rows — the measured cost-model floor, #227) and is scored natively by
the #287 maximum reducer. An explicit `gpu` with a metric that has no device fit
objective is a usage error, because the kernel would otherwise fit a different
loss; use `--metric squared`, `--metric log_squashed`, `--metric absolute`,
`--metric log_loss`, a rank metric, `--metric median`, `--metric
interquartile`, `--metric maximum` or a CPU strategy.
Every arm falls back to the default LM path when it cannot prove a safe result,
and the default remains the historical LM path bit for bit. `--metric signed`
(the absolute mean residual) is the one non-squared metric with an exact closed
form: its constants are projected in f64 onto the zero-mean-residual manifold by
a constrained least-squares step (two passes, no optimiser), and when no
constant can move the mean the arm says so instead of fabricating a fit — see
[`docs/optimisation.md`](docs/optimisation.md) §3.0.1.
  analytic derivatives for every differentiable primitive (`erf`/`erfc`,
  `atan2`, `asinh`/`acosh`/`atanh`, and documented subgradients for
  `mod`/`min`/`max` and `and`/`or`/`xor`; `0` a.e. for the piecewise-constant
  `floor`/`ceil`/`round`/`sign`/`step`, the comparisons and `not`);
- an **island-model, multi-objective** evolutionary search over an
  error/complexity **Pareto front**, with mutation, crossover, migration and
  restarts, run in parallel across islands;
- **ε-dominance** archiving and **lexicographic parsimony pressure**, so the
  front stays small and expressions stay short;
- **Pareto-front parent feedback** (15%): one parent in six is drawn from the
  global archive instead of the island tournament, so a shape found on another
  island or generations earlier can be recombined rather than lost.
  Measured on the known-truth suite at three seeds: **72 → 74/75**, **72 → 75/75**
  and **72 → 74/75**, with `pythagoras` recovered in **7/9** cells instead of
  **0/9**, for about 5% more wall time. `--no-front-parents` disables it and
  restores the previous trajectory exactly;
- **opt-in quality-diversity search** (`--qd`, `--qd-cvt`, `--qd-policy`,
  `--qd-cells`, `--qd-cell-cap`): CVT-MAP-Elites over variable usage ×
  prediction skewness, with deterministic Halton initialisation, Lloyd
  centroid updates, a bounded local `(error, cost)` front per cell, UCB1
  parent selection, stratified migration and least-occupied restarts. The
  passive layer reports reachable-support coverage, occupancy entropy and a
  QD-score in text and `--json` without changing the Pareto front; every run
  also reports a fixed-reference validation hypervolume that is comparable
  across arms. `--qd-policy` is the opt-in search change. See
  [`docs/qd.md`](docs/qd.md);
- **configurable complexity** (`--op-complexity`, `--complexity-weights`,
  `--depth-weight`): per-operator costs, variable/constant weights and a
  nesting-depth penalty, with the node-count default preserved exactly;
- **structural de-duplication** and **convergence/time-budget stopping**, so the
  same expression is never evaluated twice and the search stops when there is
  nothing left to gain;
- **subsampled fitness** (`--predictor-samples N`): each generation scores and
  fits candidates on `N` training rows drawn without replacement, while the
  archive and every reported error are re-scored on the full training set.
  Carried individuals are refitted on the generation's subset before they are
  scored, so everyone is compared as `fit(S_g) -> score(S_g)` rather than with
  stale constants. A request at least the training size is an exact no-op, and
  a request above the measured break-even (3/5 of the rows) is widened to the
  full sample, because rebuilding the sample and refitting the carried
  population cost more than the rows saved;
- **screened constant fitting** (`--fit-samples N`): within each generation,
  all offspring are fitted and scored on a fresh deterministic `N`-row subset,
  the top 1/16 are promoted to a full-data refit and re-score, and the next
  population and archive are selected from those full-data scores. This is a
  throughput-for-statistics trade, not a free win: the sampled best is
  optimistically biased (the winner's curse), promotion reduces but does not
  remove it, and on deterministic, noiseless data the screen is a pure loss.
  `0` (the default) keeps the historical path; the flag is independent of
  `--predictor-samples`;
- **seeded starts** (`--seed-expr`, `--seed-file`): prior solutions are parsed,
  simplified, fitted and inserted into the archive and every island before
  generation 1; a seed is an elite, kept even above `--max-complexity`;
- **split modes for time series and panel data** (`random`, `temporal`, `group`,
  `temporal-group`, `auto`) with an embargo gap, per-block test errors and
  **delayed variables** (`--lags`, `--history-frac`, `--gap-break`), so a
  forecast is never scored on data it has already seen;
- **opt-in data preparation** — missing-value fills (`--fill`, missing marked
  by an empty cell, `NA`, `NaN` or `null`), IQR outlier removal, moving-average
  smoothing, median/IQR/range scaling and expression row filters (`--filter`) —
  with every statistic computed from the training rows only and a preparation
  report in text, JSON and NDJSON;
- **multi-output regression** (`--target y1,y2,...`): one shared tree shape,
  one fitted constant vector per output, combined error = mean per-output
  error; see [`docs/multi-output.md`](docs/multi-output.md);
- **overfitting diagnostics** on every solution (AICc, BIC, Durbin–Watson,
  overfit index, and per-constant **standard errors** from the Gauss–Newton
  Jacobian that flag constants the data cannot pin down), **null baselines**
  (mean, least-squares line, 5-NN) and **residual tests** (Ljung–Box,
  Breusch–Pagan, skewness/kurtosis) reported automatically, plus a
  `--null-runs` permutation control that runs the identical search on shuffled
  targets, a `--bootstrap-error` `.632+` bootstrap estimate that refits each
  solution's constants on resamples, `--cv` cross-validation (`k-fold`,
  `rolling`, `purged`) that reports the mean and spread of the held-out error,
  and `--stability` bootstrap selection that ranks forms and variables by how
  often they are rediscovered;
- **Markdown reports** (`--report summary|sensitivity|benchmark`): a
  self-contained model summary with a statistics-driven trust verdict, an
  analytic per-input sensitivity ranking (`mean(|∂f/∂x_i| · σ_x_i / σ_f)`, no
  finite differences), and a repeated-search benchmark with per-complexity
  median/IQR error and wall-clock time;
- a sound, value-preserving bottom-up normal form for scoring/dedup/CSE
  (`expr::simplify`) plus an aggressive display relation with affine
  normalisation (`expr::simplify_display`), and denormalisation back to original
  units.

See [`docs/optimisation.md`](docs/optimisation.md) for the measured before/after
numbers and the hardware plan.

## Backends: GPU by default when available

A binary built with `--features cuda` probes the device once at startup and uses
the GPU to guide the search whenever one is available **and the run's metric is
scored on the device**; a build without the feature, or a machine with no usable
device, falls back to the CPU and records the reason. `--gpu` asserts that a
device must exist (exit 2 otherwise), `--no-gpu`/`--cpu` forces the CPU, and
`RCSR_GPU=0` simulates a missing device. Non-quiet runs print one resource line
with the backend, device, available threads and training rows/columns, and
`--json` records `backend`, `backend_reason` and `device`.

The kernel scores the configured metric natively through the #287
`Metric -> ScoreDispatch` table. The objective kernel covers `squared`,
`absolute`, `log_squashed`, `signed`, `maximum`, `correlation`,
`correlation_signed`, `aic`/`aicc`/`bic` and the classification family
(`log_loss`, `brier`, `hinge*`, #289). The exact `f64` order-statistic/rank
batch entry point (#288) covers `median`, `interquartile`, `rank_correlation`,
`rank_correlation_signed` and `auc` from 4 096 training rows (`auc` also needs a
two-valued target). Every other metric, and the order/rank metrics below their
floor, keep the CPU evaluator, print the *named* reason and record it in
`backend_reason`; a device run is never ranked by a substitute loss.

Cross-backend runs are **not bit-identical**: objective-kernel fitness is `f32`,
so it can change the search trajectory. The #288 statistics are exact `f64` and
match the CPU bit for bit for the same prediction vector (median and
interquartile through a radix select from 4 096 and 65 536 rows, rank and AUC
through a sort), but the prediction itself may round differently on the two
devices. Reported metrics, constants and diagnostics are always rebuilt on the
CPU in exact `f64`; use `--cpu` for a byte-identical reproduction. A metric may
use the GPU only when the score table names a native scorer for it, so no run is
ever ranked by a proxy loss (#236). Native scoring returns `f64` plus a status; a
non-finite prediction is `+inf` on both CPU and device, never a finite sentinel
(#287, #288). A GPU run also reports a
measured `gpu` block (allocation, H2D, launch, sync, D2H, wall, throughput,
evaluator-window kernel duty, quality per GPU-hour) and a run-scoped
`gpu.ledger`: exclusive stage seconds, device-busy as the **union** of device
intervals over run wall (never their sum), fit/eval/fallback seconds with
labelled causes, fit kernel versus CPU-fallback time, and
numerator/denominator/units/scope on every derived ratio. `Σ stage_s +
unaccounted_s = wall_s` within 1 %; `gpu.eval_kernel_duty` (with `gpu.eval_window_s`) is explicitly the evaluator
window, while `gpu.ledger.duty_run` is the run-level headline. `gpu.device_seconds_incl_fit`,
`gpu.device_fraction_of_run` and `gpu.eval_fraction_of_run` give the device
seconds including the fit kernels and their share of `timing.total_seconds`,
the one runtime of the process; `gpu.batches_per_s` counts evaluator batches,
not generations (#256). The packed batch
path allocates a persistent device arena once, stages host bytes in a <=64 MiB
pinned ring and copies on `S_h2d`/`S_d2h` with event dependencies, so `alloc_s`
is ~0 in steady state; `RCSR_GPU_SYNC=1` forces the labelled blocking fallback
(#285). The fit, rank, order-statistic and maximum batches use persistent device and
pinned regions grown at the high-water mark and one event synchronisation per
batch; `gpu.ledger.fit.alloc.steady_alloc_calls` is 0 in steady state (#334). See
[`docs/gpu-backend.md`](docs/gpu-backend.md#measuring-the-full-gpu-path-issue-118).
`sm_occupancy` is `null` unless measured separately with `ncu`; `nvidia-smi`
sampled utilisation is not occupancy. See
[`docs/gpu-backend.md`](docs/gpu-backend.md).

## Is the search converging?

`--progress` reports four bounded numbers every generation — **confidence**
(the best error stopped moving), **stability** (the front's structural forms
survive), **maturity** (the oldest surviving form's age) and **percent
converged** (their weighted headline) — and the final `--json`/NDJSON output
carries the same `"convergence"` object. They are computed from the archive
alone, cost O(front size) and never touch a training row. See
[`docs/convergence.md`](docs/convergence.md) for the exact formulas, the
20-generation window, how to read them and the easy/hard evidence.

## Why did the run behave like that?

Every run writes a bounded NDJSON **development log** — on by default at
`summary` level, the in-engine replacement for an external run monitor.
`run_start`, one generation event every few generations (generation, elapsed,
best/median error, Pareto size, phase times, fit/GPU/fallback counts, GPU duty,
convergence), `checkpoint` events and `run_end`. `--dev-log-level full` writes
every generation with the complete record (all error quantiles, archive
hypervolume/hash, de-duplication rate, admissions, fallback-by-cause, QD
coverage/entropy, per-island best, RNG key). The file is `rcsr.dev.ndjson` next
to `--json-out`/`--save`, else in the working directory; `--no-dev-log` turns it
off. It never changes the archive, the results or the configuration hash, each
event is one atomic line, and the file is bounded by `--dev-log-max-bytes`.
`rcsr dev-report <log>` (pure Rust, no Python) turns the file into a phase table,
a fit-versus-evaluation share, fallback causes, convergence and archive growth,
and a ranked list of bottlenecks. See [`docs/dev-log.md`](docs/dev-log.md).

## Layout

```
SPEC.md                       behaviour specification
docs/how-symbolic-regression-works.md   plain-language explainer
docs/optimisation.md          theory, literature and measurements behind the search
docs/data-splitting.md        train/test splitting for time series and panel data
docs/time-series.md           time-series/panel checklist: purge, look-ahead, lockbox, costs
docs/data-preparation.md      missing values, outliers, smoothing, scaling and filters
docs/memory-model.md          bytes per cell, dictionary encoding and row selectors
docs/data-format.md           versioned .rcsb column store, import and the memory guard
docs/walkforward.md           walk-forward evaluation, costs, metrics and the long/flat rule
docs/variable-types.md        boolean and categorical inputs, one-hot encoding
docs/weights.md               per-row weights, weighted metrics and weighted least squares
docs/implicit.md              implicit relations f(x, y) = 0 and the level-set metric
docs/overfitting.md           avoiding and detecting overfitting, with references
docs/meta-operators.md        structural meta operators and their cache rule
docs/aggregation.md           dataset aggregates: semantics, cache and leakage rules
docs/seeds.md                 start the search from a known formula (--seed-expr/--seed-file)
docs/checkpoint.md            save/resume state, atomic writes and backend identity
docs/run-integrity.md         run manifests, verification and reproducible launchers
docs/building-blocks.md       user-defined reusable sub-formulas (--function)
docs/multi-output.md          one shared shape, per-output constants (--target y1,y2)
docs/benchmarks.md            known-truth recovery benchmarks and recorded results
docs/convergence.md           confidence, stability, maturity and percent converged
docs/dev-log.md               optional per-generation development log and `dev-report`
docs/qd.md                    opt-in CVT-MAP-Elites quality-diversity layer (--qd/--qd-policy)
docs/gui.md                   design for a GUI over the JSON/NDJSON contract
docs/gpu-backend.md           the optional CUDA evaluator and the DGX Spark plan
my-version/
  Cargo.toml
  build.rs      compiles the CUDA backend when --features cuda is used
  src/
    rng.rs        MT19937 RNG for reproducible runs
    expr.rs       expression trees, primitives, evaluator, mutation, simplification
    fit.rs        Levenberg–Marquardt constant fitting
    metrics.rs    error metrics
    pareto.rs     the error/complexity Pareto front
    data.rs       table loading, split modes, normalisation
    bench.rs      known-truth benchmark suite: problems, generators, scoring
    search.rs     island-model GP loop
    gpu.rs        bytecode compiler + FFI for the optional GPU evaluator
    cuda/eval.cu  the CUDA kernel
    main.rs       the `rcsr` CLI
    bin/
      gpu_bench.rs  GPU vs CPU evaluation benchmark (--features cuda)
      rcsr_bench.rs the `rcsr-bench` known-truth benchmark harness
  tests/
    recover.rs    end-to-end recovery tests + unit tests
    split.rs      split-mode tests (temporal, group, panel, folds)
    prep.rs       data-preparation tests (fill, outliers, smoothing, scaling, filters)
    weights.rs    row weights, weighted metrics and weighted least squares
    bench.rs      known-truth benchmark harness tests
  examples/
    line.csv, sine.csv
```

## License

MIT.
