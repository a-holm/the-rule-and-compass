# Walk-forward benchmark protocol: Rust vs the row-wise Python reference

Issue #124 asks whether `rcsr walkforward` is at least **10x faster** than the
Python evaluation path on the 4M-row panel, while reading the store directly.
This document pins the corpus, the reference, the command and the exact
acceptance check so the number cannot drift to a different fixture or a
different Python program.

The executable form is:

```sh
# from the repository root
python3 benchmarks/eval_10x.py --repeats 2 \
  --json benchmarks/results/issue-124-eval-10x.json \
  --markdown benchmarks/results/issue-124-eval-10x.md
```

or, as a long-running ignored test:

```sh
/mnt/lager/tmp/opencode/one-full-suite.sh bash -c \
  'cd worktrees/issue-124-benchmark/my-version && \
   cargo test --release --test walkforward_bench -- --ignored --nocapture'
```

The test fails unless the q70 threshold agrees within `1e-12` and the median
wall ratio is at least `10x`. The driver writes a JSON report with every
repeat, both thresholds, peak RSS, hashes and the ratios.

The pilot corpus is git-ignored and lives in the main checkout, so a linked
worktree must point at it explicitly:

```sh
RCSR_BENCH_CSV=/mnt/lager/eureqa/arbeidsmappe/crypto-rcsr/pilot/panel_hourly_compact.csv \
RCSR_BENCH_STORE=/mnt/lager/eureqa/arbeidsmappe/crypto-rcsr/pilot/panel_hourly_v2.rcsb \
python3 benchmarks/eval_10x.py --repeats 2
```

## Corpus

| item | value |
|---|---|
| CSV | `arbeidsmappe/crypto-rcsr/pilot/panel_hourly_compact.csv` |
| CSV size | 4,233,483,400 bytes |
| CSV SHA-256 | `7a3233822af84dbe68f73f005627713fdc74f4ad31e0e4db023f6dd047ed5bca` |
| rows x columns | 3,943,856 x 56 |
| store | `arbeidsmappe/crypto-rcsr/pilot/panel_hourly_v2.rcsb` |
| store SHA-256 | `7b98a2836f97b4d6ac6385443486a81f73fec929de310f28d6a4efa3cb78a6f4` |

The CSV is the projection of the same 56 columns as the `.rcsb` store (header
`ts,symbol,r16,...,fwd60,y60`). Use the **v2** store: it is self-describing
(`ts` as `i64`, `symbol` as `u32`) and is the layout the Rust evaluator reads.
The older `panel_hourly_compact.rcsb` in the same directory is a legacy v1
all-`f64` copy of the same rows; it gives identical thresholds and metrics but
is a slower physical layout, so it is not the pinned benchmark store. If the v2
store is absent, rebuild it from the pinned CSV with
`rcsr import arbeidsmappe/crypto-rcsr/pilot/panel_hourly_compact.csv
arbeidsmappe/crypto-rcsr/pilot/panel_hourly_v2.rcsb --group-rows 65536`.
It is git-ignored pilot data, not a fixture to commit; the hashes above are the
pin. The earlier PR comment claimed no CSV projection of the hourly panel
existed. That was false: this file was on the host, and it is the artifact the
pilot itself would have read.

## Window and arithmetic

| field | value |
|---|---|
| train | `[1609459200, 1704067200)` — 2021-01-01 to 2024-01-01 |
| test | `[1704240000, 1735689600)` — 2024-01-03 to 2025-01-01 |
| purge | 172,800 s (the `y60` horizon), train ends 2024-01-01, test starts 2024-01-03 |
| score | `r16` |
| target | `y60` |
| group | `symbol` |
| position | `1` if `score > q70(train scores)` and finite, else `0` (long/flat only) |
| quantile | q70, numpy type-7 linear interpolation |
| cost | 10 bp per side on `abs(position - previous position)` inside each symbol |
| aggregation | equal-weight mean net return per timestamp |
| annualisation | 8,760 hourly periods per year |

Both sides start every symbol flat and charge a cost whenever the position
changes. The Rust evaluator additionally charges each symbol's final exit, as
documented in `walkforward.md`; the pinned Python loop only charges changes it
observes, so its final-exit term is absent. The difference is one side per
symbol at the end of the window (negligible against 173k trades), and it is
part of why the pooled Sharpe differs in the fifth decimal rather than being
bit-identical. The threshold, which is the equality gate, is identical.

## Why the row-wise reference is the fair one

The issue says the walk-forward **used Python to read a 25 GB CSV, evaluate
expressions row by row and compute P&L**, and that the Rust tool should replace
that path. The fair comparison is therefore the same program at the same
semantic level:

* `benchmarks/eval_python_reference.py` is the pilot loop, not a rewrite: it
  uses `csv.reader` (never `csv.DictReader`), the standard library only, one
  `array('d')` of finite training scores, per-timestamp `(sum, count)`
  dictionaries and one last position per symbol. It does not use pandas,
  numpy, memory maps, multiprocessing or vectorised expression evaluation.
* Both sides perform the same work: read the corpus, evaluate `r16` over the
  training rows, compute q70, evaluate the test rows, apply the long/flat rule,
  charge per-symbol position changes and aggregate by timestamp.
* The Rust side is allowed to be faster because it avoids text parsing and
  interprets the `.rcsb` value columns directly through the engine's own
  `Expr` evaluator. That is exactly the improvement the issue asks for.

A vectorised or pandas reference would be a **different program**. It could be
made very fast, but then the comparison would no longer answer “did replacing
the pilot's row-wise loop with the Rust evaluator make it 10x faster?” If the
project ever wants a second, “fastest possible Python” comparison, that is a
new protocol amendment and a new benchmark, not a substitute for this one.

The pinned reference is:

`benchmarks/eval_python_reference.py`, SHA-256 recorded in every report
(`reference_sha256`). Changing that file changes the reference hash and the
evidence no longer applies.

## Measurement

* Wall clock and peak RSS come from GNU `time -f '%e %M'` around each child.
  RSS is maximum resident set size in KiB.
* Two repeats are run and both are reported; the first repeat warms the page
  cache for the second, and the median of the two is the gate. A cold-cache run
  is not part of the pass/fail gate.
* CPython 3.12, single-threaded. The Rust release binary is built from the
  branch under test.
* The ratio is `median(Python wall) / median(Rust wall)`. The gate is
  `ratio >= 10` **and** `abs(tau_python - tau_rust) <= 1e-12`.
* The report also records the binary, store, CSV and script hashes, the host
  platform and interpreter version.

## Result (measured)

The committed artifact is `benchmarks/results/issue-124-eval-10x.md`, generated
by the pinned protocol above on the host that owns the pilot data (Intel
i7-3770K, 31 GiB, Linux). Warm-cache, two repeats, medians:

| side | wall | peak RSS | q70 threshold |
|---|---:|---:|---:|
| Python row-wise reference | 81.4 s | 104.9 MiB | `0.007034873088967409` |
| Rust `rcsr walkforward` | 4.06 s | 107.7 MiB | `0.007034873088967409` |

The committed wall ratio is **20.1x**, the threshold is bit-identical
(`delta = 0`), and the RSS ratio is 0.97x. This is the reproducible gate and it
passes the issue's **at least 10x** requirement with margin.

The feasibility pass that motivated this protocol measured a colder cache and
reported **51x–75x** for the two pilot Python variants: 125.5 s / 2.46 s = 51x
(test from 2024-01-01) and 185.3 s / 2.46 s = 75x (purge-matched 2024-01-03,
the window pinned here). That range is a second measurement of the same
artifact, not a substitute for the committed report; it shows that the ratio is
cache- and load-sensitive while every measurement remains well above 10x. The
committed result deliberately reports every repeat and the median, so the
reader can see both warm runs rather than a cherry-picked maximum.

Test rows differ by 6 out of 845,928 because the Rust evaluator drops rows
whose score or target is non-finite while the Python reference still charges
their cost; this is a row-filter difference, not a semantic divergence, and it
is recorded in the report (`test_row_delta`).

The committed result file is the source of truth for the exact run. Different
hosts will give different absolute seconds, so the gate is the ratio, not a
fixed wall time.
