#!/usr/bin/env python3
"""Pinned row-wise Python reference for the #124 walk-forward comparison.

This is the pilot's *row-wise* path frozen as a benchmark artifact. It is
deliberately not vectorised and uses the standard library only:

* `csv.reader`, never `csv.DictReader` and never pandas/numpy;
* one pass over the CSV to collect finite training scores in an
  `array('d')`, then one pass to build the test P&L;
* per-timestamp `(sum, count)` dictionaries and one last position per symbol;
* q70 by numpy's default linear interpolation (type 7), written out by hand.

Do not "optimise" this file. The comparison in `eval_10x.py` measures the
Rust walk-forward evaluator against *this* row-wise program, which is the
program the Rust evaluator replaces. A vectorised rewrite would be a different
reference and would invalidate the protocol in `docs/eval-benchmark.md`.

Protocol (see `docs/eval-benchmark.md` for the full story):

    corpus:  panel_hourly_compact.csv (3,943,856 x 56, 4,233,483,400 bytes)
    window:  test [2024-01-03, 2025-01-01), purge-matched
    target:  y60, score: r16, long if score > q70(train scores)
    costs:   10 bp per side on |position - previous position| per symbol
    annualisation: 8,760 hourly periods per year

Output is a single JSON object on stdout; timings and peak RSS are measured by
the caller with `/usr/bin/time`, but the program also reports its own peak RSS
for corroboration.
"""
import argparse
import csv
import json
import math
import os
import resource
import sys
import time
from array import array

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_CSV = os.path.join(
    ROOT, "arbeidsmappe", "crypto-rcsr", "pilot", "panel_hourly_compact.csv"
)


def open_reader(path):
    f = open(path, newline="")
    r = csv.reader(f)
    header = next(r)
    return f, r, {name: i for i, name in enumerate(header)}


def train_scores(path, train):
    f, r, ix = open_reader(path)
    it, ir = ix["ts"], ix["r16"]
    out = array("d")
    rows = 0
    t0 = time.perf_counter()
    for row in r:
        ts = int(int(row[it]) / 1e9) if len(row[it]) > 12 else int(row[it])
        if train[0] <= ts < train[1]:
            v = float(row[ir])
            if math.isfinite(v):
                out.append(v)
        rows += 1
    f.close()
    out = array("d", sorted(out))
    return out, rows, time.perf_counter() - t0


def q70(sorted_scores):
    """numpy's default linear interpolation (type 7)."""
    pos = 0.70 * (len(sorted_scores) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_scores[lo]
    frac = pos - lo
    return sorted_scores[lo] * (1 - frac) + sorted_scores[hi] * frac


def run_test(path, test, tau, cost):
    f, r, ix = open_reader(path)
    it, ir, iy, isym = ix["ts"], ix["r16"], ix["y60"], ix["symbol"]
    sums, counts, last = {}, {}, {}
    rows = 0
    t0 = time.perf_counter()
    for row in r:
        ts = int(int(row[it]) / 1e9) if len(row[it]) > 12 else int(row[it])
        if not (test[0] <= ts < test[1]):
            continue
        sym = row[isym]
        score = float(row[ir])
        pos = 1.0 if (math.isfinite(score) and score > tau) else 0.0
        prev = last.get(sym, 0.0)
        fee = cost * abs(pos - prev)
        last[sym] = pos
        y = float(row[iy])
        pnl = pos * y - fee if math.isfinite(y) else -fee
        sums[ts] = sums.get(ts, 0.0) + pnl
        counts[ts] = counts.get(ts, 0) + 1
        rows += 1
    f.close()
    daily = [sums[t] / counts[t] for t in sorted(sums)]
    return daily, rows, time.perf_counter() - t0


def stats(d):
    m = sum(d) / len(d)
    var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
    sd = math.sqrt(var)
    return m, sd, (m / sd * math.sqrt(8760) if sd else float("nan"))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=os.environ.get("RCSR_BENCH_CSV", DEFAULT_CSV))
    p.add_argument("--train-start", type=int, default=1609459200)  # 2021-01-01
    p.add_argument("--train-end", type=int, default=1704067200)    # 2024-01-01
    p.add_argument("--test-start", type=int, default=1704240000)   # 2024-01-03
    p.add_argument("--test-end", type=int, default=1735689600)     # 2025-01-01
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--periods-per-year", type=float, default=8760.0)
    args = p.parse_args(argv)

    t0 = time.perf_counter()
    scores, rows_scanned, t_train = train_scores(
        args.csv, (args.train_start, args.train_end)
    )
    tau = q70(scores)
    daily, test_rows, t_test = run_test(
        args.csv,
        (args.test_start, args.test_end),
        tau,
        args.cost_bps / 10_000.0,
    )
    total = time.perf_counter() - t0
    mu, sd, sr = stats(daily)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux

    out = {
        "reference": "row-wise csv.reader (pinned, stdlib only)",
        "csv_bytes": os.path.getsize(args.csv),
        "rows_scanned": rows_scanned,
        "train_scored": len(scores),
        "test_rows": test_rows,
        "timestamps": len(daily),
        "threshold": tau,
        "mean_return": mu,
        "sd_return": sd,
        "sharpe": sr,
        "periods_per_year": args.periods_per_year,
        "timings": {
            "train_seconds": t_train,
            "test_seconds": t_test,
            "total_seconds": total,
        },
        "peak_rss_kib_self": rss,
        "window": {
            "train_start": args.train_start,
            "train_end": args.train_end,
            "test_start": args.test_start,
            "test_end": args.test_end,
            "cost_bps": args.cost_bps,
            "score": "r16",
            "target": "y60",
        },
    }
    json.dump(out, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
