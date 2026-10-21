# Benchmarks and reference programs

`eval_python_reference.py` is the **pinned row-wise Python reference** for
issue #124. It is deliberately the pilot's streaming `csv.reader` loop, not a
vectorised rewrite; changing its arithmetic changes the reference and therefore
invalidates the 10x comparison. Do not optimise it.

`eval_10x.py` runs that reference and `rcsr walkforward` over the same
3,943,856 x 56 hourly panel projection, records wall time, peak RSS, the q70
thresholds and hashes, and writes the JSON/Markdown evidence under `results/`.

```sh
# from the repository root, with the git-ignored pilot data present
python3 benchmarks/eval_10x.py --repeats 2 --check
```

The full protocol and the argument for the row-wise reference are in
[`../docs/eval-benchmark.md`](../docs/eval-benchmark.md). The long run is also
available as the ignored Rust test
`my-version/tests/walkforward_bench.rs::eval_10x_against_pinned_python_reference`.
