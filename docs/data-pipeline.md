# Data pipeline

The crypto data pipeline is Rust. It uses `rcsr prep`, `rcsr panel`, `rcsr enrich`, `rcsr walkforward` and `rcsr verify-manifest`. The acquisition step is the standalone tool `rcsr-acquire` in `tools/acquire/`, which shells out to `curl` (the core crate stays pure `std`, so there is no hidden HTTP client behind a Rust name).

## Decisions

* **Downloads.** No HTTP dependency is added to the core crate. Acquisition is the
  separate pure-std crate `tools/acquire/` (`rcsr-acquire`), which runs `curl` against
  `https://data.binance.vision/`. `list` reads the bucket listing, `fetch` skips
  existing zips and records every request, 404 included, in a status CSV, and
  `verify` checks each zip against its vendor `.CHECKSUM` and writes the snapshot
  JSON. `universe` filters the listed USDT symbols to genuine crypto pairs from pinned
  file inputs and does no network access. Details: [README, "Acquiring the Binance Vision archive"](../README.md#acquiring-the-binance-vision-archive).
* **Panels and membership.** `rcsr prep` owns raw decoding, resampling, causal
  joins, features and labels (#123/#125); `rcsr panel` owns point-in-time
  membership, costs and the lockbox (#145).
* **Enrichment.** `rcsr enrich` appends pre-computed columns without rebuilding
  the store (#200); it does not duplicate the #125 feature library, which it
  reuses for `ta:*` names.
* **Evaluation and orchestration.** `rcsr walkforward` (#124), checkpoints and
  run manifests (#140) run evaluation. Launchers are `scripts/run-long.sh` and `rcsr-launch`.
