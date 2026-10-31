# Python pipeline retirement (issue #200)

The crypto pipeline used to be driven by 41 Python scripts in the git-ignored
`arbeidsmappe/crypto-rcsr/pilot/` directory. This page is the complete
classification. The **documented pipeline is Rust-first**: it uses `rcsr prep`,
`rcsr panel`, `rcsr enrich`, `rcsr walkforward` and `rcsr verify-manifest`, and
the acquisition step is the standalone tool `rcsr-acquire` in `tools/acquire/`, which
shells out to `curl` (the core crate stays pure `std`, so there is no hidden HTTP
client behind a Rust name).

One-off research probes are **archived, not ported**. The archive is the
git-ignored `arbeidsmappe/crypto-rcsr/pilot/archive/`; nothing in `docs/`,
`README.md` or the CLI invokes a script from it. A script marked *superseded*
was used in production at least once; its behaviour now lives in a Rust command
or in `rcsr-acquire`.

| # | script | class | target / decision |
|---:|---|---|---|
| 1 | `build_hourly_all.py` | superseded | `rcsr prep --input raw_15m_all --resample 1h --features ... --out panel_hourly_all.rcsb` |
| 2 | `build_panel.py` | superseded | `rcsr prep` (resample + features) and `rcsr panel`; historical 20-coin config archived |
| 3 | `build_panel_15m.py` | superseded | `rcsr prep --resample 15m --features ...`; 10-coin config archived |
| 4 | `build_panel_15m_100.py` | superseded | same `rcsr prep` path; the 100-coin universe is a `rcsr panel` membership input |
| 5 | `build_panel_4h.py` | superseded | `rcsr prep --resample 4h`; the long/flat label is `rcsr prep --label` |
| 6 | `build_panel_daily_100.py` | superseded | `rcsr prep --resample 1d --label ...` |
| 7 | `build_panel_daily_all.py` | superseded | `rcsr prep --resample 1d` + `rcsr panel` membership |
| 8 | `build_panel_daily_long.py` | superseded | `rcsr prep --resample 1d --label ...`; long-history config archived |
| 9 | `build_universe_pit.py` | superseded | `rcsr panel` point-in-time membership |
| 10 | `compact_hourly.py` | superseded | `rcsr prep --resample 1h`; extra hourly features are `rcsr enrich` |
| 11 | `dl_15m_all.py` | superseded | `rcsr-acquire fetch` (`curl` transport, no Rust HTTP client) |
| 12 | `download_15m_100.py` | superseded | `rcsr-acquire fetch` |
| 13 | `download_15m_long.py` | superseded | `rcsr-acquire fetch` |
| 14 | `download_daily.py` | superseded | `rcsr-acquire fetch` |
| 15 | `download_daily_all.py` | superseded | `rcsr-acquire fetch` |
| 16 | `download_hourly_long.py` | superseded | `rcsr-acquire fetch` |
| 17 | `download_minute.py` | superseded | `rcsr-acquire fetch` |
| 18 | `download_new_tickers.py` | one-off | archived (ad-hoc ticker list) |
| 19 | `download_spot.py` | superseded | `rcsr-acquire fetch` |
| 20 | `download_tokenized_stocks.py` | one-off | archived (tokenised-stock research) |
| 21 | `enrich_panel_15m.py` | superseded | `rcsr enrich --features py15m` |
| 22 | `evaluate.py` | superseded | `rcsr walkforward --markdown ...` |
| 23 | `evaluate_100.py` | superseded | `rcsr walkforward --markdown ...` |
| 24 | `evaluate_4h.py` | superseded | `rcsr walkforward --markdown ...` |
| 25 | `evaluate_daily_100.py` | superseded | `rcsr walkforward --markdown ...` |
| 26 | `evaluate_walkforward.py` | superseded | `rcsr walkforward --windows ...` |
| 27 | `launch_10h_all.py` | superseded | `rcsr walkforward` with `--run-manifest`, `--save`/`--resume` (#124, #140); detached-launch wrapper is shell |
| 28 | `list_archive_keys.py` | superseded | `rcsr-acquire list` |
| 29 | `make_manifest_daily_all.py` | superseded | the manifest written by `rcsr prep`; integrity via `rcsr verify-manifest` |
| 30 | `monitor_10h.py` | one-off | archived; progress is `--progress`/`--format ndjson` |
| 31 | `organize_tokenized_stocks.py` | one-off | archived |
| 32 | `per_asset_4h.py` | one-off | archived; per-group evaluation is `rcsr walkforward --group-by symbol` when needed |
| 33 | `resume_10h.py` | superseded | `rcsr walkforward --resume <checkpoint>` |
| 34 | `resume_daily.py` | superseded | `rcsr walkforward --resume <checkpoint>` |
| 35 | `resume_long.py` | superseded | `rcsr walkforward --resume <checkpoint>` |
| 36 | `run_gpu_8h.py` | superseded | `rcsr walkforward --gpu ...` |
| 37 | `run_gpu_daily_8h.py` | superseded | `rcsr walkforward --gpu ...` |
| 38 | `run_gpu_daily_extra.py` | superseded | `rcsr walkforward --gpu ...` |
| 39 | `run_walkforward.py` | superseded | `rcsr walkforward --windows ...` |
| 40 | `select_universe.py` | superseded | `rcsr panel` membership selection |
| 41 | `verify_hourly.py` | superseded | `FileStore::open` validates magic/version/header/block offsets; `rcsr verify-manifest` checks the provenance. The redundant script is archived |

## Decisions

* **Downloads.** No HTTP dependency is added to the core crate. Acquisition is the
  separate pure-std crate `tools/acquire/` (`rcsr-acquire`), which runs `curl` against
  `https://data.binance.vision/`. `list` reads the bucket listing, `fetch` skips
  existing zips and records every request, 404 included, in a status CSV, and
  `verify` checks each zip against its vendor `.CHECKSUM` and writes the snapshot
  JSON. Details: [README, "Acquiring the Binance Vision archive"](../README.md#acquiring-the-binance-vision-archive).
* **Panels and membership.** `rcsr prep` owns raw decoding, resampling, causal
  joins, features and labels (#123/#125); `rcsr panel` owns point-in-time
  membership, costs and the lockbox (#145).
* **Enrichment.** `rcsr enrich` appends pre-computed columns without rebuilding
  the store (#200); it does not duplicate the #125 feature library, which it
  reuses for `ta:*` names.
* **Evaluation and orchestration.** `rcsr walkforward` (#124), checkpoints and
  run manifests (#140) replace the `evaluate*` and `run_*`/`resume_*` scripts.
* **One-offs.** The six one-off scripts above are archived and are not part of
  any documented command. Re-running them is unsupported.
