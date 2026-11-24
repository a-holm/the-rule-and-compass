# h1-mini: a small run fixture for bit-identity proofs

A bit-identical engine change proves itself by hash equality on one small run. h1-mini is that run: a seeded synthetic hourly panel with the h1 book (minimum holding 12, plain long-only band 0, QD with the book descriptor, device fit thresholds).

Files:

* `rcsr-engine/tests/common/h1_mini.rs`: the panel generator, 8 groups of 5,000 bars (40,000 rows), on top of the synthetic panel of `tests/common/synth_panel.rs`. The same seed gives the same bytes (`tests/h1_mini.rs`). No data file is committed.
* `scripts/h1-mini/panel.rs`: prints the panel CSV for a seed. `h1-mini.sh prep` builds it with `rustc -O`.
* `scripts/h1-mini/h1-mini.sh`: `prep` (import and enrich with the binary under test) and `run` (one arm).
* `scripts/h1-mini/hash.sh`, `strip.jq`, `strip-chunk.jq` and `strip-ticket.jq`: the hash set H and the strip lists.

## The run

`h1-mini.sh run <rcsr> <dir> <seed> <chunks> <out>` runs `<rcsr>` on the store made by `prep`. Flags: `--fitness pnl --pnl-blocks 8 --band 0 --min-holding 12 --qd-policy --qd-descriptor book --fit-thresholds --fit-accept fitness`, the 14 operators (`min`, `max`, `greater`, `if` included), `--pop 32 --islands 4 --explore-islands 1`, `--min-trades 10`, `--fit-rows 1024`, `--generations 30`, `--save-every 10`, seed 11 or 12. `--migration-interval` is 25 (`H1_MINI_MIGRATION_INTERVAL`); no field of `run.json` or the logs records a migration, so the run does not prove one happened. `H1_MINI_BACKEND=gpu` (default) uses the device flags of the pilot (`--gpu --fit-strategy gpu --fit-max-iters 12 --gpu-memory 6GiB --pipeline-depth 0`); `cpu` uses `--cpu`. A gpu arm holds the device: start it through the shared test lock.

`<chunks>` 1 runs 30 generations in one process. `<chunks>` 3 runs up to three processes as a pilot unit does: the first two with `--time-budget 10 --interim-chunk`, the last resumes and runs to generation 30. The generation each chunk stopped at goes to `<out>/chunks.txt`. The chunk boundaries depend on the clock; the hashes must not.

## The hash set H

`hash.sh <out>` prints nine lines `<name> <sha256>`. It reads `grep`, `jq` and `sha256sum` only and prints no run value.

| name | input |
|---|---|
| `ckpt_front` | `unit.ckpt` lines `archive`, `archive_hash`, `final`, `sol`, `fsol` |
| `ckpt_ind` | `ind` lines (exit population) |
| `ckpt_rng_mix` | `rng` and `mix` lines |
| `ckpt_qd` | `qd_*` lines |
| `run_pareto` | `jq -S -c .pareto` of `run.json` |
| `run_pareto_expressions` | `[.pareto[].expression_exact]` |
| `run_finalist` | `.objective.finalist.expression_exact` |
| `run_stripped` | `run.json` after the strip list S, `jq -S -c` |
| `run_stripped_chunked` | `run.json` after `strip-chunk.jq` (S plus the per-process counters `.search.books` and `.search.fit_accept`) |

`build` and `config` lines of the checkpoint are not hashed.

`hash.sh keys <a> <b>` prints the key paths of two `run.json` files whose values differ after S (names only). `H1_MINI_STRIP=<jq file>` replaces S; a file holding `.` shows every differing key.

## The strip list S

`strip.jq` deletes: `.timing`, `.elapsed_seconds`, `.gpu`, `.dispatch`, `.checkpoint`, `.tool`, `.device_probed`, `.backend_reason`, `.threads`, `.process_backend`, `.baselines.knn.seconds`, `.fitting.gpu_batches`, `.preparation.store_memory`, `.search.cumulative.chunks`, `.search.cumulative.device_hours`, `.search.cumulative.elapsed_seconds`, `.search.cumulative.ledger_wall_s`. Every other key stays. `tests/h1_mini.rs` pins the file text and the effect.

`.gpu` goes whole: its counters are per chunk, or work counters that a device scorer changes by design.

## Controls

The pinned base binary must reproduce itself before any new binary is judged against it:

1. Determinism: two runs of one seed give equal H.
2. Chunk boundaries: one run with `<chunks>` 1 and one with `<chunks>` 3 give equal H.

If a control fails, no new binary can meet D70 condition 3.

Control 1 gives equal H with `strip.jq` (S). Control 2 gives equal values on every line of H except `run_stripped`: `.search.books` and `.search.fit_accept` count per process, so a chunked run differs from a straight run there. `run_stripped_chunked` is equal. Compare runs of different chunking with `run_stripped_chunked`.

`run_stripped` and `run_stripped_chunked` depend on the store directory: the stripped `run.json` keeps `.target_spec.path` and `.lockbox.declared_in[0].file`. Runs compared by these two lines must use the same store directory; the other seven lines do not depend on it. The strip lists stay the frozen D73 lists.

`strip-ticket.jq` is the shorter list proposed in the issue (`.timing`, `.elapsed_seconds`, `.gpu`, `.dispatch.route_ledger`, `.checkpoint.resumed_from`, `.checkpoint.saved_to` and every key ending in `_s`, `_seconds` or `_per_s`). It leaves `.baselines.knn.seconds` and `.search.cumulative.device_hours` unstable in control 1, so it is not used for H.
