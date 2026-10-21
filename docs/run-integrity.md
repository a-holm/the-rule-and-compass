# Run integrity: manifests, atomic output and reproducible launchers

A long run is evidence only if you can name the binary, the data and the
configuration that produced it. Issue #140 adds four guarantees:

1. a checkpoint records its **backend** (`cpu`/`gpu`) and a canonical
   **config hash**, and `--resume` refuses a silent backend change;
2. every checkpoint/metadata write is **atomic** (temp file, `fsync`, rename)
   and the previous checkpoint generation is kept as `<path>.prev`;
3. `--run-manifest <path>` records a verifiable JSON manifest at the start and
   updates it atomically at the end;
4. the final JSON result can be published atomically with `--json-out <path>`,
   and progress is never mixed into the final document.

See [`checkpoint.md`](checkpoint.md) for the checkpoint format and the resume
rules, and [`seeds.md`](seeds.md) for the deterministic seed contract.

## Run manifest

```sh
rcsr --data data.csv --target y --vars x --seed 7 --gpu \
     --generations 500 --run-manifest run.manifest.json \
     --json-out run.result.json --progress
```

The manifest is written before the search starts (so a killed process still
leaves evidence) and rewritten atomically at the end with `end_utc` and phase
timings. It contains:

| field | meaning |
|---|---|
| `tool`, `version` | engine name and crate version |
| `start_utc`, `end_utc`, `completed` | UTC timestamps; `end_utc` is `null` until the run finishes |
| `commit`, `dirty` | `git rev-parse HEAD` and whether the work tree had uncommitted files |
| `binary.path`, `binary.sha256` | the actual `current_exe` and its SHA-256 (two CUDA-linked binaries at the same commit can differ) |
| `data.path`, `data.sha256` | raw data file and its SHA-256 (`null` for stdin) |
| `data.train`, `data.test` | prepared-sample row count and FNV-1a fingerprint, exactly as the checkpoint records them |
| `config.hash`, `config.seed`, `config.canonical` | the canonical configuration text, its FNV-1a hash and the search seed |
| `backend`, `device` | `cpu`/`gpu`, device name, driver version and CUDA version |
| `resume.from`, `resume.backend_change` | checkpoint path and an allowed backend change, if any |
| `phases_seconds` | `data_prepare`, `search`, `total` |

`rcsr verify-manifest <path>` re-checks the manifest against the **current**
environment:

* the SHA-256 of the running `rcsr` binary;
* the current git commit;
* the data file named in the manifest (or `--data <file>`) by SHA-256;
* the canonical configuration text by recomputing its hash.

It prints one `PASS`/`FAIL` line per check and exits `0` only if every check
passes, `1` on any mismatch, `2` on a usage error. A stale binary, changed data
or edited configuration therefore fails loudly instead of being mistaken for a
reproduction.

```console
$ rcsr verify-manifest run.manifest.json
PASS binary sha256 9c1f...
PASS commit 4e2a...
PASS data sha256 data.csv 7b31...
PASS config hash 4f0c2a1c74b1e0a9
PASS manifest run.manifest.json
```

## Atomic publication

All engine-owned files use the same pattern: write `<path>.tmp`, `fsync`,
rename over `<path>`; the destination is never a partially written stream.

* `--save` checkpoints retain `<path>.prev`;
* `--run-manifest` is published at start and updated at the end;
* `--json-out` is the final JSON object; without it, `--json` prints one object
  to stdout;
* `--out` (the Pareto TSV) is atomic too.

A killed process leaves the old file (and, for a checkpoint, `.prev`) intact.
The output stream rules are unchanged: stdout carries the machine-readable
result, stderr carries the human log. `--progress` writes generation lines to
stderr, or NDJSON lines to stdout when `--ndjson` is selected; the final result
is never interleaved with progress in the same file.

## Reproducible launcher contract

A launcher should:

* choose `(label, seed)` and derive the numeric seed with a **stable** hash —
  never Python's salted `hash(label)`;
* state the backend explicitly (`--gpu` or `--cpu`), even when CPU is the
  default, and pass the same one on resume;
* record the exact command line and the resolved seed;
* pass `--run-manifest`, `--json-out` and `--progress`;
* keep stderr in a file rather than `/dev/null`;
* resume with `--resume checkpoint.rcsr` and, only if intentional, with
  `--allow-backend-change`.

Two tracked examples implement the contract:

* [`scripts/run-long.sh`](../scripts/run-long.sh) — one uninterrupted or
  resumed run; requires `--gpu`/`--cpu` and writes a manifest, atomic result,
  command record and progress log.
* [`scripts/launch-windows.py`](../scripts/launch-windows.py) — several
  walk-forward windows; derives `SHA-256(label:seed)`, reuses the same backend
  on resume, and writes one manifest/log/result set per window. `--dry-run`
  prints the resolved commands without searching.

The scratch launchers in the git-ignored `arbeidsmappe/crypto-rcsr/pilot/`
directory are **retired**. Their classification and the Rust command that
replaces each one are in
[`python-pipeline-retirement.md`](python-pipeline-retirement.md); one-off probes
are archived under `pilot/archive/` and must not be used. In particular, do not
resume a long run with them (`hash(label)` seeds, `/dev/null` stderr, plain
`--json` redirected to a file without a manifest). Use the tracked launchers
above, or the `rcsr walkforward` command directly. The tracked
`scripts/launch-windows.py` is the one remaining Python helper; it already
follows the contract (SHA-256 seeds, explicit backend, manifest, atomic JSON,
progress log) and is scheduled for a shell/Rust replacement as a separate
follow-up, not as part of the data pipeline.

`my-version/tests/launchers.rs` enforces the contract on every tracked
`scripts/*.sh`/`*.py`: no `DEVNULL`, no Python `hash(`, an explicit backend, and
`--run-manifest`/`--json-out`/`--progress`. It also runs `run-long.sh`
end-to-end and checks the manifest with `rcsr verify-manifest`.

## Acceptance checks

The integration branch runs the tests that pin this behaviour:

* `tests/checkpoint.rs`: backend/config-hash round trip, backend-mismatch
  refusal and override, injected partial write leaving the previous generation
  intact, `.prev` retention;
* `tests/manifest.rs`: manifest pass, stale binary hash, changed data and
  changed config all fail the verifier; `--json-out` is one atomic JSON object
  and never streams to stdout.
