# 01_lock: everything through the lock

One job: show that heavy processes run at the lowest priority behind the lock.

## Inputs

**This run:**
- `ps -eo pid,ni,comm,args`

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`
- `../../../_tools/machine-sync.sh`
- `../../../_tools/rc-run.sh`
- `../../../_tools/sanitize.sh`

Do NOT load: stopping or reniceing processes.

## Process

1. Two scripts serialise heavy work. The test lock `one-full-suite.sh` runs one command at a time. The build slot `build-slot.sh` allows up to 3 builds at once. Both set autogroup 19 and run the command under `chrt -i 0` (SCHED_IDLE), `nice -n 19` and `ionice -c 3`, and wait while `/mnt/lager/tmp/opencode/PAUSE` exists. While `desktop-busy.sh` exits 0 (GPU video decoder or encoder busy, or the file `/mnt/lager/tmp/opencode/STREAMING` exists) they also pin the command to CPUs 2,3,6,7 and set `RUST_TEST_THREADS=4`.
2. Builds (`cargo build`, `cargo test --no-run`) use the build slot. Tests, `rcsr-bench`, sanitizer and GPU runs use the test lock. CPU-only tests may use the slot. `../../../_tools/rc-run.sh` and `../../../_tools/sanitize.sh` pick the right one. GPU failure-path tests run under `/usr/local/cuda-13.0/bin/compute-sanitizer` first. The Xid count (`journalctl -k | grep -c "NVRM: Xid"`) is checked before and after GPU steps. Look before starting: no other cargo or rustc process.
3. Check the live scripts against the canonical copies: `../../../_tools/machine-sync.sh --check` prints per file `same`, `whitespace only` or a diff. Until the orchestrator installs the fixed copies after the pilot, `build-slot.sh` reports `whitespace only`. `--install` is the orchestrator's step after the pilot, never an agent's.
4. Write `lock.md`: running cargo, rustc and rcsr-bench processes with their nice values, and the `machine-sync.sh --check` output.
5. A direct `cargo` call is the loophole; report it, do not copy it.

## Outputs

- `lock.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
ps -eo ni=,comm= | awk '$2=="rustc" && $1!=19 {bad=1} END{exit bad}'
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
