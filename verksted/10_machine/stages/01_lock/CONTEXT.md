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

1. Two scripts serialise heavy work. The test lock `TEST_LOCK` runs one command at a time. The build slot `BUILD_SLOT` allows up to `BUILD_SLOTS` builds at once (`BUILD_SLOTS_BUSY` while the desktop is busy). Both set autogroup 19, run the command under `chrt -i 0` (SCHED_IDLE), `nice -n 19` and `ionice -c 3`, pin it to `PIN_CORES_BUILD` (`PIN_CORES_BUSY` while the desktop is busy) and wait while the `PAUSE` file exists. The values are in `machine.md`; the keys are listed in `../../../_shared/local.md`.
2. Builds (`cargo build`, `cargo test --no-run`) use `BUILD_SLOT`. Tests, `rcsr-bench`, sanitizer and GPU runs use `TEST_LOCK`. CPU-only tests may use the slot. `../../../_tools/rc-run.sh` and `../../../_tools/sanitize.sh` pick the right one. Resources are checked before work and the lanes stay closed: [rules 64 and 65](../../../_shared/rules.md). GPU failure-path tests run under `$SANITIZER` first. Before a GPU step record the time; after it `journalctl -k --since <time> | grep -c "NVRM: Xid"` must be 0, else stop all GPU work and report (rule 31). Look before starting: no other cargo or rustc process.
3. Check the live scripts against the canonical copies: `../../../_tools/machine-sync.sh --check` prints per file `same`, `whitespace only` or a diff. `--install` is the orchestrator's step, never an agent's.
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
