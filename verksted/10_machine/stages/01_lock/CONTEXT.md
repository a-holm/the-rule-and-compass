# 01_lock: everything through the lock

One job: show that heavy processes run at the lowest priority behind the lock.

## Inputs

**This run:**
- `ps -eo pid,ni,comm,args`

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: stopping or reniceing processes.

## Process

1. The lock script `one-full-suite.sh` serialises cargo, sets autogroup 19 and runs the command under `chrt -i 0` (SCHED_IDLE), `nice -n 19` and `ionice -c 3`. While `desktop-busy.sh` exits 0 (GPU video decoder or encoder busy, or the file `/mnt/lager/tmp/opencode/STREAMING` exists) it also pins the command to CPUs 2,3,6,7 and sets `RUST_TEST_THREADS=4`. `build-slot.sh` does the same and allows 1 slot instead of 3. Every cargo build, cargo test and `rcsr-bench` goes through it. Look before starting: no other cargo or rustc process.
2. Write `lock.md`: running cargo, rustc and rcsr-bench processes with their nice values.
3. A direct `cargo` call is the loophole; report it, do not copy it.

## Outputs

- `lock.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
ps -eo ni=,comm= | awk '$2=="rustc" && $1!=19 {bad=1} END{exit bad}'
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
