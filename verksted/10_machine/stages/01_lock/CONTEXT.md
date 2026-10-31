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

1. The lock script `one-full-suite.sh` serialises cargo, sets autogroup 19, `nice -n 19` and `ionice -c 3`. Every cargo build, cargo test and `rcsr-bench` goes through it. Look before starting: no other cargo or rustc process.
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
