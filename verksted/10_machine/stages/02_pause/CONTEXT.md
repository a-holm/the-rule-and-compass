# 02_pause: new runs wait

One job: create the pause file so no new run queues.

## Inputs

**This run:**
- `../01_lock/output/lock.md`
- `../01_lock/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/rules.md`
- `../../CONTEXT.md`

Do NOT load: killing running jobs.

## Process

1. If an input is missing or empty, stop and say which.
2. `touch /mnt/lager/tmp/opencode/PAUSE`. The lock script waits while the file exists. Running jobs finish; nothing new starts.
3. Write `pause.md`: the time and the running lock holder.

## Outputs

- `pause.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
test -e /mnt/lager/tmp/opencode/PAUSE
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
