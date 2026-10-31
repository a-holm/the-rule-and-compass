# 06_medium_run: a medium store, not the huge test

One job: run the real path on a medium store in under 10 minutes.

## Inputs

**This run:**
- `../05_fix_bounded/output/fixed.md`
- `../05_fix_bounded/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: the 39.8 GB test, a second lock holder.

## Process

1. If an input is missing or empty, stop and say which.
2. Build a store of at least 200 groups and 100 feature columns. Run `rcsr enrich --gpu on` on it through the lock, each step with its own `timeout`, wrapped in `/usr/bin/time -f 'wall=%es maxrss=%MkB'`.
3. Write `medium.md`: `wall=<s>`, `exit: <code>`, `gpu_used: yes|no`, the commit.

## Outputs

- `medium.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
m="$V/05_diagnose/stages/06_medium_run/output/medium.md"; grep -qx 'exit: 0' "$m" && grep -qx 'gpu_used: yes' "$m" && awk -F'[=s]' '/^wall=/{exit !($2+0 < 600)}' "$m"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
