# 03_measure: measure before accusing

One job: measure load, GPU use and the top consumers.

## Inputs

**This run:**
- `../02_pause/output/pause.md`
- `../02_pause/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/failure-modes.md`
- `../../CONTEXT.md`
- `../../../_tools/status.sh`

Do NOT load: stopping anything, guessing the culprit.

## Process

1. If an input is missing or empty, stop and say which.
2. Run `nproc` and `../../../_tools/status.sh --no-fetch --section machine` (load, GPU use and Xid count, lock holder, build slots, `PAUSE` and `STREAMING`). Then `ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu | head -8` for the top consumers.
3. Write `measure.md`: lines `load: <1 min>`, `cores: <n>` and a table of the top consumers `| pid | owner | cpu | command |`.

## Outputs

- `measure.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
m="$V/10_machine/stages/03_measure/output/measure.md"; grep -q '^load: ' "$m" && grep -q '^cores: ' "$m" && test "$(grep -c '^| [0-9]' "$m")" -ge 3
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
