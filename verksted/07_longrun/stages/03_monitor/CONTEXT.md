# 03_monitor: alive and writing

One job: show that the run is alive and writing its log and checkpoints.

## Inputs

**This run:**
- `../02_launch/output/launch.md`
- `../02_launch/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../../docs/dev-log.md`
- `../../CONTEXT.md`

Do NOT load: extra monitoring scripts, polling loops.

## Process

1. If an input is missing or empty, stop and say which.
2. `tail -n 5 runs/<label>.dev.ndjson` shows a summary event every 10 to 25 generations. The checkpoint is rewritten every `--save-every` generations.
3. Check with one call per look. No wait loops. A stuck run is `../../05_diagnose/CONTEXT.md`.
4. Write `monitor.md`: time, generation, error, front size, GPU duty.

## Outputs

- `monitor.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
l="$RUN/runs/$LABEL.dev.ndjson"; test -n "$(find "$l" -mmin -10 2> /dev/null)" && kill -0 "$(sed -n 's/^pid: //p' "$V/07_longrun/stages/02_launch/output/launch.md")"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
