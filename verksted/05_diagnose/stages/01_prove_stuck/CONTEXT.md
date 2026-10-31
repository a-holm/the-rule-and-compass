# 01_prove_stuck: stuck is a number, not a feeling

One job: measure whether the process is stuck or only slow.

## Inputs

**This run:**
- `/proc/<pid>/io`, `/proc/<pid>/task/*/stat`, `nvidia-smi pmon -c 1 -s u` of the suspect process

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: fixes, theories about the cause.

## Process

1. Sample twice, 10 seconds apart: `/proc/<pid>/io` counters, CPU percent, thread states and `wchan`, SM percent from `nvidia-smi pmon`, newest file mtime of the run's outputs.
2. Write `stuck.md` with the lines `io_delta: <n>`, `cpu_pct: <n>`, `threads: <states>`, `sm_pct: <n>` and `verdict: stuck` or `verdict: slow`.
3. A slow run is reported and left alone. Kill only a process you started; never `pkill` or `killall`.

## Outputs

- `stuck.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
s="$V/05_diagnose/stages/01_prove_stuck/output/stuck.md"; for k in io_delta cpu_pct threads sm_pct; do grep -q "^$k: " "$s" || exit 1; done; grep -qx 'verdict: stuck' "$s"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
