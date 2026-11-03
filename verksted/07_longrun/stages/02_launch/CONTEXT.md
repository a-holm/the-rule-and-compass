# 02_launch: explicit backend, tracked launcher

One job: launch the run through the tracked launcher.

## Inputs

**This run:**
- `../01_preconditions/output/precond.md`
- `../01_preconditions/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../../my-version/src/bin/rcsr_launch.rs`
- `../../CONTEXT.md`
- `../../../_tools/runlib.sh`

Do NOT load: ad-hoc shell launchers, stderr to /dev/null.

## Process

1. If an input is missing or empty, stop and say which.
2. Dry run: `rcsr-launch --dry-run --data ... --backend gpu --window ...`. Save the output as `dry.txt`. It must show the backend, `--run-manifest`, `--json-out`, `--progress` and `--dev-log`.
3. Put the launch in a new runbook that sources `../../../_tools/runlib.sh`: steps with done-files, protected steps behind `SEALED=1`, `rl_gpu` holds (test lock, Xid count checked before and after). Each hold stays under 10 minutes. The pilot v2 runbook in the git-excluded work folder (`arbeidsmappe/crypto-rcsr/pilot/v2/runbook/`) is the reference example. The runbook runs the pinned `<run>/bin/rcsr`. Stdout and stderr go to a per-window log file. The seed is derived from (label, seed) with SHA-256.
4. Write `launch.md`: `windows: <K>`, `pid: <pid>`, the command, the log paths.

## Outputs

- `launch.md` -> `output/`
- `dry.txt` -> `output/`
- `check.md` -> `output/`

## Check

```sh
d="$V/07_longrun/stages/02_launch/output/dry.txt"; for f in --run-manifest --json-out --progress --dev-log; do grep -q -- "$f" "$d" || exit 1; done; grep -qE -- '--(gpu|cpu)' "$d" && test "$(ps -eo comm= | grep -cx rcsr)" = "$(sed -n 's/^windows: //p' "$V/07_longrun/stages/02_launch/output/launch.md")"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
