# 04_resume: same backend, same build

One job: resume a stopped run from its checkpoint.

## Inputs

**This run:**
- `../03_monitor/output/monitor.md`
- `../03_monitor/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../../docs/run-integrity.md`
- `../../CONTEXT.md`

Do NOT load: changing backend, data or build without a reason.

## Process

1. If an input is missing or empty, stop and say which.
2. A `time_budget` stop is not terminal. Resume with `--resume <checkpoint.rcsr>` on the same backend. The run refuses a different binary, commit, device, driver, thread count or target spec.
3. `--allow-backend-change` and `--allow-build-change` only on purpose: write `backend-change: intended <reason>` in `resume.md`.
4. Write `resume.md`: the command, the checkpoint, the generation resumed from.

## Outputs

- `resume.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
r="$V/07_longrun/stages/04_resume/output/resume.md"; "$RUN/my-version/target/release/rcsr" verify-manifest "$RUN/runs/$LABEL.manifest.json" && { ! grep -q -- '--allow-backend-change' "$r" || grep -q '^backend-change: intended ' "$r"; }
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
