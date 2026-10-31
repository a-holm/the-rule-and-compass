# 05_audit: independent verdicts per axis

One job: audit the finished run with independent auditors.

## Inputs

**This run:**
- `../04_resume/output/resume.md`
- `../04_resume/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/failure-modes.md`
- `../../CONTEXT.md`

Do NOT load: fixing findings inside this run.

## Process

1. If an input is missing or empty, stop and say which.
2. `rcsr dev-report runs/<label>.dev.ndjson`. Dispatch independent auditors (Opus with tractatus_thinking is the research exception, no code) with the dev report and the result JSON.
3. Write `SYNTHESIS.md` with one `## <axis>` section per axis and a `verdict: pass|fail: <reason>` line in each. Required axes: `Leakage`, `GPU use`, `Baselines`.
4. Label every claim measured, derived or inferred. An inferred claim is not a finding.

## Outputs

- `SYNTHESIS.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
s="$V/07_longrun/stages/05_audit/output/SYNTHESIS.md"; for h in Leakage 'GPU use' Baselines; do grep -q "^## $h" "$s" || exit 1; done; test "$(grep -c '^## ' "$s")" = "$(grep -c '^verdict: ' "$s")"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
