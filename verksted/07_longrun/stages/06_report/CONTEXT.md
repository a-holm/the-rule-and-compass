# 06_report: baselines beside the result

One job: report the result with baselines and the honest result fields.

## Inputs

**This run:**
- `../05_audit/output/SYNTHESIS.md`
- `../05_audit/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/output-format.md`
- `../../CONTEXT.md`

Do NOT load: claims without a baseline.

## Process

1. If an input is missing or empty, stop and say which.
2. Report the best model next to the constant, the zero-parameter and the affine baselines. A gain that a baseline reproduces is not a gain.
3. Quote `stop_reason`, `converged_early` and the units of every figure. A time-budget stop is not convergence.
4. Write `report.md` with the lines `best_model:`, `baseline_constant:`, `baseline_zero_parameter:`, `baseline_affine:`.

## Outputs

- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
r="$V/07_longrun/stages/06_report/output/report.md"; for k in best_model baseline_constant baseline_zero_parameter baseline_affine; do grep -q "^$k: " "$r" || exit 1; done; grep -q '"stop_reason"' "$RUN/runs/$LABEL.json" && grep -q '"converged_early"' "$RUN/runs/$LABEL.json"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
