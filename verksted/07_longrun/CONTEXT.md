# 07_longrun: from merged work to an audited multi-hour run

One job: start one long search run only when its preconditions hold, keep it observable and resumable, and report it with baselines and honest fields.

Evidence: the 10-hour run of 2026-10-20 to 2026-10-22; eight `rcsr` processes where one was meant; a run started before the required issues were merged; an audit that found the "gain" was the label's own normalisation recovered by a zero-parameter formula and 95.4 % of wall time in a serial CPU fallback (`../../docs/run-integrity.md`). The launcher is `rcsr-launch`.

## Inputs

**This run:**
- the label `LABEL`, the data store, the target and the backend (`gpu` by default)

**Reference (every run):**
- `../_shared/commands.md`
- `../_shared/rules.md`
- `../_shared/output-format.md`
- `../../docs/run-integrity.md`
- `../../docs/dev-log.md`
- `../../docs/convergence.md`
- `../../scripts/run-long.sh`

Do NOT load: the ticket and data pipelines' stages, earlier runs' `output/`, the rest of `../_shared/`.

## Process

1. `stages/01_preconditions/`: no open work left, binary from the new `main`, nothing old running.
2. `stages/02_launch/`: launch through `rcsr-launch` with an explicit backend and a dry run first.
3. `stages/03_monitor/`: watch the development log, not an external script.
4. `stages/04_resume/`: resume on the same backend after a stop.
5. `stages/05_audit/`: `rcsr dev-report` and independent auditors with a verdict per axis.
6. `stages/06_report/`: baselines beside the best model, honest result fields.

## Outputs

- `stages/06_report/output/report.md`: result, baselines, audit verdicts, deviations
- `stages/06_report/output/check.md`: `pass` or `fail: <reason>`

## Check

```sh
for f in "$V"/07_longrun/stages/*/output/check.md; do test "$(head -1 "$f")" = pass || exit 1; done
```
