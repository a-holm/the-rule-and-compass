# 07_classify_deltas: bug or tier, never a drive-by

One job: give every parity delta a class and an owner.

## Inputs

**This run:**
- `../06_medium_run/output/medium.md`
- `../06_medium_run/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/output-format.md`
- `../../CONTEXT.md`

Do NOT load: fixing a delta inside this branch.

## Process

1. If an input is missing or empty, stop and say which.
2. For each delta between the CPU reference and the GPU result decide: production bug (the device used wrong or future rows; own issue, priority by blast radius) or numeric sensitivity (declare a parity tier; own issue).
3. Write `deltas.md`: rows `| <column or test> | <delta> | bug | #N |` or `| ... | tier | #N |`. Create the issues with `gh issue create` first.
4. Reported values stay exact CPU f64. Write `report.md`.

## Outputs

- `deltas.md` -> `output/`
- `report.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
d="$V/05_diagnose/stages/07_classify_deltas/output/deltas.md"; test "$(grep -c '^| .* | \(bug\|tier\) | #[0-9]* |$' "$d")" -ge 1 && for n in $(grep -o '| #[0-9]* |$' "$d" | tr -d '| #'); do gh issue view "$n" --json number > /dev/null || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
