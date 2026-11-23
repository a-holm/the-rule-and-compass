# 09_lessons: record what this dataset taught

One job: write each new lesson once, in the home its scope decides.

## Inputs

**This run:**
- `../08_verify_manifest/output/report.md`
- `../08_verify_manifest/output/check.md` (must be `pass`)
- every `deviations` line of this run's outputs

**Reference (every run):**
- `../../../_shared/local.md`
- `../../../_templates/lesson.md`
- `../../CONTEXT.md`

Do NOT load: locked-window results, raw data files, other datasets' lessons.

## Process

1. If an input is missing or empty, stop and say which.
2. List candidate lessons: every deviation, every stop, every engine gap, every guess that proved wrong.
3. Scope each one. True for any dataset of the type: propose it for the tracked `references/lessons.md` of the type stage in `output/lessons-general.md` (an issue and a pull request carry it in). True for this dataset or this machine only: append it to `lessons/<type>.md` of the local area.
4. Each lesson is one present-tense rule that ends with `Source: <date>, <artefact kind>.`
5. Refuse content: no value, date or symbol from inside a locked range before the final seal and reads are posted; no raw row; no secret; no personal data; no machine path in a general lesson.
6. An engine gap becomes an issue, not a lesson.

## Outputs

- `lessons-general.md` -> `output/` (may say `none`)
- `lessons-local.md` -> `output/` (the lines appended to the local area; may say `none`)
- `check.md` -> `output/`

## Check

```sh
o="$V/06_data/stages/09_lessons/output"; test -s "$o/lessons-general.md" && test -s "$o/lessons-local.md" && ! grep -qE '/mnt/|/home/' "$o/lessons-general.md" && { grep -qx none "$o/lessons-general.md" || grep -q 'Source: ' "$o/lessons-general.md"; }
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
