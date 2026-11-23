# 01_intake: the dataset card before any download

One job: write the dataset card: question, source, licence, earliest bar, lock window and what must not be read.

## Inputs

**This run:**
- `DS`, the dataset id; the source description from the dispatcher
- the decisions register `decisions.md` of the local area, when it exists

**Reference (every run):**
- `../../../_templates/dataset-card.md`
- `../../references/time-rules.md`
- `../../references/data-types.md`
- `../../CONTEXT.md`

Do NOT load: any data file, any type stage's references, earlier datasets' cards.

## Process

1. Copy `dataset-card.md` to `output/card.md`.
2. Write the question first: target idea, decision time, horizon, what a useful result looks like.
3. Source and licence: name the vendor, the terms and what they allow. Research-only data never enters git; record the attribution text the licence asks for.
4. History: the first bar the source offers. The store starts there. A later start needs `approved_start` with the approver and date.
5. Lock: the declared locked test `START,END`, or `none` with the reason. State where development ends.
6. Not to be read: files and directories inside the lock, sealed directories.
7. Guess the type from the source description (the profile confirms it).

## Outputs

- `card.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
c="$V/06_data/stages/01_intake/output/card.md"; for k in id question source licence first_bar locked_test dev_end type_guess not_to_be_read; do grep -qE "^$k: .+" "$c" || exit 1; done; ! grep -qE '^(licence|first_bar|locked_test): *<' "$c"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
