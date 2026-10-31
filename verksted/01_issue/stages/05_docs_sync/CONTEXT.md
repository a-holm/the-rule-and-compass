# 05_docs_sync: docs match the behaviour

One job: bring help, schema, README and docs in line with the change.

## Inputs

**This run:**
- `../04_implement/output/implementation.md`
- `../04_implement/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/output-format.md`
- `../../CONTEXT.md`

Do NOT load: code changes, tests, other tickets.

## Process

1. If an input is missing or empty, stop and say which.
2. If a flag, output field or behaviour changed: update `--help`, the `rcsr schema` text, `README.md` and the file in `docs/` that owns it. Docs describe the present state only.
3. If nothing user-visible changed and the docs already state the behaviour (a fix that makes the code match them), write `docs already correct: <path of that doc>` as line 1 of `docs.md`.
4. List the files touched in `docs.md`.

## Outputs

- `docs.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
d=$(git -C "$W" diff --name-only "$BASE"...HEAD); echo "$d" | grep -qE '^(README\.md|docs/|verksted/)' || { l=$(head -1 "$V/01_issue/stages/05_docs_sync/output/docs.md"); case "$l" in "docs already correct: "*) test -e "$W/${l#docs already correct: }" ;; *) false ;; esac; }
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
