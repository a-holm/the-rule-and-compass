# 02_issue_body: freeze the issue into a run card

One job: turn the issue into a run card with one proof line per acceptance criterion.

## Inputs

**This run:**
- `../01_worktree/output/worktree.md`
- `../01_worktree/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_templates/run.md`
- `../../CONTEXT.md`

Do NOT load: the code, other issues, the PR template.

## Process

1. If an input is missing or empty, stop and say which.
2. Copy `_templates/run.md` to `output/run.md`.
3. Read the issue (`gh issue view "$N" --json title,body,labels`). Fill Goal and Scope from it. Out-of-scope items keep their `#N` target.
4. Write one `- [ ]` Proof line per acceptance criterion: a command and its expected exit code or number. A criterion without a command is a question for the dispatcher: stop.
5. Fill the Stop rule: unmet criteria are listed in the PR and the issue stays open.

## Outputs

- `run.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
ac=$(gh issue view "$N" --json body -q .body | sed -n '/^## Acceptance criteria/,/^## /p' | grep -c '^- \[[ x]\]'); proof=$(sed -n '/^## Proof/,/^## /p' "$V/01_issue/stages/02_issue_body/output/run.md" | grep -c '^- \[ \]'); test "$ac" -ge 1 && test "$ac" = "$proof"
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
