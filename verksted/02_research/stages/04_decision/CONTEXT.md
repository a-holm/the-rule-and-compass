# 04_decision: binding design for the implementer

One job: write the decision: scope split, spec and measurable acceptance criteria.

## Inputs

**This run:**
- `../03_premise/output/premise.md`
- `../03_premise/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_templates/research-decision.md`
- `../../CONTEXT.md`

Do NOT load: new research, code, the issue comment.

## Process

1. If an input is missing or empty, stop and say which.
2. Copy `_templates/research-decision.md` to `output/decision.md`.
3. Split the scope by issue number. Out-of-scope items name the issue where they land.
4. Spec: files, functions, flags and tests by name. Every acceptance criterion carries a command or a number.

## Outputs

- `decision.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
d="$V/02_research/stages/04_decision/output/decision.md"; for h in 'Scope split' Spec 'Acceptance criteria'; do grep -q "^## $h" "$d" || exit 1; done; test "$(grep -c '^- \[ \]' "$d")" -ge 1 && test "$(grep '^- \[ \]' "$d" | grep -vcE '`|[0-9]')" = 0
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
