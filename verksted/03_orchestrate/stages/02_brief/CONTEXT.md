# 02_brief: one complete brief per ticket

One job: write the brief of each ticket from the template.

## Inputs

**This run:**
- `../01_wave_plan/output/wave.md`
- `../01_wave_plan/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_templates/brief.md`
- `../../../_shared/brief.md`
- `../../CONTEXT.md`

Do NOT load: other tickets' briefs, the agents' reports.

## Process

1. If an input is missing or empty, stop and say which.
2. For each row of `wave.md` start `output/brief-<N>.md` as a copy of `_shared/brief.md`, then append `_templates/brief.md` with every placeholder filled.
3. Line 1 of every brief is the no-resume line. The ticket section names the worktree, owned files, forbidden files, lock command, git rules and final-report format.
4. A research brief names the model policy and says "no code, no branch, no commit". An implementation brief quotes the decision comment URL.

## Outputs

- `brief-<N>.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
for f in "$V"/03_orchestrate/stages/02_brief/output/brief-*.md; do head -1 "$f" | grep -qxF 'You are not resumed automatically; block in your own turn until everything is done.' || exit 1; for k in 'Worktree:' 'Owned files:' 'Forbidden files:' 'Final report:'; do grep -q "^$k" "$f" || exit 1; done; ! grep -q '<' "$f" || exit 1; done; ls "$V"/03_orchestrate/stages/02_brief/output/brief-*.md > /dev/null
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
