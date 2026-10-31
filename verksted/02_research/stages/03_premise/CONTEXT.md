# 03_premise: measure before building

One job: measure each premise with a small Rust test or an existing log.

## Inputs

**This run:**
- `../01_question/output/question.md`
- `../02_method/output/method.md`
- `../02_method/output/check.md` (must be `pass`)

**Reference (every run):**
- `../../../_shared/rules.md`
- `../../../_shared/commands.md`
- `../../CONTEXT.md`

Do NOT load: the full engine change, the decision text.

## Process

1. If an input is missing or empty, stop and say which.
2. For every `P<n>` in `question.md`, run the cheapest measurement: an existing test, a log, `/proc` numbers, or a single-file Rust program built with `nice -n 19 rustc -O` outside the repo. No Python.
3. Write `premise.md`: a table `| P1 | <command or log> | <result> | holds |`. The last column is `holds` or `fails`.
4. A failed premise shrinks or removes the scope. Say so in one line below the table.

## Outputs

- `premise.md` -> `output/`
- `check.md` -> `output/`

## Check

```sh
q="$V/02_research/stages/01_question/output/question.md"; p="$V/02_research/stages/03_premise/output/premise.md"; for id in $(grep -o '^- P[0-9]*' "$q" | cut -c3-); do grep -qE "^\| $id \|.*\| (holds|fails) \|$" "$p" || exit 1; done
```

Exit 0 writes `pass`. Anything else writes `fail: <what was measured and what was wrong>` to `output/check.md`; `_tools/check.sh` does this. A red result is never rewritten as green.
