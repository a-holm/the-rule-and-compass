#!/usr/bin/env bash
# Usage: test-issue-lint.sh
# Hermetic test of issue-lint.sh: a filled fixture passes, each broken fixture gives its FAIL line, flags, stdin, stubbed gh.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LINT=$HERE/../issue-lint.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  local name=$1 got=$2 want=$3
  if [ "$got" != "$want" ]; then
    echo "FAIL $name: got '$got' want '$want'"
    fails=$((fails + 1))
  fi
}
# lint <fixture> [args]: sets OUT and RC.
lint() {
  OUT=$(bash "$LINT" "$@" 2>&1)
  RC=$?
}
# expect <name> <rc> <fixture> <pattern> [args]: rc matches and OUT contains the pattern.
expect() {
  local name=$1 rc=$2 fx=$3 pat=$4
  shift 4
  lint "$fx" "$@"
  check "$name-rc" "$RC" "$rc"
  case $OUT in
    *"$pat"*) ;;
    *) echo "FAIL $name: output lacks '$pat': $OUT"; fails=$((fails + 1)) ;;
  esac
}

cat > "$T/good.md" <<'EOF'
| Field | Value |
|---|---|
| Milestone | none (the repo has no milestones) |
| Epic/Area | #666, area:docs |
| Priority | P3 |
| Size | S |
| Estimate | 0.5 d |
| Planned start | 2026-11-05 |
| Planned end | 2026-11-06 |
| Blocked by | #3, #5 |
| Labels | roadmap, area:docs |

## Goal and background

Why.

## Scope

In scope:
- the script

Out of scope:
- other things (#9)
- more things, see #10

## Design proposal (proposal, not mandate)

```
## Not a heading
Closes nothing here
```

## Acceptance criteria

- [ ] exits 0

## Test plan

- the test

## Dependencies

- Blocked by: #3 first, #5 second.
- Blocks: #20.
EOF

# derive <name> <sed-expr>: a broken copy of the good fixture.
derive() { sed -E "$2" "$T/good.md" > "$T/$1.md"; }

lint "$T/good.md" --offline
check good-rc "$RC" 0
check good-out "$OUT" OK

derive no-estimate '/^\| Estimate/d'
expect missing-field 1 "$T/no-estimate.md" "FAIL: field-missing: Estimate" --offline

derive bad-dates 's/^(\| Planned end \| ).*/\12026-11-01 |/'
expect date-order 1 "$T/bad-dates.md" "FAIL: date-order" --offline

derive bad-date-format 's/^(\| Planned start \| ).*/\12026-13-45 |/'
expect date-invalid 1 "$T/bad-date-format.md" "FAIL: date-invalid: Planned start" --offline

derive bad-size 's/^(\| Size \| ).*/\1Huge |/'
expect bad-size 1 "$T/bad-size.md" "FAIL: size-invalid" --offline

derive bad-priority 's/^(\| Priority \| ).*/\1P4 |/'
expect bad-priority 1 "$T/bad-priority.md" "FAIL: priority-invalid" --offline

derive bad-estimate 's/^(\| Estimate \| ).*/\1soon |/'
expect bad-estimate 1 "$T/bad-estimate.md" "FAIL: estimate-invalid" --offline

derive bad-blocked 's/^(\| Blocked by \| ).*/\1later |/'
expect bad-blocked 1 "$T/bad-blocked.md" "FAIL: blocked-by-invalid" --offline

derive none-blocked 's/^(\| Blocked by \| ).*/\1none |/'
expect none-blocked 0 "$T/none-blocked.md" OK --offline

derive no-test-plan '/^## Test plan/d'
expect missing-heading 1 "$T/no-test-plan.md" "FAIL: heading-missing: ## Test plan" --offline

# swap Scope and Goal headings
derive swapped 's/^## Goal and background/## TMPX/; s/^## Scope$/## Goal and background/; s/^## TMPX/## Scope/'
expect heading-order 1 "$T/swapped.md" "FAIL: heading-order" --offline

derive out-no-number 's/^- other things \(#9\)/- other things/'
expect out-of-scope 1 "$T/out-no-number.md" "FAIL: scope-out-without-issue: - other things" --offline

derive no-out '/^Out of scope:/d'
expect no-out 1 "$T/no-out.md" "FAIL: scope-out-missing" --offline

derive closes 's/^Why\./Why. Closes #12/'
expect closes 1 "$T/closes.md" "FAIL: closing-keyword" --offline

derive fixes-lower 's/^Why\./why fixes #12/'
expect fixes-lower 1 "$T/fixes-lower.md" "FAIL: closing-keyword" --offline

derive quoted-closes 's/^Why\./Why. The fixture has `Closes #12` in it./'
expect quoted-inline 0 "$T/quoted-closes.md" OK --offline
derive quoted-then-bare 's/^Why\./Why. `Closes #12` is quoted but Fixes #13 is bare./'
expect quoted-then-bare 1 "$T/quoted-then-bare.md" "FAIL: closing-keyword" --offline
derive fenced-closes 's/^Closes nothing here/Closes #12/'
expect quoted-fenced 0 "$T/fenced-closes.md" OK --offline

dash=$(printf '\xe2\x80\x94')
derive dash "s/^Why\./Why $dash really./"
expect em-dash 1 "$T/dash.md" "FAIL: em-dash" --offline

derive no-box 's/^- \[ \] exits 0/- exits 0/'
expect no-checkbox 1 "$T/no-box.md" "FAIL: acceptance-missing" --offline

derive no-dep-blocked '/^- Blocked by:/d'
expect no-dep-blocked 1 "$T/no-dep-blocked.md" "FAIL: dependencies-blocked-by-missing" --offline

printf 'Just prose.\n' > "$T/prose.md"
expect no-table 1 "$T/prose.md" "FAIL: table-missing" --offline

# a heading inside a code fence does not count; a fenced Closes is still found
derive fence-heading 's/^## Test plan/## Test plans/'
expect heading-exact 1 "$T/fence-heading.md" "FAIL: heading-missing: ## Test plan" --offline

# --number and --allow-blocker
expect number-ok 0 "$T/good.md" OK --offline --number 6
expect number-bad 1 "$T/good.md" "FAIL: blocker-order: blocker #5 is not lower than #5" --offline --number 5
lint "$T/good.md" --offline --number 3
check number-two-rc "$RC" 1
case $OUT in
  *"blocker #3 "*"blocker #5 "*) ;;
  *) echo "FAIL number-two: $OUT"; fails=$((fails + 1)) ;;
esac
expect allow-one 1 "$T/good.md" "blocker #3 " --offline --number 3 --allow-blocker 5
expect allow-all 0 "$T/good.md" OK --offline --number 3 --allow-blocker 3,5
expect allow-hash 0 "$T/good.md" OK --offline --number 3 --allow-blocker '#3' --allow-blocker 5

# stdin and CRLF
lint - --offline < "$T/good.md"
check stdin-rc "$RC" 0
sed 's/$/\r/' "$T/good.md" > "$T/crlf.md"
expect crlf 0 "$T/crlf.md" OK --offline
lint - --offline < "$T/no-estimate.md"
check stdin-bad-rc "$RC" 1

# usage errors
lint
check no-args "$RC" 2
lint "$T/nope.md" --offline
check unreadable "$RC" 2
lint "$T/good.md" --bogus
check bad-flag "$RC" 2
lint "$T/good.md" --number x
check bad-number "$RC" 2
lint "$T/good.md" --number
check missing-number "$RC" 2
lint -h
check help-rc "$RC" 0
case $OUT in
  "Usage: issue-lint.sh"*) ;;
  *) echo "FAIL help: $OUT"; fails=$((fails + 1)) ;;
esac

# online checks with a stubbed gh
cat > "$T/gh" <<'EOF'
#!/usr/bin/env bash
case "$1 $2" in
  "label list") printf 'roadmap\narea:docs\nenhancement\n' ;;
  "issue view") case "$3" in 3|5) echo '{}' ;; *) exit 1 ;; esac ;;
  *) exit 1 ;;
esac
EOF
chmod +x "$T/gh"
export RCSR_GH=$T/gh
lint "$T/good.md"
check online-ok-rc "$RC" 0
derive bad-label 's/^(\| Labels \| ).*/\1roadmap, area:nowhere |/'
expect label-unknown 1 "$T/bad-label.md" "FAIL: label-unknown: area:nowhere"
derive bad-blocker 's/#5/#7/g'
expect blocker-unknown 1 "$T/bad-blocker.md" "FAIL: blocker-unknown: #7"
expect offline-skips-gh 0 "$T/bad-label.md" OK --offline
export RCSR_GH=$T/missing-gh
expect gh-unavailable 1 "$T/good.md" "FAIL: gh-unavailable"

if [ "$fails" -ne 0 ]; then
  echo "test-issue-lint: $fails failed"
  exit 1
fi
echo "test-issue-lint: ok"
