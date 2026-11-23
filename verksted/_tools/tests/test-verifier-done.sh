#!/usr/bin/env bash
# Usage: test-verifier-done.sh
# Hermetic test of verifier-done.sh and of the verifier routine text: merged and hold forms, missing, empty and differing files,
# a detached worktree as cwd, argument errors, the grep guards on CONTEXT.md and the two templates. gh is a stub that serves fixtures
# and fails like gh does when it must read the current branch.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/machine.sh"
S=$HERE/../verifier-done.sh
V=$HERE/../..
ROUTINE=$V/08_close/stages/01_verifier/CONTEXT.md
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
mkdir -p "$T/bin" "$T/out"
cat > "$T/bin/gh" <<'GH'
#!/usr/bin/env bash
echo "$*" >> "$FAKE_DIR/gh-calls"
case "$*" in
  "pr view --json"*|"issue view --json"*) echo "could not determine current branch: failed to run git: not on any branch" >&2; exit 1 ;;
esac
case "$*" in
  *"-R "*) ;;
  *) echo "fake gh: no -R in $*" >&2; exit 1 ;;
esac
case "$*" in
  "pr view "*) echo "${FAKE_PR_STATE:-MERGED}" ;;
  *"--json body"*) cat "$FAKE_BODY" ;;
  *"--json comments"*) cat "$FAKE_COMMENT" ;;
  *) echo "fake gh: unexpected $*" >&2; exit 1 ;;
esac
GH
chmod +x "$T/bin/gh"
export FAKE_DIR=$T PATH="$T/bin:$PATH" RCSR_REPO=o/r FAKE_BODY=$T/body.md FAKE_COMMENT=$T/out/verification.md
printf '## Goal\n\n- [x] goal box\n\n## Acceptance criteria\n\n- [x] first\n- [x] second\n- [ ] third\n\n## Test plan\n\n- [ ] plan\n' > "$T/body.md"
merged() {
  cat > "$T/out/verification.md" <<'MD'
Merged into integration/gpu-hardening as abc1234 (PR #9).

# Verification: #5 title

| Row | Criterion | Evidence | Status |
|---|---|---|---|
| AC1 | first | log a | met |
| AC2 | second | test b | met |
| AC3 | third | needs the full gate | not run (#7) |

Verdict: CLOSE
MD
}
hold() {
  cat > "$T/out/verification.md" <<'MD'
Hold: PR #9 kept open, the lint set is red.

# Verification: #5 title

| Row | Criterion | Evidence | Status |
|---|---|---|---|
| AC1 | first | log a | met |
| AC2 | second | test b | unmet |
| AC3 | third | needs the full gate | not run (#7) |

Verdict: HOLD
MD
}
fresh() { printf 'report\n' > "$T/out/report.md"; rm -f "$T/out/check.md"; }
run() { (cd "${CWD:-$T}" && bash "$S" "$@" > "$T/stdout" 2> "$T/stderr"); }

merged; fresh
run 5 9 "$T/out"; check merged-rc "$?" 0
check merged-check-pass "$(head -1 "$T/out/check.md")" pass
check merged-gh-repo "$(grep -vc -- ' -R o/r' "$T/gh-calls")" 0

hold; fresh
FAKE_PR_STATE=OPEN run 5 9 "$T/out"; check hold-rc "$?" 0
check hold-check-pass "$(head -1 "$T/out/check.md")" pass
merged; fresh
FAKE_PR_STATE=OPEN run 5 9 "$T/out"; check merged-form-on-open-pr "$?" 1
check merged-form-on-open-fail "$(head -1 "$T/out/check.md" | cut -c1-5)" fail:
hold; fresh
run 5 9 "$T/out"; check hold-form-on-merged-pr "$?" 1
FAKE_PR_STATE=CLOSED run 5 9 "$T/out"; check closed-pr "$?" 1
sed -i 's/^Verdict: HOLD/Verdict: CLOSE/' "$T/out/verification.md"
FAKE_PR_STATE=OPEN run 5 9 "$T/out"; check hold-with-close-verdict "$?" 1

# files: missing, empty, differing comment
merged; fresh
rm "$T/out/report.md"; run 5 9 "$T/out"; check no-report "$?" 1
check no-report-names-file "$(grep -c 'report.md' "$T/stderr")" 1
: > "$T/out/report.md"; run 5 9 "$T/out"; check empty-report "$?" 1
fresh; : > "$T/out/verification.md"; run 5 9 "$T/out"; check empty-verification "$?" 1
merged; fresh; rm "$T/out/verification.md"; run 5 9 "$T/out"; check no-verification "$?" 1
merged; fresh
printf 'other comment\n' > "$T/other.md"
FAKE_COMMENT=$T/other.md run 5 9 "$T/out"; check comment-differs "$?" 1
check comment-differs-fail "$(head -1 "$T/out/check.md" | cut -c1-5)" fail:
fresh
sed -i 's/^| AC3 .*/| AC3 | third | log c | met |/' "$T/out/verification.md"
run 5 9 "$T/out"; check met-rows-differ-from-ticks "$?" 1
sed -i 's/^- \[ \] third/- [x] third/' "$T/body.md"
run 5 9 "$T/out"; check all-met-all-ticked "$?" 0
printf '## Goal\n\nno criteria\n' > "$T/body.md"
run 5 9 "$T/out"; check no-criteria "$?" 1

# a detached worktree as cwd: gh must not need the current branch
printf '## Acceptance criteria\n\n- [x] first\n- [x] second\n- [ ] third\n\n## Test plan\n' > "$T/body.md"
git init -q "$T/repo"
git -C "$T/repo" -c user.name=t -c user.email=t@t.invalid commit -q --allow-empty -m init
git -C "$T/repo" worktree add -q --detach "$T/det" HEAD
check detached "$(git -C "$T/det" symbolic-ref -q HEAD || echo detached)" detached
merged; fresh
CWD=$T/det run 5 9 "$T/out"; check detached-rc "$?" 0
check detached-no-branch-error "$(cat "$T/stdout" "$T/stderr" | grep -c 'not on any branch')" 0

# the unfixed failure: gh with an empty PR argument list reads the branch
(cd "$T/det" && gh pr view --json state -q .state > /dev/null 2>&1); check empty-pr-fails-like-gh "$?" 1

# arguments
run 5 "" "$T/out"; check empty-pr-arg "$?" 2
run 5 x "$T/out"; check text-pr-arg "$?" 2
run "" 9 "$T/out"; check empty-n-arg "$?" 2
run 5 9; check missing-out-arg "$?" 2
run 5 9 "$T/out" extra; check extra-arg "$?" 2
run -h; check help "$?" 0

# routine text
check no-output-path "$(grep -c 'output/' "$ROUTINE" "$V/_templates/verification-comment.md" "$V/_templates/brief-verifier.md" | grep -vc ':0$')" 0
check no-dollar-v "$(grep -cE '\$V([^A-Za-z_]|$)' "$ROUTINE" "$V/_templates/verification-comment.md" "$V/_templates/brief-verifier.md" | grep -vc ':0$')" 0
check no-expected-red "$(grep -ci 'expected red' "$ROUTINE")" 0
check no-stage-references-old-output "$(grep -rl --include=*.md 01_verifier/output "$V" | wc -l | tr -d " ")" 0
check names-wait-for "$(grep -c 'wait-for.sh' "$ROUTINE")" 1
check one-long-wait "$(grep -c 'one long tracked wait' "$ROUTINE")" 1
check names-out "$(grep -c '\$OUT' "$ROUTINE" | awk '$1 > 0 { print "yes" }')" yes
check final-step "$(grep -cE '^[0-9]+\. .*verifier-done.sh.*exits 0' "$ROUTINE")" 1
check hold-outcome "$(grep -c 'Hold: PR #<PR> kept open, <reasons>\.' "$V/_templates/verification-comment.md")" 1
check merged-outcome "$(grep -c '^Merged into <base> as <short sha> (PR #<PR>)\.$' "$V/_templates/verification-comment.md")" 1
check walk-strict-named "$(grep -c 'strict walk' "$ROUTINE")" 1

if [ "$fails" -ne 0 ]; then
  echo "test-verifier-done: $fails failed"
  exit 1
fi
echo "test-verifier-done: ok"
