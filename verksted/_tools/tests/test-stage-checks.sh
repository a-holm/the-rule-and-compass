#!/usr/bin/env bash
# Usage: test-stage-checks.sh
# Hermetic test of the Check commands of 08_close/stages/02_close and 03_cleanup, and 04_gate/stages/03_full_gate.
# Each Check is read from its CONTEXT.md and run on fixture outputs with a fake gh.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/lib/machine.sh"
V=$HERE/../..
T=$(cd "$(mktemp -d)" && pwd -P)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
# check_cmd <relative CONTEXT.md>: the first fenced sh block under `## Check`
check_cmd() {
  awk '/^## Check$/ { c = 1; next } c && /^```sh$/ { b = 1; next } b && /^```$/ { exit } b { print }' "$V/$1"
}
mkdir -p "$T/bin" "$T/out/verifier" "$T/out/04_gate/stages/03_full_gate/output"
cat > "$T/bin/gh" <<'GH'
#!/usr/bin/env bash
args="$*"
case "$args" in
  "pr view"*) echo "${FAKE_PR_STATE:-MERGED}" ;;
  "issue view 7 "*) echo "${FAKE_OWNER_STATE:-OPEN}" ;;
  *"--json body"*) cat "$FAKE_BODY" ;;
  *"--json comments"*) cat "$FAKE_COMMENT" ;;
  *"--json state,stateReason"*) echo "${FAKE_ISSUE_STATE:-CLOSED COMPLETED}" ;;
  *) echo "fake gh: unexpected $args" >&2; exit 1 ;;
esac
GH
chmod +x "$T/bin/gh"
run_check() { # run_check <CONTEXT.md> [env assignments...]: exit code of the Check command
  local rel=$1
  shift
  (cd "$T" && env V="$T/out" OUT="$T/out/verifier" N=5 PR=9 WT="$T/wt" PATH="$T/bin:$PATH" FAKE_BODY="$T/body.md" FAKE_COMMENT="$T/out/verifier/verification.md" "$@" bash -c "$(check_cmd "$rel")" > /dev/null 2>&1)
}

# 01_verifier: its Check is verifier-done.sh, tested in test-verifier-done.sh
ver=$T/out/verifier/verification.md

# 02_close: every row met, or not run with an open owner issue
V2=08_close/stages/02_close/CONTEXT.md
cat > "$ver" <<'MD'
| AC1 | first | log a | met |
| AC2 | second | needs a measurement | not run (#7) |

Verdict: CLOSE
MD
run_check "$V2"; check close-ok "$?" 0
run_check "$V2" FAKE_OWNER_STATE=CLOSED; check close-owner-closed "$?" 1
run_check "$V2" FAKE_ISSUE_STATE=OPEN; check close-not-closed "$?" 1
sed -i 's/not run (#7)/not run/' "$ver"
run_check "$V2"; check close-ownerless-not-run "$?" 1
sed -i 's/| not run |/| unmet |/' "$ver"
run_check "$V2"; check close-unmet "$?" 1
printf '| AC1 | first | log a | met |\n\nVerdict: KEEP OPEN\n' > "$ver"
run_check "$V2"; check close-keep-open "$?" 1

# 03_cleanup: no ticket worktree, no verifier worktree, no branch
V3=08_close/stages/03_cleanup/CONTEXT.md
git init -q "$T/repo"
git -C "$T/repo" -c user.name=t -c user.email=t@t.invalid commit -q --allow-empty -m init
run3() { (cd "$T/repo" && env N=5 PR=9 WT="$T/wt" bash -c "$(check_cmd "$V3")" > /dev/null 2>&1); }
run3; check cleanup-clean "$?" 0
mkdir -p "$T/wt/verifier-9"
run3; check cleanup-verifier-worktree-left "$?" 1
rmdir "$T/wt/verifier-9"
git -C "$T/repo" branch issue/5-x
run3; check cleanup-branch-left "$?" 1
git -C "$T/repo" branch -q -D issue/5-x

# 04_gate/stages/03_full_gate: an ignored test that ran nothing fails the Check
V4=04_gate/stages/03_full_gate/CONTEXT.md
g=$T/out/04_gate/stages/03_full_gate/output/gate.md
sha=0123456789abcdef0123456789abcdef01234567
printf 'cpu: ok=10 failed=0\ncuda: ok=5 failed=0\nrecovered 27/27\ntested: %s\n' "$sha" > "$g"
run_check "$V4"; check gate-ok "$?" 0
printf 'ignored: x passed=2\n' >> "$g"
run_check "$V4"; check gate-ignored-ran "$?" 0
printf 'ignored: x passed=0\n' >> "$g"
run_check "$V4"; check gate-ignored-zero "$?" 1
printf 'cpu: ok=10 failed=0\ncuda: ok=5 failed=0\nrecovered 27/27\ntested: %s\nignored: y passed=0\n' "$sha" > "$g"
run_check "$V4"; check gate-ignored-zero-alone "$?" 1
printf 'cpu: ok=10 failed=0\ncuda: ok=5 failed=0\nrecovered 27/27\ntested: %s\nignored: y passed=10\n' "$sha" > "$g"
run_check "$V4"; check gate-ignored-ten "$?" 0

if [ "$fails" -ne 0 ]; then
  echo "test-stage-checks: $fails failed"
  exit 1
fi
echo "test-stage-checks: ok"
