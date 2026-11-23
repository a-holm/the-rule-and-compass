#!/usr/bin/env bash
# Usage: verifier-done.sh <issue N> <PR> <out dir>
# Last step of the verifier routine (08_close/stages/01_verifier): the Check, as a script that reads no current branch.
# Refuses (exit 1, naming the file) unless verification.md and report.md exist and are non-empty in <out dir>. Then it measures, through check.sh,
# and writes `pass` or `fail: <what was measured>` to <out dir>/check.md. Every gh call names the repository (RCSR_REPO, else the origin remote)
# and takes N and PR from the arguments, so the cwd may be a detached worktree.
# Two outcomes pass. Merged: PR state MERGED, line 1 `Merged into <base> as <sha> (PR #<PR>).`, `Verdict: CLOSE` or `Verdict: KEEP OPEN`, and the number of
# `met` rows equals the number of ticked acceptance boxes of the issue. Hold: PR state OPEN, line 1 `Hold: PR #<PR> kept open, <reasons>.`, `Verdict: HOLD`
# (a CONFLICTING PR is a hold). Both: one `| AC<n>` row per acceptance box, and the last issue comment equals verification.md.
# Writes: <out dir>/check.md
# Exit: 0 pass; 1 refused or fail; 2 usage.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ "${1:-}" = --measure ]; then
  shift
  n=$1 pr=$2 out=$3 repo=$4
  v=$out/verification.md
  state=$(gh pr view "$pr" -R "$repo" --json state -q .state) || { echo "gh pr view $pr failed"; exit 1; }
  ac=$(gh issue view "$n" -R "$repo" --json body -q .body | sed -n '/^## Acceptance criteria/,/^## /p')
  boxes=$(printf '%s\n' "$ac" | grep -c '^- \[[ x]\]')
  ticked=$(printf '%s\n' "$ac" | grep -c '^- \[x\]')
  rows=$(grep -c '^| AC[0-9]' "$v")
  met=$(grep -cE '^\| AC[0-9]+ .*\| met \|$' "$v")
  [ "$boxes" -ge 1 ] || { echo "issue $n has no acceptance box"; exit 1; }
  [ "$rows" = "$boxes" ] || { echo "$rows AC rows for $boxes acceptance boxes"; exit 1; }
  case "$state" in
    MERGED)
      grep -qE '^Merged into .* as [0-9a-f]{7,40} \(PR #[0-9]+\)\.$' "$v" || { echo "line for a merged PR missing in verification.md"; exit 1; }
      grep -qE '^Verdict: (CLOSE|KEEP OPEN)$' "$v" || { echo "verdict of a merged PR is not CLOSE or KEEP OPEN"; exit 1; }
      [ "$met" = "$ticked" ] || { echo "$met met rows but $ticked ticked boxes"; exit 1; }
      ;;
    OPEN)
      grep -qE "^Hold: PR #$pr kept open, .+\.$" "$v" || { echo "hold line missing in verification.md for the OPEN PR"; exit 1; }
      grep -qE '^Verdict: HOLD$' "$v" || { echo "verdict of a hold is not HOLD"; exit 1; }
      ;;
    *) echo "PR $pr state is $state, not MERGED or OPEN"; exit 1 ;;
  esac
  [ "$(gh issue view "$n" -R "$repo" --json comments -q '.comments[-1].body')" = "$(cat "$v")" ] || { echo "the last issue comment differs from verification.md"; exit 1; }
  exit 0
fi
source "$here/lib.sh"
case "${1:-}" in -h|--help) usage; exit 0 ;; esac
[ $# -eq 3 ] || { usage >&2; exit 2; }
n=$1 pr=$2 out=$3
case "$n" in ''|*[!0-9]*) echo "verifier-done: issue number must be digits" >&2; usage >&2; exit 2 ;; esac
case "$pr" in ''|*[!0-9]*) echo "verifier-done: PR number must be digits" >&2; usage >&2; exit 2 ;; esac
for f in verification.md report.md; do
  [ -s "$out/$f" ] || { echo "verifier-done: $out/$f is missing or empty" >&2; exit 1; }
done
repo=${RCSR_REPO:-$(git -C "$RCSR_ROOT" remote get-url origin 2> /dev/null | sed -E 's#^.*github\.com[:/]##; s#\.git$##')}
[ -n "$repo" ] || { echo "verifier-done: no repository: set RCSR_REPO or an origin remote on github.com" >&2; exit 1; }
exec bash "$here/check.sh" "$out" "bash $(printf '%q' "$here/verifier-done.sh") --measure $n $pr $(printf '%q' "$out") $(printf '%q' "$repo")"
