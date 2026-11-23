#!/usr/bin/env bash
# Usage: test-lints.sh
# Hermetic test of lints.sh: the full list, a fixture tree with one test file missing, an empty tests dir, usage errors.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
S=$HERE/../lints.sh
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fails=0
check() {
  if [ "$2" != "$3" ]; then
    echo "FAIL $1: got '$2' want '$3'"
    fails=$((fails + 1))
  fi
}
NAMES="test_attributes rcsr_command_lint test_hygiene device_skip_lint no_python sanitizer_plan sanitizer_child_env stdin_write_lint conflict_markers no_banned_words lockbox_literal cpu_config_lint"
want=
for n in $NAMES; do want="$want --test $n"; done
want=${want# }

check full-list "$(env -u RCSR_CRATE bash "$S" 2> "$T/err")" "$want"
check full-list-silent "$(wc -c < "$T/err" | tr -d ' ')" 0
check full-list-count "$(bash "$S" 2> /dev/null | grep -o -e '--test' | wc -l | tr -d ' ')" 12

mkdir -p "$T/tree/pkg/tests"
printf '[package]\nname = "rule-and-compass"\nversion = "0.1.0"\n' > "$T/tree/pkg/Cargo.toml"
for n in $NAMES; do : > "$T/tree/pkg/tests/$n.rs"; done
check tree-all "$(env -u RCSR_CRATE bash "$S" --tree "$T/tree" 2> "$T/err")" "$want"
check tree-all-silent "$(wc -c < "$T/err" | tr -d ' ')" 0

rm "$T/tree/pkg/tests/device_skip_lint.rs"
out=$(env -u RCSR_CRATE bash "$S" --tree "$T/tree" 2> "$T/err"); rc=$?
check tree-missing-rc "$rc" 0
check tree-missing-out "$out" "${want/ --test device_skip_lint/}"
check tree-missing-err "$(cat "$T/err")" "lints.sh: skipped device_skip_lint (no tests/device_skip_lint.rs in $T/tree)"

mkdir -p "$T/empty/pkg"
cp "$T/tree/pkg/Cargo.toml" "$T/empty/pkg/"
out=$(env -u RCSR_CRATE bash "$S" --tree "$T/empty" 2> "$T/err"); rc=$?
check tree-empty-rc "$rc" 0
check tree-empty-out "$out" ""
check tree-empty-skips "$(wc -l < "$T/err" | tr -d ' ')" 12

mkdir -p "$T/nocrate"
env -u RCSR_CRATE bash "$S" --tree "$T/nocrate" > /dev/null 2>&1; check no-crate-rc $? 1
bash "$S" --tree "$T/missing-dir" > /dev/null 2>&1; check no-dir-rc $? 2
bash "$S" --tree > /dev/null 2>&1; check tree-no-value $? 2
bash "$S" --bogus > /dev/null 2>&1; check bad-flag $? 2
bash "$S" --tree a b > /dev/null 2>&1; check extra-arg $? 2
bash "$S" -h > /dev/null 2>&1; check help $? 0

[ "$fails" -eq 0 ] && echo "test-lints: ok"
exit "$fails"
