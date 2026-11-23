#!/usr/bin/env bash
# Usage: test-no-unbounded-fifo.sh
# Lint: no script test reads a FIFO with a plain `read` redirect or opens one for writing with a plain `>`; tests use the helpers in lib/fifo.sh.
# Writes: nothing.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fails=0
self=$(basename "${BASH_SOURCE[0]}")
for f in "$HERE"/test-*.sh "$HERE"/lib/fakes.sh; do
  [ "$(basename "$f")" != "$self" ] || continue
  hits=$(grep -nE '(^|[^_[:alnum:]])read .*< *"?[^ ]*(fifo|ready|go|FAKE_GO)|(echo|printf) .*> *"?[^ ]*(fifo/|/ready|/go|FAKE_READY)' "$f" | grep -vE '1<>|2>|/dev/null')
  if [ -n "$hits" ]; then
    echo "FAIL unbounded-fifo $f:"
    echo "$hits"
    fails=$((fails + 1))
  fi
done
if [ "$fails" -ne 0 ]; then
  echo "test-no-unbounded-fifo: $fails failed"
  exit 1
fi
echo "test-no-unbounded-fifo: ok"
