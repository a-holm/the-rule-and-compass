#!/usr/bin/env bash
# Usage: test-lib.sh
# Hermetic test of lib.sh: overrides, defaults, need_gates, rc_done, log_path, xid_count, usage.
# Writes: a temp dir that it removes.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LIB=$HERE/../lib.sh
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
# Run a snippet with lib.sh sourced and every RCSR_* variable unset except those passed as args.
lib() {
  local snippet=$1
  shift
  env -u RCSR_ROOT -u RCSR_BASE -u RCSR_GATES -u RCSR_MACHINE_DIR -u RCSR_SLOT_SCRIPT \
    -u RCSR_LOCK_SCRIPT -u RCSR_SANITIZER -u RCSR_XID_CMD "$@" \
    bash -c "set -uo pipefail; source '$LIB'; $snippet"
}
git_q() { git -c user.name=t -c user.email=t@t.invalid "$@"; }

# defaults
want_top=$(git -C "$HERE" rev-parse --show-toplevel)
check default-RCSR_ROOT "$(lib 'echo $RCSR_ROOT')" "$want_top"
check default-BASE "$(lib 'echo $BASE')" "integration/gpu-hardening"
check default-GATES "$(lib 'echo $GATES')" "/mnt/lager/tmp/rcsr-gates"
check default-G "$(lib 'echo $G')" "/mnt/lager/tmp/rcsr-gates"
check default-OC "$(lib 'echo $OC')" "/mnt/lager/tmp/opencode"
check default-B "$(lib 'echo $B')" "/mnt/lager/tmp/opencode/build-slot.sh"
check default-L "$(lib 'echo $L')" "/mnt/lager/tmp/opencode/one-full-suite.sh"
check default-SANITIZER "$(lib 'echo $SANITIZER')" "/usr/local/cuda-13.0/bin/compute-sanitizer"
check default-CUDA_ARCH "$(lib 'echo $CUDA_ARCH')" "sm_86"
check default-LOCK_HOLD_MAX "$(lib 'echo $LOCK_HOLD_MAX')" "600"
check default-VERKSTED "$(lib 'echo $VERKSTED')" "$(cd "$HERE/../.." && pwd)"

# overrides
check RCSR_BASE "$(lib 'echo $BASE' RCSR_BASE=b/x)" "b/x"
check RCSR_GATES "$(lib 'echo "$GATES $G"' RCSR_GATES=/g/h)" "/g/h /g/h"
check RCSR_MACHINE_DIR "$(lib 'echo "$OC $PAUSE $STREAMING $LOCKFILE $SLOTDIR $B $L"' RCSR_MACHINE_DIR=/m)" \
  "/m /m/PAUSE /m/STREAMING /m/full-suite.lock /m/build-slots /m/build-slot.sh /m/one-full-suite.sh"
check RCSR_SLOT_SCRIPT "$(lib 'echo $B' RCSR_SLOT_SCRIPT=/s/b.sh)" "/s/b.sh"
check RCSR_LOCK_SCRIPT "$(lib 'echo $L' RCSR_LOCK_SCRIPT=/s/l.sh)" "/s/l.sh"
check RCSR_SANITIZER "$(lib 'echo $SANITIZER' RCSR_SANITIZER=/s/san)" "/s/san"

# RCSR_ROOT, ROOT and WT from a scratch repository and its worktree
mkdir -p "$T/main"
git -C "$T/main" init -q
git_q -C "$T/main" commit -q --allow-empty -m init
git -C "$T/main" worktree add -q -b side "$T/side"
main_top=$(git -C "$T/main" rev-parse --show-toplevel)
side_top=$(git -C "$T/side" rev-parse --show-toplevel)
check RCSR_ROOT-override "$(lib 'echo $RCSR_ROOT' RCSR_ROOT="$main_top")" "$main_top"
check ROOT-main "$(lib 'echo $ROOT' RCSR_ROOT="$main_top")" "$main_top"
check ROOT-from-worktree "$(lib 'echo $ROOT' RCSR_ROOT="$side_top")" "$main_top"
check RCSR_ROOT-from-worktree "$(lib 'echo $RCSR_ROOT' RCSR_ROOT="$side_top")" "$side_top"
check WT "$(lib 'echo $WT' RCSR_ROOT="$side_top")" "$main_top/worktrees"

# need_gates
out=$(lib 'need_gates; echo reached' RCSR_GATES="$T/nogates" 2>&1); rc=$?
check need_gates-rc "$rc" "1"
check need_gates-msg "$out" "gates dir $T/nogates missing; #684 creates the symlink"
check need_gates-creates-nothing "$([ -e "$T/nogates" ] && echo yes || echo no)" "no"
mkdir "$T/gates"
check need_gates-ok "$(lib 'need_gates; echo reached' RCSR_GATES="$T/gates")" "reached"

# rc_done
got=$(lib 'rc_done 3')
if ! [[ $got =~ ^RC_DONE\ rc=3\ at=[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]]; then
  echo "FAIL rc_done: '$got'"
  fails=$((fails + 1))
fi

# log_path
check log_path-issue "$(lib 'log_path 12 red' RCSR_GATES=/g)" "/g/issue12-red.log"
check log_path-name "$(lib 'log_path research-x run1' RCSR_GATES=/g)" "/g/research-x-run1.log"

# xid_count
check xid-cmd "$(lib 'xid_count' RCSR_XID_CMD='echo 7')" "7"
check xid-cmd-empty "$(lib 'xid_count' RCSR_XID_CMD='true')" "0"
mkdir "$T/bin"
printf '#!/bin/sh\nprintf "a\\nNVRM: Xid (PCI): 79\\nb\\nNVRM: Xid (PCI): 13\\n"\n' > "$T/bin/journalctl"
chmod +x "$T/bin/journalctl"
check xid-journal "$(lib 'xid_count' PATH="$T/bin:$PATH")" "2"
printf '#!/bin/sh\necho quiet\n' > "$T/bin/journalctl"
check xid-journal-none "$(lib 'xid_count' PATH="$T/bin:$PATH")" "0"

# usage prints the calling script's header without the comment prefix
cat > "$T/hdr.sh" <<HDR
#!/usr/bin/env bash
# Usage: hdr.sh [-h]
# Does a thing.
# Writes: nothing.
source '$LIB'
usage
HDR
check usage "$(bash "$T/hdr.sh")" "$(printf 'Usage: hdr.sh [-h]\nDoes a thing.\nWrites: nothing.')"

if [ "$fails" -ne 0 ]; then
  echo "test-lib: $fails failed"
  exit 1
fi
echo "test-lib: ok"
