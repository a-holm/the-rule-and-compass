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
# synthetic machine.md every lib() call reads unless the call passes its own RCSR_LOCAL
DEFLOCAL=$T/mdef
mkdir -p "$DEFLOCAL"
printf -- '- GATES: /def/gates\n- MACHINE_SCRIPTS: /def/scripts\n- SANITIZER: /def/sanitizer\n- CUDA_ARCH: sm_def\n' > "$DEFLOCAL/machine.md"
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
    -u RCSR_LOCK_SCRIPT -u RCSR_SANITIZER -u RCSR_XID_CMD -u RCSR_CRATE -u RCSR_LOCAL RCSR_LOCAL="$DEFLOCAL" "$@" \
    bash -c "set -uo pipefail; source '$LIB'; $snippet"
}
git_q() { git -c user.name=t -c user.email=t@t.invalid "$@"; }

# defaults
want_top=$(git -C "$HERE" rev-parse --show-toplevel)
check default-RCSR_ROOT "$(lib 'echo $RCSR_ROOT')" "$want_top"
check default-BASE "$(lib 'echo $BASE')" "integration/gpu-hardening"
check default-LOCK_HOLD_MAX "$(lib 'echo $LOCK_HOLD_MAX')" "600"
check default-VERKSTED "$(lib 'echo $VERKSTED')" "$(cd "$HERE/../.." && pwd)"


# machine.md keys: present, missing, placeholder, overrides
mach() { mkdir -p "$T/m$1"; printf '%s' "$2" > "$T/m$1/machine.md"; echo "$T/m$1"; }
all_keys='# Profile
- GATES: /fx/gates
- MACHINE_SCRIPTS: /fx/scripts
- SANITIZER: /fx/cuda/compute-sanitizer
- CUDA_ARCH: sm_fx
'
d=$(mach 1 "$all_keys")
check key-GATES "$(lib 'echo "$GATES $G"' RCSR_LOCAL="$d")" "/fx/gates /fx/gates"
check key-SANITIZER "$(lib 'echo $SANITIZER' RCSR_LOCAL="$d")" "/fx/cuda/compute-sanitizer"
check key-CUDA_ARCH "$(lib 'echo $CUDA_ARCH' RCSR_LOCAL="$d")" "sm_fx"
check key-MACHINE_SCRIPTS "$(lib 'echo "$OC $PAUSE $STREAMING $LOCKFILE $SLOTDIR $B $L"' RCSR_LOCAL="$d")" \
  "/fx/scripts /fx/scripts/PAUSE /fx/scripts/STREAMING /fx/scripts/full-suite.lock /fx/scripts/build-slots /fx/scripts/build-slot.sh /fx/scripts/one-full-suite.sh"
check machine_key-direct "$(lib 'machine_key CUDA_ARCH' RCSR_LOCAL="$d")" "sm_fx"
d=$(mach 2 '- GATES:   /a b/c  
- GATES: /second
- MACHINE_SCRIPTS: /s
- SANITIZER: /z
- CUDA_ARCH: sm_1
')
check key-trim-first-wins "$(lib 'echo "[$GATES]"' RCSR_LOCAL="$d")" "[/a b/c]"
for k in GATES MACHINE_SCRIPTS SANITIZER CUDA_ARCH; do
  d=$(mach "miss$k" "$(grep -v "^- $k:" <<< "$all_keys")")
  out=$(lib 'echo reached' RCSR_LOCAL="$d" 2>&1); rc=$?
  check "missing-$k-rc" "$rc" "2"
  check "missing-$k-names" "$(grep -c "key $k missing in $d/machine.md" <<< "$out")" "1"
  check "missing-$k-no-run" "$(grep -c reached <<< "$out")" "0"
  d=$(mach "ph$k" "$(sed "s/^- $k: .*/- $k: <set me>/" <<< "$all_keys")")
  out=$(lib 'echo reached' RCSR_LOCAL="$d" 2>&1); rc=$?
  check "placeholder-$k-rc" "$rc" "2"
  check "placeholder-$k-names" "$(grep -c "key $k missing in $d/machine.md" <<< "$out")" "1"
done
out=$(lib 'echo reached' RCSR_LOCAL="$T/none" 2>&1); rc=$?
check no-machine-md-rc "$rc" "2"
check no-machine-md-names "$(grep -c "key GATES missing in $T/none/machine.md" <<< "$out")" "1"
d=$(mach empty "- GATES:
$(grep -v '^- GATES:' <<< "$all_keys")")
check empty-value-rc "$(lib 'echo reached' RCSR_LOCAL="$d" >/dev/null 2>&1; echo $?)" "2"
# interactive shell: the source reports status 2 and the shell survives; a script still stops at the source line
ilib() { env -u RCSR_GATES -u RCSR_MACHINE_DIR -u RCSR_SANITIZER "$@" bash -ic "source '$LIB'; r=\$?; echo alive; echo rc=\$r" 2> /dev/null; }
check interactive-missing "$(ilib RCSR_LOCAL="$T/none" | tr '\n' ' ')" "alive rc=2 "
check interactive-ok "$(ilib RCSR_LOCAL="$DEFLOCAL" | tr '\n' ' ')" "alive rc=0 "
d=$(mach iph "$(sed 's/^- SANITIZER: .*/- SANITIZER: <set me>/' <<< "$all_keys")")
check interactive-placeholder "$(ilib RCSR_LOCAL="$d" | tr '\n' ' ')" "alive rc=2 "
printf '#!/usr/bin/env bash\nsource "%s"\necho continued\n' "$LIB" > "$T/src.sh"
out=$(env -u RCSR_GATES RCSR_LOCAL="$T/none" bash "$T/src.sh" 2> /dev/null); rc=$?
check script-stops-at-source "$rc:$out" "2:"
out=$(env RCSR_LOCAL="$T/none" bash -c "source '$LIB'; echo continued" 2> /dev/null); rc=$?
check bash-c-stops-at-source "$rc:$out" "2:"
# --help of a tool with an unseeded machine.md: the source stops it first, exit 2
out=$(RCSR_LOCAL="$T/none" bash "$HERE/../resources.sh" -h 2>&1); rc=$?
check tool-help-unseeded-rc "$rc" "2"
check tool-help-unseeded-names "$(grep -c 'key GATES missing' <<< "$out")" "1"
# an override replaces its key, the other keys still come from machine.md
d=$(mach ov "- CUDA_ARCH: sm_fx
")
check override-all "$(lib 'echo "$GATES $OC $SANITIZER $CUDA_ARCH"' RCSR_LOCAL="$d" RCSR_GATES=/g RCSR_MACHINE_DIR=/m RCSR_SANITIZER=/s)" "/g /m /s sm_fx"
check override-gates-only-rc "$(lib 'echo reached' RCSR_LOCAL="$d" RCSR_GATES=/g >/dev/null 2>&1; echo $?)" "2"
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

# VLOCAL
mkdir -p "$main_top/verksted/_local"
cp "$DEFLOCAL/machine.md" "$main_top/verksted/_local/machine.md"
check VLOCAL-default "$(lib 'echo $VLOCAL' RCSR_ROOT="$side_top" RCSR_LOCAL=)" "$main_top/verksted/_local"
check VLOCAL-override "$(lib 'echo $VLOCAL' RCSR_LOCAL="$DEFLOCAL")" "$DEFLOCAL"

# crate_dir: one match, none, two, a [lib] name alone, subdirectory cwd, override
fx=$T/fx
mkdir -p "$fx/pkg/src" "$fx/other" "$fx/sub"
git -C "$fx" init -q
printf '[package]\nname = "rule-and-compass"\n\n[[bin]]\nname = "x"\n' > "$fx/pkg/Cargo.toml"
printf '[package]\nname = "else"\n\n[lib]\nname = "rule-and-compass"\n' > "$fx/other/Cargo.toml"
check crate_dir-root "$(cd "$fx" && lib 'crate_dir "$PWD"')" "$fx/pkg"
check crate_dir-subdir "$(cd "$fx/pkg/src" && lib 'crate_dir "$(dirname "$(dirname "$PWD")")"')" "$fx/pkg"
check crate_dir-default-root "$(cd "$fx/sub" && lib 'crate_dir' RCSR_ROOT="$fx")" "$fx/pkg"
check crate_dir-lib-name-alone "$(cd "$fx" && lib 'rm -rf pkg; crate_dir "$PWD"; echo rc=$?' 2>&1 | tail -n 1)" "rc=1"
mkdir -p "$fx/pkg2" "$fx/pkg3"
for d in pkg2 pkg3; do printf '[package]\nname = "rule-and-compass"\n' > "$fx/$d/Cargo.toml"; done
out=$(cd "$fx" && lib 'crate_dir "$PWD"; echo rc=$?' 2>&1)
check crate_dir-two-rc "$(tail -n 1 <<< "$out")" "rc=1"
check crate_dir-two-msg "$(head -n 1 <<< "$out" | grep -c 'declare package')" "1"
check crate_dir-override "$(cd "$fx" && lib 'crate_dir "$PWD"' RCSR_CRATE=pkg3)" "$fx/pkg3"

# need_gates
out=$(lib 'need_gates; echo reached' RCSR_GATES="$T/nogates" 2>&1); rc=$?
check need_gates-rc "$rc" "1"
check need_gates-msg "$out" "gates dir $T/nogates missing; link it to the log directory or set RCSR_GATES"
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

# xid_new_since: counts new lines since the start, fails closed
check xid-new-cmd "$(lib 'xid_new_since "2026-01-01 00:00:00"' RCSR_XID_CMD='echo 3')" "3"
check xid-new-missing-start "$(lib 'xid_new_since ""' RCSR_XID_CMD='echo 0' 2>/dev/null; echo "rc=$?")" "rc=1"
check xid-new-cmd-fails "$(lib 'xid_new_since "t"' RCSR_XID_CMD='exit 3' 2>/dev/null; echo "rc=$?")" "rc=1"
check xid-new-cmd-garbage "$(lib 'xid_new_since "t"' RCSR_XID_CMD='echo x' 2>/dev/null; echo "rc=$?")" "rc=1"
# the stub selects its output by the --since argument: quiet after 12:00, one new line after 11:00
cat > "$T/bin/journalctl" <<'JC'
#!/bin/sh
[ "$1" = -k ] || exit 2
[ "$2" = --since ] || exit 2
case "$3" in
  "2026-01-01 12:00:00") printf 'quiet\n' ;;
  "2026-01-01 11:00:00") printf 'quiet\nNVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus\n' ;;
  *) exit 1 ;;
esac
JC
chmod +x "$T/bin/journalctl"
check xid-new-old-line-ignored "$(lib 'xid_new_since "2026-01-01 12:00:00"' PATH="$T/bin:$PATH")" "0"
check xid-new-after-start "$(lib 'xid_new_since "2026-01-01 11:00:00"' PATH="$T/bin:$PATH")" "1"
check xid-new-journalctl-fails "$(lib 'xid_new_since "2030-01-01 00:00:00"' PATH="$T/bin:$PATH" 2>/dev/null; echo "rc=$?")" "rc=1"
check xid-start-format "$(lib 'xid_start' | grep -cE '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$')" "1"

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
