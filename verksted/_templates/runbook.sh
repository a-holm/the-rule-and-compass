#!/usr/bin/env bash
# Starter runbook for a long run. Copy it to the run folder, rename the steps, fill the commands; not run from here.
# Usage: RUN=<run dir> RL_BIN=<run dir>/bin/rcsr bash runbook.sh [list | <step>...]    (SEALED=1 runs the protected step)
# Writes: $RUN/done/<step>.done, $RUN/run.log, $RUN/<hold name>.log, $RUN/run.pid.
set -uo pipefail
RUN=${RUN:?set RUN to the run directory}
STEPS=(preflight launch seal)
PROTECTED=(seal)
# runlib.sh is a sibling of the verksted tools: adjust the path to where the runbook lives.
RUNLIB=${RUNLIB:?set RUNLIB to the absolute path of verksted/_tools/runlib.sh}

step_preflight() {
  [ -x "${RL_BIN:?set RL_BIN to the pinned binary}" ] || return 1
  rl_cpu preflight 600 -- "$RL_BIN" --help
}

step_launch() {
  echo $$ >> "$RUN/run.pid"
  rl_gpu launch 540 -- "$RL_BIN" '<launch flags from the dry run>'
}

step_seal() {
  rl_cpu seal 600 -- '<the command that reads the protected rows once>'
}

source "$RUNLIB"
runlib_main "$@"
