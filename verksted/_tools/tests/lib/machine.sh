#!/usr/bin/env bash
# Sourced by verksted/_tools/tests/test-*.sh after HERE is set; not run.
# Writes: nothing.
# Points RCSR_LOCAL at the synthetic machine.md of the guard crate, so lib.sh finds its four keys (and BUILD_JOBS) in every test
# and no test reads the real one. A test that needs other keys exports its own RCSR_LOCAL afterwards.
export RCSR_LOCAL=$(cd "$HERE/../verksted-check/tests/fixtures/local" && pwd)
