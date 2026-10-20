#!/usr/bin/env bash
# A minimal reproducible launcher for a long rcsr run (issue #140).
#
# It demonstrates the contract: an explicit backend, a stable seed derived from
# (label, seed) with SHA-256 (never Python's salted hash), a run manifest, an
# atomic final JSON file, a progress/error log instead of /dev/null, and a
# checkpoint that can be resumed on the same backend.
#
# Usage:
#   scripts/run-long.sh --data data.csv --target y --vars x \
#       --label btc-usd-2019h1 --seed 7 --generations 1000 --gpu
#
# Resume continues the checkpoint and may raise --generations:
#   scripts/run-long.sh ... --generations 2000 --gpu --resume

set -euo pipefail

DATA=""
TARGET=""
VARS=""
LABEL=""
SEED="0"
GENERATIONS="200"
BACKEND=""
RESUME="0"
OUT_DIR="${OUT_DIR:-runs}"

usage() {
    sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --data) DATA="${2:?--data needs a value}"; shift 2 ;;
        --target) TARGET="${2:?--target needs a value}"; shift 2 ;;
        --vars) VARS="${2:?--vars needs a value}"; shift 2 ;;
        --label) LABEL="${2:?--label needs a value}"; shift 2 ;;
        --seed) SEED="${2:?--seed needs a value}"; shift 2 ;;
        --generations) GENERATIONS="${2:?--generations needs a value}"; shift 2 ;;
        --gpu) BACKEND="--gpu"; shift ;;
        --cpu) BACKEND="--cpu"; shift ;;
        --resume) RESUME="1"; shift ;;
        -h|--help) usage ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$DATA" ] || [ -z "$TARGET" ] || [ -z "$VARS" ] || [ -z "$LABEL" ]; then
    echo "error: --data, --target, --vars and --label are required" >&2
    exit 2
fi
if [ -z "$BACKEND" ]; then
    echo "error: pass --gpu or --cpu explicitly; the backend is part of the run identity" >&2
    exit 2
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "error: --seed must be a non-negative integer, got '$SEED'" >&2
    exit 2
fi

# Deterministic derivation: SHA-256(label:seed) -> first 8 hex digits -> u32.
# Python's hash(label) is salted per process and must never be used for this.
HEX="$(printf '%s:%s' "$LABEL" "$SEED" | sha256sum | cut -c1-8)"
RESOLVED_SEED=$((16#$HEX % 4294967296))

mkdir -p "$OUT_DIR"
CHECKPOINT="$OUT_DIR/$LABEL.rcsr"
MANIFEST="$OUT_DIR/$LABEL.manifest.json"
RESULT="$OUT_DIR/$LABEL.result.json"
LOG="$OUT_DIR/$LABEL.stderr.log"
CMD="$OUT_DIR/$LABEL.command.txt"

ARGS=(
    --data "$DATA" --target "$TARGET" --vars "$VARS"
    --seed "$RESOLVED_SEED" --generations "$GENERATIONS"
    --save "$CHECKPOINT" --run-manifest "$MANIFEST"
    --json-out "$RESULT" --progress
    "$BACKEND"
)
if [ "$RESUME" = "1" ]; then
    ARGS+=(--resume "$CHECKPOINT")
fi

{
    echo "# resolved_seed=$RESOLVED_SEED (label=$LABEL seed=$SEED backend=$BACKEND)"
    printf 'rcsr'
    printf ' %q' "${ARGS[@]}"
    printf '\n'
} | tee "$CMD"

# stderr carries the human log and --progress NDJSON; keep it, never /dev/null.
if ! rcsr "${ARGS[@]}" 2> >(tee "$LOG" >&2); then
    echo "error: rcsr failed; see $LOG" >&2
    exit 1
fi
echo "# result: $RESULT"
echo "# manifest: $MANIFEST (verify with: rcsr verify-manifest $MANIFEST)"
