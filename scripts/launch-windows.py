#!/usr/bin/env python3
"""Tracked multi-window launcher for long `rcsr` runs (issue #140).

This is the tracked replacement for the scratch launchers in the git-ignored
`arbeidsmappe/crypto-rcsr/pilot/` directory. It follows the launcher contract in
`docs/run-integrity.md`:

* the backend is explicit (`--gpu` or `--cpu`) and is reused on resume;
* the numeric seed is `SHA-256(label:seed)`, never Python's salted built-in hash;
* every run writes `--run-manifest`, `--json-out` and `--progress`;
* the final JSON is published atomically by the engine (`--json-out`);
* stdout/stderr go to a per-window log, never `/dev/null`;
* the exact command line and resolved seed are recorded next to the outputs.

Usage:

    scripts/launch-windows.py \
        --data panel.rcsb --target y1 --vars close,volume \
        --time ts --backend gpu --seed 7 \
        --window 2024H1,1704067200,1719792000 \
        --window 2024H2,1719792000,1735689600 \
        --out-dir runs

Use `--dry-run` to print the resolved commands without starting a search.
"""

import argparse
import hashlib
import os
import shlex
import subprocess
import sys


def stable_seed(label: str, seed: int) -> int:
    """A process-independent seed: SHA-256(label:seed), first 8 hex digits."""
    digest = hashlib.sha256(f"{label}:{seed}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100000


def parse_window(text: str):
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "window must be LABEL,START,END (epoch seconds)"
        )
    label, start, end = parts
    if not label:
        raise argparse.ArgumentTypeError("window label is empty")
    try:
        return label, int(float(start)), int(float(end))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


def build_command(args, label: str, start: int, end: int, out_dir: str):
    checkpoint = os.path.join(out_dir, f"{label}.ckpt")
    manifest = os.path.join(out_dir, f"{label}.manifest.json")
    result = os.path.join(out_dir, f"{label}.result.json")
    log = os.path.join(out_dir, f"{label}.stderr.log")
    dev_log = os.path.join(out_dir, f"{label}.dev.ndjson")
    command_file = os.path.join(out_dir, f"{label}.command.txt")
    resolved_seed = stable_seed(label, args.seed)

    train_end = start - args.label_horizon_seconds
    cmd = [
        args.rcsr,
        "--data", args.data,
        "--target", args.target,
        "--vars", args.vars,
        "--time", args.time,
        "--train-select", args.train_select.format(start=start, train_end=train_end),
        "--test-select", args.test_select.format(start=start, end=end),
        "--seed", str(resolved_seed),
        "--generations", str(args.generations),
        "--save", checkpoint,
        "--save-every", str(args.save_every),
        "--run-manifest", manifest,
        "--json-out", result,
        "--progress",
        # #211: the engine's development log, one file per window (the default
        # rcsr.dev.ndjson next to --json-out would be shared by every window).
        "--dev-log", dev_log,
        "--dev-log-level", "summary",
        "--gpu" if args.backend == "gpu" else "--cpu",
    ]
    if args.time_budget:
        cmd += ["--time-budget", str(args.time_budget)]
    if args.resume:
        cmd += ["--resume", checkpoint]
    if args.allow_backend_change:
        cmd += ["--allow-backend-change"]
    return cmd, checkpoint, manifest, result, log, dev_log, command_file, resolved_seed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rcsr", default="rcsr", help="path to the rcsr binary")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--vars", required=True)
    parser.add_argument("--time", default="ts")
    parser.add_argument(
        "--train-select",
        default="ts >= 0 and ts < {train_end}",
        help="expression template with {start} and {train_end}",
    )
    parser.add_argument(
        "--test-select",
        default="ts >= {start} and ts < {end}",
        help="expression template with {start} and {end}",
    )
    parser.add_argument(
        "--label-horizon-seconds",
        type=int,
        default=86400,
        help="rows/label look-ahead removed before the test window",
    )
    parser.add_argument("--backend", choices=("gpu", "cpu"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=1_000_000)
    parser.add_argument("--time-budget", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument(
        "--window",
        action="append",
        type=parse_window,
        required=True,
        metavar="LABEL,START,END",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-backend-change", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    failed = []
    for label, start, end in args.window:
        cmd, ck, manifest, result, log, dev_log, command_file, seed = build_command(
            args, label, start, end, args.out_dir
        )
        line = "# resolved_seed={} label={} backend={}\n{}".format(
            seed, label, args.backend, shlex.join(cmd)
        )
        with open(command_file, "w") as fh:
            fh.write(line + "\n")
        print(line, flush=True)
        if args.dry_run:
            continue
        # Keep every byte of output; never redirect to /dev/null.
        with open(log, "a") as err:
            rc = subprocess.run(cmd, stdout=err, stderr=err).returncode
        if rc != 0:
            failed.append((label, rc))
            print(f"error: {label} exited {rc}; see {log}", file=sys.stderr)
        else:
            print(f"# result: {result}", flush=True)
            print(f"# manifest: {manifest}", flush=True)
    if failed:
        for label, rc in failed:
            print(f"failed: {label} rc={rc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
