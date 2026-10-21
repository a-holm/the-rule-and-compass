#!/usr/bin/env python3
"""Sample the RSS and thread count of a child process (issue #123).

This is the measurement harness for the `rcsr prep` peak-RSS claim. It runs the
command after `--`, polls `/proc/<pid>/status` every interval, and writes one
CSV row per sample:

    elapsed_s,vmrss_kib,vmhwm_kib,threads

The thread count identifies the pipeline phases without instrumenting the
binary: parse/resample/features run with at most `--jobs` workers, while the
final assembly pass is single-threaded. The summary is printed as JSON so the
peak (`ru_maxrss * 1024` bytes) and the store size can be recorded together.

Usage:
    scripts/prep_rss.py --csv /tmp/prep-rss.csv --interval 0.025 -- \
        target/release/rcsr prep --input raw_15m ... --out panel.rcsb

Exit status is the child's. The script is deliberately stdlib-only.
"""

import argparse
import json
import os
import subprocess
import sys
import time


def read_status(pid):
    """Return (VmRSS KiB, VmHWM KiB, Threads) or None if the process is gone."""
    try:
        with open(f"/proc/{pid}/status", "r", encoding="ascii", errors="replace") as f:
            rss = hwm = threads = None
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1])
                elif line.startswith("Threads:"):
                    threads = int(line.split()[1])
            if rss is None or hwm is None or threads is None:
                return None
            return rss, hwm, threads
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", help="write the per-sample CSV here")
    ap.add_argument("--interval", type=float, default=0.025)
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="command to run, after --")
    args = ap.parse_args()

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("nothing to run; put the command after --")

    started = time.monotonic()
    child = subprocess.Popen(cmd)
    samples = []
    peak_rss = peak_hwm = peak_threads = 0
    csv_file = open(args.csv, "w", encoding="ascii") if args.csv else None
    if csv_file:
        csv_file.write("elapsed_s,vmrss_kib,vmhwm_kib,threads\n")
    try:
        while True:
            status = read_status(child.pid)
            if status is not None:
                elapsed = time.monotonic() - started
                rss, hwm, threads = status
                samples.append((elapsed, rss, hwm, threads))
                peak_rss = max(peak_rss, rss)
                peak_hwm = max(peak_hwm, hwm)
                peak_threads = max(peak_threads, threads)
                if csv_file:
                    csv_file.write(f"{elapsed:.3f},{rss},{hwm},{threads}\n")
                    csv_file.flush()
            if child.poll() is not None:
                break
            time.sleep(args.interval)
    finally:
        if csv_file:
            csv_file.close()

    code = child.wait()
    wall = time.monotonic() - started
    # VmHWM is already the kernel's high-water mark; keep the sampled maximum
    # too so a sample taken just before exit is not lost.
    summary = {
        "command": cmd,
        "exit_code": code,
        "wall_s": round(wall, 3),
        "samples": len(samples),
        "interval_s": args.interval,
        "peak_rss_kib_sampled": peak_rss,
        "peak_vmhwm_kib": peak_hwm,
        "peak_threads": peak_threads,
        "peak_rss_bytes": peak_hwm * 1024,
        "csv": args.csv,
    }
    print(json.dumps(summary, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
