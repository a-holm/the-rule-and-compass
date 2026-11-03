#!/usr/bin/env bash
# Exit 0 while the desktop needs priority: a manual STREAMING flag, or the GPU
# video decoder or encoder in use (watching or sending a stream).
[ -e /mnt/lager/tmp/opencode/STREAMING ] && exit 0
read -r dec enc < <(nvidia-smi --query-gpu=utilization.decoder,encoder.stats.sessionCount --format=csv,noheader,nounits 2>/dev/null | tr -d ',')
[ "${dec:-0}" -gt 0 ] 2>/dev/null && exit 0
[ "${enc:-0}" -gt 0 ] 2>/dev/null && exit 0
exit 1
