#!/bin/bash
# Recover tnWF sweep data from Modal volume tnwf-results, paced to avoid
# the VolumeListFiles rate limit (which surfaces as a misleading
# "Is a directory" CLI error).

set -e
cd "$(dirname "$0")/.."

VOL=tnwf-results
DELAY=8  # seconds between top-level downloads

for top in "$@"; do
  local_dst="results/$top"
  if [ -d "$local_dst" ] && [ "$(find "$local_dst" -type f | wc -l | tr -d ' ')" -gt 0 ]; then
    echo "== /$top already populated ($(find "$local_dst" -type f | wc -l) files), skipping"
    continue
  fi
  rm -rf "$local_dst"
  echo "== /$top → $local_dst"
  python3 -m modal volume get "$VOL" "/$top" "$local_dst" 2>&1 \
    | tail -1
  echo "    files: $(find "$local_dst" -type f 2>/dev/null | wc -l | tr -d ' '), $(du -sh "$local_dst" 2>/dev/null | awk '{print $1}')"
  sleep $DELAY
done
