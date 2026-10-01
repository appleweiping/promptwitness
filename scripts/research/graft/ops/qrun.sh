#!/bin/bash
# usage: qrun.sh <queue-file>
# Run queued shell commands one at a time: pop the first line of the queue file (under an
# flock, so the queue can be edited while it runs), run it, log START/END with exit code.
# Exits when the queue is empty. Lines starting with '#' are skipped.
Q=$1
LOG=$Q.log
while true; do
  LINE=$(flock "$Q.lock" bash -c 'line=$(head -n 1 "$0"); [ -n "$line" ] && sed -i "1d" "$0"; printf "%s" "$line"' "$Q")
  if [ -z "$LINE" ]; then
    echo "QUEUE EMPTY $(date -u +%FT%TZ)" >> "$LOG"
    break
  fi
  case "$LINE" in \#*) continue;; esac
  echo "START $(date -u +%FT%TZ) $LINE" >> "$LOG"
  bash -c "$LINE" >> "$LOG" 2>&1
  echo "END $? $(date -u +%FT%TZ) $LINE" >> "$LOG"
done
