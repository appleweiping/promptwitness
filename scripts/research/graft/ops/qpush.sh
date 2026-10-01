#!/bin/bash
# usage: qpush.sh <queue-file> front|back <command line ...>
# Add one job to an ops/qrun.sh queue under its lock.
Q=$1; WHERE=$2; shift 2
LINE="$*"
if [ "$WHERE" = front ]; then
  flock "$Q.lock" bash -c 'tmp=$(mktemp); { printf "%s\n" "$1"; cat "$0"; } > "$tmp" && cat "$tmp" > "$0" && rm -f "$tmp"' "$Q" "$LINE"
else
  flock "$Q.lock" bash -c 'printf "%s\n" "$1" >> "$0"' "$Q" "$LINE"
fi
flock "$Q.lock" cat "$Q" | cut -c1-160
