#!/bin/bash
# Read-only status of the GRAFT runtime: GPUs, our jobs, queue log, newest outputs.
R=/media/lenovo/data2/promptwitness-graft-runtime
echo "== $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "== queue log"
tail -n ${1:-6} $R/cc/gpu1.queue.log 2>/dev/null | cut -c1-180
echo "== queue (pending)"
cut -c1-140 $R/cc/gpu1.queue 2>/dev/null
echo "== outputs"
ls -t $R/dist $R/reroll2 2>/dev/null | head -n 14
for f in $(ls -t $R/reroll2/*-null*.json 2>/dev/null | head -n 3); do
  echo "-- $f"; python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['summary'].get('null'), d['null_reads'])" $f
done
echo "== our python jobs"
ps -u lenovo -o pid,etime,args --no-headers | grep -E "is_study|reroll_study|run_graft|select_and_test|main.py|graft_genserver" | grep -v grep | cut -c1-170
