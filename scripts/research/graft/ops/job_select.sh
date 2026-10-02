#!/bin/bash
# usage: job_select.sh <model_key> <runs-dir> <output.json>   (inside with_server.sh with INPROC=1 MAXLEN=6144: deterministic in-process engine)
# Dev selection among checkpoint incumbents and one test read per search run, deterministic server,
# 4,096-token budget. Resumable: runs already in the output are skipped.
R=/media/lenovo/data2/promptwitness-graft-runtime
V=/media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python
RUNS=$(ls $2/$1-*.json 2>/dev/null)
[ -z "$RUNS" ] && { echo "SELECT $1 no runs"; exit 0; }
$V scripts/research/graft/select_and_test.py $RUNS --data-dir $R/data/greater-42a22d9 --gen-server $GEN \
  --max-new-tokens 4096 --output $3 >> ${3%.json}.log 2>&1
echo "SELECT $1 $(basename $2) exit $? $(date -u +%FT%TZ)"
