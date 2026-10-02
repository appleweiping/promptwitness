#!/bin/bash
# usage: job_greater_eval.sh   (inside with_server.sh for llama3 with INPROC=1 MAXLEN=6144: deterministic in-process engine)
# Read the final prompt of every completed official-GReaTer rerun (gradient or random shortlist)
# with the shared deterministic reader and a 4,096-token budget. Skips runs already evaluated.
R=/media/lenovo/data2/promptwitness-graft-runtime
V=/media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python
mkdir -p $R/eval2/greater
for P in $R/greater_rerun/*.prompt.json; do
  N=$(basename $P .prompt.json)
  case $N in *_steps*) continue;; esac  # smoke runs
  OUT=$R/eval2/greater/$N.json
  [ -f $OUT ] && continue
  TASK=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(next(iter(d)) if d and next(iter(d.values())) else '')" $P)
  [ -z "$TASK" ] && { echo "GREATER-EVAL $N no prompt"; continue; }
  $V scripts/research/graft/evaluate_prompts.py --engine server --gen-server $GEN --model-path $MODEL --model-key llama3 \
    --data-dir $R/data/greater-42a22d9 --published $P --tasks $TASK --sets greater_published \
    --max-new-tokens 4096 --reroll-chunks ${RECHUNKS:-20,34,50} --output $OUT > $R/eval2/greater/$N.log 2>&1
  echo "GREATER-EVAL $N exit $? $(date -u +%FT%TZ)"
done
