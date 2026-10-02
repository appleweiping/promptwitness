#!/bin/bash
# usage: job_reroll.sh <record> [<record> ...]   (inside with_server.sh, INPROC=1)
# env: BUDGET (default 384); RECHUNKS (e.g. 16,25,40,64,100: numerical re-rolls of the incumbent);
#      NULLONLY=1 (skip the edits) with SUFFIX (output name suffix).
# Re-roll study under the deterministic (in-process) reader: the incumbent and every edit of a
# validity record re-read on its held-out questions (per-question correctness kept), plus two
# same-prompt null reads. Outputs go to reroll3/ (reroll2/ holds the multiprocess-engine runs).
R=/media/lenovo/data2/promptwitness-graft-runtime
V=/media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python
mkdir -p $R/reroll3
for REC in "$@"; do
  N=$(basename "$REC" .json)$SUFFIX
  $V scripts/research/graft/reroll_study.py "$REC" --model-path $MODEL --data-dir $R/data/greater-42a22d9 \
    --gen-server $GEN --null-reads 2 --max-new-tokens ${BUDGET:-384} ${RECHUNKS:+--reroll-chunks $RECHUNKS} \
    ${NULLONLY:+--null-only} --output $R/reroll3/$N.json > $R/reroll3/$N.log 2>&1
  echo "REROLL $N exit $? $(date -u +%FT%TZ)"
done
