#!/bin/bash
# usage: job_reroll.sh <record> [reasoning budget, default 384]   (inside with_server.sh, INPROC=1)
# Re-roll study under the deterministic (in-process) reader: the incumbent and every edit of a
# validity record re-read on its held-out questions (per-question correctness kept), plus two
# same-prompt null reads. Outputs go to reroll3/ (reroll2/ holds the multiprocess-engine runs).
R=/media/lenovo/data2/promptwitness-graft-runtime
V=/media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python
N=$(basename "$1" .json)
mkdir -p $R/reroll3
$V scripts/research/graft/reroll_study.py "$1" --model-path $MODEL --data-dir $R/data/greater-42a22d9 \
  --gen-server $GEN --null-reads 2 --max-new-tokens ${2:-384} --output $R/reroll3/$N.json > $R/reroll3/$N.log 2>&1
echo "REROLL $N exit $? $(date -u +%FT%TZ)"
