#!/bin/bash
# usage: job_dist.sh <model_key> <null-record> <is-records...> [-- <reroll-records...>]
# Run inside with_server.sh (uses $MODEL and $GEN). Order: same-prompt null on one record
# (reader determinism), then the H-dist study on every pool, then re-roll studies with the
# same-prompt null. Existing outputs are skipped, so the job can be resumed.
R=/media/lenovo/data2/promptwitness-graft-runtime
V=/media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python
D=$R/data/greater-42a22d9
KEY=$1; NULLREC=$2; shift 2
mkdir -p $R/dist $R/reroll2
name() { basename "$1" .json; }
N=$(name $NULLREC)
if [ ! -f $R/reroll2/$N-null.json ]; then
  $V scripts/research/graft/reroll_study.py $NULLREC --model-path $MODEL --data-dir $D --gen-server $GEN \
    --null-reads 2 --null-only --output $R/reroll2/$N-null.json > $R/reroll2/$N-null.log 2>&1
  echo "NULL $N exit $? $(date -u +%FT%TZ)"
fi
REROLL=0
for REC in "$@"; do
  if [ "$REC" = "--" ]; then REROLL=1; continue; fi
  N=$(name $REC)
  if [ $REROLL = 0 ]; then
    [ -f $R/dist/$N.json ] && continue
    $V scripts/research/graft/is_study.py $REC --model-path $MODEL --data-dir $D --gen-server $GEN \
      --output $R/dist/$N.json > $R/dist/$N.log 2>&1
    echo "IS $N exit $? $(date -u +%FT%TZ)"
  else
    [ -f $R/reroll2/$N.json ] && continue
    $V scripts/research/graft/reroll_study.py $REC --model-path $MODEL --data-dir $D --gen-server $GEN \
      --null-reads 2 --output $R/reroll2/$N.json > $R/reroll2/$N.log 2>&1
    echo "REROLL $N exit $? $(date -u +%FT%TZ)"
  fi
done
