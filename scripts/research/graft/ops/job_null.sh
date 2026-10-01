#!/bin/bash
# usage: job_null.sh <record> <suffix> [reasoning budget, default 384] [extra reads, default 2]   (inside with_server.sh)
# Same-prompt null only: read the record's incumbent three times on its held-out questions.
R=/media/lenovo/data2/promptwitness-graft-runtime
V=/media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python
N=$(basename "$1" .json)-null-$2
mkdir -p $R/reroll2
$V scripts/research/graft/reroll_study.py "$1" --model-path $MODEL --data-dir $R/data/greater-42a22d9 \
  --gen-server $GEN --null-reads ${4:-2} --null-only --max-new-tokens ${3:-384} --output $R/reroll2/$N.json > $R/reroll2/$N.log 2>&1
echo "NULL $N exit $? $(date -u +%FT%TZ)"
grep -h "Maximum concurrency" $R/cc/logs/genserver-*-$(echo $GEN | cut -d: -f2).log 2>/dev/null | tail -n 1 | cut -c1-160
