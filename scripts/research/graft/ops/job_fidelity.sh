#!/bin/bash
# usage: job_fidelity.sh <gpu>
# One-pass (superposed) likelihood ratios vs exact teacher forcing on every H-dist pool whose
# is_study output stored its traces. HF model only (no generation server).
R=/media/lenovo/data2/promptwitness-graft-runtime
V=/media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python
cd /media/lenovo/data2/promptwitness-graft
export CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PWD/src:$PWD/scripts/research/graft
mkdir -p $R/dist_onepass
for D in $R/dist/*.json; do
  N=$(basename $D .json)
  [ -f $R/dist_onepass/$N.json ] && continue
  grep -q '"traces"' $D || continue
  case $N in
    llama3-*) M=/media/lenovo/data2/graft-hf-cache/models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/53346005fb0ef11d3b6a83b12c895cca40156b6c; A=sdpa;;
    qwen3-*) M=/media/lenovo/data2/graft-hf-cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218; A=sdpa;;
    gemma2-*) M=/media/lenovo/data2/graft-hf-cache/models--unsloth--gemma-2-9b-it/snapshots/fc7d4737cda11c3a19af2b722319e846670b4d89; A=eager;;
  esac
  REC=$(ls $R/v3/$N.json $R/ts3/$N.json 2>/dev/null | head -n 1)
  $V scripts/research/graft/trace_fidelity.py $REC $D --model-path $M --data-dir $R/data/greater-42a22d9 --attn $A \
    --output $R/dist_onepass/$N.json > $R/dist_onepass/$N.log 2>&1
  echo "FIDELITY $N exit $? $(date -u +%FT%TZ)"
done
