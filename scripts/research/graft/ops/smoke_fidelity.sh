#!/bin/bash
# Plumbing test (no result value): is_study (HF path) and trace_fidelity with the tiny model.
R=/media/lenovo/data2/promptwitness-graft-runtime
V=/media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python
T=$R/cc/tiny-qwen3
REC=$R/v3/llama3-tracking_shuffled_objects_seven_objects-dec.json
cd /media/lenovo/data2/promptwitness-graft
export CUDA_VISIBLE_DEVICES=${1:-1} PYTHONPATH=$PWD/src:$PWD/scripts/research/graft
$V scripts/research/graft/is_study.py $REC --model-path $T --data-dir $R/data/greater-42a22d9 --samples 2 \
  --fresh-samples 1 --max-new-tokens 12 --output $R/cc/smoke-dist.json > $R/cc/smoke-dist.log 2>&1
echo "is_study exit $?"; tail -n 2 $R/cc/smoke-dist.log | cut -c1-300
$V scripts/research/graft/trace_fidelity.py $REC $R/cc/smoke-dist.json --model-path $T \
  --data-dir $R/data/greater-42a22d9 --output $R/cc/smoke-onepass.json > $R/cc/smoke-onepass.log 2>&1
echo "trace_fidelity exit $?"; tail -n 14 $R/cc/smoke-onepass.log | cut -c1-200
