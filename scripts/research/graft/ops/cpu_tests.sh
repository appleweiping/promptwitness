#!/bin/bash
# Run the model-dependent unit tests on the server CPU (no GPU use).
cd /media/lenovo/data2/promptwitness-graft
export CUDA_VISIBLE_DEVICES= PYTHONPATH=src
export PW_TEST_TOKENIZER=/media/lenovo/data2/graft-hf-cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
timeout 3000 /media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python -m pytest \
  tests/test_superposed_gates.py tests/test_graft_dist.py tests/test_graft_eval.py -q -p no:cacheprovider --no-cov "$@" 2>&1 | tail -n 15
