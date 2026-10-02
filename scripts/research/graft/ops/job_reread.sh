#!/bin/bash
# usage: job_reread.sh <model_key>   (inside with_server.sh started with MAXLEN=6144)
# Plan amendment 2026-10-01: re-read every fixed prompt (ZS-CoT, GReaTer initial, GReaTer
# published; all 23 tasks) and every completed Stage C v1 run of this model with the
# deterministic generation server and a 4,096-token reasoning budget. Outputs in eval2/.
R=/media/lenovo/data2/promptwitness-graft-runtime
V=/media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python
DATA=$R/data/greater-42a22d9
PUB=$R/data/greater_published_prompts.json
KEY=$1
TASKS="boolean_expressions causal_judgement date_understanding disambiguation_qa formal_fallacies geometric_shapes hyperbaton logical_deduction_five_objects movie_recommendation multistep_arithmetic_two navigate object_counting penguins_in_a_table reasoning_about_colored_objects ruin_names salient_translation_error_detection snarks sports_understanding temporal_sequences tracking_shuffled_objects_five_objects web_of_lies gsm8k folio"
mkdir -p $R/eval2
$V scripts/research/graft/evaluate_prompts.py --engine server --gen-server $GEN --model-path $MODEL --model-key $KEY \
  --data-dir $DATA --published $PUB --tasks $TASKS --sets zs_cot greater_init greater_published \
  --max-new-tokens 4096 --reroll-chunks ${RECHUNKS:-20,34,50} --output $R/eval2/baselines_$KEY.json > $R/eval2/baselines_$KEY.log 2>&1
echo "BASELINES $KEY exit $? $(date -u +%FT%TZ)"
RUNS=$(ls $R/stageC/runs/$KEY-*.json 2>/dev/null)
if [ -n "$RUNS" ]; then
  $V scripts/research/graft/select_and_test.py $RUNS --data-dir $DATA --gen-server $GEN --max-new-tokens 4096 \
    --output $R/eval2/selected_${KEY}_v1.json > $R/eval2/selected_${KEY}_v1.log 2>&1
  echo "SELECT $KEY v1 exit $? $(date -u +%FT%TZ)"
fi
