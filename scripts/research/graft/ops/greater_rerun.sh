#!/bin/bash
# usage: greater_rerun.sh <gpu> <task> [gradient|random] [steps] [rep]
# Stage F: official GReaTer (Llama-3-8B, our train split); 'random' = same loop with a random shortlist.
# 2026-10-01: the claim/skip name includes the steps suffix (a smoke run no longer claims the
# full run's name); main.py reuses the single local model as GReaTer's reasoning worker, and the
# reasoning worker regenerates 1/8 of the rows per call instead of 1/2 (memory only).
# Later the same day (memory only, see greater_logits_batch.patch): candidate logits in batches of
# 3 and freed before the next candidate; no expandable_segments (GReaTer shares CUDA tensors with
# its worker process, which expandable segments forbid); per-process GPU memory logged every 15 s;
# a failed run is archived under failed/ and its claim released, so re-queueing retries it.
# Smoke 5 (2026-10-02): the gradient worker froze no weights, so every backward pass computed and
# kept ~16 GB of unused weight gradients (worker at 27-32 GB next to the 16 GB main process);
# greater_freeze_weights.patch freezes them (one_hot.grad unchanged), memory only. Smoke 6 then
# peaked at 33 GB in total (worker 17 GB), so candidate logits are back to GReaTer's batch of 9.
G=/media/lenovo/data2/greater-official/GreaTer
R=/media/lenovo/data2/promptwitness-graft-runtime
GPU=$1; TASK=$2; SHORT=${3:-gradient}; STEPS=${4:-106}; REP=${5:-1}
NAME=llama3_${TASK}_${SHORT}
[ "$REP" = 1 ] || NAME=${NAME}_r$REP
[ "$STEPS" = 106 ] || NAME=${NAME}_steps$STEPS
# Claim the run (two GPUs may run this script): skip if done or claimed elsewhere.
[ -f $R/greater_rerun/$NAME.time.json ] && exit 0
mkdir -p $R/greater_rerun $G/data/BBH_graft
mkdir $R/greater_rerun/$NAME.claim 2>/dev/null || exit 0
# Our 50 train rows, in our seeded order, copied verbatim from GReaTer's own CSV.
PYTHONPATH=/media/lenovo/data2/promptwitness-graft/src /media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python - << PY
import csv
from pathlib import Path
from promptwitness import graft_tasks
task = "$TASK"
data = Path("$R/data/greater-42a22d9")
rows = list(csv.DictReader(open(data / "BBH" / f"{task}.csv", encoding="utf-8", newline="")))
train = graft_tasks.load_splits(data, task)["train"]
out = Path("$G/data/BBH_graft") / f"{task}.json"
with out.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=["goal", "target", "final_target"])
    writer.writeheader()
    for example in train:
        writer.writerow(rows[int(example.example_id.rsplit(":", 1)[1])])
print("wrote", out, len(train))
PY
EXTRACTOR=$(grep -F "[\"$TASK\"]=" $G/experiments/run_llama3_all.sh | head -1 | sed -E 's/^[^=]*="(.*)"$/\1/' | sed 's/\\[$]/$/g')
echo "extractor: $EXTRACTOR"
cd $G/experiments
START=$(date +%s)
echo "START greater-$NAME gpu$GPU $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
(while true; do echo "$(date -u +%T) $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' ')"; sleep 15; done) > $R/greater_rerun/$NAME.mem.log 2>&1 &
MEMLOG=$!
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 GREATER_LOGITS_BATCH=9 GREATER_GEN_FRACTION=0.125 GREATER_SHORTLIST=$SHORT CUDA_VISIBLE_DEVICES=$GPU /media/lenovo/data2/greater-official/venv/bin/python main.py \
  --config="./configs/local_llama3_1gpu.py" \
  --config.train_data="../data/BBH_graft/${TASK}.json" --config.test_data="../data/BBH_graft/${TASK}.json" \
  --config.result_prefix="$R/greater_rerun/$NAME" \
  --config.progressive_goals=True --config.stop_on_success=False --config.allow_non_ascii=False \
  --config.num_train_models=1 --config.n_train_data=50 --config.n_test_data=50 --config.n_steps=$STEPS \
  --config.test_steps=500 --config.anneal=True --config.batch_size=64 --config.topk=40 --config.topq=6 \
  --config.control_init=" proper logical reasoning and think step by step. Finally give the actual correct answer." \
  --config.extractor_text="$EXTRACTOR" --config.control_weight=0.20 --config.target_weight=1.0 \
  > $R/greater_rerun/$NAME.log 2>&1
CODE=$?
kill $MEMLOG 2>/dev/null
echo "{\"task\": \"$TASK\", \"shortlist\": \"$SHORT\", \"steps\": $STEPS, \"rep\": $REP, \"exit\": $CODE, \"seconds\": $(( $(date +%s) - START ))}" > $R/greater_rerun/$NAME.time.json
if [ $CODE -ne 0 ]; then
  F=$R/greater_rerun/failed/$NAME-$(date -u +%Y%m%dT%H%M%S)
  mkdir -p $F && mv $R/greater_rerun/$NAME.log $R/greater_rerun/$NAME.time.json $R/greater_rerun/$NAME.mem.log $F/
  rmdir $R/greater_rerun/$NAME.claim
  echo "FAILED greater-$NAME exit $CODE (archived $F) $(date -u +%FT%TZ)"; exit $CODE
fi
# Final prompt = "Use" + last reported best control (as in GReaTer's published prompts).
python3 - << PY
import json, re
text = open("$R/greater_rerun/$NAME.log", encoding="utf-8", errors="replace").read()
best = re.findall(r"Best Control:(.*)", text)
json.dump({"$TASK": ["Use" + best[-1].rstrip()] if best else []}, open("$R/greater_rerun/$NAME.prompt.json", "w"))
print("final prompt:", ("Use" + best[-1].rstrip()) if best else None)
PY
echo "DONE greater-$NAME exit $CODE $(date -u +%FT%TZ)"
