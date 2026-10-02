#!/bin/bash
# usage: stage_gpu.sh <gpu> <model_key> <plan.json> <out_dir> <port> [server_mem]
# One GPU, one model: start a vLLM generation server, run that model's plan entries
# sequentially (HF scores, the server generates), then stop the server.
# INPROC=1 runs the vLLM engine core in the server process (graft_genserver --inproc), which makes
# repeated generation token-identical (verified by same-prompt re-reads, GRAFT_STATUS item 56).
cd /media/lenovo/data2/promptwitness-graft
V=/media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python
VV=/media/lenovo/data2/graft-vllm/venv/bin/python
R=/media/lenovo/data2/promptwitness-graft-runtime
GPU=$1; KEY=$2; PLAN=$3; OUT=$4; PORT=$5; MEM=${6:-0.40}
case $KEY in
  llama3) M=/media/lenovo/data2/graft-hf-cache/models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/53346005fb0ef11d3b6a83b12c895cca40156b6c;;
  gemma2) M=/media/lenovo/data2/graft-hf-cache/models--unsloth--gemma-2-9b-it/snapshots/fc7d4737cda11c3a19af2b722319e846670b4d89;;
  qwen3) M=/media/lenovo/data2/graft-hf-cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218;;
esac
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD/src
mkdir -p $OUT
LOG=$OUT/genserver-$KEY-gpu$GPU.log
if [ "$MEM" = "0" ]; then  # HF-only (Gemma-2: weights + eager attention leave no room for a server)
  echo "HF-ONLY $KEY gpu$GPU $(date -u +%FT%TZ)"
  $V scripts/research/graft/stage_queue.py $PLAN --out $OUT --data-dir $R/data/greater-42a22d9 --model $KEY
  echo "STAGE DONE $KEY gpu$GPU $(date -u +%FT%TZ)"; exit 0
fi
$VV -m promptwitness.graft_genserver --model-path $M --port $PORT --gpu-memory-utilization $MEM ${INPROC:+--inproc} > $LOG 2>&1 &
SERVER=$!
for i in $(seq 200); do grep -q READY $LOG && break; kill -0 $SERVER 2>/dev/null || break; sleep 3; done
if ! grep -q READY $LOG; then echo "SERVER FAILED $KEY gpu$GPU"; exit 1; fi
echo "SERVER READY $KEY gpu$GPU $(date -u +%FT%TZ)"
$V scripts/research/graft/stage_queue.py $PLAN --out $OUT --data-dir $R/data/greater-42a22d9 --model $KEY --gen-server 127.0.0.1:$PORT
$VV -c "from promptwitness.graft_genserver import GenClient; GenClient('127.0.0.1:$PORT').shutdown()" || kill $SERVER
wait $SERVER
echo "STAGE DONE $KEY gpu$GPU $(date -u +%FT%TZ)"
