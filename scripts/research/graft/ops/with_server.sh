#!/bin/bash
# usage: with_server.sh <gpu> <model_key> <port> <mem> [--prefix-caching] -- <command ...>
# Start a vLLM generation server (graft_genserver) for one model on one GPU, wait until it
# is ready, run the command with GEN=127.0.0.1:<port> exported, then shut the server down
# and make sure its engine child processes are gone (an orphaned VLLM::EngineCore once held
# 30 GB). Only processes started here are signalled.
R=/media/lenovo/data2/promptwitness-graft-runtime
VV=/media/lenovo/data2/graft-vllm/venv/bin/python
GPU=$1; KEY=$2; PORT=$3; MEM=$4; shift 4
EXTRA=""
if [ "$1" = "--prefix-caching" ]; then EXTRA="--prefix-caching"; shift; fi
[ "$1" = "--" ] && shift
case $KEY in
  llama3) M=/media/lenovo/data2/graft-hf-cache/models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/53346005fb0ef11d3b6a83b12c895cca40156b6c;;
  gemma2) M=/media/lenovo/data2/graft-hf-cache/models--unsloth--gemma-2-9b-it/snapshots/fc7d4737cda11c3a19af2b722319e846670b4d89;;
  qwen3) M=/media/lenovo/data2/graft-hf-cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218;;
  *) echo "unknown model $KEY"; exit 2;;
esac
cd /media/lenovo/data2/promptwitness-graft
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD/src:$PWD/scripts/research/graft MODEL=$M GEN=127.0.0.1:$PORT
mkdir -p $R/cc/logs
SLOG=$R/cc/logs/genserver-$KEY-gpu$GPU-$PORT.log
setsid $VV -m promptwitness.graft_genserver --model-path $M --port $PORT --gpu-memory-utilization $MEM \
  --max-model-len ${MAXLEN:-4096} ${MAXSEQS:+--max-num-seqs $MAXSEQS} $EXTRA > $SLOG 2>&1 &
SERVER=$!
for i in $(seq 200); do grep -q READY $SLOG && break; kill -0 $SERVER 2>/dev/null || break; sleep 3; done
if ! grep -q READY $SLOG; then
  echo "SERVER FAILED $KEY gpu$GPU"; kill -- -$SERVER 2>/dev/null; exit 1
fi
echo "SERVER READY $KEY gpu$GPU port $PORT $EXTRA $(date -u +%FT%TZ)"
"$@"
CODE=$?
$VV -c "from promptwitness.graft_genserver import GenClient; GenClient('127.0.0.1:$PORT').shutdown()" >/dev/null 2>&1
for i in $(seq 40); do kill -0 $SERVER 2>/dev/null || break; sleep 3; done
# The server ran in its own session (setsid): signal that whole group, engine child included.
kill -- -$SERVER 2>/dev/null
sleep 5
kill -9 -- -$SERVER 2>/dev/null
echo "SERVER DOWN $KEY gpu$GPU $(date -u +%FT%TZ)"
exit $CODE
