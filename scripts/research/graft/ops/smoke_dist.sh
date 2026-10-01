#!/bin/bash
# Code-path smoke test of run_graft --method dist with a tiny random model (no result value).
R=/media/lenovo/data2/promptwitness-graft-runtime
V=/media/lenovo/data2/promptwitness-delta-20260926/venv-torch271/bin/python
Q=/media/lenovo/data2/graft-hf-cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
T=$R/cc/tiny-qwen3
cd /media/lenovo/data2/promptwitness-graft
export CUDA_VISIBLE_DEVICES=${1:-1} PYTHONPATH=$PWD/src:$PWD/scripts/research/graft
if [ ! -f $T/config.json ]; then
$V - <<PY
import transformers, torch
tok = transformers.AutoTokenizer.from_pretrained("$Q")
cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=64, max_position_embeddings=4096)
torch.manual_seed(0)
model = transformers.Qwen3ForCausalLM(cfg).to(torch.bfloat16)
model.generation_config = transformers.GenerationConfig.from_pretrained("$Q")
model.save_pretrained("$T"); tok.save_pretrained("$T")
print("saved tiny model")
PY
fi
for M in dist patch; do
  $V scripts/research/graft/run_graft.py --task object_counting --model-path $T --data-dir $R/data/greater-42a22d9 \
    --method $M --rounds 2 --batch 8 --k 2 --dist-rows 3 --dist-samples 4 --max-new-tokens 24 --eval-max-new-tokens 24 \
    --defer-eval --output $R/cc/smoke-$M.json > $R/cc/smoke-$M.log 2>&1
  echo "$M exit $?"; tail -n 3 $R/cc/smoke-$M.log | cut -c1-200
done
python3 -c "import json; d=json.load(open('$R/cc/smoke-dist.json')); print([ (t['round'], t.get('dist_beta'), t.get('accepted')) for t in d['trajectory']]); print({k: v for k, v in d.get('cost', {}).items() if 'dist' in k})"
