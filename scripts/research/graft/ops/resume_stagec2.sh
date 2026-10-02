#!/bin/bash
# usage: resume_stagec2.sh <plan.json>   (GPU-1 queue line)
# Release the runs deferred by defer_stage.py, then run the Stage C v2 plan on GPU 1 with the
# in-process (deterministic) vLLM engine.
R=/media/lenovo/data2/promptwitness-graft-runtime
python3 $R/cc/defer_stage.py "$1" $R/stageC2/runs release
INPROC=1 bash $R/stage_gpu.sh 1 llama3 "$1" $R/stageC2/runs 47600
