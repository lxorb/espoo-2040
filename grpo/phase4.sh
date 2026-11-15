#!/bin/bash
# 4B think+anon, training seeds 0 and 1 (50 steps each): after both finish, test seeds for the best-validation and final checkpoints.
cd "$(dirname "$0")/.."
export PYTHONPATH=. VLLM_USE_FLASHINFER_SAMPLER=0
while pgrep -f "name 4b-think-anon" > /dev/null; do sleep 10; done
.venv/bin/python -m grpo.reason_eval test --runs 4b-think-anon 4b-think-anon-s1 --workers 28 --mem 0.3 >> runs/logs/test.log 2>&1
echo PHASE4 DONE
