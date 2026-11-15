#!/bin/bash
# 4B think+anon: resume the 30-step run to 50 steps (same budget as the 1B cells), hard stop at DEADLINE, then test seeds.
cd "$(dirname "$0")/.."
export PYTHONPATH=. VLLM_USE_FLASHINFER_SAMPLER=0
DEADLINE=${DEADLINE:-"04:12"}
while pgrep -f "name 4b-think-anon --size" > /dev/null; do sleep 5; done
.venv/bin/python -m grpo.reason --name 4b-think-anon --size 4b --think --anon --steps 50 --every 5 --prompts 8 --g 8 --lr 5e-5 --workers 28 --mem 0.25 --mb 1 --seed 0 >> runs/logs/4b-think-anon.log 2>&1 &
P=$!
while kill -0 $P 2>/dev/null; do
  [ "$(date +%H:%M)" \> "$DEADLINE" ] && { echo "deadline: stopping 4B at $(date)"; kill $P; sleep 10; pkill -f "VLLM::EngineCore"; }
  sleep 10
done
.venv/bin/python -m grpo.reason_eval test --runs 4b-think-anon --workers 28 --mem 0.3 >> runs/logs/test.log 2>&1
echo PHASE3 DONE
