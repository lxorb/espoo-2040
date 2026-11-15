#!/bin/bash
# 4B think+anon with the strict format gate (--strict), training seeds 0 and 1 in parallel, up to 100 steps,
# hard stop of training at DEADLINE (UTC), then test seeds for the best-validation and final checkpoints.
cd "$(dirname "$0")/.."
export PYTHONPATH=. VLLM_USE_FLASHINFER_SAMPLER=0
DEADLINE=${DEADLINE:-"07:45"}
for s in 0 1; do
  .venv/bin/python -m grpo.reason --name 4b-think-anon-strict-s$s --size 4b --think --anon --strict --steps 100 --every 5 \
    --prompts 8 --g 8 --lr 5e-5 --workers 14 --mem 0.25 --mb 1 --seed $s >> runs/logs/4b-think-anon-strict-s$s.log 2>&1 &
  sleep 90
done
while pgrep -f "name 4b-think-anon-strict" > /dev/null; do
  if [ "$(date -u +%H:%M)" \> "$DEADLINE" ]; then
    echo "deadline: stopping training at $(date -u)"; pkill -f "name 4b-think-anon-strict"; sleep 15; pkill -f "VLLM::EngineCore"; break
  fi
  sleep 10
done
sleep 10
.venv/bin/python -m grpo.reason_eval test --runs 4b-think-anon-strict-s0 4b-think-anon-strict-s1 --workers 28 --mem 0.3 >> runs/logs/test.log 2>&1
echo PHASE5 DONE $(date -u)
