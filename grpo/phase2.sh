#!/bin/bash
# After the two think cells finish (GPU memory): no-think cells, 4B zero-shot, test-seed evaluation of the think cells.
cd "$(dirname "$0")/.."
export PYTHONPATH=. VLLM_USE_FLASHINFER_SAMPLER=0
while pgrep -f "name 1b-think" > /dev/null; do sleep 15; done
C="--size 1b --steps ${STEPS:-50} --every 5 --prompts 8 --g 8 --lr 5e-5 --workers 9 --seed 0"
nohup .venv/bin/python -m grpo.reason --name 1b-nothink-anon  --anon $C --mem 0.10 --mb 8 >> runs/logs/1b-nothink-anon.log 2>&1 &
sleep 60
nohup .venv/bin/python -m grpo.reason --name 1b-nothink-named        $C --mem 0.10 --mb 8 >> runs/logs/1b-nothink-named.log 2>&1 &
sleep 60
.venv/bin/python -m grpo.reason_eval test --runs 1b-think-anon 1b-think-named --workers 6 --mem 0.12 >> runs/logs/test.log 2>&1
.venv/bin/python -m grpo.reason_eval zero --sizes 4b 1b --workers 6 --mem 0.15 >> runs/logs/zero.log 2>&1
while pgrep -f "name 1b-nothink" > /dev/null; do sleep 15; done
.venv/bin/python -m grpo.reason_eval test --runs 1b-nothink-anon 1b-nothink-named --workers 20 --mem 0.12 >> runs/logs/test.log 2>&1
echo PHASE2 DONE
