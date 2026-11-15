#!/bin/bash
# After the two think cells finish (GPU memory): no-think cells and a 100-step continuation of think+anon,
# then test-seed evaluation and zero-shot base models. Resumable.
cd "$(dirname "$0")/.."
export PYTHONPATH=. VLLM_USE_FLASHINFER_SAMPLER=0
while pgrep -f "name 1b-think" > /dev/null; do sleep 10; done
C="--size 1b --every 5 --prompts 8 --g 8 --lr 5e-5 --seed 0"
[ -d runs/reason/1b-think-anon-long ] || { cp -r runs/reason/1b-think-anon runs/reason/1b-think-anon-long; rm -rf runs/reason/1b-think-anon-long/cur; }
nohup .venv/bin/python -m grpo.reason --name 1b-think-anon-long --think --anon $C --steps 100 --workers 10 --mem 0.12 --mb 2 >> runs/logs/1b-think-anon-long.log 2>&1 &
sleep 50
nohup .venv/bin/python -m grpo.reason --name 1b-nothink-anon  --anon $C --steps 50 --workers 9 --mem 0.10 --mb 8 >> runs/logs/1b-nothink-anon.log 2>&1 &
sleep 50
nohup .venv/bin/python -m grpo.reason --name 1b-nothink-named        $C --steps 50 --workers 9 --mem 0.10 --mb 8 >> runs/logs/1b-nothink-named.log 2>&1 &
sleep 30
while pgrep -f "name 1b-nothink" > /dev/null; do sleep 15; done
.venv/bin/python -m grpo.reason_eval test --runs 1b-think-anon 1b-think-named 1b-nothink-anon 1b-nothink-named --workers 18 --mem 0.12 >> runs/logs/test.log 2>&1
.venv/bin/python -m grpo.reason_eval zero --sizes 4b 1b --workers 18 --mem 0.15 >> runs/logs/zero.log 2>&1
while pgrep -f "name 1b-think-anon-long" > /dev/null; do sleep 15; done
.venv/bin/python -m grpo.reason_eval test --runs 1b-think-anon-long --workers 28 --mem 0.12 >> runs/logs/test.log 2>&1
echo PHASE2 DONE
