#!/bin/bash
# Gemma 3 1B 2x2 factorial (think x anon), same budget per cell. Resumable: rerun the same command.
cd "$(dirname "$0")/.."
export PYTHONPATH=. VLLM_USE_FLASHINFER_SAMPLER=0
C="--size 1b --steps ${STEPS:-50} --every 5 --prompts 8 --g 8 --lr 5e-5 --workers 7 --seed 0"
mkdir -p runs/logs
nohup .venv/bin/python -m grpo.reason --name 1b-think-anon  --think --anon $C --mem 0.12 --mb 2 >> runs/logs/1b-think-anon.log 2>&1 &
sleep 90
nohup .venv/bin/python -m grpo.reason --name 1b-think-named --think        $C --mem 0.12 --mb 2 >> runs/logs/1b-think-named.log 2>&1 &
sleep 90
nohup .venv/bin/python -m grpo.reason --name 1b-nothink-anon         --anon $C --mem 0.08 --mb 8 >> runs/logs/1b-nothink-anon.log 2>&1 &
sleep 90
nohup .venv/bin/python -m grpo.reason --name 1b-nothink-named               $C --mem 0.08 --mb 8 >> runs/logs/1b-nothink-named.log 2>&1 &
