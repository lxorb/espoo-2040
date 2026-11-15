#!/bin/bash
# Hard stop for the 4B seed-1 run at DEADLINE (the hour granted for this session); phase4.sh then evaluates its last checkpoint.
DEADLINE=${DEADLINE:-"04:57"}
while pgrep -f "name 4b-think-anon-s1 --size" > /dev/null; do
  if [ "$(date +%H:%M)" \> "$DEADLINE" ]; then
    echo "deadline: stopping 4B seed 1 at $(date)"
    pkill -f "name 4b-think-anon-s1 --size"; sleep 15; pkill -f "VLLM::EngineCore"
    break
  fi
  sleep 10
done
