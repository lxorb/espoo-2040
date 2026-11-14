#!/bin/bash
set -e
H=${H:-espoo-vm3}
D=grporun/${R:-main}
case "$1" in
start)
  shift
  ~/espoo-ops/sync.sh $H
  ssh $H "mkdir -p ~/$D && rsync -a --exclude .venv --exclude runs ~/espoo/ ~/$D/ && ln -sfn ~/espoo/.venv ~/$D/.venv && mkdir -p ~/$D/runs/grpo && cd ~/$D && (PYTHONUNBUFFERED=1 nohup .venv/bin/python -m grpo.train $* >> runs/grpo/train.out 2>&1 &)"
  ;;
stop)
  ssh $H 'pkill -f "[p]ython -m grpo\.train"; true'
  ;;
pull)
  rsync -az $H:$D/runs/grpo/ ~/espoo/runs/grpo/
  ;;
esac
