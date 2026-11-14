#!/bin/bash
set -e
H=${H:-espoo-vm3}
R=${R:-main}
O=${O:-runs/grpo}
case "$1" in
start)
  shift
  ~/espoo-ops/sync.sh $H
  ssh $H "mkdir -p ~/grporun/$R && rsync -a --exclude .venv --exclude runs ~/espoo/ ~/grporun/$R/ && ln -sfn ~/espoo/.venv ~/grporun/$R/.venv && cd ~/grporun/$R && mkdir -p $O && (GRPO_OUT=$O PYTHONUNBUFFERED=1 nohup .venv/bin/python -m grpo.train $* >> $O/train.out 2>&1 &)"
  ;;
eval)
  shift
  ssh $H "cd ~/grporun/$R && (GRPO_OUT=$O PYTHONUNBUFFERED=1 nohup .venv/bin/python -m grpo.eval $* >> $O/eval.out 2>&1 &)"
  ;;
stop)
  ssh $H 'pkill -f "[p]ython -m grpo\.(train|eval)"; true'
  ;;
pull)
  mkdir -p ~/espoo/$O
  rsync -az $H:grporun/$R/$O/ ~/espoo/$O/
  ;;
esac
