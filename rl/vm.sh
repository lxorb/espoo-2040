#!/bin/bash
set -e
case "$1" in
start)
  ~/espoo-ops/sync.sh
  ssh espoo-vm 'bash -s' <<'EOF'
set -e
V=$(cd ~/espoo && .venv/bin/python -c "from rl.env import version; print(version())")
D=~/rlrun/$V
mkdir -p $D/runs/rl
rsync -a --exclude .venv --exclude runs ~/espoo/ $D/
ln -sfn ~/espoo/.venv $D/.venv
cd $D
export PYTHONUNBUFFERED=1
.venv/bin/python -c "from rl.eval import warm; warm(range(16))"
export NUMBA_NUM_THREADS=1
nohup .venv/bin/python -m rl.train --workers 22 --episodes 660 --sims 4 --consider 4 --depth 3 --steps 50 --keep 150 --search-every 5 >> runs/rl/gumbel.out 2>&1 &
nohup .venv/bin/python -m rl.ppo --episodes 600 --every 25 >> runs/rl/ppo.out 2>&1 &
nohup .venv/bin/python -m rl.eval --policies nothing,random,greedy --workers 6 --out base >> runs/rl/base.out 2>&1 &
echo $D
EOF
  ;;
stop)
  ssh espoo-vm 'pkill -f "[p]ython -m rl\.(train|ppo|eval)" || true'
  ;;
pull)
  mkdir -p ~/espoo/runs/rl/vm
  rsync -az espoo-vm:rlrun/ ~/espoo/runs/rl/vm/ --include '*/' --include '*.json' --include '*.pt' --include '*.out' --exclude '*' --prune-empty-dirs
  ;;
esac
