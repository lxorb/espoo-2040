#!/bin/bash
set -e
H=${H:-espoo-vm2}
cmd=$1
shift || true
case "$cmd" in
start)
  ~/espoo-ops/sync.sh $H
  ssh $H "bash -s -- $*" <<'EOF'
set -e
cd ~/espoo
V=$(.venv/bin/python -c "from rl.train import version; print(version())")
D=~/rlrun/$V
mkdir -p $D/runs/rl
rsync -a --exclude .venv --exclude runs ~/espoo/ $D/
mkdir -p $D/runs/expert && rsync -a ~/espoo/runs/expert/ $D/runs/expert/ 2>/dev/null || true
ln -sfn ~/espoo/.venv $D/.venv
cd $D
export PYTHONUNBUFFERED=1 RL_EXPERT=~/espoo/runs/expert
ulimit -n 65536
nohup .venv/bin/python -m rl.train "$@" >> runs/rl/train-$(date +%H%M%S).out 2>&1 &
echo $D
EOF
  ;;
ppo|eval)
  ssh $H "bash -s -- $cmd $*" <<'EOF'
m=$1
shift
cd ~/rlrun/$(cd ~/espoo && .venv/bin/python -c "from rl.train import version; print(version())")
ulimit -n 65536
PYTHONUNBUFFERED=1 nohup .venv/bin/python -m rl.$m "$@" >> runs/rl/$m-$(date +%H%M%S).out 2>&1 &
EOF
  ;;
code)
  V=$(ssh $H 'cd ~/espoo && .venv/bin/python -c "from rl.train import version; print(version())"')
  rsync -a --exclude __pycache__ ~/espoo/rl/ $H:rlrun/$V/rl/
  ;;
stop)
  ssh $H 'pkill -f "[p]ython -m rl\.(train|ppo|eval)"; pkill -f "[r]lrun/.*multiprocessing"; true'
  ;;
pull)
  mkdir -p ~/espoo/runs/rl/vm
  rsync -az --prune-empty-dirs --include '*/' --include '*.json' --include '*.npz' --include '*.out' --exclude '*' $H:rlrun/ ~/espoo/runs/rl/vm/
  ;;
esac
