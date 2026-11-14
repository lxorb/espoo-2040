#!/bin/bash
set -e
H=${H:-espoo-vm2}
cmd=$1
shift || true
case "$cmd" in
run)
  ~/espoo-ops/sync.sh $H
  ssh $H "bash -s -- $*" <<'EOF'
cd ~/espoo
D=~/rlrun/$(.venv/bin/python -c "from rl.train import version; print(version())")
mkdir -p $D/runs/rl
rsync -a --exclude .venv --exclude runs ~/espoo/ $D/
ln -sfn ~/espoo/.venv $D/.venv
cd $D
ulimit -n 65536
m=$1
shift
RL_EXPERT=~/espoo/runs/expert PYTHONUNBUFFERED=1 nohup .venv/bin/python -m rl.$m "$@" >> runs/rl/$m-$(date +%H%M%S).out 2>&1 &
echo $D
EOF
  ;;
stop)
  ssh $H 'pkill -f "[p]ython -m rl\."; pkill -f "[r]lrun/.*multiprocessing"; true'
  ;;
pull)
  rsync -az --prune-empty-dirs --include '*/' --include '*.json' --include '*.npz' --include '*.out' --exclude '*' $H:rlrun/ ~/espoo/runs/rl/vm/
  ;;
esac
