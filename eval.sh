#!/bin/bash
# Usage: SEED=0 MODEL=provider/model ./eval.sh   (or ./eval.sh sync to only refresh the task copies)
set -uo pipefail
MODEL=${MODEL:-openai/gpt-6.1-sol}
cd "$(dirname "$0")"
for f in sim/norms.json data/out/net.json data/out/zones.json data/out/od.npy data/out/truck.npy data/out/pt.npy data/catalogue.json rl/policy.py rl/policy.npz rl/places.json; do
  [ -f "$f" ] || { echo "eval.sh: missing $f (needs a full clone); task copies left untouched" >&2; exit 1; }
done
set -e
for s in task/environment/server task/tests; do
  rm -rf $s/sim $s/data
  mkdir -p $s/sim $s/data/out
  cp sim/*.py $s/sim/
  cp sim/norms.json $s/sim/
  cp data/out/{net,zones}.json data/out/{od,truck,pt}.npy $s/data/out/
  cp data/catalogue.json $s/data/
done
cp rl/policy.py rl/policy.npz rl/places.json task/solution/
set +e
[ "${1:-}" = sync ] && exit 0
id=$(date +%m%d-%H%M)-s${SEED:-0}
h() { harbor run -y -p task --job-name "$id-$1" "${@:2}"; }
h reference -a oracle
h agent -a opencode -m "$MODEL" -k 3 -n "${N:-1}"
h nothing -a nop
MODE=early h early -a oracle
python3 - "$id" <<'PY'
import json
import sys
from pathlib import Path

Path("runs").mkdir(exist_ok=True)
for p in sorted(Path("jobs").glob(f"{sys.argv[1]}-*/*/result.json"), key=lambda p: p.stat().st_mtime):
    r = json.loads(p.read_text())
    reward = (r.get("verifier_result") or {}).get("rewards", {}).get("reward")
    who = p.parts[1][len(sys.argv[1]) + 1:]
    log = p.parent / "verifier" / "run.json"
    if log.exists():
        Path(f"runs/{who}-{p.parent.name}.json").write_text(json.dumps({**json.loads(log.read_text()), "who": who}))
    out = (p.parent / "verifier" / "test-stdout.txt").read_text().splitlines()[0] if log.exists() else ""
    print(f"{who:10} {reward}  {out}")
PY
