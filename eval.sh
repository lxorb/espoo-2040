#!/bin/bash
MODEL=${MODEL:-openai/gpt-6.1-sol}
cd "$(dirname "$0")"
for s in task/environment/server task/tests; do
  rm -rf $s/sim $s/data
  mkdir -p $s/sim $s/data/out
  cp sim/*.py $s/sim/
  cp sim/norms.json $s/sim/ 2>/dev/null
  cp data/out/{net,zones}.json data/out/{od,truck,pt}.npy $s/data/out/
  cp data/catalogue.json $s/data/
done
cp rl/policy.py rl/policy.npz rl/places.json task/solution/ 2>/dev/null
[ "$1" = sync ] && exit
id=$(date +%m%d-%H%M)-s${SEED:-0}
h() { harbor run -y -p task --job-name "$id-$1" "${@:2}"; }
h reference -a oracle
h agent -a opencode -m "$MODEL" -k 3 -n 3
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
