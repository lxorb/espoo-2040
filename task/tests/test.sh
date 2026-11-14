#!/bin/bash
cd /app
python3 - <<'PY'
import json
import math

from sim import City

r, log = 0.0, None
try:
    st = json.load(open("/var/lib/env/state.json"))
    c = City(int(st["seed"]), int(st["years"]), int(st["runout"]))
    for a in st["plan"]:
        if not c.done:
            c.step(a)
    seen = [x["delay"] for x in c.recs]
    while not c.done:
        c.step()
    res = c.result()
    r = c.score()
    r = float(res["saving"] / res["nothing"] if r is None else r)
    log = c.log()
    off = max([abs(a - b) / max(abs(b), 1) for a, b in zip(seen, st["seen"])], default=0)
    print(json.dumps({**res, "replay_off": off}))
except Exception as e:
    print(f"error {e!r}")
r = min(1.0, max(0.0, r)) if math.isfinite(r) else 0.0
if log:
    log["score"] = r
    json.dump(log, open("/logs/verifier/run.json", "w"))
print(f"reward {r}")
json.dump({"reward": r}, open("/logs/verifier/reward.json", "w"))
PY
