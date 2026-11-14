import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim import City

out = Path(__file__).parent / "data"
c, rows = City(1), []
while not c.done:
    c.step([{"maintain": 1}] if c.s["year"] < c.years else [])
    rows.append({a["name"]: (a["pop"] + a["jobs"]) / (a["pop0"] + a["jobs0"]) for a in c.state()["areas"]})
(out / "growth.json").write_text(json.dumps({"1": dict(index=rows, rate={n: float(c.rp[m].mean()) for n, m in c.areas})}))
booms = {}
for seed in range(16):
    c = City(seed)
    booms[seed] = [n for n, m in c.areas if c.rp[m].mean() > 0.01]
(out / "booms.json").write_text(json.dumps(booms))
