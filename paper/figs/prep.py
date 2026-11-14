import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim import City

out = {}
for seed in range(5):
    c = City(seed)
    rows = []
    while not c.done:
        c.step([{"maintain": 1}] if c.s["year"] < c.years else [])
        st = c.state()
        rows.append({a["name"]: (a["pop"] + a["jobs"]) / (a["pop0"] + a["jobs0"]) for a in st["areas"]})
    rate = {n: float(c.rp[m].mean()) for n, m in c.areas}
    out[seed] = dict(index=rows, rate=rate)
(Path(__file__).parent / "data" / "growth.json").write_text(json.dumps(out))

booms = {}
for seed in range(16):
    c = City(seed)
    booms[seed] = [n for n, m in c.areas if c.rp[m].mean() > 0.01]
(Path(__file__).parent / "data" / "booms.json").write_text(json.dumps(booms))
