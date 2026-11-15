import json
import multiprocessing as mp
import os
import platform
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
REBUILD = ('''        s["cash"] -= P.REBUILD * cost[bad].sum()
        ev += [f"{s['segname'].get(int(g), g)} emergency rebuild" for g in ids[bad]]
        cut[bad[inv]] *= P.REBUILD_CAP
        s["cond"][np.isin(inv, fix) | bad[inv]] = 1.0''', '''        pay = bad & (np.cumsum(np.where(bad, P.REBUILD * cost, 0)) <= max(s["cash"], 0))
        s["cash"] -= P.REBUILD * cost[pay].sum()
        ev += [f"{s['segname'].get(int(g), g)} emergency rebuild" for g in ids[pay]] + [f"{s['segname'].get(int(g), g)} rebuild deferred" for g in ids[bad & ~pay]]
        cut[bad[inv]] *= P.REBUILD_CAP
        s["cond"][np.isin(inv, fix) | pay[inv]] = 1.0''')
WAVE = [('''        n, L = len(self.nodes), s["L"]
''', '''        w = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(2,)))
        rest = [i for i in range(len(self.areas)) if i not in boom]
        self.wave = (int(w.integers(*P.WAVE_YEARS)), [(self.areas[i][1], w.uniform(*gr["areas"][i]["pop"])) for i in w.choice(rest, w.integers(1, 3), replace=False)])
        n, L = len(self.nodes), s["L"]
'''), ('''        if y:
            s["fp"] *= 1 + self.rp
            s["fj"] *= 1 + self.rj''', '''        s["wave"] = s.get("wave") or y == self.wave[0] and self.who != "forecast"
        rp, rj = self.rp, self.rj
        for m, r in self.wave[1] if s["wave"] else []:
            rp, rj = (np.where(np.isin(np.arange(x.size), m), r, x) for x in (rp, rj))
        if y:
            s["fp"] *= 1 + rp
            s["fj"] *= 1 + rj''')]
TEXT = [(" gets an emergency rebuild that year, at a multiple of its resurfacing cost and with reduced capacity.",
         " gets an emergency rebuild that year, at a multiple of its resurfacing cost and with reduced capacity, if your cash covers it; otherwise the rebuild is deferred, the segment keeps decaying and keeps the reduced capacity until it is rebuilt.")]
TEXT_WAVE = [(" It shows year by year in `areas` of `get_state` (now against the base year).",
              " It shows year by year in `areas` of `get_state` (now against the base year). In one of years `rules.wave_years` one or two further areas start to grow at a boom rate; this second wave is not announced and shows only in `areas`.")]


def sub(p, pairs):
    s = p.read_text()
    for a, b in pairs:
        assert a in s, a[:60]
        s = s.replace(a, b)
    p.write_text(s)


def gen(name, wave):
    D = ROOT / name
    shutil.rmtree(D, ignore_errors=True)
    for f in ("sim", "data/out", "data/catalogue.json", "rl", "task", "tools/norms.py", "tools/v2.py", "tools/experts.py",
              *[f"evidence/gate4-s{k}" for k in range(5)]):
        (D / f).parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if (ROOT / f).is_dir() else shutil.copy)(ROOT / f, D / f, **({"ignore": shutil.ignore_patterns("__pycache__")} if (ROOT / f).is_dir() else {}))
    (D / "sim/norms.json").write_text("{}")
    sub(D / "sim/city.py", [REBUILD] + (WAVE if wave else []))
    with open(D / "sim/params.py", "a") as f:
        f.write(f"WAVE_YEARS = {(6, 10) if wave else (99, 100)}\n")
    sub(D / "task/instruction.md", TEXT + (TEXT_WAVE if wave else []))
    sub(D / "task/environment/server/server.py", [('"capacity": P.REBUILD_CAP}}', '"capacity": P.REBUILD_CAP, "deferred_if_no_cash": True}' + (', "wave_years": [6, 9]}' if wave else "}"))])
    for t in ("task/environment/server", "task/tests"):
        for x in ("sim", "data"):
            shutil.rmtree(D / t / x, ignore_errors=True)
        shutil.copytree(D / "sim", D / t / "sim", ignore=shutil.ignore_patterns("__pycache__"))
        (D / t / "data/out").mkdir(parents=True)
        for f in ("net.json", "zones.json", "od.npy", "truck.npy", "pt.npy"):
            shutil.copy(D / "data/out" / f, D / t / "data/out" / f)
        shutil.copy(D / "data/catalogue.json", D / t / "data/catalogue.json")
    for f in ("policy.py", "policy.npz", "places.json"):
        shutil.copy(D / "rl" / f, D / "task/solution" / f)


def one(a):
    seed, who = a
    sys.path[:0] = [str(ROOT / "tools")]
    import norms as N
    f = {"nothing": (lambda c: [], None), "planner": (N.planner(False), None), "learned": (N.learned(N.Policy()), None), "reactive": (N.reactive(), 1),
         "learned-m2": (lambda c, f=N.learned(N.Policy()): [a for a in f(c) if "maintain" not in a], 2), "maintain-0": (lambda c: [], 0), "maintain-2": (lambda c: [], 2)}[who]
    loss, plan, c = N.play(seed, *f)
    cash = [r["budget"] for r in c.recs]
    return dict(seed=seed, who=who, loss=loss, plan=plan, min_cash=min(cash), first_negative=next((i for i, x in enumerate(cash) if x < 0), None),
                deferred=sum(e.endswith("rebuild deferred") for r in c.recs for e in r["events"]), rebuilds=sum(e.endswith("emergency rebuild") for r in c.recs for e in r["events"]),
                score=c.score())


def measure(seeds):
    whos = ["nothing", "planner", "learned", "learned-m2", "reactive", "maintain-0", "maintain-2"]
    with mp.get_context("spawn").Pool(min(32, len(seeds) * len(whos))) as pool:
        res = pool.map(one, [(s, w) for s in seeds for w in whos], chunksize=1)
    out = {}
    for r in res:
        out.setdefault(str(r["seed"]), {})[r["who"]] = {k: v for k, v in r.items() if k not in ("seed", "who")}
    for d in out.values():
        for v in d.values():
            v["saved"] = 1 - v["loss"] / d["nothing"]["loss"]
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs/v2measure.json").write_text(json.dumps(out))
    return out


def collect(names):
    res = {n: json.loads((ROOT / ("" if n == "v1" else f"task-{n}/") / "runs/v2measure.json").read_text()) for n in names}
    meta = dict(script="tools/v2.py", command=" ".join(["python", "tools/v2.py"] + sys.argv[1:]), seeds=sorted(map(int, res["v1"])), date=time.strftime("%Y-%m-%d"),
                machine="Intel Xeon Platinum 8573C (espoo-vm3)", rules=dict(v2="emergency rebuild only if cash covers it, else deferred: the segment keeps decaying and keeps the 0.8 capacity cut; no debt from rebuilds",
                                                                                 v2e="v2 plus a second growth wave: in a year drawn from 6-9 one or two non-boom areas start growing at a boom rate (seed stream 2), hidden from forecasts until it starts"))
    for n in names[1:]:
        p = ROOT / f"task-{n}/sim/norms.json"
        if p.exists():
            nm = json.loads(p.read_text())
            for s, d in res[n].items():
                for v in d.values():
                    v["score"] = (nm[s][0] - v["loss"]) / (nm[s][0] - nm[s][1])
            meta[f"norms_{n}"] = nm
            ev = ROOT / f"task-{n}/evidence/norms.json"
            if ev.exists():
                meta[f"norms_{n}_source"] = {k: v["source"] for k, v in json.loads(ev.read_text())["seeds"].items()}
        ex = ROOT / f"task-{n}/evidence/exploits.json"
        if ex.exists():
            meta[f"exploits_{n}"] = {k: {f: v[f] for f in ("seeds", "score", "mean", "score_mean", "wins_over_ref", "ties_with_ref", "min_cash", "replay_off")} for k, v in json.loads(ex.read_text())["strategies"].items()}
    tmp = ROOT / "evidence/rules_v2.json.tmp"
    tmp.write_text(json.dumps(dict(meta=meta, variants=res)))
    os.replace(tmp, ROOT / "evidence/rules_v2.json")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        gen("task-v2", False)
        gen("task-v2e", True)
    elif cmd == "measure":
        measure(list(range(16)))
    else:
        collect(["v1", "v2", "v2e"])
