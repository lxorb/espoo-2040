import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "task/solution"), str(ROOT / "rl")]
import numpy as np

from plan import decide
from policy import Policy
from sim import City
from sim import params as P

CACHE = ROOT / "runs" / "norms"


def B(k):
    return {"build": k}


def play(seed, choose, mt=None, income=None):
    inc, P.INCOME = P.INCOME, income or P.INCOME
    try:
        c = City(seed)
        while not c.done:
            y = c.s["year"]
            c.step((choose(c) + ([{"maintain": mt}] if mt is not None else [])) if y < c.years else [])
    finally:
        P.INCOME = inc
    return sum(r["delay"] for r in c.recs), sorted((y, a["build"]) for y, r in enumerate(c.recs) for a in r["actions"] if a.get("ok") and "build" in a), c


def fixed(plan):
    return lambda c: [B(k) for y, k in plan if y == c.s["year"]]


def build(order):
    def f(c):
        out = []
        for k in order:
            if k not in c.s["built"] and k not in c.s["prog"] and len(out) < 2 and all(w is None for w in c.check(out + [B(k)])):
                out.append(B(k))
        return out
    return f


def tot(c, plan, h):
    f = c.fork(True)
    f.gap = P.FAST
    return [f.step(plan.get(k, [])) for k in range(min(h, f.years + f.runout - f.s["year"]))]


def greedy(la):
    def f(c):
        m = [k for k, ok in zip(c.menu, c.mask()) if ok]
        sc, cache = [], {}
        for k in m:
            h = c.cat[k]["years"] + 1 + la
            cache.setdefault(h, [r["delay"] for r in tot(c, {}, h)])
            d = [r["delay"] for r in tot(c, {0: [B(k)]}, h)]
            g = cache[h][-1] - d[-1] if la == 0 else sum(cache[h]) - sum(d)
            sc.append((g / c.cat[k]["cost"], g, k))
        out = []
        for _, g, k in sorted(sc, reverse=True):
            if g > 0 and len(out) < 2 and all(w is None for w in c.check(out + [B(k)])):
                out.append(B(k))
        return out
    return f


def reactive():
    foot = {}

    def f(c):
        if not c.recs:
            return []
        if not foot:
            xy = c.nodes[c.s["a"].astype(int)]
            for k, p in c.cat.items():
                osm = [o for e in p.get("mod", []) + p.get("works", []) for o in e["osm"]]
                pts = [q for e in p.get("new", []) for q in e["geom"]]
                m = np.isin(c.s["osm"], osm) if osm else np.zeros(xy.shape[0], bool)
                foot[k] = m | (np.linalg.norm(xy[:, None] - np.array(pts, float)[None], axis=2) < 1000).any(1) if pts else m
        r, n = c.recs[-1], len(c.s["osm"])
        late = r["flow"][:n] * (r["time"][:n] - c._t0()[:n] * 60)
        m = [k for k, ok in zip(c.menu, c.mask()) if ok]
        return [B(max(m, key=lambda k: late[:foot[k].size][foot[k][:n]].sum()))] if m else []
    return f


def rand(seed):
    rng = np.random.default_rng(seed + 99)

    def f(c):
        out = []
        for k in rng.permutation([k for k, ok in zip(c.menu, c.mask()) if ok]):
            if rng.random() < 0.3 and len(out) < 2 and all(w is None for w in c.check(out + [B(str(k))])):
                out.append(B(str(k)))
        return out
    return f


def planner(late):
    def f(c):
        st = dict(c.state(), startable=[k for k, ok in zip(c.menu, c.mask()) if ok])
        return [a for a in decide(st, [[p["id"], p["name"], p["kind"], p["cost"], p["years"], p["opex"], "open", p["jam"]] for p in c.catalogue()], late)]
    return f


def learned(pol):
    def f(c):
        cat = {p["id"]: p for p in c.catalogue()}
        proj = [dict(id=k, kind=q["kind"], cost=q["cost"], years=q["years"], opex=q.get("opex", 0), jam=cat[k]["jam"] if k in cat else 0.0) for k, q in c.cat.items()]
        return pol(c.state(), proj, c.mask(), years=c.years)
    return f


def moves(plan, menu, rng, years=15):
    out, have = [], {k for _, k in plan}
    cnt = lambda pl, y: sum(1 for yy, _ in pl if yy == y)
    for i, (y, k) in enumerate(plan):
        rest = plan[:i] + plan[i + 1:]
        out.append(rest)
        out += [rest + [(y + d, k)] for d in (-2, -1, 1, 2) if 0 <= y + d < years and cnt(rest, y + d) < 2]
        out += [rest + [(y, str(q))] for q in rng.choice([q for q in menu if q not in have], 3, replace=False)]
    for q in rng.choice([q for q in menu if q not in have], 6, replace=False):
        out += [plan + [(int(y), str(q))] for y in rng.choice(years, 2, replace=False) if cnt(plan, int(y)) < 2]
    return [sorted(m) for m in out]


def one(seed):
    path = CACHE / f"{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    t, cand = time.time(), []
    add = lambda src, r, mt: cand.append(dict(loss=r[0], plan=[list(x) for x in r[1]], maint=mt, src=src))
    nothing = play(seed, lambda c: [])[0]
    menu = City(seed).menu
    hi = play(seed, lambda c: [], income=1e6)[0]
    val = {k: hi - play(seed, lambda c, k=k: [B(k)] if c.s["year"] == 0 else [], income=1e6)[0] for k in menu}
    o = [k for k in sorted(val, key=lambda k: -val[k]) if val[k] > 0]
    big = [k for k in o if val[k] > 3e6]
    for order in (o, o[:8], o[:6], o[:4], o[:3], big + [k for k in o if k not in big]):
        for mt in (1, 2):
            add("single-value order", play(seed, build(order), mt), mt)
    for mt in (1, 2):
        for name, f in (("random", rand(seed)), ("greedy", greedy(0)), ("greedy5", greedy(5)), ("reactive", reactive())):
            add(name, play(seed, f, mt), mt)
    for late in (False, True):
        add("planner", play(seed, planner(late)), 2)
    r = play(seed, learned(Policy()))
    ms = [a["maintain"] for x in r[2].recs[:15] for a in x["actions"] if "maintain" in a and a.get("ok")]
    add("learned policy", r, max(set(ms), key=ms.count) if ms else 1)
    rng = np.random.default_rng([1, seed])
    for _ in range(3):
        cur = min(cand, key=lambda c: c["loss"])
        for m in moves([tuple(x) for x in cur["plan"]], menu, rng) + [None]:
            mt = cur["maint"] if m is not None else 3 - cur["maint"]
            add("local search", play(seed, fixed(m if m is not None else cur["plan"]), mt), mt)
    best = min(cand, key=lambda c: c["loss"])
    res = dict(seed=seed, nothing=nothing, best=best["loss"], source=best["src"], plan=best["plan"], maint=best["maint"],
               candidates=len(cand), seconds=round(time.time() - t))
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, path)
    return res


def write(path, data):
    tmp = path.with_suffix(f".{os.getpid()}")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="16-47")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    with mp.get_context("spawn").Pool(min(a.jobs, len(seeds))) as pool:
        res = {r["seed"]: r for r in pool.imap_unordered(one, seeds)}
    norms = json.loads((ROOT / "sim/norms.json").read_text())
    if a.check:
        for s in seeds:
            f = norms.get(str(s))
            print(s, "nothing", res[s]["nothing"], f and res[s]["nothing"] == f[0], "best", res[s]["best"], f and f[1], res[s]["source"])
    else:
        new = {**{str(s): [res[s]["nothing"], res[s]["best"]] for s in seeds}, **norms}
        write(ROOT / "sim/norms.json", {k: new[k] for k in sorted(new, key=int)})
        ev = ROOT / "evidence/norms.json"
        old = json.loads(ev.read_text())["seeds"] if ev.exists() else {}
        old.update({str(s): {k: res[s][k] for k in ("nothing", "best", "source", "plan", "maint", "candidates", "seconds")} for s in seeds if str(s) not in norms})
        write(ev, dict(meta=dict(script="tools/norms.py", command=" ".join(["python", "tools/norms.py"] + sys.argv[1:]), seeds=sorted(old, key=int),
                                 date=time.strftime("%Y-%m-%d"), machine=platform.processor() or platform.machine()), seeds={k: old[k] for k in sorted(old, key=int)}))
        print({s: (round(res[s]["nothing"]), round(res[s]["best"]), res[s]["source"]) for s in seeds})
