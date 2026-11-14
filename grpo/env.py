import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / os.environ.get("GRPO_OUT", "runs/grpo")
AUTO, SCALE = 2, 0.3
FIXED = ["olari-link", "mankkaa-link", "espoonvayla-2x2", "otaniemi-access"]
PLACES = json.loads((ROOT / "rl" / "places.json").read_text())


def near(pid):
    p = np.array(PLACES["projects"][pid])
    return min(PLACES["areas"], key=lambda a: np.linalg.norm(p - PLACES["areas"][a]))


def city(seed):
    from sim import City
    c = City(seed)
    for _ in range(AUTO):
        c.step([{"maintain": 1}])
    return c


def info(seed):
    c = city(seed)
    s = c.state()
    return dict(seed=seed, budget=s["budget"], year=s["year"], menu=[dict(id=p["id"], cost=p["cost"], years=p["years"], near=near(p["id"])) for p in c.catalogue()],
                areas=sorted(([a["name"], a["pop"] / a["pop0"] - 1, a["jobs"] / a["jobs0"] - 1] for a in s["areas"]), key=lambda a: -a[1]))


def prompt(i):
    a = "\n".join(f"{n}: pop {p:+.1%}, jobs {j:+.1%}" for n, p, j in i["areas"])
    m = "\n".join(f"{p['id']} {p['cost']:g}M {p['years']}y {p['near']}" for p in i["menu"])
    return (f"You plan road projects in Espoo until 2040. Goal: least total traffic delay until 2045. It is {i['year']}. "
            f"Budget {i['budget']:.0f} MEUR, income 40 MEUR per year, savings carry over. Each year the next projects in your list start "
            "when affordable, at most 2 per year. Roads under construction lose capacity. Maintenance level 0 spends nothing and roads decay, "
            "1 spends 4 MEUR per year, 2 spends 10 MEUR per year.\n\nGrowth since 2026, fastest first:\n" + a +
            "\n\nProjects (id, cost, build years, nearest area):\n" + m + "\n\nAnswer with one line: plan: id, id, id; maintain: k")


def parse(text, menu):
    m = re.search(r"plan:(.*?)(?:;|\n|$)\s*(?:maintain:\s*(\d))?", text)
    if not m:
        return [], 1, 1.0
    ids, bad = [], 0
    for t in (t.strip() for t in m.group(1).split(",")):
        if t in menu and t not in ids:
            ids.append(t)
        else:
            bad += bool(t)
    k = int(m.group(2)) if m.group(2) and int(m.group(2)) < 3 else None
    return ids[:8], 1 if k is None else k, 0.05 * bad + 0.05 * (k is None)


def play(seed, ids, k, tries=3):
    try:
        return run(seed, ids, k)
    except ValueError:
        if tries:
            return play(seed, ids, k, tries - 1)
        raise


def run(seed, ids, k):
    c = city(seed)
    while not c.done:
        acts = [{"maintain": k}]
        if c.state()["step"] < c.years:
            for p in [p for p in ids if p not in c.s["built"] and p not in c.s["prog"]]:
                if len(acts) > 2 or any(c.check(acts[1:] + [{"build": p}])):
                    break
                acts.append({"build": p})
        c.step(acts)
    r = c.result()
    return dict(seed=seed, plan=ids, maintain=k, score=c.score(), frac=r["saving"] / r["nothing"], saving=r["saving"] / 1e6,
                last=c.recs[-1]["delay"] / 1e6, spent=sum(x["spent"] for x in c.recs), built=list(c.s["built"]))


def reward(r, pen):
    return float(np.clip(r["frac"] / SCALE - pen, -0.5, 1.5)) if r else -0.5


class Pool:
    def __init__(self, n):
        self.ex, self.cache = ProcessPoolExecutor(n, mp_context=get_context("spawn")), {}

    def infos(self, seeds):
        return list(self.ex.map(info, seeds))

    def run(self, jobs):
        keys = [(s, tuple(ids), k) for s, ids, k in jobs]
        todo = {q: self.ex.submit(play, q[0], list(q[1]), q[2]) for q in set(keys) if q not in self.cache and q[1]}
        self.cache.update({q: f.result() for q, f in todo.items()})
        return [self.cache.get(q) for q in keys]


def refs(i, rng):
    ids = [p["id"] for p in i["menu"]]
    grow = [f"{n.split()[0].lower().replace('ä', 'a')}-access" for n, _, _ in i["areas"][:3]]
    return dict(random=[str(x) for x in rng.permutation(ids)[:6]], fixed=FIXED, growth=FIXED[:2] + grow, growfirst=grow + FIXED[:2])


def search(seed, start, n=60, rng=None):
    rng = rng or np.random.default_rng(seed)
    best, menu = play(seed, start, 1), [p["id"] for p in info(seed)["menu"]]
    for _ in range(n):
        ids = list(best["plan"])
        op = rng.integers(3)
        if op == 0 and len(ids) < 8:
            ids.insert(rng.integers(len(ids) + 1), rng.choice([p for p in menu if p not in ids]))
        elif op == 1 and len(ids) > 1:
            ids.pop(rng.integers(len(ids)))
        elif len(ids) > 1:
            a, b = rng.choice(len(ids), 2, replace=False)
            ids[a], ids[b] = ids[b], ids[a]
        r = play(seed, ids, best["maintain"] if rng.random() > 0.1 else int(rng.integers(3)))
        best = r if r["frac"] > best["frac"] else best
    return best
