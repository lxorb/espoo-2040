import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from sim import City

from .core import Net
from .policy import Policy, decode, jmask, joints
from .train import OUT, ROOT, play, score, version


def rows(c, st, jam):
    def status(k):
        return f"building {st['works'][k]}" if k in st["works"] else "built" if k in st["built"] else "open"
    return [[k, q["name"], q["kind"], q["cost"], q["years"], q.get("opex", 0), status(k), jam.get(k, 0.0), []] for k, q in c.cat.items() if k in c.menu]


def footprint(c, q):
    m = np.isin(c.s["osm"], [o for e in q.get("mod", []) + q.get("works", []) for o in e["osm"]])
    pts = [g for e in q.get("new", []) for g in e["geom"]]
    return m | (np.linalg.norm(c.nodes[c.s["a"].astype(int)][:, None] - np.array(pts, float)[None], axis=2) < 1000).any(1) if pts else m


def baseline(name, seed, npz, maint):
    sys.path.insert(0, str(ROOT / "task" / "solution"))
    import plan
    c, rng = City(seed), np.random.default_rng(seed)
    menu, pol = list(c.menu), Policy(npz) if name == "deploy" else None
    proj = [dict(id=k, **{f: c.cat[k].get(f, 0) for f in ("kind", "cost", "years", "opex")}) for k in menu]
    foot = [footprint(c, c.cat[k]) for k in menu] if name == "reactive" else []
    while not c.done:
        st, acts = c.state(), []
        if st["step"] < c.years and name != "nothing":
            m, jam = np.array(c.mask(), bool), {q["id"]: q["jam"] for q in c.catalogue()}
            if name == "random":
                acts = decode(joints(len(menu)), menu, int(rng.choice(np.flatnonzero(jmask(joints(len(menu)), m)))))
            elif name == "deploy":
                acts = pol(st, [dict(q, jam=jam.get(q["id"], 0.0)) for q in proj], m)
            elif name == "planner":
                acts = plan.decide(dict(st, startable=[k for k, ok in zip(menu, m) if ok]), rows(c, st, jam))
            elif name == "reactive":
                r, n = c.recs[-1] if c.recs else None, foot[0].size
                late = r and r["flow"][:n] * np.maximum(r["time"][:n] - c._t0()[:n] * 60, 0)
                ok = [i for i, v in enumerate(m) if v]
                acts = [{"build": menu[max(ok, key=lambda i: late[foot[i]].sum())]}] if ok and r else []
            else:
                la, gain = 0 if name == "greedy1" else 5, []
                for k in (k for k, v in zip(menu, m) if v):
                    ref, d = ([r["delay"] for r in c.forecast(p, c.cat[k]["years"] + 1 + la)] for p in ({}, {0: [{"build": k}]}))
                    gain.append(((ref[-1] - d[-1] if la == 0 else sum(ref) - sum(d)) / max(c.cat[k]["cost"], 0.1), k))
                for g, k in sorted(gain)[::-1]:
                    if g > 0 and len(acts) < 2 and not any(c.check(acts + [{"build": k}])):
                        acts.append({"build": k})
            if maint is not None and name != "deploy":
                acts = [a for a in acts if "maintain" not in a] + [{"maintain": maint}]
        c.step(acts)
    c.log(f"rl-{name}-{seed}")
    return score(c), sum(r["delay"] for r in c.recs)


def job(name, seed, sd, npz, n, depth, maint):
    if name == "search":
        return play(seed, sd, "search", n, depth, f"rl-search-{seed}")[1:]
    return baseline(name, seed, npz, maint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="nothing,random,greedy1,greedy5,reactive,planner,deploy,search")
    ap.add_argument("--seeds", default=",".join(map(str, range(16))))
    ap.add_argument("--npz", default=str(ROOT / "rl" / "policy.npz"))
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--maintain", type=int)
    ap.add_argument("--out", default="eval")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    seeds, sd = [int(s) for s in a.seeds.split(",")], Net.npz(a.npz)
    nothing = {s: sum(City(s).baseline()) for s in seeds}
    os.environ["NUMBA_NUM_THREADS"] = "1"
    jobs = [(p, s) for p in a.policies.split(",") for s in seeds]
    with ProcessPoolExecutor(a.workers, get_context("spawn")) as pool:
        res = list(pool.map(job, *zip(*jobs), *([x] * len(jobs) for x in (sd, a.npz, a.sims, a.depth, a.maintain))))
    out = {}
    for (p, s), (sc, d) in zip(jobs, res):
        r = out.setdefault(p, dict(seeds=[], score=[], saved_frac=[], delay=[]))
        r["seeds"].append(s), r["score"].append(sc), r["saved_frac"].append(1 - d / nothing[s]), r["delay"].append(d / 1e6)
    for p, r in out.items():
        r.update(mean=float(np.mean(r["score"])), mean_saved=float(np.mean(r["saved_frac"])))
        print(f"{p:9s} {r['mean']:+.4f} {r['mean_saved']:.4f} " + " ".join(f"{x:+.3f}" for x in r["score"]))
    (OUT / f"{a.out}-{version()}.json").write_text(json.dumps(dict(maintain=a.maintain, policies=out)))


if __name__ == "__main__":
    main()
