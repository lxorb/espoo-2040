import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon
from sim import City

from .core import Net
from .policy import Policy, decode, jmask, joints
from .train import OUT, ROOT, play, version


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
    return c.score(), sum(r["delay"] for r in c.recs)


def job(spec, seed, npz, n, depth):
    name, _, maint = spec.partition(":")
    f = OUT / "cache" / version(False) / f"{spec}-{seed}.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        r = play(seed, Net.npz(npz), "search", n, depth)[1:] if name == "search" else baseline(name, seed, npz, int(maint) if maint else None)
        f.write_text(json.dumps(dict(delay=r[1])))
    return json.loads(f.read_text())


def ci(x, rng):
    m = rng.choice(x, (10000, len(x))).mean(1)
    return [float(np.mean(x)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="deploy,planner,reactive:2,greedy1:2,greedy5:2,random,nothing")
    ap.add_argument("--seeds", default="0-47")
    ap.add_argument("--npz", default=str(ROOT / "rl" / "policy.npz"))
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "evidence" / "test.json"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    norms = json.loads((ROOT / "sim" / "norms.json").read_text())
    seeds, pols = list(range(lo, hi + 1)), a.policies.split(",")
    os.environ["NUMBA_NUM_THREADS"] = "1"
    jobs = [(p, s) for p in pols + ["nothing"] * ("nothing" not in pols) for s in seeds]
    with ProcessPoolExecutor(a.workers, get_context("spawn")) as pool:
        res = dict(zip(jobs, pool.map(job, *zip(*jobs), *([x] * len(jobs) for x in (a.npz, a.sims, a.depth)))))
    out = dict(meta=dict(script="rl/eval.py", command="python -m rl.eval " + " ".join(sys.argv[1:]), seeds=a.seeds, version=version(),
                         norms=dict(file="sim/norms.json", seeds=sorted(int(k) for k in norms)), date=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), machine=platform.processor() or platform.machine(),
                         cpu=next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), ""),
                         saved_frac="1 - delay / delay of do-nothing on the same seed", ci="95% bootstrap over seeds, 10000 resamples, rng 0", score_clip="score clipped to [0, 1] per seed",
                         tests="paired vs deploy on saved_frac: one-sided sign test and Wilcoxon signed-rank, deploy better"))
    for part, ss in (("validation", [s for s in seeds if s < 16]), ("test", [s for s in seeds if s >= 16])):
        if not ss:
            continue
        rows = {p: {s: dict(delay=res[p, s]["delay"], saved_frac=1 - res[p, s]["delay"] / res["nothing", s]["delay"],
                            score=(norms[str(s)][0] - res[p, s]["delay"]) / (norms[str(s)][0] - norms[str(s)][1]) if str(s) in norms else None)
                    for s in ss} for p in pols}
        st = {}
        for p in pols:
            v = np.array([rows[p][s]["saved_frac"] for s in ss])
            sc = [rows[p][s]["score"] for s in ss]
            st[p] = dict(saved_frac=ci(v, np.random.default_rng(0)), score=ci(np.array(sc), np.random.default_rng(0)) if None not in sc else None,
                         score_clip=ci(np.clip(sc, 0, 1), np.random.default_rng(0)) if None not in sc else None,
                         delay_m=float(np.mean([rows[p][s]["delay"] for s in ss]) / 1e6))
            if p != "deploy" and "deploy" in pols:
                d = np.array([rows["deploy"][s]["saved_frac"] for s in ss]) - v
                w = int((d > 0).sum())
                st[p].update(deploy_wins=w, ties=int((d == 0).sum()), sign_p=float(binomtest(w, int((d != 0).sum()), alternative="greater").pvalue) if (d != 0).any() else None,
                             wilcoxon_p=float(wilcoxon(d, alternative="greater").pvalue) if (d != 0).any() else None)
            print(part, f"{p:11s}", " ".join(f"{k}={v}" for k, v in st[p].items()), flush=True)
        out[part] = dict(policies=rows, stats=st)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
