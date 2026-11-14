import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch

from sim import City
from sim import params as P

from .core import footprint, jam
from .policy import Policy, decode, jmask, joints
from .train import OUT, play, score, version


def total(c, plan, h):
    return sum(r["delay"] for r in c.forecast(plan, h))


def simple(name, seed, npz=None):
    c, rng = City(seed), np.random.default_rng(seed)
    menu = list(c.menu)
    proj = [dict(id=k, **{f: c.cat[k].get(f, 0) for f in ("kind", "cost", "years", "opex")}) for k in menu]
    J, pol = joints(len(menu)), Policy(npz, name == "deploycheck", 2 if name == "deploy2" else None) if name.startswith("deploy") else None
    n, foot = len(c.s["osm"]), [footprint(c, c.cat[k]) for k in menu]
    while not c.done:
        t, acts = c.state()["step"], []
        if t < c.years and name != "nothing":
            m = np.array(c.mask(), bool)
            if name == "random":
                acts = decode(J, menu, int(rng.choice(np.flatnonzero(jmask(J, m)))))
            elif name == "reactive":
                ok, jm = np.flatnonzero(m), jam(c, foot)
                acts = [{"build": menu[max(ok, key=lambda i: jm[i])]}] if ok.size and c.recs else []
            elif name.startswith("deploy"):
                jm = {q["id"]: q["jam"] for q in c.catalogue()}
                acts = pol(c.state(), [dict(q, jam=jm.get(q["id"], 0.0)) for q in proj], m, lambda a: total(c, {0: a}, P.FORECAST_YEARS), P.FORECASTS, c.years)
            else:
                la, gain = 0 if name == "greedy1" else 5, []
                for k in (k for k, ok in zip(menu, m) if ok):
                    h = c.cat[k]["years"] + 1 + la
                    ref, d = ([r["delay"] for r in c.forecast(pl, h)] for pl in ({}, {0: [{"build": k}]}))
                    gain.append(((ref[-1] - d[-1] if la == 0 else sum(ref) - sum(d)) / max(c.cat[k]["cost"], 0.1), k))
                gain.sort()
                for g, k in gain[::-1]:
                    if g > 0 and len(acts) < 2 and not any(c.check(acts + [{"build": k}])):
                        acts.append({"build": k})
        c.step(acts)
    c.log(f"rl-{name}-{seed}")
    return score(c)


def job(name, seed, sd, npz, n, depth):
    if name in ("net", "search", "expert"):
        return play(seed, sd, dict(search="searcheval").get(name, name), n, depth, f"rl-{name}-{seed}")[1]
    return simple(name, seed, npz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="nothing,random,greedy1,greedy5,reactive,net,search,deploy,deploycheck")
    ap.add_argument("--seeds", default=",".join(map(str, range(16))))
    ap.add_argument("--name", default="gumbel")
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()
    ck = Path(a.ckpt) if a.ckpt else OUT / f"{a.name}-{version()}.pt"
    sd = {k: v.numpy() for k, v in torch.load(ck, weights_only=False)["net"].items()} if ck.exists() else None
    seeds = [int(s) for s in a.seeds.split(",")]
    for s in seeds:
        City(s).baseline()
    os.environ["NUMBA_NUM_THREADS"] = "1"
    jobs = [(p, s) for p in a.policies.split(",") for s in seeds]
    with ProcessPoolExecutor(a.workers, get_context("spawn")) as pool:
        res = list(pool.map(job, *zip(*jobs), *([x] * len(jobs) for x in (sd, ck.with_suffix(".npz"), a.sims, a.depth))))
    out = {p: [r for (q, _), r in zip(jobs, res) if q == p] for p in dict.fromkeys(p for p, _ in jobs)}
    (OUT / f"eval-{a.name}-{version()}.json").write_text(json.dumps(dict(seeds=seeds, scores=out)))
    for p, r in out.items():
        print(f"{p:8s} {np.nanmean(r):+.4f} " + " ".join(f"{x:+.3f}" for x in r))


if __name__ == "__main__":
    main()
