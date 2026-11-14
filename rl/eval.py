import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from sim import params as P

from .env import K, M, Game, fork, version
from .net import Net
from .search import year

OUT = Path(__file__).resolve().parent.parent / "runs" / "rl"


def load(name):
    net = Net()
    net.load_state_dict(torch.load(OUT / f"{name}-{version()}.pt", weights_only=False)["net"])
    return net.eval()


def greedy(g, c, rec):
    _, mask, cand = g.view(c, rec, [])
    rem, ref, score = c.years + c.runout - c.s["year"], {}, []
    segs = {s["id"]: s["resurface"] for s in c.state()["segs"]}
    n, keep = len(g.pj) + 1, np.zeros(mask.size, bool)
    keep[1:n + 8] = keep[n + 3 * K: n + 3 * K + 8] = True
    for i in np.flatnonzero(mask & keep):
        a = cand[i]
        h = min(rem, c.cat[a["build"]]["years"] + 2 if "build" in a else 2)
        if h not in ref:
            ref[h] = sum(r["delay"] for r in c.forecast({}, h))
        gain = ref[h] - sum(r["delay"] for r in c.forecast({0: [a]}, h))
        k = c.cat[a["build"]]["cost"] if "build" in a else segs[a["resurface"]] if "resurface" in a else P.RETIME_COST
        score.append((gain / k, gain, a))
    f, acts = fork(c), []
    for _, gain, a in sorted(score, key=lambda x: -x[0]):
        if gain > 0 and len(acts) < M and f._act(a, []) is None:
            acts.append(a)
    return acts


def walk(g, c, rec, pick):
    acts = []
    while True:
        obs, mask, cand = g.view(c, rec, acts)
        i = pick(obs, mask)
        if i == 0:
            return acts
        acts.append(cand[i])


def run(policy, seed, net=None, n=8, k=4, log=False, depth=1):
    torch.set_num_threads(1)
    g = Game(seed, policy == "search")
    c, rec, rng = g.c, g.rec, np.random.default_rng(seed)
    if policy in ("net", "ppo"):
        net = net or load(policy if policy == "ppo" else "gumbel")

        def pick(obs, mask):
            with torch.no_grad():
                lg = net(*(torch.from_numpy(x)[None] for x in obs))[0][0].numpy()
            return int(np.argmax(np.where(mask, lg, -np.inf)))
    while not c.done:
        acts = []
        if c.s["year"] < c.years:
            if policy == "random":
                acts = walk(g, c, rec, lambda o, m: int(rng.choice(np.flatnonzero(m))))
            elif policy == "greedy":
                acts = greedy(g, c, rec)
            elif policy == "search":
                acts = year(g, net or load("gumbel"), c, rec, n, k, 0.0, None, depth)[0]
            elif policy != "nothing":
                acts = walk(g, c, rec, pick)
        rec = c.step(acts)
    if log:
        c.log(f"rl-{policy}-{seed}")
    r = c.result()
    return dict(policy=policy, seed=seed, delay=r["delay"], nothing=r["nothing"], saving=r["saving"] / r["nothing"])


def warm(seeds):
    for s in seeds:
        Game(s, False).c._nothing()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", default="nothing,random,greedy,ppo,net,search")
    ap.add_argument("--seeds", default=",".join(map(str, range(16))))
    ap.add_argument("--sims", type=int, default=8)
    ap.add_argument("--net", default="gumbel")
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="eval")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    warm(seeds)
    jobs = [(p, s, load(a.net) if p in ("net", "search") else None) for p in a.policies.split(",") for s in seeds]
    with ProcessPoolExecutor(a.workers, get_context("spawn")) as pool:
        out = []
        for r in pool.map(run, *zip(*jobs), [a.sims] * len(jobs), [4] * len(jobs), [True] * len(jobs), [a.depth] * len(jobs)):
            out.append(r)
            print(json.dumps(r), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.out}-{version()}.json").write_text(json.dumps(out))
    for p in dict.fromkeys(r["policy"] for r in out):
        print(p, round(float(np.mean([r["saving"] for r in out if r["policy"] == p])), 4))


if __name__ == "__main__":
    main()
