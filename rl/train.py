import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from .env import Game, version
from .net import Net
from .search import year

OUT = Path(__file__).resolve().parent.parent / "runs" / "rl"


def episode(seed, sd, n, k, depth):
    torch.set_num_threads(1)
    net = Net()
    net.load_state_dict(sd)
    net.eval()
    t, g, rng = time.time(), Game(seed), np.random.default_rng(seed)
    c, rec, smp, vs, d = g.c, g.rec, [], [], []
    c.gap = c.fast
    while not c.done:
        vs.append(g.view(c, rec, [])[0])
        acts, s = year(g, net, c, rec, n, k, 1.0, rng, depth) if c.s["year"] < c.years else ([], [])
        smp += s
        rec = c.step(acts)
        d.append(rec["delay"])
    ret = np.cumsum(((np.array(g.base) - d) / g.scale)[::-1])[::-1]
    out = [(o, m, p, None) for o, m, p in smp] + [(o, None, None, float(r)) for o, r in zip(vs, ret)]
    return out, 1 - sum(d) / sum(g.base), time.time() - t


def batch(xs):
    return [torch.from_numpy(np.stack([x[0][i] for x in xs])) for i in range(3)]


def learn(net, opt, buf, steps, bs=64):
    pol = [x for x in buf if x[2] is not None]
    val = [x for x in buf if x[3] is not None]
    tot = []
    for _ in range(steps):
        lp = lv = torch.tensor(0.0)
        if pol:
            b = [pol[i] for i in np.random.randint(len(pol), size=min(bs, len(pol)))]
            lg, _ = net(*batch(b))
            m = torch.from_numpy(np.stack([x[1] for x in b]))
            pi = torch.from_numpy(np.stack([x[2] for x in b])).float()
            lp = -(pi * torch.log_softmax(lg.masked_fill(~m, -1e9), 1)).sum(1).mean()
        if val:
            b = [val[i] for i in np.random.randint(len(val), size=min(bs, len(val)))]
            _, v = net(*batch(b))
            lv = ((v - torch.tensor([x[3] for x in b])) ** 2).mean()
        opt.zero_grad()
        (lp + lv).backward()
        opt.step()
        tot.append((lp.item(), lv.item()))
    return np.mean(tot, 0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gumbel")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--sims", type=int, default=8)
    ap.add_argument("--consider", type=int, default=4)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--keep", type=int, default=30)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--search-every", type=int, default=5)
    ap.add_argument("--eval", default=",".join(map(str, range(16))))
    a = ap.parse_args()
    from .eval import run, warm
    OUT.mkdir(parents=True, exist_ok=True)
    ck, cj = (OUT / f"{a.name}-{version()}{x}" for x in (".pt", ".json"))
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    st = dict(ep=0, bufs=[], curve=[])
    if ck.exists():
        st = torch.load(ck, weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
    if a.workers > 1:
        os.environ["NUMBA_NUM_THREADS"] = str(max(1, os.cpu_count() // a.workers))
    pool = ProcessPoolExecutor(a.workers, get_context("spawn")) if a.workers > 1 else None
    seeds = [int(s) for s in a.eval.split(",") if s]
    warm(seeds)
    it = len({r.get("it") for r in st["curve"]})
    pmap = pool.map if pool else map
    while st["ep"] < a.episodes:
        it += 1
        sd = {k: v.clone() for k, v in net.state_dict().items()}
        ss = [1000 + st["ep"] + i for i in range(a.workers)]
        res = list(pool.map(episode, ss, [sd] * len(ss), [a.sims] * len(ss), [a.consider] * len(ss), [a.depth] * len(ss))) if pool else [episode(ss[0], sd, a.sims, a.consider, a.depth)]
        rows = []
        for s, (smp, sav, dt) in zip(ss, res):
            st["bufs"] = (st["bufs"] + [smp])[-a.keep:]
            st["ep"] += 1
            rows.append(dict(it=it, episode=st["ep"], seed=s, train_saving=sav, secs=dt))
        row = rows[-1]
        net.train()
        row["loss_pi"], row["loss_v"] = learn(net, opt, [x for b in st["bufs"] for x in b], a.steps * len(res))
        net.eval()
        for p, e in (("net", a.every), ("search", a.search_every)):
            if e and it % e == 0 and seeds:
                row[f"eval_{p}"] = [r["saving"] for r in pmap(run, [p] * len(seeds), seeds, [net] * len(seeds), [a.sims] * len(seeds),
                                                               [a.consider] * len(seeds), [False] * len(seeds), [a.depth] * len(seeds))]
        st["curve"] += rows
        print(json.dumps(rows), flush=True)
        st.update(net=net.state_dict(), opt=opt.state_dict())
        torch.save(st, ck.with_suffix(".tmp"))
        ck.with_suffix(".tmp").replace(ck)
        cj.write_text(json.dumps(st["curve"]))


if __name__ == "__main__":
    main()
