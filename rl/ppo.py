import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch

from sim import City

from .core import Agent, Net, batch, jlogits
from .policy import decode
from .train import OUT, play, score, version


def rollout(seed, sd):
    torch.set_num_threads(1)
    net = Net()
    net.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
    net.eval()
    c, rng = City(seed, fast=True), np.random.default_rng(seed)
    ag, base = Agent(c, net), c.baseline()
    scale, tr = np.mean(base), []
    while not c.done:
        t, acts = c.state()["step"], []
        if t < c.years:
            ob, jm = ag.look(c)
            lg, v = ag.eval(ob)
            p = np.exp(np.where(jm, lg, -np.inf) - lg[jm].max())
            a = int(rng.choice(p.size, p=p / p.sum()))
            tr.append([ob, jm, a, float(np.log(p[a] / p.sum())), v, 0.0])
            acts = decode(ag.J, ag.menu, a)
        tr[-1][5] += np.nan_to_num((base[t] - c.step(acts)["delay"]) / scale)
    return tr, score(c)


def update(net, opt, trs, J, epochs=4, bs=64, lam=0.95):
    data = []
    for tr in trs:
        v, adv, g = np.array([t[4] for t in tr] + [0.0]), np.zeros(len(tr)), 0.0
        for i in reversed(range(len(tr))):
            g = tr[i][5] + v[i + 1] - v[i] + lam * g
            adv[i] = g
        data += [t + [ad, ad + t[4]] for t, ad in zip(tr, adv)]
    A = np.array([d[6] for d in data])
    A = (A - A.mean()) / (A.std() + 1e-8)
    for _ in range(epochs):
        for ix in np.array_split(np.random.permutation(len(data)), max(1, len(data) // bs)):
            b = [data[i] for i in ix]
            a_, c_, val, b_ = net(*batch(b))
            m = torch.from_numpy(np.stack([d[1] for d in b]))
            lp = torch.log_softmax(jlogits(J, a_, c_, b_).masked_fill(~m, -1e9), 1)
            ratio = torch.exp(lp.gather(1, torch.tensor([d[2] for d in b])[:, None])[:, 0] - torch.tensor([d[3] for d in b]))
            adv = torch.from_numpy(A[ix]).float()
            ent = -(lp.exp() * lp.masked_fill(~m, 0)).sum(1).mean()
            loss = (-torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
                    + 0.5 * ((val - torch.tensor([d[7] for d in b]).float()) ** 2).mean() - 0.01 * ent)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--init", default="")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--name", default="ppo")
    ap.add_argument("--eval", default=",".join(map(str, range(16))))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ck, seeds = OUT / f"{a.name}-{version()}.pt", [int(s) for s in a.eval.split(",")]
    net = Net()
    opt = torch.optim.Adam(net.parameters(), a.lr)
    st = torch.load(ck, weights_only=False) if ck.exists() else dict(curve=[])
    if "opt" in st:
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
    elif a.init:
        net.load_state_dict(torch.load(a.init, weights_only=False)["net"])
    os.environ["NUMBA_NUM_THREADS"] = "1"
    pool, J = ProcessPoolExecutor(a.workers, get_context("spawn")), Agent(City(seeds[0]), net).J
    for it in range(len(st["curve"]) + 1, a.iters + 1):
        t = time.time()
        sd = {k: v.numpy().copy() for k, v in net.state_dict().items()}
        ss = [2000 + it * a.workers + i for i in range(a.workers)]
        res = list(pool.map(rollout, ss, [sd] * len(ss)))
        update(net.train(), opt, [r[0] for r in res], J)
        net.eval()
        row = dict(it=it, episodes=it * a.workers, train=float(np.nanmean([r[1] for r in res])), secs=time.time() - t)
        if it % a.every == 0:
            sd = {k: v.numpy().copy() for k, v in net.state_dict().items()}
            row["net"] = [r[1] for r in pool.map(play, seeds, [sd] * len(seeds), ["net"] * len(seeds))]
            row["mean_net"] = float(np.nanmean(row["net"]))
        st["curve"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "net"}), flush=True)
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), curve=st["curve"]), ck)
        ck.with_suffix(".json").write_text(json.dumps(st["curve"]))


if __name__ == "__main__":
    main()
