import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from sim import City

from .core import Agent, Net, batch, jlogits
from .policy import decode, encode
from .search import Search

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "rl"
EXPERT = Path(os.environ.get("RL_EXPERT", ROOT / "runs" / "expert"))
SEEDS = list(range(16))


def version():
    files = sorted(ROOT.glob("sim/*.py")) + sorted((ROOT / "data" / "out").glob("*")) + [ROOT / "data" / "catalogue.json", ROOT / "sim" / "norms.json"]
    return hashlib.sha1(b"".join(f.read_bytes() for f in files if f.is_file())).hexdigest()[:8]


def score(c):
    r = c.result()
    return r["score"] if r["score"] is not None else r["saving"] / r["nothing"]


def load(sd):
    torch.set_num_threads(1)
    net = Net()
    net.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
    return net.eval()


def weights(net):
    return {k: v.numpy().copy() for k, v in net.state_dict().items()}


def play(seed, sd, mode="net", n=16, depth=6, log=None):
    c = City(seed, fast=mode == "expert")
    ag, base = Agent(c, load(sd)), c.baseline()
    sr, plan = Search(ag, base, np.mean(base), n, depth), mode == "expert" and json.loads((EXPERT / f"{seed}.json").read_text())["years"]
    pol, obs, d = [], [], []
    while not c.done:
        t, acts = c.state()["step"], []
        ob, jm = ag.look(c)
        obs.append(ob)
        if t < c.years:
            if plan:
                acts = plan[t]["actions"]
                pol.append((ob, jm, np.eye(jm.size, dtype=np.float32)[encode(ag.J, ag.menu, acts)]))
            else:
                a = sr(c) if mode == "search" else int(np.argmax(np.where(jm, ag.eval(ob)[0], -np.inf)))
                acts = decode(ag.J, ag.menu, a)
        d.append(c.step(acts)["delay"])
    if log:
        c.log(log)
    z = np.cumsum(((np.array(base) - d) / np.mean(base))[::-1])[::-1]
    return [(o, m, p, None) for o, m, p in pol] + [(o, None, None, float(v)) for o, v in zip(obs, z)], score(c), sum(d)


def learn(net, opt, buf, steps, J, bs=128):
    pol, val = [x for x in buf if x[2] is not None], [x for x in buf if x[3] is not None]
    for _ in range(steps):
        b = [pol[i] for i in np.random.randint(len(pol), size=bs)]
        A, C, _, B = net(*batch(b))
        m = torch.from_numpy(np.stack([x[1] for x in b]))
        lp = -(torch.from_numpy(np.stack([x[2] for x in b])) * torch.log_softmax(jlogits(J, A, C, B).masked_fill(~m, -1e9), 1)).sum(1).mean()
        b = [val[i] for i in np.random.randint(len(val), size=bs)]
        lv = ((net(*batch(b))[2] - torch.tensor([x[3] for x in b])) ** 2).mean()
        opt.zero_grad()
        (lp + lv).backward()
        opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="bc")
    ap.add_argument("--experts", type=int, default=200)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--nojam", action="store_true")
    a = ap.parse_args()
    if a.nojam:
        os.environ["RL_NOJAM"] = "1"
    OUT.mkdir(parents=True, exist_ok=True)
    net, ck = Net(), OUT / f"{a.name}-{version()}"
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    os.environ["NUMBA_NUM_THREADS"] = "1"
    ex = [s for s in range(1000, 1000 + a.experts) if (EXPERT / f"{s}.json").exists()]
    with ProcessPoolExecutor(a.workers, get_context("spawn")) as pool:
        buf = [x for r in pool.map(play, ex, [weights(net)] * len(ex), ["expert"] * len(ex)) for x in r[0]]
        learn(net.train(), opt, buf, a.steps, Agent(City(0), net).J)
        res = [r[1] for r in pool.map(play, SEEDS, [weights(net.eval())] * len(SEEDS))]
    row = dict(experts=len(ex), jam=not os.environ.get("RL_NOJAM"), net=res, mean_net=float(np.mean(res)))
    print(json.dumps(row), flush=True)
    ck.with_suffix(".json").write_text(json.dumps(row))
    net.export(ck.with_suffix(".npz"))


if __name__ == "__main__":
    main()
