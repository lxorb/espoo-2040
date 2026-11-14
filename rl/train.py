import argparse
import json
import os
import time
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


def version():
    import hashlib
    files = sorted(ROOT.glob("sim/*.py")) + sorted((ROOT / "data" / "out").glob("*")) + [ROOT / "data" / "catalogue.json", ROOT / "sim" / "norms.json"]
    return hashlib.sha1(b"".join(f.read_bytes() for f in files if f.is_file())).hexdigest()[:8]


def score(c):
    r = c.result()
    return r["score"] if r["score"] is not None else r["saving"] / r["nothing"]


def expert(seed):
    d = json.loads((EXPERT / f"{seed}.json").read_text())
    return [y["actions"] if isinstance(y, dict) else y for y in d.get("years", d.get("plan", d))]


def play(seed, sd, mode="search", n=16, depth=4, log=None):
    torch.set_num_threads(1)
    net = Net()
    net.load_state_dict({k: torch.as_tensor(v) for k, v in sd.items()})
    net.eval()
    c = City(seed, fast=mode in ("search", "expert"))
    ag, base = Agent(c, net), c.baseline()
    scale = np.mean(base)
    sr, rng, plan = Search(ag, base, scale, n, depth), np.random.default_rng(seed), expert(seed) if mode == "expert" else None
    pol, obs, d = [], [], []
    while not c.done:
        t = c.state()["step"]
        ob, jm = ag.look(c)
        obs.append(ob)
        acts = []
        if t < c.years:
            if mode == "expert":
                acts = [a for a in plan[t] if "build" in a or "maintain" in a]
                pol.append((ob, jm, np.eye(jm.size)[encode(ag.J, ag.menu, acts)]))
            elif mode in ("search", "searcheval"):
                a, pi, _, _ = sr(c, 1.0 if mode == "search" else 0.0, rng)
                pol.append((ob, jm, pi))
                acts = decode(ag.J, ag.menu, a)
            elif mode == "net":
                lg, _ = ag.eval(ob)
                acts = decode(ag.J, ag.menu, int(np.argmax(np.where(jm, lg, -np.inf))))
        d.append(c.step(acts)["delay"])
    if log:
        c.log(log)
    bad = int((~np.isfinite(d)).sum())
    z = np.cumsum(np.nan_to_num((np.array(base) - d) / scale)[::-1])[::-1]
    return [(o, m, p.astype(np.float32), None) for o, m, p in pol] + [(o, None, None, float(v)) for o, v in zip(obs, z)], score(c), bad


def learn(net, opt, buf, steps, J, bs=128):
    pol, val, out = [x for x in buf if x[2] is not None], [x for x in buf if x[3] is not None], []
    for _ in range(steps):
        b = [pol[i] for i in np.random.randint(len(pol), size=bs)]
        A, C, _, B = net(*batch(b))
        m = torch.from_numpy(np.stack([x[1] for x in b]))
        lp = -(torch.from_numpy(np.stack([x[2] for x in b])).float() * torch.log_softmax(jlogits(J, A, C, B).masked_fill(~m, -1e9), 1)).sum(1).mean()
        b = [val[i] for i in np.random.randint(len(val), size=bs)]
        lv = ((net(*batch(b))[2] - torch.tensor([x[3] for x in b])) ** 2).mean()
        opt.zero_grad()
        (lp + lv).backward()
        opt.step()
        out.append((lp.item(), lv.item()))
    return np.mean(out, 0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gumbel")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--sims", type=int, default=16)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bc", type=int, default=3000)
    ap.add_argument("--keep", type=int, default=400)
    ap.add_argument("--search-every", type=int, default=3)
    ap.add_argument("--init", default="")
    ap.add_argument("--selfplay", type=int, default=1)
    ap.add_argument("--experts", type=int, default=400)
    ap.add_argument("--eval", default=",".join(map(str, range(16))))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ck = OUT / f"{a.name}-{version()}.pt"
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    st = dict(it=0, ep=0, buf=[], curve=[])
    if ck.exists():
        st = torch.load(ck, weights_only=False)
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
    elif a.init:
        net.load_state_dict(torch.load(a.init, weights_only=False)["net"])
    seeds = [int(s) for s in a.eval.split(",") if s]
    for s in seeds:
        City(s).baseline()
    os.environ["NUMBA_NUM_THREADS"] = "1"
    pool = ProcessPoolExecutor(a.workers, get_context("spawn"))
    J = Agent(City(seeds[0]), net).J

    def run(mode, ss):
        sd = {k: v.numpy().copy() for k, v in net.state_dict().items()}
        return list(pool.map(play, ss, [sd] * len(ss), [mode] * len(ss), [a.sims] * len(ss), [a.depth] * len(ss)))

    while st["it"] <= a.iters:
        t, row = time.time(), dict(it=st["it"])
        new = [s for s in range(1000, 1000 + a.experts) if (EXPERT / f"{s}.json").exists() and s not in st.setdefault("eseen", [])]
        if len(new) >= 20 or new and not st.get("ebuf"):
            steps = a.bc if not st.get("ebuf") else a.bc // 3
            st["ebuf"] = st.get("ebuf", []) + [x for r in run("expert", new) for x in r[0]]
            st["eseen"] += new
            row.update(phase="bc", experts=len(st["eseen"]))
        elif st["it"] == 0:
            row.update(phase="init")
            steps = 0
        elif not a.selfplay:
            time.sleep(60)
            continue
        else:
            ss = list(range(1000 + st["ep"], 1000 + st["ep"] + a.workers))
            res = run("search", ss)
            st["ep"] += len(ss)
            st["buf"] = (st["buf"] + [x for r in res for x in r[0]])[-a.keep * 20:]
            row.update(phase="search", episodes=st["ep"], train=float(np.nanmean([r[1] for r in res])), nan=sum(r[2] for r in res))
            steps = a.steps
        net.train()
        if steps:
            row["loss_pi"], row["loss_v"] = learn(net, opt, st["buf"] + st.get("ebuf", []), steps, J)
        net.eval()
        row["net"] = [r[1] for r in run("net", seeds)]
        if a.search_every and st["it"] % a.search_every == 0:
            row["search"] = [r[1] for r in run("searcheval", seeds)]
        row.update({f"mean_{k}": float(np.nanmean(row[k])) for k in ("net", "search") if k in row}, secs=time.time() - t)
        st["curve"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ("net", "search")}), flush=True)
        st["it"] += 1
        st.update(net=net.state_dict(), opt=opt.state_dict())
        torch.save(st, ck.with_suffix(".tmp"))
        ck.with_suffix(".tmp").replace(ck)
        ck.with_suffix(".json").write_text(json.dumps(st["curve"]))
        net.export(ck.with_suffix(".npz"))
        (OUT / "snap").mkdir(exist_ok=True)
        net.export(OUT / "snap" / f"{a.name}-{version()}-{st['ep']:06d}.npz")


if __name__ == "__main__":
    main()
