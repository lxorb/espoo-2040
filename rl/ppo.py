import argparse
import json
import time

import numpy as np
import torch

from .env import Game, version
from .eval import OUT, run, warm
from .net import Net
from .train import batch


def rollout(net, seed, rng):
    g = Game(seed)
    c, rec, tr = g.c, g.rec, []
    while not c.done:
        acts = []
        while c.s["year"] < c.years:
            obs, mask, cand = g.view(c, rec, acts)
            with torch.no_grad():
                lg, v = net(*(torch.from_numpy(x)[None] for x in obs))
            p = torch.softmax(lg[0].masked_fill(~torch.from_numpy(mask), -1e9), 0).numpy().astype(float)
            a = int(rng.choice(p.size, p=p / p.sum()))
            tr.append([obs, mask, a, np.log(p[a]), float(v[0]), 0.0])
            if a == 0:
                break
            acts.append(cand[a])
        rec = c.step(acts)
        tr[-1][5] += g.reward(c, rec)
    return tr, 1 - sum(r["delay"] for r in c.recs) / sum(g.base)


def update(net, opt, tr, epochs=4, bs=64, lam=0.95):
    v = np.array([t[4] for t in tr] + [0.0])
    r = np.array([t[5] for t in tr])
    adv, run_ = np.zeros(len(tr)), 0.0
    for i in reversed(range(len(tr))):
        run_ = r[i] + v[i + 1] - v[i] + lam * run_
        adv[i] = run_
    ret = adv + v[:-1]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    for _ in range(epochs):
        for ix in np.array_split(np.random.permutation(len(tr)), max(1, len(tr) // bs)):
            lg, val = net(*batch([tr[i] for i in ix]))
            m = torch.from_numpy(np.stack([tr[i][1] for i in ix]))
            lp = torch.log_softmax(lg.masked_fill(~m, -1e9), 1)
            a = torch.tensor([tr[i][2] for i in ix])
            ratio = torch.exp(lp.gather(1, a[:, None])[:, 0] - torch.tensor([tr[i][3] for i in ix]).float())
            A = torch.from_numpy(adv[ix]).float()
            ent = -(lp.exp() * lp.masked_fill(~m, 0)).sum(1).mean()
            loss = (-torch.min(ratio * A, ratio.clamp(0.8, 1.2) * A).mean() + 0.5 * ((val - torch.from_numpy(ret[ix]).float()) ** 2).mean()
                    - 0.01 * ent)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--eval", default=",".join(map(str, range(16))))
    a = ap.parse_args()
    seeds = [int(s) for s in a.eval.split(",")]
    warm(seeds)
    OUT.mkdir(parents=True, exist_ok=True)
    net, rng, ck = Net(), np.random.default_rng(0), OUT / f"ppo-{version()}.pt"
    opt = torch.optim.Adam(net.parameters(), 3e-4)
    st = torch.load(ck, weights_only=False) if ck.exists() else dict(curve=[])
    if "opt" in st:
        net.load_state_dict(st["net"])
        opt.load_state_dict(st["opt"])
    curve = st["curve"]
    for ep in range(len(curve) + 1, a.episodes + 1):
        t = time.time()
        tr, sav = rollout(net.eval(), 2000 + ep, rng)
        update(net.train(), opt, tr)
        row = dict(episode=ep, train_saving=sav, secs=time.time() - t)
        if ep % a.every == 0:
            row["eval_net"] = [run("ppo", s, net.eval())["saving"] for s in seeds]
        curve.append(row)
        print(json.dumps(row), flush=True)
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), curve=curve), ck)
        ck.with_suffix(".json").write_text(json.dumps(curve))


if __name__ == "__main__":
    main()
