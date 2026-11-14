import argparse
import json

import numpy as np
import torch

from .env import OUT, ROOT, Pool, parse, prompt, reward
from .eval import share
from .train import load, sample


def stats(e, infos, f, menu):
    g = [(s["greedy"]["plan"], s["greedy"]["maintain"]) if s["greedy"] else ([], parse(s["text"], menu)[1]) for s in e["seeds"]]
    sc = [s["score"] * f[s["seed"]] for s in e["seeds"]]
    acc = [[k for k, p in enumerate(pl) if p.endswith("-access")] for pl, _ in g]
    return dict(plans=[dict(seed=s["seed"], plan=pl, maintain=k) for s, (pl, k) in zip(e["seeds"], g)],
                distinct=len({(tuple(pl), k) for pl, k in g}), valid=e["valid"], len=float(np.mean([len(pl) for pl, _ in g])),
                maintain={k: sum(m == k for _, m in g) / len(g) for k in range(3)},
                freq={p: sum(p in pl for pl, _ in g) / len(g) for p in menu}, first={p: sum(pl[:1] == [p] for pl, _ in g) / len(g) for p in menu},
                access_share=float(np.mean([len(a) / len(pl) for a, (pl, _) in zip(acc, g) if pl])),
                **{f"boom_{k}": v for k, v in share([pl for pl, _ in g], infos).items()},
                first_access_pos=float(np.mean([a[0] + 1 for a in acc if a])) if any(acc) else None,
                score=float(np.mean(sc)), scores=sc, saved_ph=float(np.mean([s["greedy"]["saving"] if s["greedy"] else 0.0 for s in e["seeds"]])))


def spread(ckpt, infos, pool, n):
    tok, m = load(ckpt)
    rep = [i for i in infos for _ in range(n)]
    ps = [parse(t, [p["id"] for p in i["menu"]]) for i, t in zip(rep, sample(m, tok, [prompt(i) for i in infos], n)[2])]
    R = np.array([reward(r, p[2]) for r, p in zip(pool.run([(i["seed"], p[0], p[1]) for i, p in zip(rep, ps)]), ps)]).reshape(len(infos), n)
    return dict(sample_distinct=float(np.mean([len({(tuple(p[0]), p[1]) for p in ps[k * n:(k + 1) * n]}) for k in range(len(infos))])),
                sample_std=float(R.std(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    torch.set_num_threads(32)
    pool, infos, path = Pool(a.workers), json.loads((OUT / "infos.json").read_text()), OUT / "strategy.json"
    new, menu, names = json.loads((ROOT / "sim" / "norms.json").read_text()), [p["id"] for p in infos[0]["menu"]], [r.split("=")[0] for r in a.runs]
    out = [r for r in json.loads(path.read_text()) if r["run"] not in names] if path.exists() else []
    for name, d in (r.split("=") for r in a.runs):
        d = ROOT / d
        log = json.loads((d / "curve.json").read_text())
        old = json.loads((d / "norms-train.json").read_text()) if (d / "norms-train.json").exists() else new
        f = {int(k): (v[0] - old[k][1]) / (v[0] - new[k][1]) for k, v in old.items()}
        for e in log["evals"]:
            st = [s for s in log["steps"] if e["step"] <= s["step"] < e["step"] + 10]
            row = dict(run=name, step=e["step"], name=e["name"], **stats(e, infos, f, menu),
                       train_std=float(np.mean([s["std"] for s in st])) if st else None, train_reward=float(np.mean([s["reward"] for s in st])) if st else None)
            if (d / f"ckpt-{e['step']}").exists():
                row.update(spread(d / f"ckpt-{e['step']}", infos, pool, 8))
            out.append(row)
            print(name, e["step"], row["score"], row["distinct"], row.get("sample_distinct"), flush=True)
            path.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
