import argparse
import json

import numpy as np
import torch

from .env import OUT, Pool, access, execute, parse, prompt, refs, search
from .train import EVAL, evaluate, load, sample


def share(plans, infos):
    b = [{access(n) for n, p, _ in i["areas"] if p > 0.02} for i in infos]
    acc = [p in bb for plan, bb in zip(plans, b) for p in plan if p.endswith("-access")]
    return dict(share=float(np.mean(acc)) if acc else 0.0, hit=float(np.mean([len(bb & set(plan)) / len(bb) for plan, bb in zip(plans, b)])))


def probe(m, tok, infos, plans, off, k=3):
    sh = []
    for i in infos:
        for j in range(k):
            g = np.random.default_rng(i["seed"] * 10 + j).permutation(len(i["areas"]))
            sh.append(dict(i, areas=sorted(([a[0], *i["areas"][q][1:]] for a, q in zip(i["areas"], g)), key=lambda a: -a[1])))
    with m.disable_adapter() if off else torch.no_grad():
        new = [parse(t, [p["id"] for p in i["menu"]])[0] for t, i in zip(sample(m, tok, [prompt(i) for i in sh], 1, True)[2], sh)]
    return dict(share(new, sh), changed=float(np.mean([p != plans[n // k] for n, p in enumerate(new)])))


def table(name, rs, infos):
    print(f"{name:14s} {np.mean([r['score'] for r in rs]):.3f}", flush=True)
    return dict({k: float(np.mean([r[k] for r in rs])) for k in ("score", "saving", "last", "spent")}, **share([r["plan"] for r in rs], infos), seeds=rs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*")
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--log", nargs=2)
    ap.add_argument("--search", type=int, default=120)
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    torch.set_num_threads(32)
    path, ip = OUT / "eval.json", OUT / "infos.json"
    pool = Pool(a.workers)
    infos = json.loads(ip.read_text()) if ip.exists() else pool.infos(EVAL)
    ip.write_text(json.dumps(infos))
    if a.log:
        i = infos[int(a.log[1])]
        tok, m = load(a.log[0])
        t = sample(m, tok, [prompt(i)], 1, True)[2][0].strip()
        c = execute(i["seed"], *parse(t, [p["id"] for p in i["menu"]])[:2])
        c.who = "gemma"
        (OUT / f"gemma-seed{i['seed']}.json").write_text(json.dumps(dict(c.log(), plan=t, ckpt=a.log[0])))
        return print(t, c.score())
    out = json.loads(path.read_text()) if path.exists() else {}
    out["example"] = dict(seed=infos[0]["seed"], prompt=prompt(infos[0]))
    if a.refs:
        R = [refs(i, np.random.default_rng(i["seed"])) for i in infos]
        for k in R[0]:
            for mt in (1, 2):
                out[f"{k}-m{mt}"] = table(f"{k}-m{mt}", pool.run([(i["seed"], r[k], mt) for i, r in zip(infos, R)]), infos)
        start = [max((out[f"{k}-m{mt}"]["seeds"][j] for k in R[0] for mt in (1, 2)), key=lambda r: r["frac"]) for j in range(len(infos))]
        out["search"] = table("search", list(pool.ex.map(search, EVAL, [s["plan"] for s in start], [a.search] * len(EVAL))), infos)
    for c in a.ckpts:
        tok, m = load(None if c == "zero-shot" else c)
        e = evaluate(m, tok, pool, infos, off=c == "zero-shot")
        plans = [(s["greedy"] or {}).get("plan", []) for s in e["seeds"]]
        out[c] = dict(e, **share(plans, infos), probe=probe(m, tok, infos, plans, c == "zero-shot"))
        print(c, round(e["score"], 3), round(e["sampled"], 3), round(e["valid"], 2), {k: out[c][k] for k in ("share", "hit", "probe")}, flush=True)
    path.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
