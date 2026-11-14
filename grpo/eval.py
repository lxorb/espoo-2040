import argparse
import json

import numpy as np
import torch

from .env import OUT, ROOT, Pool, parse, prompt, refs, search
from .train import EVAL, evaluate, load, sample


def booms(i):
    return {n.split()[0].lower().replace("ä", "a") + "-access" for n, p, _ in i["areas"] if p > 0.02}


def share(plans, infos):
    acc = [[p in booms(i) for p in plan if p.endswith("-access")] for plan, i in zip(plans, infos)]
    hit = [len(booms(i) & set(plan)) / len(booms(i)) for plan, i in zip(plans, infos)]
    return dict(share=float(np.mean([x for a in acc for x in a])) if any(acc) else 0.0, hit=float(np.mean(hit)))


def shuffle(i, rng):
    g = [a[1:] for a in i["areas"]]
    return dict(i, areas=sorted(([a[0], *g[k]] for a, k in zip(i["areas"], rng.permutation(len(g)))), key=lambda a: -a[1]))


def probe(m, tok, infos, plans, off, k=3):
    sh = [shuffle(i, np.random.default_rng(i["seed"] * 10 + j)) for i in infos for j in range(k)]
    with m.disable_adapter() if off else torch.no_grad():
        ts = sample(m, tok, [prompt(i) for i in sh], 1, True)[2]
    new = [parse(t, [p["id"] for p in i["menu"]])[0] for t, i in zip(ts, sh)]
    return dict(share(new, sh), changed=float(np.mean([p != plans[n // k] for n, p in enumerate(new)])))


def table(name, rs, infos):
    s = [r["score"] for r in rs]
    print(f"{name:14s} {np.mean(s):.3f}  " + " ".join(f"{x:.2f}" for x in s), flush=True)
    return dict(score=float(np.mean(s)), saving=float(np.mean([r["saving"] for r in rs])), last=float(np.mean([r["last"] for r in rs])),
                spent=float(np.mean([r["spent"] for r in rs])), **share([r["plan"] for r in rs], infos), seeds=rs)


def curve(d, infos):
    log = json.loads((d / "curve.json").read_text())
    new = json.loads((ROOT / "sim" / "norms.json").read_text())
    old = json.loads((d / "norms-train.json").read_text()) if (d / "norms-train.json").exists() else new
    f = {int(k): (v[0] - old[k][1]) / (v[0] - new[k][1]) for k, v in old.items()}
    ev = [dict(step=e["step"], name=e["name"], greedy=float(np.mean([s["score"] * f[s["seed"]] for s in e["seeds"]])),
               sample_mean=float(np.mean([s["sampled"] * f[s["seed"]] for s in e["seeds"]])), valid=e["valid"],
               **share([s["greedy"]["plan"] if s["greedy"] else [] for s in e["seeds"]], infos)) for e in log["evals"]]
    st = [{k: s[k] for k in ("step", "reward", "best", "std", "frac", "valid")} for s in log["steps"]]
    (d / "curve-frozen.json").write_text("{\"evals\": [\n" + ",\n".join(map(json.dumps, ev)) + "],\n\"steps\": [\n" + ",\n".join(map(json.dumps, st)) + "]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="*")
    ap.add_argument("--refs", action="store_true")
    ap.add_argument("--curve", nargs="*")
    ap.add_argument("--search", type=int, default=80)
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    torch.set_num_threads(32)
    pool, path = Pool(a.workers), OUT / "eval.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    ip = ROOT / "runs" / "grpo" / "infos.json"
    infos = json.loads(ip.read_text()) if ip.exists() else pool.infos(EVAL)
    ip.write_text(json.dumps(infos))
    out["example"] = dict(seed=infos[0]["seed"], prompt=prompt(infos[0]), booms=sorted(booms(infos[0])))
    for d in a.curve or []:
        curve(ROOT / d, infos)
    if a.refs:
        R = [refs(i, np.random.default_rng(i["seed"])) for i in infos]
        for k in R[0]:
            for mt in (1, 2):
                out[f"{k}-m{mt}"] = table(f"{k}-m{mt}", pool.run([(i["seed"], r[k], mt) for i, r in zip(infos, R)]), infos)
        path.write_text(json.dumps(out))
        start = [max((out[f"{k}-m{mt}"]["seeds"][j] for k in R[0] for mt in (1, 2)), key=lambda r: r["frac"]) for j in range(len(infos))]
        best = [pool.ex.submit(search, i["seed"], s["plan"], a.search) for i, s in zip(infos, start)]
        out["search"] = table("search", [b.result() for b in best], infos)
    for c in a.ckpts:
        tok, m = load(None if c == "zero-shot" else c)
        e = evaluate(m, tok, pool, infos, off=c == "zero-shot")
        plans = [s["greedy"]["plan"] if s["greedy"] else [] for s in e["seeds"]]
        out[c] = dict(e, **share(plans, infos), probe=probe(m, tok, infos, plans, c == "zero-shot"))
        print(f"{c:14s} {e['score']:.3f} sampled {e['sampled']:.3f} valid {e['valid']:.2f}", {k: out[c][k] for k in ("share", "hit", "probe")}, flush=True)
    path.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
