import argparse
import json
import platform
import subprocess
import sys
from datetime import date

import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon

from .env import OUT, ROOT, Pool, parse, prompt
from .train import load, sample

VAL, TEST = list(range(16)), list(range(16, 48))


def variants(i):
    names = sorted(a[0] for a in i["areas"])
    out = {"orig": i, "zero": dict(i, areas=[[n, 0.0, 0.0] for n in names])}
    for j in range(3):
        g = np.random.default_rng(i["seed"] * 10 + j).permutation(len(i["areas"]))
        out[f"shuf{j}"] = dict(i, areas=sorted(([a[0], *i["areas"][q][1:]] for a, q in zip(i["areas"], g)), key=lambda a: -a[1]))
    return out


def cached(path, keys, fn):
    c = json.loads(path.read_text()) if path.exists() else {}
    todo = [k for k in keys if k not in c]
    if todo:
        c.update(zip(todo, fn(todo)))
        path.write_text(json.dumps(c))
    return c


def boot(x, n=10000):
    x = np.asarray(x, float)
    m = np.random.default_rng(0).choice(x, (n, x.size)).mean(1)
    return dict(mean=float(x.mean()), ci=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], n=int(x.size))


def paired(a, b):
    d = np.subtract(a, b)
    return dict(wins=int((d > 0).sum()), losses=int((d < 0).sum()), sign_p=float(binomtest(int((d > 0).sum()), int((d != 0).sum()), alternative="greater").pvalue) if (d != 0).any() else 1.0,
                wilcoxon_p=float(wilcoxon(d, alternative="greater").pvalue) if (d != 0).any() else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    pool, d = Pool(a.workers), OUT / "rigor"
    d.mkdir(exist_ok=True)
    infos = cached(d / "infos.json", [str(s) for s in VAL + TEST], lambda ks: pool.infos([int(k) for k in ks]))
    infos = [infos[str(s)] for s in VAL + TEST]
    menu = [p["id"] for p in infos[0]["menu"]]
    V = [variants(i) for i in infos]

    def gen(ckpt):
        def f(keys):
            tok, m = load(ROOT / ckpt)
            return sample(m, tok, [prompt(V[int(k.split("|")[1])][k.split("|")[2]]) for k in keys], 1, True)[2]
        return f

    texts = {}
    for c in a.ckpts:
        texts.update(cached(d / "texts.json", [f"{c}|{n}|{v}" for n in range(len(infos)) for v in V[0]], gen(c)))
    plans = {k: parse(t, menu)[:2] for k, t in texts.items()}
    final = a.ckpts[1]
    const = max(set(tuple(plans[f"{final}|{n}|orig"][0]) + (plans[f"{final}|{n}|orig"][1],) for n in range(16)),
                key=lambda p: sum(tuple(plans[f"{final}|{n}|orig"][0]) + (plans[f"{final}|{n}|orig"][1],) == p for n in range(16)))
    rnd = {i["seed"]: [(list(r.permutation(menu)[:r.integers(2, 7)]), int(r.integers(3))) for r in [np.random.default_rng(i["seed"])] for _ in range(8)] for i in infos}
    methods = {"const": {i["seed"]: [(list(const[:-1]), const[-1])] for i in infos}, "random": rnd}
    methods.update({c: {i["seed"]: [plans[f"{c}|{n}|orig"]] for n, i in enumerate(infos)} for c in a.ckpts})
    jobs = sorted({f"{s}|{','.join(p)}|{k}" for m in methods.values() for s, ps in m.items() for p, k in ps if p})
    res = cached(d / "plays.json", jobs, lambda ks: pool.run([(int(k.split("|")[0]), k.split("|")[1].split(","), int(k.split("|")[2])) for k in ks]))
    norms = json.loads((ROOT / "sim" / "norms.json").read_text())

    def score(s, p, k):
        r = res.get(f"{s}|{','.join(p)}|{k}") if p else None
        base, best = norms.get(str(s), (None, None))
        return dict(frac=r["frac"] if r else 0.0, saved_ph=r["saving"] if r else 0.0, delay=r["delay"] if r else None,
                    score=(r["saving"] * 1e6 / (base - best) if r else 0.0) if base else None,
                    score_clip=float(np.clip(r["saving"] * 1e6 / (base - best), 0, 1) if r else 0.0) if base else None)

    out = dict(meta=dict(script="grpo/rigor.py", command="python -m grpo.rigor " + " ".join(sys.argv[1:]), seeds=dict(val=VAL, test=TEST), date=str(date.today()),
                         machine=subprocess.run(["lscpu"], capture_output=True, text=True).stdout.split("Model name:")[1].split("\n")[0].strip() + " " + platform.node(),
                         const_plan=dict(plan=list(const[:-1]), maintain=const[-1]), random="8 plans per seed, ids = permutation(menu)[:U{2..6}], maintain U{0..2}, rng seed = env seed"),
               methods={}, sensitivity={}, tests={})
    for name, m in methods.items():
        rows = {s: [score(s, p, k) for p, k in ps] for s, ps in m.items()}
        per = {str(s): {f: (float(np.mean([x[f] for x in r])) if all(x[f] is not None for x in r) else None) for f in ("score", "score_clip", "frac", "saved_ph", "delay")} for s, r in rows.items()}
        out["methods"][name] = dict(seeds=per, plans={str(s): [dict(plan=p, maintain=k) for p, k in ps] for s, ps in m.items()},
                                    **{sp: {f: boot([per[str(s)][f] for s in ss]) for f in ("score", "score_clip", "frac") if all(per[str(s)][f] is not None for s in ss)} for sp, ss in (("val", VAL), ("test", TEST))})
    for c in a.ckpts:
        out["sensitivity"][c] = {v: dict(changed=float(np.mean([plans[f"{c}|{n}|{v}"] != plans[f"{c}|{n}|orig"] for n in range(len(infos))])), n=len(infos)) for v in V[0] if v != "orig"}
        out["sensitivity"][c]["shuffle"] = float(np.mean([out["sensitivity"][c][f"shuf{j}"]["changed"] for j in range(3)]))
    for x, o in [(final, "const"), (final, "random"), (final, a.ckpts[0])] + [(c, "const") for c in a.ckpts[2:]]:
        for sp, ss in (("val", VAL), ("test", TEST)):
            f = "score" if all(out["methods"][y]["seeds"][str(s)]["score"] is not None for y in (x, o) for s in ss) else "frac"
            out["tests"][f"{x} vs {o} {sp}"] = dict(metric=f, **paired(*[[out["methods"][y]["seeds"][str(s)][f] for s in ss] for y in (x, o)]))
    out["seed_runs"] = dict(runs=a.ckpts[1:], **{sp: {f: dict(values=v, mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else None)
                                                    for f in out["methods"][final][sp] for v in [[out["methods"][c][sp][f]["mean"] for c in a.ckpts[1:]]]} for sp in ("val", "test")})
    (ROOT / "evidence").mkdir(exist_ok=True)
    (ROOT / "evidence" / "grpo.json").write_text(json.dumps(out, indent=1))
    for name, m in out["methods"].items():
        print(name, {sp: {f: round(v["mean"], 3) for f, v in m[sp].items()} for sp in ("val", "test")})
    print(json.dumps(out["sensitivity"]), json.dumps(out["tests"]))


if __name__ == "__main__":
    main()
