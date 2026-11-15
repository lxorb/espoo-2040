"""Reference points and held-out evaluation for the reasoning runs (grpo/reason.py).

  refs : oracle / random / run-A constant plan under the same single-turn protocol, named and anon catalogues, seeds 0-47
  zero : zero-shot base models (1B, 4B) with --think and without, named and anon, seeds 0-15 (and test for 1B)
  test : for each run, the final and the best-validation checkpoint on test seeds 16-47 (selection on validation only)
Everything is cached in runs/reason/eval/*.json, so reruns resume.
"""
import argparse
import json

import numpy as np

from .reason import ACC, RUNS, TEST, VAL, Gen, Pool, base_name, evaluate, infos_cached

OUT = RUNS / "eval"
CONST = (["otaniemi-access", "kiviruukki-access", "espoonvayla-2x2"], 2)  # run A final plan (evidence/grpo.json const_plan)


def jload(p):
    return json.loads(p.read_text()) if p.exists() else {}


def refs(pool):
    p = OUT / "refs.json"
    out = jload(p)
    run_a = json.loads((RUNS.parent.parent / "evidence" / "grpo.json").read_text())["meta"]["const_plan"]
    const = (run_a["plan"], run_a["maintain"])
    for anon in (False, True):
        tag = "anon" if anon else "named"
        infos = infos_cached(pool, VAL + TEST, anon)
        lists = {}
        for i in infos:
            rng = np.random.default_rng(i["seed"])
            menu = [q["id"] for q in i["menu"]]
            boom = [a for a, n in ACC.items() if n in i["booms"]]
            lists.setdefault("oracle", []).append([(boom + ["olari-link", "mankkaa-link"], 1)])
            lists.setdefault("const", []).append([const])
            lists.setdefault("random", []).append([(list(rng.permutation(menu)[:rng.integers(2, 7)]), int(rng.integers(3))) for _ in range(4)])
        for k, L in lists.items():
            key = f"{k}-{tag}"
            if key in out:
                continue
            flat = [(i["seed"], ids, mt) for i, ps in zip(infos, L) for ids, mt in ps]
            rs = pool.run(flat, anon, False)
            it = iter(rs)
            per = [[next(it) for _ in ps] for ps in L]
            out[key] = {str(i["seed"]): dict(frac=float(np.mean([r["frac"] for r in rr])), score=float(np.mean([r["score"] for r in rr])) if rr[0]["score"] is not None else None,
                                             boom_hit=float(np.mean([len(set(i["booms"]) & {ACC[q] for q in r["plan"] if q in ACC}) / len(i["booms"]) for r in rr])))
                        for i, rr in zip(infos, per)}
            print(key, np.mean([v["frac"] for v in out[key].values()]), flush=True)
            p.write_text(json.dumps(out))


def zero(pool, sizes, mem):
    p = OUT / "zero.json"
    out = jload(p)
    for size in sizes:
        gen = None
        for think in (True, False):
            for anon in (True, False):
                for split, seeds in (("val", VAL), ("test", TEST)):
                    key = f"{size}-{'think' if think else 'nothink'}-{'anon' if anon else 'named'}-{split}"
                    if key in out:
                        continue
                    gen = gen or Gen(base_name(size), mem, 3072, lora=False)
                    e = evaluate(gen, pool, infos_cached(pool, seeds, anon), think, anon, 1024 if think else 96, None, n=4 if split == "val" else 1, edits=split == "val")
                    out[key] = e
                    print(key, round(e["score"], 3), round(e["frac"], 3), flush=True)
                    p.write_text(json.dumps(out))
        del gen


def test(pool, runs, mem):
    p = OUT / "test.json"
    out = jload(p)
    for name in runs:
        log = json.loads((RUNS / name / "log.json").read_text())
        a = log["args"]
        ev = [e for e in log["evals"]]
        best = max(ev, key=lambda e: (e["score"], e["step"]))
        last = ev[-1]
        gen = None
        for tag, e in (("best", best), ("final", last)):
            key = f"{name}|{tag}|{e['step']}"
            if key in out:
                continue
            gen = gen or Gen(base_name(a["size"]), mem, 3072)
            ck = RUNS / name / f"ckpt-{e['step']}"
            r = evaluate(gen, pool, infos_cached(pool, TEST, a["anon"]), a["think"], a["anon"], a["maxtok"] or (1024 if a["think"] else 96), ck, n=1, edits=True)
            out[key] = r
            print(key, round(r["score"], 3), round(r["frac"], 3), flush=True)
            p.write_text(json.dumps(out))
        del gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["refs", "zero", "test"])
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--sizes", nargs="*", default=["1b"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mem", type=float, default=0.1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pool = Pool(a.workers)
    dict(refs=lambda: refs(pool), zero=lambda: zero(pool, a.sizes, a.mem), test=lambda: test(pool, a.runs, a.mem))[a.what]()


if __name__ == "__main__":
    main()
