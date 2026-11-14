import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from sim import City

from .policy import Policy, forward, jmask, joint, joints, obs
from .train import OUT, version


def area_of(k):
    return k.split("-")[0] if k.endswith("-access") else None


def boom(c):
    return {n.split()[0].lower().replace("ä", "a") for n, m in c.areas if np.mean((1 + c.rp[m]) ** c.years) >= 1.5}


def plan(path, seed):
    c, pol = City(seed), Policy(path, maintain=None)
    proj = [dict(id=k, **{f: c.cat[k].get(f, 0) for f in ("kind", "cost", "years", "opex")}) for k in c.menu]
    J, hot, ys, ent, legal, cash = joints(len(c.menu)), boom(c), [], [], [], None
    while not c.done:
        t, acts = c.state()["step"], []
        if t < c.years:
            jm = {q["id"]: q["jam"] for q in c.catalogue()}
            pj, m = [dict(q, jam=jm.get(q["id"], 0.0)) for q in proj], c.mask()
            A, C, _, B = forward(pol.W, *obs(c.state(), pj, m, c.years, None, [q["jam"] for q in pj]))
            ok = jmask(J, m)
            lg = joint(J, A, C, B)[ok]
            p = np.exp(lg - lg.max())
            p /= p.sum()
            ent.append(float(-(p * np.log(p + 1e-12)).sum()))
            legal.append(int(ok.sum()))
            acts = pol(c.state(), pj, m)
        elif cash is None:
            cash = c.state()["budget"]
        r = c.step(acts)
        ys.append([a for a in r["actions"] if a.get("ok")])
    res = c.result()
    b = [(t, a["build"]) for t, y in enumerate(ys[: c.years]) for a in y if "build" in a]
    mt = [next((a["maintain"] for a in y if "maintain" in a), 1) for y in ys[: c.years]]
    acc = [(t, k) for t, k in b if area_of(k)]
    return dict(seed=seed, score=c.score(), saved=res["saving"], builds=[k for _, k in b], years=[t for t, _ in b],
                spend=float(sum(c.cat[k]["cost"] for _, k in b)), idle=c.years - len({t for t, _ in b}), cash=cash,
                maintain=mt, access=len(acc), hit=len([1 for _, k in acc if area_of(k) in hot]), boom=len(hot),
                wrong=len([1 for _, k in acc if area_of(k) not in hot]), first_access=min([t for t, _ in acc], default=c.years),
                boom_access=float(np.mean([min([t for t, k in acc if area_of(k) == h], default=c.years) for h in hot])) if hot else None,
                first_build=min([t for t, _ in b], default=c.years), entropy=float(np.mean(ent)), legal=float(np.mean(legal)), menu=list(c.menu))


def summary(rows):
    n, menu = len(rows), rows[0]["menu"]
    mt = np.concatenate([r["maintain"] for r in rows])
    allb = [k for r in rows for k in r["builds"]]
    return dict(score=float(np.mean([r["score"] for r in rows])), scores=[r["score"] for r in rows],
                saved_ph=float(np.mean([r["saved"] for r in rows])), builds=float(np.mean([len(r["builds"]) for r in rows])),
                spend=float(np.mean([r["spend"] for r in rows])), idle_years=float(np.mean([r["idle"] for r in rows])),
                cash_end=float(np.mean([r["cash"] for r in rows])), maintain=[float(np.mean(mt == k)) for k in range(3)],
                freq={k: allb.count(k) / n for k in menu}, access_share=sum(r["access"] for r in rows) / max(len(allb), 1),
                boom_hit=sum(r["hit"] for r in rows) / max(sum(r["boom"] for r in rows), 1),
                wrong_access=float(np.mean([r["wrong"] for r in rows])),
                first_access_year=float(np.mean([r["first_access"] for r in rows])),
                boom_access_year=float(np.mean([r["boom_access"] for r in rows if r["boom_access"] is not None])),
                first_build_year=float(np.mean([r["first_build"] for r in rows])),
                distinct=len({tuple(zip(r["years"], r["builds"])) + tuple(r["maintain"]) for r in rows}),
                entropy=float(np.mean([r["entropy"] for r in rows])), legal=float(np.mean([r["legal"] for r in rows])),
                plans=[dict(seed=r["seed"], builds=list(zip(r["years"], r["builds"]))) for r in rows])


def main():
    names, seeds = sys.argv[1].split(","), list(range(16))
    os.environ["NUMBA_NUM_THREADS"] = "1"
    out_f = OUT / "strategy.json"
    out = json.loads(out_f.read_text()) if out_f.exists() else {}
    snaps = sorted((OUT / "snap").glob(f"*-{version()}-*.npz"))
    with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count(), get_context("spawn")) as pool:
        for f in snaps:
            name, ep = re.match(r"(.+)-[0-9a-f]{8}-(\d+)\.npz", f.name).groups()
            if name not in names or any(r["episodes"] == int(ep) for r in out.get(name, [])):
                continue
            rows = list(pool.map(plan, [f] * len(seeds), seeds))
            out.setdefault(name, []).append(dict(episodes=int(ep), **summary(rows)))
            out[name].sort(key=lambda r: r["episodes"])
            out_f.write_text(json.dumps(out))
            print(name, ep, round(out[name][-1]["score"], 3), flush=True)


if __name__ == "__main__":
    main()
