import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from failures import EV, acc, by_seed, ci, groups, meta, paired, play, runs, truth

ARMS = {"a_access": lambda r, t: dict(add=acc(t["fastest"])), "b_maint1": lambda r, t: dict(maint=1), "c_both": lambda r, t: dict(add=acc(t["fastest"]), maint=1)}


def job(x):
    return play(*x[:2], **x[2])


def summary(grp, rs, rows, base, ref):
    ag = by_seed([r for r in rs if r["role"] == grp], lambda r: base[r["job"]]["score"])
    common = sorted(set(ref) & set(ag))
    gap = {s: ref[s] - ag[s] for s in common}
    summ = dict(agent=ci([ag[s] for s in common], 10000), reference=ci([ref[s] for s in common], 10000), gap=dict(ci(list(gap.values())), per_seed=gap))
    for k in list(ARMS) + ["d_ref_maint2"]:
        rr = [x for x in rows if x["arm"] == k and (x["role"] == grp or k == "d_ref_maint2")]
        cf = by_seed(rr, lambda x: x["score"])
        d = {s: cf[s] - (ag if k != "d_ref_maint2" else ref)[s] for s in common if s in cf}
        share = {s: (d[s] if k != "d_ref_maint2" else -d[s]) / gap[s] if gap[s] else None for s in d}
        v, g = np.array([d[s] for s in d]), np.array([gap[s] for s in d])
        boot = np.random.default_rng(0).integers(0, len(d), (10000, len(d)))
        sg = 1 if k != "d_ref_maint2" else -1
        bs = sg * v[boot].sum(1) / g[boot].sum(1)
        summ[k] = dict(diff=dict(ci(list(d.values())), per_seed=d), gap_share_per_seed=share, gap_share_pooled=float(sg * v.sum() / g.sum()),
                       gap_share_lo=float(np.nanpercentile(bs, 2.5)), gap_share_hi=float(np.nanpercentile(bs, 97.5)),
                       test=paired([cf[s] for s in d], [ag[s] for s in d]) if sg > 0 else paired([ref[s] for s in d], [cf[s] for s in d]),
                       cf_beats_reference=sum(cf[s] > ref[s] for s in d) if sg > 0 else None)
    return summ


def main():
    rs = [r for r in runs() if r["role"] in ("a1", "gate4", "reference")]
    seeds = sorted({r["seed"] for r in rs})
    with Pool(16) as p:
        tr = dict(zip(seeds, p.map(truth, seeds)))
        tasks = [(r["seed"], r["plan"], {}) for r in rs] + [(r["seed"], r["plan"], kw) for r in rs
                                                           for kw in ([f(r, tr[r["seed"]]) for f in ARMS.values()] if r["role"] != "reference" else [dict(maint=2)])]
        res = p.map(job, tasks)
    base = {r["job"]: x for r, x in zip(rs, res)}
    it = iter(res[len(rs):])
    rows = []
    for r in rs:
        arms = {k: next(it) for k in ARMS} if r["role"] != "reference" else {"d_ref_maint2": next(it)}
        for k, v in arms.items():
            rows.append(dict(job=r["job"], seed=r["seed"], role=r["role"], arm=k, add=v["add"], added=None if v["added"] is None else 2026 + v["added"],
                             fastest=tr[r["seed"]]["fastest"], base=base[r["job"]]["score"], score=v["score"], diff=v["score"] - base[r["job"]]["score"],
                             raw_base=base[r["job"]]["raw"], raw=v["raw"], rejected=v["rejected"] - base[r["job"]]["rejected"], started=sorted(v["started"])))
    ref = by_seed([r for r in rs if r["role"] == "reference"], lambda r: base[r["job"]]["score"])
    summ = {g: summary(g, rs, rows, base, ref) for g in groups(rs)}
    out = dict(meta=meta(seeds, "counterfactual"), arms=dict(a_access="add the access upgrade of the truly fastest-growing area in 2028 or the first year it is accepted",
                                                            b_maint1="maintenance 1 in all planning years", c_both="a and b",
                                                            d_ref_maint2="reference plan with maintenance 2 in all planning years"), rows=rows, summary=summ)
    (EV / "counterfactual.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    o = main()
    for x in o["rows"]:
        print(x["job"], x["arm"], x["add"], x["added"], round(x["base"], 3), round(x["score"], 3), round(x["diff"], 3), "rej", x["rejected"])
    for grp, sm in o["summary"].items():
        for k, v in sm.items():
            if "gap_share_pooled" in v:
                print(grp, k, "share", round(v["gap_share_pooled"], 3), [round(v["gap_share_lo"], 3), round(v["gap_share_hi"], 3)], {s: round(y, 3) for s, y in v["gap_share_per_seed"].items() if y is not None}, v["test"])
