"""Experiments added after an external review (CPU only).

E1  First-move counterfactuals on the frontier agent's plans (seeds 0-4).
E2  Growth sensitivity of the learned reference policy (seeds 0-15).
E3  Headroom of the single-turn GRPO format (seeds 0-15).

Writes evidence/review.json, paper/figs/review_numbers.tex and paper/figs/review_*.tex.
Usage: python paper/figs/review.py   (from the repository root)
"""
import datetime
import json
import platform
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "paper" / "figs")]
from failures import EV, ci, paired, play as replay, runs, truth  # noqa: E402

VAL = list(range(16))
CONST = ["otaniemi-access", "kiviruukki-access", "espoonvayla-2x2"]


# ---------- E1: first-move counterfactuals ----------
def access(area):
    return area.split(" and ")[0].split()[0].lower().replace("ä", "a") + "-access"


def first_move(plan, maint=None):
    p = [[dict(a) for a in acts] for acts in plan]
    m0 = next((a["maintain"] for a in p[0] if "maintain" in a), 1)
    p[0] = [{"build": "espoonvayla-2x2"}, {"maintain": m0}]
    p = [p[0]] + [[a for a in acts if a.get("build") != "espoonvayla-2x2"] for acts in p[1:]]
    if maint is not None:
        p = [[a for a in acts if "maintain" not in a] + [{"maintain": maint}] for acts in p]
    return p


def swap_targets(plan, t):
    order = sorted(t["boom"], key=lambda n: -t["rate"][n])
    targets, used, p = [access(n) for n in order], set(), []
    for acts in plan:
        row = []
        for a in acts:
            b = a.get("build", "")
            if b.endswith("-access") and b not in {access(n) for n in t["boom"]}:
                nxt = next((x for x in targets if x not in used), None)
                a = {"build": nxt} if nxt else a
            if a.get("build"):
                used.add(a["build"])
            row.append(dict(a))
        p.append(row)
    return p


def e1_job(x):
    seed, plan = x
    return replay(seed, plan)


def e1():
    rs, out = runs(), []
    agent = [r for r in rs if r["role"] == "agent"]
    ref = {r["seed"]: r for r in rs if r["role"] == "reference"}
    seeds = sorted({r["seed"] for r in agent})
    with Pool(8) as p:
        tr = dict(zip(seeds, p.map(truth, seeds)))
        arms = {"base": lambda r: r["plan"], "first": lambda r: first_move(r["plan"]), "first_m1": lambda r: first_move(r["plan"], 1),
                "targets": lambda r: swap_targets(r["plan"], tr[r["seed"]]), "first_targets": lambda r: swap_targets(first_move(r["plan"]), tr[r["seed"]])}
        tasks = [(r["seed"], arms[k](r)) for r in agent for k in arms] + [(s, ref[s]["plan"]) for s in seeds]
        res = p.map(e1_job, tasks)
    it = iter(res)
    rows = {(r["seed"], k): next(it) for r in agent for k in arms}
    refs = {s: next(it)["score"] for s in seeds}
    summ = {}
    for k in arms:
        sc = {s: rows[(s, k)]["score"] for s in seeds}
        gap = {s: refs[s] - rows[(s, "base")]["score"] for s in seeds}
        d = {s: sc[s] - rows[(s, "base")]["score"] for s in seeds}
        summ[k] = dict(score=ci(list(sc.values())), diff=ci(list(d.values())), per_seed=sc,
                       gap_share=float(sum(d.values()) / max(sum(gap.values()), 1e-12)),
                       beats_reference=int(sum(sc[s] >= refs[s] for s in seeds)), test=paired(list(sc.values()), [rows[(s, "base")]["score"] for s in seeds]),
                       started={s: sorted(rows[(s, k)]["started"]) for s in seeds}, rejected={s: rows[(s, k)]["rejected"] for s in seeds})
    out = dict(reference=refs, arms=summ, truth=tr)
    return out


# ---------- E2: growth sensitivity of the learned reference ----------
def e2_job(x):
    seed, mode = x
    sys.path.insert(0, str(ROOT))
    from rl.policy import Policy
    from sim import City
    c, pol, rng = City(seed), Policy(ROOT / "rl" / "policy.npz"), np.random.default_rng(seed)
    menu = list(c.menu)
    proj = [dict(id=k, **{f: c.cat[k].get(f, 0) for f in ("kind", "cost", "years", "opex")}) for k in menu]
    perm = rng.permutation(len(c.areas))
    builds = []
    while not c.done:
        st, acts = c.state(), []
        if st["step"] < c.years:
            ar = [dict(a) for a in st["areas"]]
            if mode == "zero":
                for a in ar:
                    a["pop"], a["jobs"] = a["pop0"], a["jobs0"]
            elif mode == "shuffle":
                rp = [a["pop"] / max(a["pop0"], 1) for a in ar]
                rj = [a["jobs"] / max(a["jobs0"], 1) for a in ar]
                for i, a in enumerate(ar):
                    a["pop"], a["jobs"] = a["pop0"] * rp[perm[i]], a["jobs0"] * rj[perm[i]]
            st = dict(st, areas=ar)
            m, jam = np.array(c.mask(), bool), {q["id"]: q["jam"] for q in c.catalogue()}
            acts = pol(st, [dict(q, jam=jam.get(q["id"], 0.0)) for q in proj], m)
            builds += [(st["step"], a["build"]) for a in acts if "build" in a]
        c.step(acts)
    return dict(seed=seed, mode=mode, score=c.score(), builds=builds)


def e2():
    with Pool(8) as p:
        res = p.map(e2_job, [(s, m) for s in VAL for m in ("normal", "shuffle", "zero")])
    by = {(r["seed"], r["mode"]): r for r in res}
    out = {}
    for m in ("normal", "shuffle", "zero"):
        sc = [by[(s, m)]["score"] for s in VAL]
        out[m] = dict(score=ci(sc), changed=float(np.mean([by[(s, m)]["builds"] != by[(s, "normal")]["builds"] for s in VAL])),
                      test=paired([by[(s, "normal")]["score"] for s in VAL], sc), builds={s: by[(s, m)]["builds"] for s in VAL})
    return out


# ---------- E3: headroom of the single-turn GRPO format ----------
def e3_job(x):
    sys.path.insert(0, str(ROOT))
    from grpo import env
    seed, name, ids, k = x
    if name == "search":
        return dict(env.search(seed, ids, 40), name=name)
    return dict(env.play(seed, ids, k), name=name)


def e3():
    sys.path.insert(0, str(ROOT))
    from grpo import env
    with Pool(8) as p:
        infos = dict(zip(VAL, p.map(env.info, VAL)))
        lists = {}
        for s in VAL:
            top = [env.access(n) for n, _, _ in infos[s]["areas"][:3]]
            lists[s] = dict(const=(CONST, 2), growth_m1=(env.FIXED[:2] + top, 1), growth_m2=(env.FIXED[:2] + top, 2),
                            growfirst_m1=(top + env.FIXED[:2], 1), growfirst_m2=(top + env.FIXED[:2], 2),
                            reflike_m1=(["espoonvayla-2x2"] + top[:2], 1), reflike_m2=(["espoonvayla-2x2"] + top[:2], 2))
        res = p.map(e3_job, [(s, n, ids, k) for s in VAL for n, (ids, k) in lists[s].items()])
        by = {(r["seed"], r["name"]): r for r in res}
        start = {s: max((by[(s, n)] for n in lists[s]), key=lambda r: r["frac"])["plan"] for s in VAL}
        srch = p.map(e3_job, [(s, "search", start[s], None) for s in VAL])
    for r in srch:
        by[(r["seed"], "search")] = r
    names = list(lists[0]) + ["search"]
    out = {n: dict(score=ci([by[(s, n)]["score"] for s in VAL]), per_seed={s: by[(s, n)]["score"] for s in VAL},
                   vs_const=paired([by[(s, n)]["score"] for s in VAL], [by[(s, "const")]["score"] for s in VAL])) for n in names}
    out["best_listed"] = dict(score=ci([max(by[(s, n)]["score"] for n in lists[s]) for s in VAL]))
    return out


def f3(x):
    return f"{x:.3f}"


def main():
    t0 = datetime.datetime.now(datetime.timezone.utc)
    r1, r2, r3 = e1(), e2(), e3()
    meta = dict(script="paper/figs/review.py", command="python paper/figs/review.py", date=t0.isoformat(timespec="seconds"),
                machine=f"{platform.node()} {platform.machine()} {platform.processor()}", note="Apple silicon; scores may differ from Intel runs by up to about 0.008")
    (EV / "review.json").write_text(json.dumps(dict(meta=meta, first_move=r1, reference_sensitivity=r2, grpo_headroom=r3), indent=1, ensure_ascii=False, default=float))
    M = {}
    a = r1["arms"]
    M.update(cfFirstScore=f3(a["first"]["score"]["mean"]), cfFirstDiff=f3(a["first"]["diff"]["mean"]), cfFirstGapShare=f"{a['first']['gap_share']:.2f}",
             cfFirstBeatsRef=f"{a['first']['beats_reference']}/5", cfFirstMOneScore=f3(a["first_m1"]["score"]["mean"]),
             cfFirstMOneGapShare=f"{a['first_m1']['gap_share']:.2f}", cfTargetsScore=f3(a["targets"]["score"]["mean"]), cfTargetsGapShare=f"{a['targets']['gap_share']:.2f}",
             cfFirstTargetsScore=f3(a["first_targets"]["score"]["mean"]), cfFirstTargetsGapShare=f"{a['first_targets']['gap_share']:.2f}",
             cfAgentBase=f3(a["base"]["score"]["mean"]))
    M.update(refSensShuffle=f"{r2['shuffle']['changed']:.2f}", refSensZero=f"{r2['zero']['changed']:.2f}",
             refScoreShuffle=f3(r2["shuffle"]["score"]["mean"]), refScoreZero=f3(r2["zero"]["score"]["mean"]), refScoreNormal=f3(r2["normal"]["score"]["mean"]),
             refShuffleLoss=f"{r2['shuffle']['test']['wins']}/16")
    for n, key in (("const", "Const"), ("growth_m2", "Growth"), ("growfirst_m2", "Growfirst"), ("reflike_m1", "ReflikeOne"), ("reflike_m2", "ReflikeTwo"),
                   ("search", "Search"), ("best_listed", "BestListed")):
        M[f"fmt{key}"] = f3(r3[n]["score"]["mean"])
    lines = "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items()))
    (ROOT / "paper" / "figs" / "review_numbers.tex").write_text("% written by paper/figs/review.py\n" + lines + "\n")
    rows = [("Agent plan as played", "base"), ("First move $\\to$ \\texttt{espoonvayla-2x2}", "first"),
            ("First move replaced, maintenance 1", "first_m1"), ("Access targets $\\to$ truly growing areas, same years", "targets"),
            ("First move replaced and targets swapped", "first_targets")]
    t = [f"{lab} & {f3(a[k]['score']['mean'])} [{f3(a[k]['score']['lo'])}, {f3(a[k]['score']['hi'])}] & {a[k]['gap_share']:.2f} & {a[k]['beats_reference']}/5 \\\\" for lab, k in rows]
    (ROOT / "paper" / "figs" / "review_firstmove.tex").write_text("\n".join(t) + "\n")
    labels = [("Constant GRPO plan, $m{=}2$", "const"), ("Cheap links, then top-3 growing areas, $m{=}2$", "growth_m2"),
              ("Top-3 growing areas first, $m{=}2$", "growfirst_m2"), ("\\texttt{espoonvayla-2x2}, then top-2 growing areas, $m{=}1$", "reflike_m1"),
              ("\\texttt{espoonvayla-2x2}, then top-2 growing areas, $m{=}2$", "reflike_m2"),
              ("Best of the lists above, per seed", "best_listed"), ("Local search per seed (40 steps, uses the simulator)", "search")]
    t = [f"{lab} & {f3(r3[k]['score']['mean'])} [{f3(r3[k]['score']['lo'])}, {f3(r3[k]['score']['hi'])}] \\\\" for lab, k in labels]
    (ROOT / "paper" / "figs" / "review_format.tex").write_text("\n".join(t) + "\n")
    for k, v in M.items():
        print(k, v)


if __name__ == "__main__":
    main()
