import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import binomtest, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
EVID = ROOT / "evidence"
Y0 = 2026

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"], "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
INK, MUTED = "#26251f", "#8a8984"
COL = dict(nothing="#9b9a94", random="#c3c2b7", greedy1="#1baf7a", greedy5="#199e70", reactive="#2a78d6", agent="#eb6834", planner="#008300",
           rl="#6250d6", rlcheck="#9085e9", rlsearch="#3f2fb0", grpo="#e87ba4", best=INK)
NAME = dict(nothing="Do nothing", random="Random", greedy1="Greedy (1-year)", greedy5="Greedy (5-year)", reactive="Reactive",
            agent="Frontier LLM agent (5 older runs)", planner="Rule planner", rl="Learned policy", rlcheck="Learned + forecast check",
            rlsearch="Learned + search", grpo="GRPO Gemma 270M", best="Best known")
KEYS = list(NAME)

load = lambda p: json.loads(Path(p).read_text())
norms, runs, growth, booms = load(ROOT / "sim" / "norms.json"), load(DATA / "runs.json"), load(DATA / "growth.json"), load(DATA / "booms.json")
RLD, GRD = load(DATA / "rl.json"), load(DATA / "grpo.json")
ASEEDS = sorted(runs["agent"], key=int)
LEVEL, M, TLEVEL = {}, {}, {}
BASE_JOBS, MAIN_JOB = {"gate4", "a1"}, "a1"
key = lambda n: n.split()[0].lower().replace("ä", "a")
tot = lambda r: sum(r["delay"])
raw = lambda s, d: (norms[s][0] - d) / (norms[s][0] - norms[s][1])
clip = lambda x: float(np.clip(x, 0, 1))
mean = lambda v: np.mean([r["score"] for r in v.values()])
first = lambda rows, k, cond: next((r[k] for r in rows if cond(r)), None)
hits = lambda s, p: p.endswith("-access") and any(p.startswith(key(n)) for n in booms[s])


def rescore():
    for k, v in runs.items():
        for s, r in (v.items() if k != "viz" else ()):
            r["raw"] = raw(s, tot(r))
            r["score"] = clip(r["raw"])
    for base in ("nothing", "random", "greedy1", "greedy5", "reactive"):
        opts = {int(k[-1]): runs[k] for k in (f"{base}@1", f"{base}@2") if k in runs}
        LEVEL[base] = max(opts, key=lambda m: mean(opts[m]))
        runs[base] = opts[LEVEL[base]]
        if base in ("reactive", "greedy5"):
            for m, v in opts.items():
                M[f"{base.replace('5', 'Five')}M{['', 'One', 'Two'][m]}"] = f"{mean(v):.2f}"
    runs.update(planner=runs["plan@x"], rl=runs["rl@x"], rlcheck=runs["rlcheck@x"])


def grpo_curve():
    g = GRD
    for e in g["evals"]:
        for x in e["seeds"]:
            x["score"] = clip(x["greedy"]["saving"] * 1e6 / (norms[str(x["seed"])][0] - norms[str(x["seed"])][1])) if x.get("greedy") else 0.0
        e["score"] = float(np.mean([x["score"] for x in e["seeds"]])) if e["seeds"] else e["score"]
    ev = [e for e in g["evals"] if e["step"] >= 0]
    zero = next(e for e in g["evals"] if e["step"] < 0)
    zs = zero["score"] if zero["score"] == zero["score"] else 0.0
    M.update(grpoZero=f"{zs:.2f}", grpoWarm=f"{ev[0]['score']:.2f}", grpoFinal=f"{ev[-1]['score']:.2f}", grpoValid=f"{ev[-1]['valid']:.2f}",
             grpoSteps=str(ev[-1]["step"]))
    last = [x.get("greedy") or {} for x in ev[-1]["seeds"]]
    plans = [tuple(g_.get("plan", [])) for g_ in last]
    M["grpoPlan"] = ", ".join(f"\\texttt{{{q}}}" for q in max(set(plans), key=plans.count))
    return ev, zs, [x["score"] for x in ev[-1]["seeds"]]


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf")


def fig_problem():
    fig, ax = plt.subplots(1, 3, figsize=(5.5, 2.0), gridspec_kw=dict(wspace=0.55))
    g = growth["1"]
    yrs = np.arange(Y0, Y0 + len(g["index"]))
    boom = [n for n, r in g["rate"].items() if r > 0.01]
    for n in g["rate"]:
        ax[0].plot(yrs, [r[n] for r in g["index"]], color="#eb6834" if n in boom else MUTED, lw=1.6 if n in boom else 1)
    ax[0].text(Y0 + 0.5, 1.95, f"{len(boom)} boom areas", fontsize=8, color="#eb6834", va="center")
    ax[0].text(Y0 + 14, 1.12, f"{len(g['rate']) - len(boom)} others", fontsize=8, color=MUTED, ha="right")
    ax[0].axvspan(Y0 + 15, yrs[-1], color="#f1f0ec", lw=0, zorder=0)
    ax[0].set(title="(a) Growth, seed 1", ylabel="pop. + jobs vs. 2026", xlim=(Y0, yrs[-1]), xticks=[2026, 2035, 2045])
    cat = load(ROOT / "data" / "catalogue.json")
    real = [p for p in cat if not p["id"].endswith("-access")]
    acc = [p for p in cat if p["id"].endswith("-access")]
    ax[1].scatter([p["years"] for p in real], [p["cost"] for p in real], s=14, color="#2a78d6", edgecolor="white", lw=0.5, zorder=3)
    ax[1].scatter([acc[0]["years"]], [acc[0]["cost"]], s=40, color="#6250d6", edgecolor="white", lw=0.5, zorder=3)
    ax[1].annotate(f"{len(acc)} access\nupgrades", (acc[0]["years"], acc[0]["cost"]), xytext=(8, 0), textcoords="offset points", fontsize=8, color="#6250d6", va="center")
    ax[1].text(3.1, 60, f"{len(real)} real\nprojects", fontsize=8, color="#2a78d6", va="center")
    ax[1].axhline(40, color=MUTED, lw=0.8, ls="--")
    ax[1].text(4.4, 30, "income/yr", fontsize=8, color=MUTED, ha="right")
    ax[1].set(title="(b) Project menu", xlabel="build years", ylabel="cost (MEUR)", xticks=[2, 3, 4], xlim=(1.6, 4.5), ylim=(0, 135))
    for s, c in zip("012", ["#9b9a94", INK, "#c3c2b7"]):
        d = np.array(runs["nothing"][s]["delay"]) / 1e6
        ax[2].plot(np.arange(Y0, Y0 + len(d)), d, color=c, lw=1.4, label=f"seed {s}")
    ax[2].set(title="(c) Delay, do nothing", ylabel="M person-hours / year", xticks=[2026, 2035, 2045])
    ax[2].legend(loc="upper left")
    save(fig, "problem")


def fig_bars(vals, lost=None, name="bars"):
    KEYS = [k for k in globals()['KEYS'] if k in vals]
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.9), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.75))
    for i, k in enumerate(KEYS):
        y, v = len(KEYS) - 1 - i, np.array(vals[k], float)
        ax[0].barh(y, v.mean(), 0.68, color=COL[k], zorder=2)
        ax[0].scatter(v, np.full(v.size, y) + np.linspace(-0.2, 0.2, v.size), s=3, color=INK, zorder=3, lw=0)
        ax[0].text(1.04, y, f"{v.mean():.2f}" + (f" (n={v.size})" if v.size != 16 else ""), va="center", fontsize=8, color=INK)
    ax[0].set_yticks(range(len(KEYS)), [NAME[k] for k in KEYS[::-1]])
    ax[0].set(xlim=(-0.02, 1.0), xlabel="normalized score (clipped to [0, 1])", title="(a) Score per seed (dots) and mean (bar)")
    ax[0].grid(axis="y", visible=False)
    ks = ("nothing", "reactive", "agent", "planner", "rl", "best")
    w = 0.8 / len(ks)
    for j, k in enumerate(ks):
        v = [lost(k, s) / 1e6 if lost else norms[s][1] / 1e6 if k == "best" else tot(runs[k][s]) / 1e6 for s in ASEEDS]
        ax[1].bar(np.arange(len(ASEEDS)) + (j - (len(ks) - 1) / 2) * w, v, w * 0.92, color=COL[k], label=NAME[k], zorder=2)
    ax[1].set_xticks(range(len(ASEEDS)), ASEEDS)
    ax[1].set(xlabel="seed", ylabel="M person-hours, 2026 to 2045", title="(b) Net person-hours lost")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2, handlelength=1, columnspacing=0.8)
    ax[1].grid(axis="x", visible=False)
    save(fig, name)


def fig_years():
    fig, ax = plt.subplots(figsize=(2.7, 2.2))
    for k in ("nothing", "agent", "rl"):
        r = runs[k]["1"]
        d = np.array(r["delay"]) / 1e6
        ax.plot(np.arange(Y0, Y0 + len(d)), d, color=COL[k], lw=1.6, label=NAME[k])
        for i, b in enumerate(r["builds"]):
            if k != "nothing" and any(p.endswith("-access") for p in b):
                ax.scatter(Y0 + i, d[i], marker="o", s=16, facecolor="white", edgecolor=COL[k], lw=1.2, zorder=5)
    ax.axvspan(Y0 + 15, Y0 + 19, color="#f1f0ec", lw=0, zorder=0)
    ax.text(Y0 + 17, 0.6, "runout", fontsize=8, color=MUTED, ha="center")
    ax.set(ylabel="M person-hours / year", ylim=(0, None), xticks=[2026, 2031, 2036, 2041, 2045])
    ax.legend(loc="upper left", handlelength=1.2)
    save(fig, "years")


def fig_curves(ev, zs):
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3), gridspec_kw=dict(wspace=0.3))
    for a in ax:
        for k in ("agent", "reactive"):
            y = mean(runs[k])
            a.axhline(y, color=COL[k], lw=0.8, ls="--")
            a.text(0.02, y + 0.01, NAME[k], transform=a.get_yaxis_transform(), fontsize=8, color=COL[k], ha="left", va="bottom")
        a.set_ylim(-0.05, 1.05)
    for n, c, lab in (("ppocold2", "#86b6ef", "PPO from scratch"), ("gz2", "#6250d6", "Gumbel self-play from scratch")):
        x, y = zip(*[(r.get("episodes", r["it"]), r["mean_net"]) for r in RLD["curves"][n]])
        ax[0].plot(x, y, color=c, lw=1.4, marker="o", ms=2.5, label=lab)
    ax[0].legend(loc="lower right")
    ax[0].set(xlabel="training episodes", ylabel="mean score, 16 seeds", title="(a) Learning without expert plans")
    ax[1].plot([e["step"] for e in ev], [e["score"] for e in ev], color=COL["grpo"], lw=1.4, marker="o", ms=2.5, label="greedy decoding")
    ax[1].scatter([ev[0]["step"]], [zs], color=INK, s=12, zorder=4, label="zero-shot")
    ax[1].legend(loc="lower right")
    ax[1].set(xlabel="GRPO step", ylabel="mean score, 16 seeds", title="(b) GRPO fine-tuning of Gemma 3 270M")
    save(fig, "curves")


def fig_evolution(rl, gs):
    fig = plt.figure(figsize=(5.5, 7.4))
    grid = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.45, height_ratios=[1, 1, 1.5])
    ax = fig.add_subplot(grid[0, :])
    x = [r["step"] for r in gs]
    for lab, c, fn, ls in (("score", COL["grpo"], lambda r: r["score"], "-"), ("distinct samples / 8", INK, lambda r: r["sample_distinct"] / 8, "-"),
                           ("share $m=2$", COL["reactive"], lambda r: r["maintain"]["2"], "-"), ("boom hit", COL["agent"], lambda r: r["boom_hit"], "-"),
                           ("reward std (train)", MUTED, lambda r: r["train_std"], "--")):
        ax.plot(x, [fn(r) for r in gs], color=c, lw=1.4, ls=ls, marker="o", ms=2.5, label=lab)
    for v, t in ((M["grpoOneAt"], "one plan"), (M["grpoFlatAt"], "std 0")):
        ax.axvline(int(v), color=MUTED, lw=0.8, ls=":")
        ax.text(int(v), 1.03, t, fontsize=8, color=MUTED, ha="center")
    ax.legend(loc="center right", ncol=1, fontsize=7)
    ax.set(xlabel="GRPO step", ylim=(-0.02, 1.1), title="(a) GRPO on Gemma")
    for j, (k, lab) in enumerate((("ppocold2", "PPO from scratch"), ("gz2", "Gumbel from scratch"), ("ppobc1s", "PPO from cloned policy"))):
        a, rows = fig.add_subplot(grid[1, j]), rl[k]
        for y, l, c in (([r["score"] for r in rows], "score", COL["rl"]), ([r["maintain"][2] for r in rows], "share $m=2$", COL["reactive"]),
                        ([r["access_share"] for r in rows], "access share", MUTED), ([r["boom_hit"] for r in rows], "boom hit", COL["agent"])):
            a.plot([r["episodes"] for r in rows], y, color=c, lw=1.3, marker="o", ms=2, label=l)
        a.set(xlabel="episodes", ylim=(-0.05, 1.05))
        a.set_title(f"({'bcd'[j]}) {lab}", fontsize=8)
        if j == 0:
            a.legend(loc="upper left", fontsize=7)
    hm, rows = fig.add_subplot(grid[2, :]), rl["ppocold2"]
    ids = list(rows[0]["freq"])
    F = np.array([[r["freq"][i] for r in rows] for i in ids], float)
    hm.imshow(F, aspect="auto", cmap="Purples", vmin=0, vmax=max(F.max(), 1e-9), interpolation="nearest")
    hm.set_yticks(range(len(ids)), ids, fontsize=7)
    hm.set_xticks(range(len(rows)), [str(r["episodes"]) for r in rows], fontsize=7, rotation=90)
    hm.set(xlabel="episodes", title="(e) Share of cities in which each project is built, PPO from scratch")
    hm.grid(False)
    save(fig, "evolution")


def strategy():
    rl, gs = RLD["strategy"], GRD["strategy"]
    for k, t in (("ppocold2", "Cold"), ("gz2", "Gumbel"), ("ppobc1s", "Bc")):
        rows, z, one = rl[k], rl[k][-1], rl[k][1]
        acc = next((r for r in rows if r["access_share"] > 0), None)
        M.update({f"evo{t}Eps": str(z["episodes"]), f"evo{t}Score": f"{z['score']:.2f}", f"evo{t}ScoreZero": f"{rows[0]['score']:.2f}",
                  f"evo{t}EpsOne": str(one["episodes"]), f"evo{t}ScoreOne": f"{one['score']:.2f}", f"evo{t}MtwoOne": f"{one['maintain'][2]:.2f}",
                  f"evo{t}BuildsOne": f"{one['builds']:.1f}", f"evo{t}MaxAcc": f"{max(r['access_share'] for r in rows):.2f}",
                  f"evo{t}EntZero": f"{rows[0]['entropy']:.2f}", f"evo{t}Ent": f"{z['entropy']:.2f}", f"evo{t}Acc": f"{z['access_share']:.2f}",
                  f"evo{t}Boom": f"{z['boom_hit']:.2f}", f"evo{t}BoomZero": f"{rows[0]['boom_hit']:.2f}", f"evo{t}Mone": f"{z['maintain'][1]:.2f}",
                  f"evo{t}Distinct": str(z["distinct"]), f"evo{t}DistinctZero": str(rows[0]["distinct"]),
                  f"evo{t}Ota": f"{z['freq'].get('otaniemi-access', 0):.2f}", f"evo{t}OtaZero": f"{rows[0]['freq'].get('otaniemi-access', 0):.2f}"})
        if acc:
            M.update({f"evo{t}AccAt": str(acc["episodes"]), f"evo{t}AccYear": f"{2026 + acc['first_access_year']:.0f}",
                      f"evo{t}AccScore": f"{acc['score']:.2f}", f"evo{t}AccShare": f"{acc['access_share']:.2f}",
                      f"evo{t}NoAccUntil": str(max([r["episodes"] for r in rows if r["episodes"] < acc["episodes"]], default=0))})
    w, z = gs[0], gs[-1]
    for nm, cond in (("grpoValidAt", lambda r: r["valid"] >= 0.9), ("grpoOneAt", lambda r: r["sample_distinct"] <= 1.5),
                     ("grpoFlatAt", lambda r: r["train_std"] == 0), ("grpoMtwoAt", lambda r: r["maintain"]["2"] == 1)):
        M[nm] = str(first(gs, "step", cond))
    kept = [p for p in z["freq"] if z["freq"][p] >= 0.999]
    M.update(grpoSdWarm=f"{w['sample_distinct']:.1f}", grpoBoomShare=f"{z['boom_share']:.2f}", grpoBoomHitEnd=f"{z['boom_hit']:.2f}",
             grpoBoomHitMax=f"{max(r['boom_hit'] for r in gs):.2f}", grpoLastStep=str(z["step"]), grpoScoreEnd=f"{z['score']:.2f}",
             grpoScoreWarmS=f"{w['score']:.2f}", grpoKept=", ".join(f"\\texttt{{{p}}}" for p in kept),
             grpoDropFirst=", ".join(f"\\texttt{{{p}}}" for p in w["freq"] if w["freq"][p] > 0 and gs[1]["freq"].get(p, 0) == 0),
             grpoKeptAt=str(first(gs, "step", lambda r: all(r["freq"].get(p, 0) >= 0.999 for p in kept))))
    fig_evolution(rl, gs)
    return z["scores"]


def ablation():
    by = {}
    for r in RLD["ablation"]:
        sc = [clip(x) for x in r["scores"]]
        by.setdefault(r["experts"], {})[r["jam"]] = (np.mean(sc), min(sc))
    es = sorted(by)
    (HERE / "ablation.tex").write_text("".join(f"{e} & {by[e][True][0]:.2f} & {by[e][True][1]:.2f} & {by[e][False][0]:.2f} & {by[e][False][1]:.2f} \\\\\n" for e in es))
    gap = lambda e: f"{by[e][True][0] - by[e][False][0]:.2f}"
    M.update(jamFew=str(es[0]), jamFewGap=gap(es[0]), jamFewMinWith=f"{by[es[0]][True][1]:.2f}", jamFewMinWithout=f"{by[es[0]][False][1]:.2f}",
             jamMany=str(es[-1]), jamManyGap=gap(es[-1]))


def heldout():
    h = {(r["policy"], r["part"]): r for r in RLD["heldout"]}
    ks = ("rl", "planner", "reactive", "greedy5", "greedy1")
    pct = lambda x: f"{100 * x:.1f}"
    (HERE / "heldout.tex").write_text("".join(f"{NAME[k]} & {pct(h[k, 'heldout']['mean'])} & {pct(min(h[k, 'heldout']['saved_frac']))} & {pct(h[k, 'eval']['mean'])} \\\\\n" for k in ks))
    wins = lambda o: sum(a > b for a, b in zip(h["rl", "heldout"]["saved_frac"], h[o, "heldout"]["saved_frac"]))
    M.update(hoN=str(len(h["rl", "heldout"]["seeds"])), hoSeeds=f"{min(h['rl', 'heldout']['seeds'])} to {max(h['rl', 'heldout']['seeds'])}",
             hoRl=pct(h["rl", "heldout"]["mean"]), hoRlEval=pct(h["rl", "eval"]["mean"]), hoGap=pct(h["rl", "eval"]["mean"] - h["rl", "heldout"]["mean"]),
             hoWinPlanner=str(wins("planner")), hoWinReactive=str(wins("reactive")))


TABLE_POL = dict(nothing=["nothing"], random=["random"], greedy1=["greedy1", "greedy1:2"], greedy5=["greedy5", "greedy5:2"], reactive=["reactive", "reactive:2"],
                 planner=["planner"], rl=["deploy"], rlcheck=["rlcheck"])


def table_rows():
    V, out = load(EVID / "test.json")["validation"]["policies"], {}
    for k, opts in TABLE_POL.items():
        p = max((o for o in opts if o in V), key=lambda o: np.mean([x["score"] for x in V[o].values()]))
        out[k] = {s: dict(score=x["score"], delay=x["delay"], saved=x["saved_frac"]) for s, x in V[p].items()}
        if ":" in p or len(opts) > 1:
            TLEVEL[k] = int(p.split(":")[1]) if ":" in p else 1
    ag = {}
    for p in EVID.glob(f"{MAIN_JOB}-s*/*/plan.json"):
        r = load(p)
        if r.get("valid", True):
            ag.setdefault(str(r["seed"]), []).append(r)
    out["agent"] = {k: dict(score=np.mean([r["score"] for r in v]), delay=np.mean([r["delay"] for r in v]), saved=np.mean([1 - r["delay"] / norms[k][0] for r in v])) for k, v in ag.items()}
    g = load(EVID / "grpo.json")["methods"]["runs/grpo/ckpt-90"]["seeds"]
    out["grpo"] = {s: dict(score=x["score"], delay=x["delay"] * 1e6, saved=x["frac"]) for s, x in g.items() if int(s) < 16}
    se = RLD["search"]
    out["rlsearch"] = {str(s): dict(score=x, delay=None, saved=None) for s, x in zip(se["seeds"], se["scores"])}
    out["best"] = {s: dict(score=1.0, delay=norms[s][1], saved=1 - norms[s][1] / norms[s][0]) for s in map(str, range(16))}
    return out


def table(vals, search, grpo):
    rows, n3 = [], sum(norms[s][0] for s in ASEEDS)
    for k in KEYS:
        if k == "grpo":  # the 270M run is no longer in the paper; the 4B runs have their own table
            continue
        name = NAME[k] + (f" ($m{{=}}{LEVEL[k]}$)" if k in LEVEL and k != "nothing" else "")
        if k in ("rlsearch", "grpo"):
            sc = [search[s] for s in ASEEDS] if k == "rlsearch" else [grpo[int(s)] for s in ASEEDS]
            rows.append(f"{name} & {np.mean(vals[k]):.2f} & {np.mean(sc):.2f} & -- & -- & -- & -- & -- \\\\")
            continue
        if k == "best":
            lost, last, spend, ext = [norms[s][1] for s in ASEEDS], runs["viz"]["best"]["delay"][-1] / 1e6, sum(runs["viz"]["best"]["spent"]), "$^\\dagger$"
        else:
            lost = [norms[s][0] if k == "nothing" else tot(runs[k][s]) for s in ASEEDS]
            last, spend, ext = np.mean([runs[k][s]["delay"][-1] for s in ASEEDS]) / 1e6, np.mean([sum(runs[k][s]["spent"]) for s in ASEEDS]), ""
        L = sum(lost)
        s16 = f"{abs(np.mean(vals[k])):.2f}" if len(vals[k]) == 16 else "n/a"
        s3 = np.mean([clip(raw(s, d)) for s, d in zip(ASEEDS, lost)])
        rows.append(f"{name} & {s16} & {s3:.2f} & {L / 1e6:.1f} & {(n3 - L) / 1e6:.1f} & {100 * (n3 - L) / n3:.1f} & {last:.1f}{ext} & {spend:.0f}{ext} \\\\")
    (HERE / "table.tex").write_text("\n".join(rows) + "\n")


def results(T):
    ag = T["agent"]
    rows = []
    for k in KEYS:
        v = T[k]
        seeds = sorted(v, key=int)
        sc = [v[s]["score"] for s in seeds]
        m, lo, hi = boot(sc)
        name = "Frontier LLM agent (3 runs per seed)" if k == "agent" else NAME[k] + (f" ($m{{=}}{TLEVEL[k]}$)" if k in TLEVEL and k != "nothing" else "")
        dl = [v[s]["delay"] for s in seeds]
        dcell = f"{np.mean(dl) / 1e6:.1f}" if None not in dl else "--"
        scell = f"{100 * np.mean([v[s]['saved'] for s in seeds]):.1f}" if None not in dl else "--"
        beats = "--" if k == "agent" else f"{sum(v[s]['score'] > ag[s]['score'] for s in seeds if s in ag)}/{sum(s in ag for s in seeds)}"
        rows.append(f"{name} & {len(seeds)} & {neg(m)} & [{neg(lo)}, {neg(hi)}] & {neg(min(sc))} & {dcell} & {scell} & {beats} \\\\")
        t = k.replace("1", "One").replace("5", "Five")
        M.update({f"tab{t[0].upper() + t[1:]}Score": neg(m), f"tab{t[0].upper() + t[1:]}Saved": scell})
    (HERE / "results.tex").write_text("\n".join(rows) + "\n")


def facts():
    a = runs["agent"]
    calls = lambda t: [r["calls"].get(t, 0) for r in a.values()]
    sc = [r["score"] for r in a.values()]
    hit = [(i, p) for s in a for i, b in enumerate(a[s]["builds"]) for p in b if hits(s, p)]
    M.update(agentUsdMin=f"{min(r['usd'] for r in a.values()):.2f}", agentUsdMax=f"{max(r['usd'] for r in a.values()):.2f}",
             agentMinMin=f"{min(r['minutes'] for r in a.values()):.0f}", agentMinMax=f"{max(r['minutes'] for r in a.values()):.0f}",
             agentScoreMean=f"{np.mean(sc):.2f}", agentN=str(len(sc)), agentSeeds=f"{ASEEDS[0]} to {ASEEDS[-1]}",
             agentGetStateMin=str(min(calls("espoo_get_state"))), agentGetStateMax=str(max(calls("espoo_get_state"))),
             agentForecastMin=str(min(calls("espoo_forecast"))), agentForecastMax=str(max(calls("espoo_forecast"))),
             nBooms=str(sum(len(booms[s]) for s in a)), agentBoomHits=str(len(hit)), agentBoomEarly=str(sum(i <= 5 for i, _ in hit)),
             agentLostC=f"{tot(a['2']) / 1e6:.1f}", rlLostC=f"{tot(runs['rl']['2']) / 1e6:.1f}", refoldLostC=f"{tot(runs['refold']['2']) / 1e6:.1f}",
             gapRlA=f"{(tot(a['0']) - tot(runs['rl']['0'])) / 1e6:.0f}")
    fh = [min([i for i, b in enumerate(v["builds"]) for p in b if hits(s, p)], default=None) for s, v in runs["rl"].items()]
    f = [(n - b) / n for n, b in norms.values()]
    d = [abs(tot(r) - tot(runs["rl"][s])) / tot(runs["rl"][s]) for s, r in runs["refrl"].items()]
    rat = [r["delay"][-1] / r["delay"][0] for r in runs["nothing"].values()]
    M.update(rlBoomEarly=str(sum(y is not None and y <= 3 for y in fh)), rlBoomAny=str(sum(y is not None for y in fh)), rlN=str(len(fh)),
             bestPctMin=f"{100 * min(f):.1f}", bestPctMax=f"{100 * max(f):.1f}", refrlN=str(len(d)), refrlGap=f"{100 * max(d):.2f}",
             nothingAbsMax=f"{max(abs(r['raw']) for r in runs['nothing'].values()):.3f}", nothingRatioMin=f"{min(rat):.1f}", nothingRatioMax=f"{max(rat):.1f}",
             rlScore=f"{mean(runs['rl']):.2f}", rlcheckScore=f"{mean(runs['rlcheck']):.2f}")


def cats():
    cat = load(ROOT / "data" / "catalogue.json")
    acc = lambda p: p["id"].endswith("-access")
    groups = {"New roads and bridges": [p for p in cat if p["kind"] == "road"], "Widenings (extra lanes)": [p for p in cat if p["kind"] == "lanes" and not acc(p)],
              "Junction rebuilds": [p for p in cat if p["kind"] == "junction"], "Area access upgrades": [p for p in cat if acc(p)]}
    rng = lambda v: f"{min(v):g}" if min(v) == max(v) else f"{min(v):g} to {max(v):g}"
    (HERE / "cats.tex").write_text("".join(f"{k} & {len(v)} & {rng([p['cost'] for p in v])} & {rng([p['years'] for p in v])} \\\\\n" for k, v in groups.items()))


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    m = v[np.random.default_rng(seed).integers(0, v.size, (n, v.size))].mean(1)
    return v.mean(), *np.percentile(m, [2.5, 97.5])


def paired(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    nz = d[d != 0]
    sign = binomtest(int((nz > 0).sum()), nz.size, 0.5, alternative="greater").pvalue if nz.size else 1.0
    wil = wilcoxon(nz, alternative="greater").pvalue if nz.size else 1.0
    return int((d > 0).sum()), d.size, sign, wil


def pf(p):
    if p >= 0.001:
        return f"{p:.2g}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times 10^{{{int(e)}}}$"


TESTKEY = {"deploy": "Rl", "rlcheck": "Rlcheck", "planner": "Planner", "reactive:2": "Reactive", "reactive": "ReactiveMOne", "greedy1:2": "GreedyOne", "greedy1": "GreedyOneMOne",
           "greedy5:2": "GreedyFive", "greedy5": "GreedyFiveMOne", "random": "Random", "nothing": "Nothing"}


def test_tables(d):
    for part, pre in (("validation", "val"), ("test", "test")):
        pol = d[part]["policies"]
        ref, rows = pol["deploy"], []
        for p, v in pol.items():
            seeds, k = sorted(v, key=int), pre + TESTKEY.get(p, p)
            sf = [v[s]["saved_frac"] for s in seeds]
            m, lo, hi = boot(sf)
            M.update({k: f"{100 * m:.1f}", f"{k}Lo": f"{100 * lo:.1f}", f"{k}Hi": f"{100 * hi:.1f}", f"{k}Worst": neg(100 * min(sf), "{:.1f}")})
            sc = [v[s]["score"] for s in seeds]
            cells = [NAME.get({"deploy": "rl"}.get(p, p).split(":")[0], p) + (f" ($m{{=}}{p.split(':')[1]}$)" if ":" in p else ""), f"{100 * m:.1f}", f"[{100 * lo:.1f}, {100 * hi:.1f}]", neg(100 * min(sf), "{:.1f}")]
            if None not in sc:
                sm, slo, shi = boot(sc)
                i = int(np.argmin(sc))
                M.update({f"{k}Score": f"{sm:.3f}", f"{k}ScoreLo": f"{slo:.3f}", f"{k}ScoreHi": f"{shi:.3f}", f"{k}ScoreClip": f"{np.mean(np.clip(sc, 0, 1)):.3f}",
                          f"{k}ScoreMin": neg(min(sc)), f"{k}ScoreMinSeed": seeds[i], f"{k}ScoreMax": neg(max(sc))})
                cells.append(f"{sm:.3f}")
            if p == "deploy":
                cells += ["--"] * 3
            else:
                w, n, sp, wp = paired([ref[s]["saved_frac"] for s in seeds], sf)
                M.update({f"{pre}RlWins{TESTKEY.get(p, p)}": str(w), f"{pre}N": str(n), f"{k}SignP": pf(sp), f"{k}WilcoxonP": pf(wp)})
                cells += [f"{w}/{n}", pf(sp), pf(wp)]
            rows.append(" & ".join(cells) + " \\\\")
        M.update({f"{pre}SignP": M[f"{pre}PlannerSignP"], f"{pre}WilcoxonP": M[f"{pre}PlannerWilcoxonP"]})
        M.update({k.replace(f"{pre}Rlcheck", f"{pre}rlcheck").replace(f"{pre}RlWinsRlcheck", f"{pre}RlWinsrlcheck"): v for k, v in list(M.items()) if k.startswith((f"{pre}Rlcheck", f"{pre}RlWinsRlcheck"))})
        (HERE / f"{part}.tex").write_text("\n".join(rows) + "\n")


def neg(x, f="{:.3f}"):
    return ("$-$" if x < 0 else "") + f.format(abs(x))


def failures(d):
    S = d["summary"]
    r = S["reference"]
    share = lambda w: S[w]["rebuild_spend"]["mean"] / S[w]["total_spend"]["mean"]
    tot = lambda w, k: sum(S[w][k]["per_seed"].values())
    M.update(failJaccardRef=f"{S['jaccard_pairs']['reference']['pairs']['mean']:.2f}", failBoomRef=f"{tot('reference', 'boom_upgraded'):.0f}",
             failBoomEarlyRef=f"{tot('reference', 'boom_upgraded_by_2031'):.0f}", failBlockedRef=f"{r['boom_access_blocked_years']['mean']:.1f}",
             failRebuildShareRef=f"{100 * share('reference'):.0f}\\%")
    for g, sfx in (("a1", ""), ("gate4", "Gate")):
        a = S[g]
        runs_ = [x for x in d["runs"] if x["role"] == g]
        M.update({f"failJaccardAgent{sfx}": f"{S['jaccard_pairs'][g]['pairs']['mean']:.2f}",
                  f"failMentionYear{sfx}": str(int(np.median([b["first_mention_msg"] for x in runs_ for b in x["sees_acts"]]))),
                  f"failBoomNeverActed{sfx}": str(a["never_upgraded_mentioned"]), f"failBoomTotal{sfx}": str(a["boom_total"]), f"failLagYears{sfx}": f"{a['lag']['mean']:.1f}",
                  f"failBoomAgent{sfx}": f"{tot(g, 'boom_upgraded'):.0f}", f"failBoomEarlyAgent{sfx}": f"{tot(g, 'boom_upgraded_by_2031'):.0f}",
                  f"failBlockedAgent{sfx}": f"{a['boom_access_blocked_years']['mean']:.1f}", f"failMaintTwoShare{sfx}": f"{100 * a['maint2_share']['mean']:.1f}\\%",
                  f"failRebuildShareAgent{sfx}": f"{100 * share(g):.0f}\\%", f"failForecastAccess{sfx}": f"{a['forecasts_with_access']['mean']:.1f}",
                  f"failForecastAccessUnbuilt{sfx}": f"{a['forecasts_access_never_started']['mean']:.1f}", f"failCallsGetState{sfx}": f"{a['calls']['get_state']['mean']:.1f}",
                  f"failCallsForecast{sfx}": f"{a['calls']['forecast']['mean']:.1f}"})
        t = S[f"reference_vs_{g}_score"]
        M.update({f"gapIntelWins{sfx}": f"{t['wins']}/{t['n']}", f"gapIntelSignP{sfx}": pf(t["sign_p"]) if t["sign_p"] < 0.001 else f"{t['sign_p']:.3f}"})
    jp = S["jaccard_pairs"]
    five = lambda g, k: np.mean([np.mean([x[k] for x in d["runs"] if x["role"] == g and x["seed"] == sd]) for sd in range(5)])
    M.update(aoneJaccard=f"{jp['a1']['pairs']['mean']:.2f}", refJaccard=f"{jp['reference']['pairs']['mean']:.2f}",
             aoneNeverActed=f"{S['a1']['never_upgraded_mentioned']}/{S['a1']['boom_total']}", aoneLag=f"{S['a1']['lag']['mean']:.1f}",
             aoneTwoBoom=f"{S['a1']['two_boom']['mean']:.2f}", refTwoBoom=f"{S['reference']['two_boom']['mean']:.2f}",
             baseBoomFive=f"{five('a1', 'boom_upgraded'):.1f}", refBoomFive=f"{five('reference', 'boom_upgraded'):.1f}", baseGetState=f"{S['a1']['calls']['get_state']['mean']:.1f}")
    for g, t in (("longforecast", "lf"), ("max", "max"), ("hint", "hint"), ("notes", "notes"), ("redteam", "redteam")):
        if g in S:
            x, c = S[g], S[f"{g}_vs_a1_score"]
            M.update({f"{t}Boom": f"{x['boom_upgraded']['mean']:.1f}", f"{t}NeverActed": f"{x['never_upgraded_mentioned']}/{x['boom_total']}",
                      f"{t}GetState": f"{x['calls']['get_state']['mean']:.1f}", f"{t}ListProjects": f"{x['calls']['list_projects']['mean']:.1f}",
                      f"{t}Score": f"{c['mean']:.2f}", f"{t}WinsAone": f"{c['wins']}/{c['n']}", f"{t}WilcoxonP": f"{c['wilcoxon_p']:.3f}".rstrip("0"),
                      f"{t}GapClosed": f"{100 * (c['mean'] - c['a1_mean']) / (c['reference_mean'] - c['a1_mean']):.0f}\\%"})
            M.update(baseScoreFive=f"{c['a1_mean']:.2f}", refScoreFive=f"{c['reference_mean']:.3f}")
    M["lfMaxP"] = M.get("lfWilcoxonP", "--")
    M["aoneListProjects"] = f"{S['a1']['calls']['list_projects']['mean']:.1f}"
    rows = [("Score", "score", "{:.2f}"), ("Plan overlap with other seeds (Jaccard)", "jaccard_other_seeds", "{:.2f}"),
            ("Growing areas upgraded (of 3)", "boom_upgraded", "{:.1f}"), ("of them by 2031", "boom_upgraded_by_2031", "{:.1f}"),
            ("Years a growing-area upgrade was not affordable", "boom_access_blocked_years", "{:.1f}"), ("Spend on maintenance level 2 (share)", "maint2_share", "{:.3f}"),
            ("Cash at the end (MEUR)", "cash_end", "{:.0f}")]
    cell = lambda w, k, f: f"{neg(S[w][k]['mean'], f)} [{neg(S[w][k]['lo'], f)}, {neg(S[w][k]['hi'], f)}]"
    (HERE / "failures.tex").write_text("".join(f"{n} & {cell('a1', k, f)} & {cell('reference', k, f)} \\\\\n" for n, k, f in rows))


def gshare(a):
    return neg(a["gap_share_pooled"], "{:.2f}" if abs(a["gap_share_pooled"]) >= 0.1 else "{:.3f}")


def counterfactual(d):
    names = dict(a_access="Agent plus an early upgrade of the fastest area", b_maint1="Agent with maintenance 1", c_both="Agent with both changes", d_ref_maint2="Reference with maintenance 2")
    for g, sfx in (("a1", ""), ("gate4", "Gate")):
        S = d["summary"][g]
        M.update({f"gapIntelMean{sfx}": f"{S['gap']['mean']:.3f}", f"gapIntelLo{sfx}": f"{S['gap']['lo']:.3f}", f"gapIntelHi{sfx}": f"{S['gap']['hi']:.3f}"})
        for arm, t in (("a_access", "Access"), ("b_maint1", "MaintOne"), ("c_both", "Both"), ("d_ref_maint2", "RefMaintTwo")):
            M.update({f"cf{t}Diff{sfx}": neg(S[arm]["diff"]["mean"]), f"cf{t}GapShare{sfx}": gshare(S[arm])})
        acc = [r for r in d["rows"] if r["arm"] == "a_access" and r["job"].startswith(g + "-")]
        M[f"cfAccessAccepted{sfx}"] = f"{sum(r['added'] is not None for r in acc)}/{len(acc)}"
        M.update({f"cfAccessWins{sfx}": str(S["a_access"]["test"]["wins"]), f"cfAccessTies{sfx}": str(S["a_access"]["test"].get("ties", 0)), f"cfAccessP{sfx}": pf(S["a_access"]["test"]["sign_p"])})
        if g == "a1":
            rows = []
            for arm in names:
                a, t = S[arm], S[arm]["test"]
                rows.append(f"{names[arm]} & {neg(a['diff']['mean'])} [{neg(a['diff']['lo'])}, {neg(a['diff']['hi'])}] & {gshare(a)} & {t['wins']}/{t['n']} & {t['sign_p']:.2g} \\\\")
            (HERE / "counterfactual.tex").write_text("\n".join(rows) + "\n")


def exploits(d):
    st, r = d["strategies"], d["strategies"]["reference"]
    sm = lambda k: np.mean(st[k]["score"])
    w = st["reference-maintain-0"]["wins_over_ref"]
    n = len(st["reference-maintain-0"]["score"])
    best = max((v["mean"], k) for k, v in st.items() if not k.startswith(("reference", "stop", "runout", "forecast", "agent")))
    M.update(ExploitRefMean=f"{r['mean']:.3f}", ExploitRefCI=f"{r['ci'][0]:.3f}--{r['ci'][1]:.3f}",
             ExploitPlanDoneYear=str(int(np.median(list(d["plan_complete_year"].values())))),
             ExploitFirstNegative=str(int(np.median(r["first_negative"])) + 1), ExploitNothingFirstNegative=str(int(np.median(st["nothing"]["first_negative"])) + 1),
             ExploitStopSevenGap="0" if st["stop-7"]["mean"] == r["mean"] else neg(st["stop-7"]["mean"] - r["mean"]), ExploitStopThree=f"{st['stop-3']['mean']:.3f}", ExploitStopFive=f"{st['stop-5']['mean']:.3f}",
             ExploitMaintZeroWins=str(w), ExploitMaintZeroN=str(n), ExploitMaintZeroGap=neg(sm("reference-maintain-0") - sm("reference"), "{:.4f}"),
             ExploitMaintZeroP=f"{binomtest(w, n, 0.5, alternative='greater').pvalue:.3f}", ExploitMaintTwoGap=("+" if sm("reference-maintain-2") > sm("reference") else "") + neg(sm("reference-maintain-2") - sm("reference"), "{:.4f}"),
             ExploitRefDebt=neg(np.mean(r["end_cash"]), "{:.1f}"), ExploitNothingDebt=neg(np.mean(st["nothing"]["end_cash"]), "{:.1f}"),
             ExploitBlindAccess=f"{st['blind-access-m0']['mean']:.3f}", ExploitBoomAccess=f"{st['boom-access-m0']['mean']:.3f}", ExploitBestNonRef=f"{best[0]:.3f}",
             ExploitRefOverOne=str(sum(x > 1 for x in r["score"])), ExploitAcceptedBad=str(len({a for v in st.values() for a in v["accepted_bad"]})))
    if "bad_inputs_tried" in d["meta"]:
        n_bad = d["meta"]["bad_inputs_tried"]
        M["ExploitRejected"] = f"{n_bad - int(M['ExploitAcceptedBad'])}/{n_bad}"
    rows, keys = [], list(st)
    i = 0
    while i < len(keys):
        k = keys[i]
        j = i
        while k.startswith("stop-") and j + 1 < len(keys) and keys[j + 1].startswith("stop-") and st[keys[j + 1]]["mean"] == st[k]["mean"]:
            j += 1
        name = f"stop after {k[5:]}" + (f" to {keys[j][5:]}" if j > i else "") + " years" if k.startswith("stop-") else k
        v = st[k]
        rows.append(f"\\texttt{{{name}}} & {v['mean']:.3f} & {v['max']:.3f} & {v['guard'] or '--'} \\\\")
        i = j + 1
    (HERE / "exploits.tex").write_text("\n".join(rows) + "\n")


def grpo_evidence(d):
    ms, ts, sens = d["methods"], d["tests"], d["sensitivity"]
    run = lambda tag: next(k for k in ms if k.endswith(tag))
    one, two = run("-s1/runs/grpo/ckpt-90"), run("-s2/runs/grpo/ckpt-90")
    ci = lambda x, f="{:.3f}", m=1: f"{f.format(m * x['mean'])} [{f.format(m * x['ci'][0])}, {f.format(m * x['ci'][1])}]"
    keys = (("runs/grpo/ckpt-0", "Warm", "Gemma after format warm-up"), ("random", "Random", "Random valid plans"),
            ("runs/grpo/ckpt-90", "Final", "Gemma after GRPO, run A"), (one, "SeedOne", "Gemma after GRPO, seed 1"), (two, "SeedTwo", "Gemma after GRPO, seed 2"),
            ("const", "Const", "Constant plan " + ", ".join(f"\\texttt{{{q}}}" for q in d["meta"]["const_plan"]["plan"])))
    rows = []
    for k, t, name in keys:
        v, te = ms[k]["val"], ms[k]["test"]
        M.update({f"grpo{t}Val": f"{v['score']['mean']:.3f}", f"grpo{t}ValLo": f"{v['score']['ci'][0]:.3f}", f"grpo{t}ValHi": f"{v['score']['ci'][1]:.3f}",
                  f"grpo{t}ValClip": f"{v['score_clip']['mean']:.3f}", f"grpo{t}Test": f"{100 * te['frac']['mean']:.1f}",
                  f"grpo{t}TestScore": f"{te['score']['mean']:.3f}", f"grpo{t}TestScoreLo": f"{te['score']['ci'][0]:.3f}", f"grpo{t}TestScoreHi": f"{te['score']['ci'][1]:.3f}",
                  f"grpo{t}TestScoreClip": f"{te['score_clip']['mean']:.3f}"})
        rows.append(f"{name} & {ci(v['score'])} & {ci(te['score'])} & {ci(te['frac'], '{:.1f}', 100)} \\\\")
    (HERE / "grpo.tex").write_text("\n".join(rows) + "\n")
    n = {p: len(d["meta"]["seeds"][p]) for p in ("val", "test")}
    for other, t in (("runs/grpo/ckpt-0", "Warm"), ("random", "Random"), ("const", "Const")):
        for part, pt in (("val", "Val"), ("test", "Test")):
            x = ts[f"runs/grpo/ckpt-90 vs {other} {part}"]
            M.update({f"grpoFinalWins{t}{pt}": f"{x['wins']}/{n[part]}", f"grpoFinalSignP{t}{pt}": pf(x["sign_p"])})
    for k, t in ((one, "SeedOne"), (two, "SeedTwo")):
        for part, pt in (("val", "Val"), ("test", "Test")):
            x = ts[f"{k} vs const {part}"]
            M.update({f"grpo{t}WinsConst{pt}": f"{x['wins']}/{n[part]}", f"grpo{t}SignPConst{pt}": pf(x["sign_p"])})
    for k, t in (("runs/grpo/ckpt-0", "Warm"), ("runs/grpo/ckpt-90", "Final"), (one, "SeedOne"), (two, "SeedTwo")):
        f = "{:.3f}" if t in ("Warm", "Final") else "{:.2f}"
        name = f"grpoSens{t}" if t in ("Warm", "Final") else f"grpo{t}Sens"
        M.update({f"{name}Shuffle": f.format(sens[k]["shuffle"]), f"{name}Zero": f.format(sens[k]["zero"]["changed"])})
    first = lambda k, s: ms[k]["plans"][s][0]["plan"][0] if isinstance(ms[k]["plans"][s], list) else ms[k]["plans"][s]["plan"][0]
    kivi = [s for s in ms[one]["plans"] if "Kiviruukki" in booms[s]]
    rest = [s for s in ms[one]["plans"] if s not in kivi]
    M.update(grpoSeedOneKiviFirst=f"{sum(first(one, s) == 'kiviruukki-access' for s in kivi)}/{len(kivi)}",
             grpoSeedOneOtaFirst=f"{sum(first(one, s) == 'otaniemi-access' for s in rest)}/{len(rest)}",
             grpoSameAsConst=str(sum(ms["runs/grpo/ckpt-90"]["plans"][s] == ms["const"]["plans"][s] for s in ms["const"]["plans"])))
    sr = d["seed_runs"]
    for part, pt in (("val", "Val"), ("test", "Test")):
        x = sr[part]["score"]
        M.update({f"grpoSeedMean{pt}": f"{x['mean']:.3f}", f"grpoSeedStd{pt}": f"{x['std']:.3f}"})


def norms_evidence(d):
    new = [v for v in d["seeds"].values() if not v.get("frozen")]
    src = lambda name: str(sum(v["source"] == name for v in new))
    M.update(normsTestSeeds=str(len(new)), normsCpuHours=f"{d['meta']['cpu_seconds_16_47'] / 3600:.1f}", normsSrcLocal=src("local search"),
             normsSrcSingle=src("single-value order"), normsSrcLearned=src("learned policy"))


def letters(x):
    return "".join(w.title() for w in re.split("[^a-zA-Z]+", str(x)) if w)


def job_of(p):
    d = p.parent if p.parent.parent == EVID else p.parent.parent
    return re.sub(r"-s\d+$", "", d.name)


def agent_summary(runs, ref, sfx):
    runs = sorted(runs, key=lambda r: int(r["seed"]))
    sc = [r["score"] for r in runs]
    by_seed = {}
    for r in runs:
        by_seed.setdefault(str(r["seed"]), []).append(r["score"])
    means = [np.mean(v) for v in by_seed.values()]
    m, lo, hi = boot(means)
    paired_runs = [(ref[str(r["seed"])], r["score"]) for r in runs if str(r["seed"]) in ref]
    seeds = sorted((s for s in by_seed if s in ref), key=int)
    w, n, sp, wp = paired([ref[s] for s in seeds], [np.mean(by_seed[s]) for s in seeds]) if seeds else (0, 0, 1.0, 1.0)
    M.update({f"agentRuns{sfx}": str(len(sc)), f"agentMeanAll{sfx}": f"{m:.3f}", f"agentCIAll{sfx}": f"{lo:.3f}--{hi:.3f}",
              f"agentWorstRun{sfx}": f"{min(sc):.3f}", f"agentBestRun{sfx}": f"{max(sc):.3f}",
              f"refWinsRuns{sfx}": f"{sum(a > b for a, b in paired_runs)}/{len(paired_runs)}", f"refWinsSeeds{sfx}": f"{w}/{n}", f"agentSignPAll{sfx}": pf(sp),
              f"agentWilcoxonPAll{sfx}": pf(wp), f"agentSeedStd{sfx}": f"{np.mean([np.std(v) for v in by_seed.values()]):.3f}",
              f"refWinsBestRun{sfx}": f"{sum(ref[s] > max(by_seed[s]) for s in seeds)}/{len(seeds)}", f"agentSeedCount{sfx}": str(len(by_seed))})
    return f"{len(sc)} & {m:.3f} [{lo:.3f}, {hi:.3f}] & {min(sc):.3f} & {max(sc):.3f} & {sum(a > b for a, b in paired_runs)}/{len(paired_runs)} & {w}/{n} & {pf(sp)}"


def agent_runs(rows):
    cost = sum(r.get("cost_usd", 0) or 0 for r in rows if r["job"] == MAIN_JOB)
    M.update(agentInvalidRuns=str(sum(r.get("valid") is False for r in rows)))
    rows = [r for r in rows if r.get("valid", True)]
    if not rows:
        return
    (HERE / "agent.tex").write_text("".join(f"{r['who']} & {r['seed']} & {r['delay'] / 1e6:.1f} & {r['score']:.3f} \\\\\n" for r in rows if r["top"]))
    (HERE / "agent_all.tex").write_text("".join(f"{r['job']} & {r['seed']} & {r['delay'] / 1e6:.1f} & {r['score']:.3f} \\\\\n" for r in rows if r["who"] != "reference"))
    ref = {str(r["seed"]): r["score"] for r in rows if r["who"] == "reference"}
    if (EVID / "test.json").exists():
        ref.update({s: v["score"] for s, v in load(EVID / "test.json")["validation"]["policies"]["deploy"].items()})
    groups = {}
    for r in rows:
        if r["who"] != "reference":
            m = r.get("model", r["who"])
            v = r.get("variant", "base" if r["job"] in BASE_JOBS else r["job"])
            groups.setdefault((m, v if r["job"] not in BASE_JOBS or r["job"] == MAIN_JOB else f"{v} ({r['job']})"), []).append(r)
    out = []
    main = [r for (m, v), g in groups.items() if v == "base" for r in g]
    if main:
        agent_summary(main, ref, "")
        by = {}
        for r in sorted(main, key=lambda r: int(r["seed"])):
            by.setdefault(str(r["seed"]), []).append(r["score"])
        (HERE / "agent_seeds.tex").write_text("".join(f"{k} & " + " & ".join(f"{x:.3f}" for x in v) + f" & {np.mean(v):.3f} & {np.std(v):.3f} & {ref[k]:.3f} \\\\\n" for k, v in by.items()))
        M.update(agentCostUsd=f"{cost:.2f}", agentSeedsAll=f"{min(r['seed'] for r in main)} to {max(r['seed'] for r in main)}")
    for (m, v), g in sorted(groups.items()):
        sfx = letters(m) + ("" if v.startswith("base") else letters(v))
        out.append(f"\\texttt{{{m}}} & {v} & " + agent_summary(g, ref, sfx) + " \\\\")
    (HERE / "agentruns.tex").write_text("\n".join(out) + "\n")
    ab = [(k, g) for k, g in groups.items() if not k[1].startswith(("base", "redteam"))]
    if ab:
        base, base3 = {}, {}
        for (m, v), g in groups.items():
            if v == "base":
                for r in sorted(g, key=lambda r: r["batch"], reverse=True):
                    base.setdefault(m, {})[str(r["seed"])] = clip(r["score"])
                    base3.setdefault(m, {}).setdefault(str(r["seed"]), []).append(clip(r["score"]))
        lines = []
        for (m, v), g in sorted(ab, key=lambda kg: ["longforecast", "max", "hint", "notes"].index(kg[0][1]) if kg[0][1] in ("longforecast", "max", "hint", "notes") else 9):
            g = sorted(g, key=lambda r: int(r["seed"]))
            seeds = [str(r["seed"]) for r in g]
            sc = [clip(r["score"]) for r in g]
            mm, lo, hi = boot(sc)
            b1 = [base[m][s] for s in seeds] if m in base and all(s in base[m] for s in seeds) else None
            bs = [np.mean(base3[m][s]) for s in seeds] if b1 else None
            rs = [clip(ref[s]) for s in seeds]
            db, dr = (np.mean(sc) - np.mean(bs) if bs else None), np.mean(sc) - np.mean(rs)
            wins = sum(a > b for a, b in zip(sc, rs))
            first = [next((x["build"] for _, acts in r["plan"] for x in acts if x.get("ok") and x.get("build", "").endswith("-access")), None) for r in g]
            M[f"abl{letters(v)}OtaFirst"] = f"{sum(f == 'otaniemi-access' for f in first)}/{len(g)}"
            t = letters(v)
            M.update({f"ablation{letters(m)}{t}": f"{mm:.3f}", f"abl{t}": f"{mm:.3f}", f"abl{t}Lo": f"{lo:.3f}", f"abl{t}Hi": f"{hi:.3f}", f"abl{t}Runs": str(len(g)),
                      f"abl{t}VsBase": neg(db) if bs else "--", f"abl{t}VsRef": neg(dr), f"abl{t}WinsRef": f"{wins}/{len(g)}",
                      f"abl{t}Min": f"{min(sc):.3f}", f"abl{t}Max": f"{max(sc):.3f}", f"abl{t}CostUsd": f"{sum(r.get('cost_usd', 0) or 0 for r in g):.2f}"})
            if bs:
                gap = lambda bb: (np.mean(sc) - np.mean(bb)) / (np.mean(rs) - np.mean(bb))
                M.update({f"abl{t}GapShare": neg(gap(bs), "{:.2f}"), f"abl{t}VsBaseFirstRun": neg(np.mean(sc) - np.mean(b1)), f"abl{t}GapShareFirstRun": neg(gap(b1), "{:.2f}"),
                          "ablBase": f"{np.mean(bs):.3f}", "ablBaseMean": f"{np.mean(bs):.3f}", "ablBaseFirstRun": f"{np.mean(b1):.3f}",
                          "ablRef": f"{np.mean(rs):.3f}", "ablSeeds": f"{min(map(int, seeds))} to {max(map(int, seeds))}"})
            lines.append(f"\\texttt{{{m}}} & {v} & {len(g)} & {mm:.3f} [{lo:.3f}, {hi:.3f}] & {neg(db) if bs else '--'} & {neg(dr)} & {wins}/{len(g)} \\\\")
        (HERE / "ablations.tex").write_text("\n".join(lines) + "\n")
    red = [r for (m, v), g in groups.items() if v.startswith("redteam") for r in g]
    if red:
        sc = [r.get("harbor_reward", r["score"]) for r in red]
        M.update(redteamMean=f"{np.mean(sc):.3f}", redteamBest=f"{max(sc):.3f}", redteamRuns=str(len(sc)),
                 redteamBeatsRef=str(sum(r["score"] > ref[str(r["seed"])] for r in red if str(r["seed"]) in ref)),
                 redteamCostUsd=f"{sum(r.get('cost_usd', 0) or 0 for r in red):.2f}")
        (HERE / "redteam.tex").write_text("".join(f"{r['job']} & {r['seed']} & {r['score']:.3f} & {r.get('harbor_reward', float('nan')):.3f} \\\\\n" for r in red))


def rules_v2(d):
    var, mt = d["variants"], d["meta"]
    per = lambda v, k: [var[v][s][k]["score"] for s in sorted(var[v], key=int)]
    ci = lambda x: "{} [{}, {}]".format(*(neg(v) for v in boot(x)))
    names = {"planner": "Rule planner", "learned": "Learned policy", "learned-m2": "Learned policy, maintenance 2", "learned-retrained": "Learned policy, retrained under v2",
             "reactive": "Reactive", "maintain-0": "Do nothing, maintenance 0", "maintain-2": "Do nothing, maintenance 2"}
    keys = [k for k in names if k in var["v2"]["0"]] + [k for k in var["v2"]["0"] if k not in names and k != "nothing"]
    have = lambda v, k: all(k in var[v][s] for s in var[v])
    rows = []
    for k in keys:
        cells = [ci(per(v, k)) if have(v, k) else "--" for v in ("v1", "v2", "v2e")]
        wins = [paired(per(v, k), per(v, "planner"))[0] if have(v, k) and k != "planner" else "--" for v in ("v2", "v2e")]
        rows.append(f"{names.get(k, k)} & " + " & ".join(cells) + f" & {wins[0]} & {wins[1]} \\\\")
    ex1 = load(EVID / "exploits.json")["strategies"]["reference-maintain-2"]["score"] if (EVID / "exploits.json").exists() else None
    lm2 = [np.mean(per(v, "learned-m2")) if have(v, "learned-m2") else mt[f"exploits_{v}"]["reference-maintain-2"]["score_mean"] for v in ("v2", "v2e")]
    if "learned-m2" not in keys:
        rows.insert(2, f"Learned policy, maintenance 2 & {ci(ex1) if ex1 else '--'} & {lm2[0]:.3f} & {lm2[1]:.3f} & -- & -- \\\\")
    if have("v2", "learned-retrained"):
        for v, t in (("v1", "One"), ("v2", "Two"), ("v2e", "Twoe")):
            if have(v, "learned-retrained"):
                M.update({f"v{t}Retrained": f"{np.mean(per(v, 'learned-retrained')):.3f}", f"v{t}RetrainedMin": f"{min(per(v, 'learned-retrained')):.3f}",
                          f"v{t}RetrainedWinsPlanner": str(paired(per(v, "learned-retrained"), per(v, "planner"))[0])})
        M["vTwoRefSeedsZeroFour"] = f"{np.mean(per('v2', 'learned')[:5]):.3f}"
    if "test" in d:
        te = d["test"]["v2"]
        sv = lambda k: [te[s][k]["saved"] for s in sorted(te, key=int)]
        M.update(vTwoTestRetrainedSaved=f"{np.mean(sv('retrained')):.3f}", vTwoTestPlannerSaved=f"{np.mean(sv('planner')):.3f}",
                 vTwoTestLearnedSaved=f"{np.mean(sv('learned')):.3f}" if all("learned" in te[s] for s in te) else "--",
                 vTwoTestRetrainedWinsPlanner=str(paired(sv("retrained"), sv("planner"))[0]), vTwoTestN=str(len(te)))
    if "agents" in d:
        ag = {v: [r for r in d["agents"][v] if r.get("valid", True)] for v in d["agents"]}
        label = lambda r: f"{r['run'].split('-s')[0]}, seed {r['seed']}"
        by = {v: {label(r): r["score"] for r in ag[v]} for v in ag}
        (HERE / "rules_v2_agents.tex").write_text("".join(f"{k} & " + " & ".join(neg(by[v][k]) if k in by[v] else "--" for v in ("v1", "v2", "v2e")) + " \\\\\n" for k in by["v2"]))
        for job, t in (("gate4", "Gate"), ("a1", "Aone")):
            sc = [r["score"] for r in ag["v2"] if r["run"].startswith(job + "-")]
            if sc:
                M[f"vTwoAgent{t}"] = f"{np.mean(sc):.3f}"
    (HERE / "rules_v2.tex").write_text("\n".join(rows) + "\n")
    mean = lambda v, k: np.mean(per(v, k))
    cash = lambda v: min(var[v][s]["learned"]["min_cash"] for s in var[v])
    M.update(vTwoPlanner=f"{mean('v2', 'planner'):.3f}", vTwoLearned=f"{mean('v2', 'learned'):.3f}", vTwoePlanner=f"{mean('v2e', 'planner'):.3f}",
             vTwoeLearned=f"{mean('v2e', 'learned'):.3f}", vOnePlanner=f"{mean('v1', 'planner'):.3f}", vOneLearned=f"{mean('v1', 'learned'):.3f}",
             vTwoPlannerWins=str(paired(per("v2", "planner"), per("v2", "learned"))[0]), vTwoePlannerWins=str(paired(per("v2e", "planner"), per("v2e", "learned"))[0]),
             vOneLearnedWins=str(paired(per("v1", "learned"), per("v1", "planner"))[0]),
             vTwoLearnedMTwo=f"{lm2[0]:.3f}", vTwoeLearnedMTwo=f"{lm2[1]:.3f}", vTwoLearnedMTwoWins=str(mt["exploits_v2"]["reference-maintain-2"]["wins_over_ref"]),
             vOneMinCash=neg(cash("v1"), "{:.0f}"), vTwoMinCash=neg(cash("v2"), "{:.1f}"), vTwoMaintZero=neg(mean("v2", "maintain-0")),
             vTwoStopSeven=f"{mt['exploits_v2']['stop-7']['score_mean']:.3f}", vTwoStopSevenWins=str(mt["exploits_v2"]["stop-7"]["wins_over_ref"]),
             vTwoeStopSeven=f"{mt['exploits_v2e']['stop-7']['score_mean']:.3f}", vTwoeStopSevenWins=str(mt["exploits_v2e"]["stop-7"]["wins_over_ref"]))
    if have("v2", "learned-m2"):
        M.update({f"v{t}LearnedMTwoWinsPlanner": str(paired(per(v, "learned-m2"), per(v, "planner"))[0]) for v, t in (("v1", "One"), ("v2", "Two"), ("v2e", "Twoe"))})


def handover(d):
    for g, sfx in (("a1", "Aone"), ("gate4", "Gate")):
        h = d["summary"][g]
        vals = {"Runs": str(h["runs"]), "BigRuns": str(h["big"]), "BreakFirst": str(min(h["break_years"])), "BreakLast": str(max(h["break_years"])),
                "RecoverMin": f"{min(h['recover']):.2f}", "RecoverMax": f"{max(h['recover']):.2f}", "AfterMin": f"{min(h['after']):.2f}", "AfterMax": f"{max(h['after']):.2f}",
                "KOneBig": f"{h['agent_first_k1_big']:.2f}", "KThreeBig": f"{h['agent_first_k3_big']:.2f}", "KFiveBig": f"{h['agent_first_k5_big']:.2f}",
                "RevYears": str(int(np.median(h["reverse_years"]))), "RevYearsRuns": str(sum(y == int(np.median(h["reverse_years"])) for y in h["reverse_years"])),
                "MixWorse": str(h["mix_worse"]), "RefAlone": f"{h['reference_alone']:.3f}", "AgentAlone": f"{h['agent_alone']:.3f}"}
        vals.update(RecoverMedian=f"{np.median(h['recover']):.2f}", BreakModal=str(max(set(h["break_years"]), key=h["break_years"].count)))
        M.update({f"handover{k}{sfx}": v for k, v in vals.items()})
        if g == "a1":
            M.update({f"handover{k}": v for k, v in vals.items()})
    sys.path.insert(0, str(HERE))
    import handover as hv
    hv.plot(d["runs"])


def seed4(d):
    a, bn, single = d["arms"], d["bottleneck"], d["single_2030"]
    b = a["base"]
    op = next(i for i, x in enumerate(b["open"]) if "otaniemi-access" in x)
    M.update(sFourBase=f"{b['score']:.2f}", sFourSwap=f"{a['swap']['score']:.2f}", sFourDropOta=f"{a['drop']['score']:.3f}",
             sFourLepAlone=f"{single['4']['leppavaara-access']:.3f}", sFourteenOtaAlone=f"{single['14']['otaniemi-access']:.2f}",
             sFourBottShare=f"{100 * bn['4']['way_share']:.0f}\\%", sFourteenBottShare=f"{100 * bn['14']['way_share']:.0f}\\%",
             sFourBottVC=f"{bn['4']['top_vc_2040']:.1f}", sFourBottRatio=f"{bn['4']['top_time_ratio_2040']:.0f}", sFourBottLen=f"{bn['4']['top_len_m']:.1f}",
             sFourBottOsm=str(bn["4"]["top_osm"]), sFourWithOtaVCMin=f"{min(b['bott_vc'][op:]):.1f}", sFourWithOtaVCMax=f"{max(b['bott_vc'][op:]):.1f}",
             sFourBottSeeds=str(sum(v["top_osm"] == bn["4"]["top_osm"] and v["way_share"] >= 0.1 for v in bn.values())))


def norms_v1(d):
    S = d["seeds"].values()
    over = lambda key: sum((v["nothing"] - (v["nothing"] - v["score_new"][key] * (v["nothing"] - v["best"]))) / (v["nothing"] - v["frozen"][1]) > 1 + 1e-9 for v in S)
    mean = lambda key: np.mean([v["score_new"][key] for v in S])
    M.update(normsVoneMaxDrop=f"{max(-v['diff'] / v['frozen'][1] * 100 for v in S):.1f}", normsVoneSeedsLower=str(sum(v["diff"] < 0 for v in S)),
             normsVoneSeedsHigher=str(sum(v["diff"] > 0 for v in S)), normsVoneLearnedMean=f"{mean('learned policy'):.3f}", normsVonePlannerMean=f"{mean('planner'):.3f}",
             normsVoneConstMean=f"{mean('grpo constant plan'):.3f}", normsVzeroOverOneLearned=str(over("learned policy")), normsVzeroOverOneConst=str(over("grpo constant plan")))
    fm = lambda key: np.mean([v["score_with_frozen"][key] for v in S])
    cmp = [v["best_with_frozen"] - v["frozen"][1] for v in S]
    M.update(normsVoneFrozenLearnedMean=f"{fm('learned policy'):.3f}", normsVoneFrozenPlannerMean=f"{fm('planner'):.3f}", normsVoneFrozenConstMean=f"{fm('grpo constant plan'):.3f}",
             normsVoneFrozenSeedsLower=str(sum(c < 0 for c in cmp)), normsVoneFrozenSeedsEqual=str(sum(c == 0 for c in cmp)), normsVoneFrozenSeedsHigher=str(sum(c > 0 for c in cmp)))


def evidence():
    f = EVID / "test.json"
    if f.exists():
        test_tables(load(f))
    for name, fn in (("failures", failures), ("counterfactual", counterfactual), ("exploits", exploits), ("grpo", grpo_evidence), ("norms", norms_evidence), ("handover", handover), ("seed4", seed4), ("norms_v1", norms_v1)):
        if (EVID / f"{name}.json").exists():
            fn(load(EVID / f"{name}.json"))
    for name in ("test", "exploits", "failures", "counterfactual", "grpo", "norms"):
        if not (EVID / f"{name}.json").exists():
            print("missing", f"evidence/{name}.json")
    agent_runs([load(p) | dict(job=job_of(p), top=p.parent.parent == EVID, batch=int(p.parent.name.split("-")[0]) if p.parent.parent != EVID else 0) for p in sorted(EVID.glob("**/plan.json"))])
    if (EVID / "rules_v2.json").exists():
        rules_v2(load(EVID / "rules_v2.json"))


def main():
    rescore()
    ev, zs, final = grpo_curve()
    d = RLD["search"]
    search = {str(s): clip(x) for s, x in zip(d["seeds"], d["scores"])}
    vals = {k: [runs[k][s]["score"] for s in sorted(runs[k], key=int)] for k in KEYS if k in runs}
    vals.update(best=[1.0] * len(norms), grpo=final, rlsearch=list(search.values()))
    if (EVID / "test.json").exists():
        # 16-seed scores come from the frozen-norm Intel evaluation (evidence/test.json), so Table main agrees with validation.tex
        V = load(EVID / "test.json")["validation"]["policies"]
        pol = dict(rl="deploy", planner="planner", nothing="nothing", random="random")
        pol.update({k: f"{k}:2" if LEVEL.get(k) == 2 else k for k in ("greedy1", "greedy5", "reactive")})
        for k, p in pol.items():
            if p in V and k in vals:
                vals[k] = [V[p][str(s)]["score"] for s in range(16)]
    worse = sorted((s for s in search if search[s] < runs["rl"][s]["score"] - 0.02), key=lambda s: search[s] - runs["rl"][s]["score"])
    M.update(rlsearchScore=f"{np.mean(vals['rlsearch']):.2f}", rlsearchWorse=", ".join(worse[:3]))
    fig_problem()
    fig_bars({k: v for k, v in vals.items() if k != "grpo"})
    fig_years()
    fig_curves(ev, zs)
    grpo = strategy()
    ablation()
    heldout()
    table(vals, search, grpo)
    facts()
    cats()
    T = table_rows()
    results(T)
    fig_bars({k: [clip(T[k][s]["score"]) for s in sorted(T[k], key=int)] for k in KEYS}, lambda k, s: T[k][s]["delay"], "bars_a1")
    evidence()
    from grpo_reason import build as grpo_reason
    grpo_reason(M)
    (HERE / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(M.items())))


if __name__ == "__main__":
    main()
