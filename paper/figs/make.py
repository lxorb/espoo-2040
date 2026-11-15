import json
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
            agent="Frontier LLM agent", planner="Rule planner", rl="Learned policy", rlcheck="Learned + forecast check",
            rlsearch="Learned + search", grpo="GRPO Gemma 270M", best="Best known")
KEYS = list(NAME)

load = lambda p: json.loads(Path(p).read_text())
norms, runs, growth, booms = load(ROOT / "sim" / "norms.json"), load(DATA / "runs.json"), load(DATA / "growth.json"), load(DATA / "booms.json")
RLD, GRD = load(DATA / "rl.json"), load(DATA / "grpo.json")
ASEEDS = sorted(runs["agent"], key=int)
LEVEL, M = {}, {}
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


def fig_bars(vals):
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
        v = [norms[s][1] / 1e6 if k == "best" else tot(runs[k][s]) / 1e6 for s in ASEEDS]
        ax[1].bar(np.arange(len(ASEEDS)) + (j - (len(ks) - 1) / 2) * w, v, w * 0.92, color=COL[k], label=NAME[k], zorder=2)
    ax[1].set_xticks(range(len(ASEEDS)), ASEEDS)
    ax[1].set(xlabel="seed", ylabel="M person-hours, 2026 to 2045", title="(b) Net person-hours lost")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2, handlelength=1, columnspacing=0.8)
    ax[1].grid(axis="x", visible=False)
    save(fig, "bars")


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


def table(vals, search, grpo):
    rows, n3 = [], sum(norms[s][0] for s in ASEEDS)
    for k in KEYS:
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


def tag(p):
    return "".join(w.title() for w in p.replace(":", " m").replace("1", " one").replace("5", " five").replace("2", " two").split())


def test_tables(d):
    for part in ("validation", "test"):
        if part not in d:
            continue
        rows, pol = [], d[part]["policies"]
        ref = pol.get("deploy")
        for p, v in pol.items():
            seeds = sorted(v, key=int)
            sf = [v[s]["saved_frac"] for s in seeds]
            m, lo, hi = boot(sf)
            t = f"{part}{tag(p)}"
            M.update({f"{t}Saved": f"{100 * m:.1f}", f"{t}SavedLo": f"{100 * lo:.1f}", f"{t}SavedHi": f"{100 * hi:.1f}", f"{t}Worst": f"{100 * min(sf):.1f}"})
            cells = [NAME.get({"deploy": "rl"}.get(p, p).split(":")[0], p) + (f" ($m{{=}}{p.split(':')[1]}$)" if ":" in p else ""), f"{100 * m:.1f}", f"[{100 * lo:.1f}, {100 * hi:.1f}]", f"{100 * min(sf):.1f}"]
            if ref and p != "deploy":
                w, n, sp, wp = paired([ref[s]["saved_frac"] for s in seeds], sf)
                M.update({f"{t}Wins": str(w), f"{t}N": str(n), f"{t}SignP": f"{sp:.2g}", f"{t}WilcoxonP": f"{wp:.2g}"})
                cells += [f"{w}/{n}", f"{sp:.2g}", f"{wp:.2g}"]
            else:
                cells += ["--"] * 3
            rows.append(" & ".join(cells) + " \\\\")
        (HERE / f"{part}.tex").write_text("\n".join(rows) + "\n")


def evidence():
    f = EVID / "test.json"
    if f.exists():
        test_tables(load(f))
    for name in ("test", "exploits", "failures", "counterfactual", "grpo", "norms"):
        if not (EVID / f"{name}.json").exists():
            print("missing", f"evidence/{name}.json")
    agents = sorted(EVID.glob("*/plan.json"))
    if agents:
        rows = [load(p) for p in agents]
        (HERE / "agent.tex").write_text("".join(f"{r['who']} & {r['seed']} & {r['delay'] / 1e6:.1f} & {r['score']:.3f} \\\\\n" for r in rows))


def main():
    rescore()
    ev, zs, final = grpo_curve()
    d = RLD["search"]
    search = {str(s): clip(x) for s, x in zip(d["seeds"], d["scores"])}
    vals = {k: [runs[k][s]["score"] for s in sorted(runs[k], key=int)] for k in KEYS if k in runs}
    vals.update(best=[1.0] * len(norms), grpo=final, rlsearch=list(search.values()))
    worse = sorted((s for s in search if search[s] < runs["rl"][s]["score"] - 0.02), key=lambda s: search[s] - runs["rl"][s]["score"])
    M.update(rlsearchScore=f"{np.mean(vals['rlsearch']):.2f}", rlsearchWorse=", ".join(worse[:3]))
    fig_problem()
    fig_bars(vals)
    fig_years()
    fig_curves(ev, zs)
    grpo = strategy()
    ablation()
    heldout()
    table(vals, search, grpo)
    facts()
    cats()
    evidence()
    (HERE / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(M.items())))


if __name__ == "__main__":
    main()
