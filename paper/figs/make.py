import glob
import json
import os
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
Y0 = 2026
RLDIR = os.environ.get("RLDIR", "0fa5d0cd")

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times", "Nimbus Roman", "DejaVu Serif"], "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
INK, MUTED, TBD = "#26251f", "#8a8984", "#d8443c"
COL = dict(nothing="#9b9a94", random="#c3c2b7", greedy1="#1baf7a", greedy5="#199e70", reactive="#2a78d6", agent="#eb6834", planner="#008300",
           rl="#6250d6", rlcheck="#9085e9", rlsearch="#3f2fb0", grpo="#e87ba4", best=INK)
NAME = dict(nothing="Do nothing", random="Random", greedy1="Greedy (1-year)", greedy5="Greedy (5-year)", reactive="Reactive",
            agent="Frontier LLM agent", planner="Rule planner", rl="Learned policy", rlcheck="Learned + forecast check",
            rlsearch="Learned + search", grpo="GRPO Gemma 270M", best="Best known")
KEYS = ["nothing", "random", "greedy1", "greedy5", "reactive", "agent", "planner", "rl", "rlcheck", "rlsearch", "grpo", "best"]

norms = json.loads((ROOT / "sim" / "norms.json").read_text())
runs = json.loads((DATA / "runs.json").read_text())
growth = json.loads((DATA / "growth.json").read_text())
ASEEDS = sorted(runs.get("agent", {}), key=int) or ["0", "1", "2"]
LEVEL, macros = {}, {}


def mac(name, value):
    macros[name] = value


def tot(r):
    return sum(r["delay"])


def raw(seed, d):
    n, b = norms[seed]
    return (n - d) / (n - b)


def score(seed, d):
    return float(np.clip(raw(seed, d), 0, 1))


def tag(k):
    return k.replace("1", "One").replace("5", "Five")


def rescore():
    for k, v in runs.items():
        if k != "viz":
            for s, r in v.items():
                r["raw"], r["score"] = raw(s, tot(r)), score(s, tot(r))
    alias = {"plan": "planner"}
    for base in sorted({alias.get(k.split("@")[0], k.split("@")[0]) for k in runs if "@" in k}):
        opts = [k for k in runs if "@" in k and alias.get(k.split("@")[0], k.split("@")[0]) == base and len(runs[k]) == 16]
        if base == "planner" and "plan@x" in opts:
            opts = ["plan@x"]
        if not opts:
            continue
        best = max(opts, key=lambda k: np.mean([r["score"] for r in runs[k].values()]))
        runs[base] = runs[best]
        if best[-1] in "12":
            LEVEL[base] = int(best[-1])
        for k in opts:
            if k[-1] in "12":
                mac(f"{tag(base)}M{'One' if k[-1] == '1' else 'Two'}", f"{np.mean([r['score'] for r in runs[k].values()]):.2f}")


def latest(pattern):
    fs = sorted(glob.glob(str(ROOT / pattern)), key=lambda f: Path(f).stat().st_mtime)
    return Path(fs[-1]) if fs else None


def rl_curves():
    out = {}
    for k in ("ppo", "gumbel"):
        f = latest(f"runs/rl/vm/{RLDIR}/runs/rl/{k}-*.json")
        if f:
            c = json.loads(f.read_text())
            out[k] = [(r.get("episodes", r["it"]), r["mean_net"]) for r in c if "mean_net" in r]
    return out


def grpo_curve():
    f = ROOT / "runs" / "grpo" / "curve.json"
    if not f.exists():
        return None
    g = json.loads(f.read_text())
    for e in g.get("evals", []):
        sc = []
        for x in e.get("seeds", []):
            n, b = norms[str(x["seed"])]
            sc.append(float(np.clip(x["greedy"]["saving"] * 1e6 / (n - b), 0, 1)) if x.get("greedy") else 0.0)
            x["score"] = sc[-1]
        if sc:
            e["score"] = float(np.mean(sc))
    ev = [e for e in g.get("evals", []) if e["step"] >= 0]
    z = [e for e in g.get("evals", []) if e["step"] < 0]
    g["zero"], g["ev"] = (z[0] if z else None), ev
    g["final"] = [x["score"] if x["score"] == x["score"] else 0.0 for x in ev[-1]["seeds"]] if ev else None
    return g


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
    cat = json.loads((ROOT / "data" / "catalogue.json").read_text())
    real = [p for p in cat if not p["id"].endswith("-access")]
    acc = [p for p in cat if p["id"].endswith("-access")]
    ax[1].scatter([p["years"] for p in real], [p["cost"] for p in real], s=14, color="#2a78d6", edgecolor="white", lw=0.5, zorder=3)
    ax[1].scatter([acc[0]["years"]], [acc[0]["cost"]], s=40, color="#6250d6", edgecolor="white", lw=0.5, zorder=3)
    ax[1].annotate(f"{len(acc)} access\nupgrades", (acc[0]["years"], acc[0]["cost"]), xytext=(8, 0), textcoords="offset points", fontsize=8, color="#6250d6", va="center")
    ax[1].text(3.1, 60, f"{len(real)} real\nprojects", fontsize=8, color="#2a78d6", va="center")
    ax[1].axhline(40, color=MUTED, lw=0.8, ls="--")
    ax[1].text(4.4, 30, "income/yr", fontsize=8, color=MUTED, ha="right")
    ax[1].set(title="(b) Project menu", xlabel="build years", ylabel="cost (MEUR)", xticks=[2, 3, 4], xlim=(1.6, 4.5), ylim=(0, 135))
    for s, c in zip(["0", "1", "2"], ["#9b9a94", INK, "#c3c2b7"]):
        d = np.array(runs["nothing"][s]["delay"]) / 1e6
        ax[2].plot(np.arange(Y0, Y0 + len(d)), d, color=c, lw=1.4, label=f"seed {s}")
    ax[2].set(title="(c) Delay, do nothing", ylabel="M person-hours / year", xticks=[2026, 2035, 2045])
    ax[2].legend(loc="upper left")
    fig.savefig(HERE / "problem.pdf")


def nothing_growth():
    rat = [runs["nothing"][s]["delay"][-1] / runs["nothing"][s]["delay"][0] for s in runs["nothing"]]
    mac("nothingRatioMin", f"{min(rat):.1f}")
    mac("nothingRatioMax", f"{max(rat):.1f}")


def vals_of(gr):
    vals = {"best": [1.0] * len(norms)}
    for k in KEYS:
        if k in runs:
            vals[k] = [runs[k][s]["score"] for s in sorted(runs[k], key=int)]
    if gr and gr.get("final"):
        vals["grpo"] = gr["final"]
    return vals


def fig_bars(vals):
    keys = [k for k in KEYS if k != "rlsearch" or k in vals]
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.9), gridspec_kw=dict(width_ratios=[1, 1], wspace=0.75))
    for i, k in enumerate(keys):
        y = len(keys) - 1 - i
        if k in vals:
            v = np.array(vals[k], float)
            ax[0].barh(y, v.mean(), 0.68, color=COL[k], zorder=2)
            ax[0].scatter(v, np.full(v.size, y) + np.linspace(-0.2, 0.2, v.size), s=3, color=INK, zorder=3, lw=0)
            ax[0].text(1.04, y, f"{v.mean():.2f}" + (f" (n={v.size})" if v.size != 16 else ""), va="center", fontsize=8, color=INK)
        else:
            ax[0].barh(y, 1.0, 0.68, color="white", edgecolor=TBD, hatch="///", lw=0.6, zorder=2)
            ax[0].text(1.04, y, "TBD", va="center", fontsize=8, color=TBD)
    ax[0].set_yticks(range(len(keys)), [NAME[k] for k in keys[::-1]])
    ax[0].set(xlim=(-0.02, 1.0), xlabel="normalized score (clipped to [0, 1])", title="(a) Score per seed (dots) and mean (bar)")
    ax[0].grid(axis="y", visible=False)
    ks = [k for k in ("nothing", "reactive", "agent", "planner", "rl", "best") if k in runs or k == "best"]
    w = 0.8 / len(ks)
    for j, k in enumerate(ks):
        v = [norms[s][1] / 1e6 if k == "best" else tot(runs[k][s]) / 1e6 for s in ASEEDS]
        ax[1].bar(np.arange(len(ASEEDS)) + (j - (len(ks) - 1) / 2) * w, v, w * 0.92, color=COL[k], label=NAME[k], zorder=2)
    ax[1].set_xticks(range(len(ASEEDS)), ASEEDS)
    ax[1].set_xlabel("seed")
    ax[1].set(ylabel="M person-hours, 2026 to 2045", title="(b) Net person-hours lost")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2, handlelength=1, columnspacing=0.8)
    ax[1].grid(axis="x", visible=False)
    fig.savefig(HERE / "bars.pdf")


def fig_years():
    seed = "1"
    src = {k: runs[k][seed] for k in ("nothing", "agent", "rl") if k in runs and seed in runs[k]}
    fig, ax = plt.subplots(figsize=(2.7, 2.2))
    for k, r in src.items():
        d = np.array(r["delay"]) / 1e6
        ax.plot(np.arange(Y0, Y0 + len(d)), d, color=COL[k], lw=1.6, label=NAME[k])
        for i, b in enumerate(r["builds"]):
            if k != "nothing" and any(p.endswith("-access") for p in b):
                ax.scatter(Y0 + i, d[i], marker="o", s=16, facecolor="white", edgecolor=COL[k], lw=1.2, zorder=5)
    ax.axvspan(Y0 + 15, Y0 + 19, color="#f1f0ec", lw=0, zorder=0)
    ax.text(Y0 + 17, 0.6, "runout", fontsize=8, color=MUTED, ha="center")
    ax.set(ylabel="M person-hours / year", ylim=(0, None), xticks=[2026, 2031, 2036, 2041, 2045])
    ax.legend(loc="upper left", handlelength=1.2)
    fig.savefig(HERE / "years.pdf")


def fig_curves(rl, gr):
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.3), gridspec_kw=dict(wspace=0.3))
    ref = {k: np.mean([r["score"] for r in runs[k].values()]) for k in ("agent", "reactive") if k in runs}
    for a in ax:
        for k, y in ref.items():
            a.axhline(y, color=COL[k], lw=0.8, ls="--")
            a.text(0.02, y + 0.01, NAME[k], transform=a.get_yaxis_transform(), fontsize=8, color=COL[k], ha="left", va="bottom")
        a.set_ylim(-0.05, 1.05)
    if rl:
        for k, c, lab in (("ppo", "#86b6ef", "PPO from scratch"), ("gumbel", "#6250d6", "Gumbel self-play from scratch")):
            if rl.get(k):
                x, y = zip(*rl[k])
                ax[0].plot(x, y, color=c, lw=1.4, marker="o", ms=2.5, label=lab)
        ax[0].legend(loc="lower right")
        ax[0].set(xlabel="training episodes", ylabel="mean score, 16 seeds")
    else:
        ax[0].text(0.5, 0.5, f"TBD: runs/rl/vm/{RLDIR}/runs/rl/", transform=ax[0].transAxes, ha="center", color=TBD)
    ax[0].set_title("(a) Learning without expert plans")
    if gr and gr["ev"]:
        ev = gr["ev"]
        ax[1].plot([e["step"] for e in ev], [e["score"] for e in ev], color=COL["grpo"], lw=1.4, marker="o", ms=2.5, label="greedy decoding")
        if gr["zero"]:
            ax[1].scatter([ev[0]["step"]], [gr["zero"]["score"] if gr["zero"]["score"] == gr["zero"]["score"] else 0.0], color=INK, s=12, zorder=4, label="zero-shot")
        ax[1].legend(loc="lower right")
        ax[1].set(xlabel="GRPO step", ylabel="mean score, 16 seeds")
    else:
        ax[1].text(0.5, 0.5, "TBD: runs/grpo/curve.json", transform=ax[1].transAxes, ha="center", color=TBD)
    ax[1].set_title("(b) GRPO fine-tuning of Gemma 3 270M")
    fig.savefig(HERE / "curves.pdf")


def table(vals):
    rows, n3 = [], sum(norms[s][0] for s in ASEEDS)
    pend = "\\pending{}"
    for k in KEYS:
        name = NAME[k] + (f" ($m{{=}}{LEVEL[k]}$)" if k in LEVEL and k != "nothing" else "")
        ext = ("", "")
        if k == "best":
            lost = [norms[s][1] for s in ASEEDS]
            last, spend = runs["viz"]["best"]["delay"][-1] / 1e6, sum(runs["viz"]["best"]["spent"])
            ext = ("$^\\dagger$", "$^\\dagger$")
        elif k in runs and all(s in runs[k] for s in ASEEDS):
            lost = [norms[s][0] if k == "nothing" else tot(runs[k][s]) for s in ASEEDS]
            last = np.mean([runs[k][s]["delay"][-1] for s in ASEEDS]) / 1e6
            spend = np.mean([sum(runs[k][s]["spent"]) for s in ASEEDS])
        else:
            s16 = f"{np.mean(vals[k]):.2f}" if k in vals and len(vals[k]) == 16 else pend
            rows.append(f"{name} & {s16} & " + " & ".join([pend] * 6) + " \\\\")
            if k in vals and len(vals[k]) == 16:
                mac(f"{tag(k)}Score", s16)
            continue
        L = sum(lost)
        s16 = f"{abs(np.mean(vals[k])):.2f}" if k in vals and len(vals[k]) == 16 else "n/a"
        s3 = np.mean([score(s, d) for s, d in zip(ASEEDS, lost)])
        rows.append(f"{name} & {s16} & {s3:.2f} & {L / 1e6:.1f} & {(n3 - L) / 1e6:.1f} & {100 * (n3 - L) / n3:.1f} & {last:.1f}{ext[0]} & {spend:.0f}{ext[1]} \\\\")
        mac(f"{tag(k)}Lost", f"{L / 1e6:.1f}")
        mac(f"{tag(k)}Saved", f"{(n3 - L) / 1e6:.1f}")
        mac(f"{tag(k)}Pct", f"{100 * (n3 - L) / n3:.1f}")
        mac(f"{tag(k)}ScoreAgentSeeds", f"{s3:.2f}")
        if s16 != "n/a":
            mac(f"{tag(k)}Score", s16)
    (HERE / "table.tex").write_text("\n".join(rows) + "\n")


def facts():
    a = runs.get("agent", {})
    if a:
        usd = [r["usd"] for r in a.values() if r.get("usd")]
        mins = [r["minutes"] for r in a.values() if r.get("minutes")]
        if usd:
            mac("agentUsdMin", f"{min(usd):.2f}")
            mac("agentUsdMax", f"{max(usd):.2f}")
        if mins:
            mac("agentMinMin", f"{min(mins):.0f}")
            mac("agentMinMax", f"{max(mins):.0f}")
        sc = [r["score"] for r in a.values()]
        mac("agentScoreMin", f"{min(sc):.2f}")
        mac("agentScoreMax", f"{max(sc):.2f}")
        mac("agentScoreMean", f"{np.mean(sc):.2f}")
        mac("agentN", str(len(sc)))
        mac("agentSeeds", f"{min(a, key=int)} to {max(a, key=int)}")
        gs = [r["calls"].get("espoo_get_state", 0) for r in a.values() if "calls" in r]
        fc = [r["calls"].get("espoo_forecast", 0) for r in a.values() if "calls" in r]
        if gs:
            mac("agentGetStateMin", str(min(gs)))
            mac("agentGetStateMax", str(max(gs)))
            mac("agentForecastMin", str(min(fc)))
            mac("agentForecastMax", str(max(fc)))
        mx = [max(r["maint"][:15]) == min(r["maint"][:15]) == 2 for r in a.values()]
        mac("agentMaxMaint", str(sum(mx)))
        booms = {s: [n for n, r in growth[s]["rate"].items() if r > 0.01] for s in a if s in growth}
        mac("nBooms", str(sum(len(b) for b in booms.values())))
        key = lambda n: n.split()[0].lower().replace("ä", "a")
        hit = early = 0
        for s, bs in booms.items():
            for i, b in enumerate(a[s]["builds"]):
                for p in b:
                    if p.endswith("-access") and any(p.startswith(key(n)) for n in bs):
                        hit += 1
                        early += i <= 5
        mac("agentBoomHits", str(hit))
        mac("agentBoomEarly", str(early))
        acc = [p for s in a for b in a[s]["builds"] for p in b if p.endswith("-access")]
        mac("agentAccessBuilt", str(len(acc)))
        mac("agentOtaniemi", str(sum(p == "otaniemi-access" for p in acc)))
        for s in a:
            L = "ABCDEFGH"[int(s)]
            mac(f"agentScore{L}", f"{a[s]['score']:.2f}")
            mac(f"agentLost{L}", f"{tot(a[s]) / 1e6:.1f}")
            mac(f"nothingLost{L}", f"{norms[s][0] / 1e6:.1f}")
            mac(f"bestLost{L}", f"{norms[s][1] / 1e6:.1f}")
            if "rl" in runs:
                mac(f"rlLost{L}", f"{tot(runs['rl'][s]) / 1e6:.1f}")
                mac(f"gapRl{L}", f"{(tot(a[s]) - tot(runs['rl'][s])) / 1e6:.0f}")
    f = [(n - b) / n for n, b in norms.values()]
    mac("bestPctMin", f"{100 * min(f):.1f}")
    mac("bestPctMax", f"{100 * max(f):.1f}")
    mac("bestPctMean", f"{100 * np.mean(f):.1f}")
    for k in ("rl", "planner", "rlcheck"):
        if k in runs:
            r = [x["raw"] for x in runs[k].values()]
            mac(f"{k}Beats", str(sum(x > 1 for x in r)))
            mac(f"{k}RawMax", f"{max(r):.2f}")
            mac(f"{k}Min", f"{min(x['score'] for x in runs[k].values()):.2f}")
    if "refrl" in runs and "rl" in runs:
        d = [abs(tot(runs["refrl"][s]) - tot(runs["rl"][s])) / tot(runs["rl"][s]) for s in runs["refrl"] if s in runs["rl"]]
        mac("refrlN", str(len(d)))
        mac("refrlGap", f"{100 * max(d):.2f}")
    nothing = [r["score"] for r in runs["nothing"].values()]
    mac("nothingAbsMax", f"{max(abs(x) for x in [r['raw'] for r in runs['nothing'].values()]):.3f}")


def cats():
    cat = json.loads((ROOT / "data" / "catalogue.json").read_text())
    groups = {"New roads and bridges": [p for p in cat if p["kind"] == "road"],
              "Widenings (extra lanes)": [p for p in cat if p["kind"] == "lanes" and not p["id"].endswith("-access")],
              "Junction rebuilds": [p for p in cat if p["kind"] == "junction"],
              "Area access upgrades": [p for p in cat if p["id"].endswith("-access")]}
    rng = lambda v: f"{min(v):g}" if min(v) == max(v) else f"{min(v):g} to {max(v):g}"
    rows = [f"{k} & {len(v)} & {rng([p['cost'] for p in v])} & {rng([p['years'] for p in v])} \\\\" for k, v in groups.items()]
    (HERE / "cats.tex").write_text("\n".join(rows) + "\n")


def main():
    rescore()
    rl, gr = rl_curves(), grpo_curve()
    if gr and gr["ev"]:
        if gr["zero"]:
            mac("grpoZero", f"{gr['zero']['score'] if gr['zero']['score'] == gr['zero']['score'] else 0.0:.2f}")
        mac("grpoWarm", f"{gr['ev'][0]['score']:.2f}")
        mac("grpoFinal", f"{gr['ev'][-1]['score']:.2f}")
        mac("grpoValid", f"{gr['ev'][-1]['valid']:.2f}")
        mac("grpoSteps", str(gr["ev"][-1]["step"]))
        booms = json.loads((DATA / "booms.json").read_text())
        key = lambda n: n.split()[0].lower().replace("ä", "a")
        for nm, e in (("Warm", gr["ev"][0]), ("Final", gr["ev"][-1])):
            plans = [tuple((x.get("greedy") or {}).get("plan", [])) for x in e["seeds"]]
            acc = [(x["seed"], q) for x, p in zip(e["seeds"], plans) for q in p if q.endswith("-access")]
            mac(f"grpoDistinct{nm}", str(len(set(plans))))
            mac(f"grpoAccess{nm}", str(len(acc)))
            mac(f"grpoBoom{nm}", str(sum(any(q.startswith(key(n)) for n in booms[str(s)]) for s, q in acc)))
        last = [x.get("greedy") or {} for x in gr["ev"][-1]["seeds"]]
        mac("grpoPlan", ", ".join(f"\\texttt{{{q}}}" for q in max(set(tuple(g.get("plan", [])) for g in last), key=lambda p: sum(tuple(g.get("plan", [])) == p for g in last))))
        mac("grpoMaint", str(sum(g.get("maintain") == 2 for g in last)))
    vals = vals_of(gr)
    nothing_growth()
    fig_problem()
    fig_bars(vals)
    fig_years()
    fig_curves(rl, gr)
    table(vals)
    facts()
    cats()
    used = set(re.findall(r"\\([a-zA-Z]+)\{\}", (HERE.parent / "main.tex").read_text()))
    for u in sorted(used):
        if re.match(r"(planner|reactive|greedy|random|agent|best|nothing|grpo|rl|rlcheck|rlsearch|refrl|gapRl|n|cat)[A-Z]", u) and u not in macros:
            macros[u] = "\\pending{}"
    (HERE / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print(json.dumps(macros, indent=0))


if __name__ == "__main__":
    main()
